import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import RK4_multistep_integrator, multisine_generator, DK_matrix_form

from  deepSI_lite import Input_output_data

"""
Purpose: Generate input/output data for multiple connected mass spring dampers.
"""

# Define system properties
class coupled_MSD():
    def __init__(self, M_vals, D_vals, K_vals, dt):
        assert M_vals.shape[0] == D_vals.shape[0] == K_vals.shape[0] # Check that all the vectors have the same size
        self.dt = dt
        self.n_sys = M_vals.shape[0]
        self.M_mat = torch.diag(M_vals)
        self.D_mat = DK_matrix_form(D_vals)
        self.K_mat = DK_matrix_form(K_vals)

        # Setup the A matrix
        self.A = torch.zeros([2*self.n_sys, 2*self.n_sys])
        self.A[:self.n_sys, self.n_sys:] = torch.eye(self.n_sys)
        self.A[self.n_sys:, :self.n_sys] = -torch.eye(self.n_sys)
        self.A[self.n_sys:, self.n_sys:] = -self.D_mat
        self.G = torch.cat((torch.zeros([self.n_sys, self.n_sys]), torch.eye(self.n_sys)), dim=0)

    def hamiltonian(self, x):
        # Position dependent grad Hamiltonian (i.e. Kq)
        q = x[:self.n_sys]
        dH_q = torch.einsum("ij, j -> i", self.K_mat, q)
        # Momentum dependent grad Hamiltonian (i.e. M^-1p)
        p = x[self.n_sys:]
        dH_p = torch.einsum("ij, j -> i", torch.inverse(self.M_mat), p)
        return torch.cat((dH_q, dH_p), dim=0)


    def step(self, x, u):
        dHdx = self.hamiltonian(x)
        yhat = torch.einsum('ij, i -> j', self.G, dHdx) # Output according to G^T dHdx

        def state_deriv(xnow):
            dHdx = self.hamiltonian(xnow)
            deriv = torch.einsum("ij, j -> i", self.A, dHdx) + torch.einsum("ij, j -> i", self.G, u)
            return deriv
        
        x_next = RK4_multistep_integrator(deriv=state_deriv, dt=self.dt, x=x)
        return x_next, yhat

    def print_arguments(self):
        print(f"Number of systems: {self.n_sys}")
        print(type(self.n_sys))
        print(f"Mass matrix: {self.M_mat.shape}")
        print(self.M_mat)
        print(f"Damping matrix: {self.D_mat.shape}")
        print(self.D_mat)
        print(f"Spring matrix: {self.K_mat.shape}")
        print(self.K_mat)
        print(f"Composite A matrix: {self.A.shape}")
        print(self.A)
        print(f"Composite G matrix: {self.G.shape}")
        print(self.G)


# Simulation of the coupled MSD system
def run_sim(sim_time, sys:coupled_MSD, x0:torch.FloatTensor, u_ext:torch.FloatTensor):
    s = torch.zeros(len(sim_time), x0.shape[0])     # Empty state tensor
    y = torch.zeros(len(sim_time), u_ext.shape[1])  # Empty output tensor
    x = x0

    for step, t_i in enumerate(sim_time):
        s[step, :] = x
        x, y[step, :] = sys.step(x, u_ext[step, :])
    return s, y



if __name__ == "__main__":
    ### ======= SETTINGS ======== ###
    sim_time = torch.linspace(0, 100, 1024)
    # Define an arbitrary system
    M_vals = torch.FloatTensor([2, 2, 2])
    D_vals = torch.FloatTensor([1, 1, 1])
    K_vals = torch.FloatTensor([0.5, 0.5, 0.5])
    sys = coupled_MSD(M_vals=M_vals, D_vals=D_vals, K_vals=K_vals, dt=sim_time[1])

    # Dataset specifications
    noise = True
    noise_sd = 0.025
    n_datasets = 8
    x0 = torch.zeros(sys.n_sys*2)
    freq_band = torch.linspace(0.1, 7, 40)
    input_mask = torch.zeros(sys.n_sys)
    input_mask[0] = 1   # Which masses to excite
    
    
    ### ====== PROCESSING ======= ###
    # Generate the datasets
    datasets = []
    for i in range(0, n_datasets):
        # Generate input signal
        u = multisine_generator(sim_time, freq_band, amplitude=1, n_inputs=sys.n_sys)
        inputs = torch.einsum("ij, j -> ij", u, input_mask)
        # Apply input and capture output
        states, output = run_sim(sim_time, sys, x0, inputs)
        # Save simulation as a list of dataset dictionaries
        dsi_IO = Input_output_data(u=inputs.numpy(), y=output.numpy(), sampling_time=(sim_time[1]-sim_time[0]))
        dataset_dict = {
            "inputs":inputs,
            "states":states,
            "output":output,
            "dsi_IO":dsi_IO
        }
        # In case of noisy measurements, save both true and noisy measurements and take the noisy output for the dsi_IO 
        if noise == True:
            max_out = torch.max(output)
            noisy_output = (noise_sd*max_out*torch.randn_like(output) - 0.5*noise_sd*max_out*torch.ones_like(output)) + output  # Note the gaussian noise
            dataset_dict["dsi_IO"] = Input_output_data(u=inputs.numpy(), y=noisy_output.numpy(), sampling_time=(sim_time[1]-sim_time[0]))
            dataset_dict["noisy_output"] = noisy_output
        datasets.append(dataset_dict)

    # Plot the system behaviour
    fig, (ax0, ax1, ax2) = plt.subplots(sys.n_sys, 1, layout='constrained') #TODO, automate ax1, ax2, ax3
    for i in range(0, sys.n_sys):
        # globals()["ax%s" % i] basically says "axi" for looped plotting
        globals()["ax%s" % i].plot(sim_time, torch.zeros_like(sim_time), "k--",label='_nolegend_')  # Plot 0-line
        globals()["ax%s" % i].plot(sim_time, states[:, i])                                          # Plot state q_i
        globals()["ax%s" % i].plot(sim_time, states[:, sys.n_sys+i])                                # Plot state p_i
        globals()["ax%s" % i].plot(sim_time, output[:, i])                                          # Plot output y_i
        globals()["ax%s" % i].scatter(sim_time, noisy_output[:, i], s=0.5, color="black")           # Plot noisy outputs y_1
        globals()["ax%s" % i].legend([f'$q_{i+1}$', f'$p_{i+1}$', f'$y_{i+1}$'], loc=1)             # Add legends
        globals()["ax%s" % i].set_xlim([0, max(sim_time)])                                          # Set xlimits
    plt.show()
    
    # Export the dataset to torch file
    PATH_DATA = "TEST_DATASET_GENERATED.pt"
    torch.save(datasets, "datasets/" + PATH_DATA)

    # MATLAB EXPORTS:
    inputs = datasets[0]["inputs"].numpy()
    np.savetxt("inputs.csv", inputs, delimiter=",")
    outputs = datasets[0]["output"].numpy()
    np.savetxt("clean_outputs.csv", outputs, delimiter=",")
    n_outputs = datasets[0]["noisy_output"].numpy()
    np.savetxt("outputs.csv", n_outputs, delimiter=",")

    sys.print_arguments()