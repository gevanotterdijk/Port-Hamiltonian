import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import RK4_multistep_integrator, multisine_generator

"""
Purpose: Generate input/output data for multiple connected mass spring dampers.
"""

# Define system properties
class coupled_MSD():
    def __init__(self, n_systems, system_MDK):
        self.n_systems = n_systems
        self.system_MDK = system_MDK

        # Init PH matrices
        self.J = torch.zeros([2*self.n_systems, 2*self.n_systems])  # Structure matrix
        self.R = torch.zeros([2*self.n_systems, 2*self.n_systems])  # Dissipation matrix
        self.C = torch.zeros([2*self.n_systems, 2*self.n_systems])  # Interconnection matrix
        self.G = torch.eye(2*self.n_systems)                        # Input matrix (Assumed to be identity)

        for sys in range(n_systems):
            # Init internal position <--> momentum influence
            self.J[2*sys, 2*sys+1] = 1
            self.J[2*sys+1, 2*sys] = -1

            # Init dampers
            try:
                self.R[2*sys+1, 2*sys+1] = system_MDK[sys, 1] + system_MDK[sys+1, 1]    # Internal damping
                self.C[2*sys+1, 2*sys+3] = -system_MDK[sys+1, 1]                        # Damped connection between sys i and sys i+1 (for cubic damping scenario)
                self.C[2*sys+3, 2*sys+1] = -system_MDK[sys+1, 1]                        # Damped connection between sys i+1 and sys i (for cubic damping scenario)
                print(f"Both damper {sys+1} and {sys+2} present")
            except:
                self.R[2*sys+1, 2*sys+1] = system_MDK[sys, [1]]
                print(f"Only damper {sys+1} present")

    def hamiltonian(self, x):
        dH = torch.zeros(2*self.n_systems)
        for sys in range(self.n_systems):
            dH[2*sys] = self.system_MDK[sys, 2]*x[2*sys]            # ki * qi
            dH[2*sys + 1] = x[(2*sys)+1] / self.system_MDK[sys, 0]  # pi / mi
        return dH

    def step(self, x, u):
        dHdx = self.hamiltonian(x)
        yhat = torch.einsum('ji, j -> i', self.G, dHdx) # Output according to G^T dHdx

        def state_deriv(xnow):
            dHdx = self.hamiltonian(xnow)
            deriv = torch.einsum("ij, j -> i", self.J - self.R + self.C, dHdx) + torch.einsum("ij, j -> i", self.G, u)
            return deriv
        
        x_next = RK4_multistep_integrator(deriv=state_deriv, dt=0.1, x=x)
        return x_next, yhat

    def print_arguments(self):
        print(f"Number of systems: {self.n_systems}")
        print(type(self.n_systems))
        print(f"MDK Matrix: {self.system_MDK}")
        print(type(self.system_MDK))
        print(f"Composite J matrix: {self.J}")
        print(self.J.shape)
        print(f"Composite R matrix: {self.R}")
        print(self.R.shape)
        print(f"Composite G matrix: {self.G}")
        print(self.G.shape)
        print(f"Composite C matrix: {self.C}")
        print(self.C.shape)

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
    # Dataset properties
    sim_time = torch.linspace(0, 800, 8192)
    noise = True
    n_datasets = 2

    # System properties
    # Declare MSD settings in the form [Mass, Damping, Spring] constants
    S1 = [2, 0.5, 1]
    S2 = [2, 0.5, 1]
    S3 = [2, 0.5, 1]
    system_tensor = torch.FloatTensor([S1])
    n_systems = system_tensor.shape[0]
    sys = coupled_MSD(n_systems, system_tensor)
    x0 = torch.zeros(n_systems*2)
    freq_band = torch.linspace(0.1, 7, 40)

    # Generate the datasets
    datasets = torch.zeros(n_datasets, 3, sim_time.shape[0], 2*n_systems)
    input_mask = torch.zeros(2*n_systems)
    input_mask[1] = 1
    for i in range(0, n_datasets):
        # Generate input signal
        u = multisine_generator(sim_time, freq_band, amplitude=5, n_states=2*system_tensor.shape[0])
        inputs = torch.einsum("ij, j -> ij", u, input_mask)

        # Apply input and capture output
        states, outputs = run_sim(sim_time, sys, x0, inputs)
        if noise == True:
            max_out = torch.max(outputs)
            print(f"STD of noise at: {0.025}, max= {max_out}")
            noisy_outputs = (0.025*torch.randn_like(outputs) - 0.0125*max_out*torch.ones_like(outputs)) + outputs
            datasets[i, :, :, :] = torch.stack([inputs, states, noisy_outputs])
        else:
            datasets[i, :, :, :] = torch.stack([inputs, states, outputs])

    # Plot the system behaviour
    fig, (ax0) = plt.subplots(n_systems, 1, layout='constrained') #TODO, ax1, ax2, ax3
    for i in range(0, n_systems):
        # globals()["ax%s" % i] basically says "axi" for looped plotting
        globals()["ax%s" % i].plot(sim_time, states[:, (2*i):(2*i)+2])                  # Plot states
        globals()["ax%s" % i].plot(sim_time, outputs[:, (2*i)+1])                       # Plot output
        globals()["ax%s" % i].scatter(sim_time, noisy_outputs[:, (2*i)+1], s=0.5, color="black")
        globals()["ax%s" % i].legend([f'$q_{i+1}$', f'$p_{i+1}$', f'$y_{i+1}$'], loc=1) # Add legends
        globals()["ax%s" % i].set_xlim([0, max(sim_time)])                              # Set xlimits
    plt.show()
    
    # Export the dataset to torch file
    PATH_DATA = "TEST_DATASET_GENERATED.pt"
    torch.save(datasets, "datasets/" + PATH_DATA)
    sys.print_arguments()
    print(sys.J - sys.R + sys.C)