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
    # Initialise a test system
    # Declare MSD settings in the form [Mass, Damping, Spring] constants
    S1 = [2, 0.5, 1]
    S2 = [5, 0.2, 1]
    S3 = [2, 0.7, 1]
    S4 = [5, 0.1, 1]
    system_tensor = torch.FloatTensor([S1, S2, S3, S4])
    sys = coupled_MSD(system_tensor.shape[0], system_tensor)

    # Example simulation:
    sim_time = torch.linspace(0, 20, 1001)
    x0 = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8])

    # Generate random phase multisine input
    freq_band = torch.linspace(0, 1, 7)
    inputs = multisine_generator(sim_time, freq_band, amplitude=7, n_states=2*system_tensor.shape[0])
    input_mask = torch.FloatTensor([0, 1, 0, 1, 0, 1, 0, 1])  # Determine which states have an input
    u = torch.einsum("ij, j -> ij", inputs, input_mask)

    # Run the simulation
    states, outputs = run_sim(sim_time, sys, x0, u)

    # Plot the system behaviour
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, layout='constrained')
    for i in range(0, 4):
        # globals()["ax%s" % i] basically says "axi" for looped plotting
        globals()["ax%s" % i].plot(sim_time, states[:, (2*i):(2*i)+2])                  # Plot states
        globals()["ax%s" % i].plot(sim_time, outputs[:, (2*i)+1])                       # Plot output
        globals()["ax%s" % i].legend([f'$q_{i+1}$', f'$p_{i+1}$', f'$y_{i+1}$'], loc=1) # Add legends
        globals()["ax%s" % i].set_xlim([0, max(sim_time)])                              # Set xlimits
    plt.show()
    