import torch
import numpy as np
import matplotlib.pyplot as plt

def RK4_multistep_integrator(deriv, dt, x, n_steps=1):
    for _ in range(n_steps):
        k1 = (dt/n_steps) * deriv(x)            # t=0
        k2 = (dt/(2*n_steps)) * deriv(x+k1/2)   # t=dt/2
        k3 = (dt/(2*n_steps)) * deriv(x+k2/2)   # t=dt/2
        k4 = (dt/n_steps) * deriv(x+k3)         # t=dt
        x = x + (k1 + 2*k2 + 2*k3 + k4)/6
    return x

def multisine_generator(t, freq_band, amplitude, n_states):
    u = torch.zeros(len(t), n_states)
    for state in range(n_states):
        for freq in freq_band:
            phase = 2*torch.pi*torch.rand(1)
            wave = torch.sin(freq*t + phase)
            u[:, state] = u[:, state] + wave
    u = u*(amplitude/torch.max(u))
    return u


### Blockify functions ###
def blockify_G(system_dim, theta):
    batch_size = theta.shape[0]
    xc_dim = np.sum(system_dim, 0)[0]
    sigc_dim = np.sum(system_dim, 0)[1]
    past_x_dim, past_u_dim, past_vals = 0, 0, 0

    G_mat = torch.zeros(batch_size, xc_dim, sigc_dim)
    for dim in system_dim:
        # Select vals for system in question
        nG = dim[0]*dim[1]

        # Assign values to correct location
        G_mat[:, past_x_dim:past_x_dim+dim[0], past_u_dim:past_u_dim+dim[1]] = theta[:, past_vals:past_vals+nG].view(batch_size, dim[0], dim[1])

        # Update counters
        past_x_dim += dim[0]
        past_u_dim += dim[1]
        past_vals += nG
    return G_mat


def blockify_J(system_dim, theta):
    batch_size = theta.shape[0]
    xc_dim = np.sum(system_dim, 0)[0]
    past_x_dim, past_vals = 0, 0

    J_mat = torch.zeros(batch_size, xc_dim, xc_dim)
    for dim in system_dim:
        # Select vals for system in question
        nJ = int((dim[0]**2 - dim[0]) / 2)
        c_theta = theta[:, past_vals:past_vals+nJ]

        # Assign values to correct locations
        if nJ > 0: # Dodge the 1D case
            for i in range(dim[0]):
                J_mat[:, past_x_dim+i, past_x_dim+1+i:past_x_dim+dim[0]] = c_theta[:, 0:dim[0]-(i+1)]
                J_mat[:, past_x_dim+1+i:past_x_dim+dim[0], past_x_dim+i] = -c_theta[:, 0:dim[0]-(i+1)]
                c_theta = c_theta[:, dim[0]-(i+1):]

        # Update counters
        past_x_dim += dim[0]
        past_vals += nJ
    return J_mat


def blockify_R(system_dim, theta):
    # TODO: Adapt this to minimize free parameters in R
    batch_size = theta.shape[0]
    xc_dim = np.sum(system_dim, 0)[0]
    past_x_dim, past_vals = 0, 0

    R_mat = torch.zeros(batch_size, xc_dim, xc_dim)
    for dim in system_dim:
        # Select vals for system in question
        # Theoretically only need: nR = int((dim[0]**2 - dim[0]) / 2)
        nR = dim[0]*dim[0]
        block = theta[:, past_vals:past_vals+nR].view(batch_size, dim[0], dim[0])

        # Assign values to correct location
        R_mat[:, past_x_dim:past_x_dim+dim[0], past_x_dim:past_x_dim+dim[0]] = torch.einsum('bik,bjk->bij', block, block)

        # Update counters
        past_x_dim += dim[0]
        past_vals += nR
    return R_mat


def blockify_H(system_dim, theta):
    batch_size = theta.shape[0]
    xc_dim = np.sum(system_dim, 0)[0]
    past_x_dim = 0

    H_mat = torch.zeros(batch_size, xc_dim)
    for dim in system_dim:
        # Assign values to correct location
        H_mat[:, past_x_dim:past_x_dim+dim[0]] = theta[:, past_x_dim:past_x_dim+dim[0]]

        # Update counters
        past_x_dim += dim[0]
    return H_mat