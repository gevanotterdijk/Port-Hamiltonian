import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

### General utility ###
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


### Variable PHNN subnetworks ###
class feed_forward_nn(nn.Module): # Standard MLP (Same as in deepSI)
    def __init__(self,n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(feed_forward_nn,self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        seq = [nn.Linear(n_in,n_nodes_per_layer), activation()]
        assert n_hidden_layers>0
        for i in range(n_hidden_layers-1):
            seq.append(nn.Linear(n_nodes_per_layer, n_nodes_per_layer))
            seq.append(activation())
        seq.append(nn.Linear(n_nodes_per_layer, n_out))
        self.net = nn.Sequential(*seq)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, val=0)
    def forward(self,X):
        return self.net(X)  


class var_J_net(nn.Module):
    def __init__(self, system_dim, net=feed_forward_nn, net_kwargs={}):
        super().__init__()
        self.system_dim = system_dim
        self.xc_dim = np.sum(system_dim, 0)[0]
        self.net_list = nn.ModuleList()

        # Create separated subnets:
        for sys, dim in enumerate(system_dim):
            nJ = int((dim[0]**2 - dim[0]) / 2)
            if nJ > 0:  # To avoid the single state system, where J has 0 elements
                self.net_list.append(net(n_in=dim[0], n_out=nJ, **net_kwargs))
    
    def forward(self, x):
        batch_size = x.shape[0]
        past_x_dim = 0
        J_mat = torch.zeros(batch_size, self.xc_dim, self.xc_dim)

        # TODO: Check speed difference between assigning parameters to J in a loop, or doing the J - J.permute(0, 2, 1)
        for sys, dim in enumerate(self.system_dim):
            nJ = int((dim[0]**2 - dim[0]) / 2) # TODO: move nJ calc out of the for loop
            if nJ > 0:  # To avoid the single state system, where J has 0 elements
                # Apply the subnetworks to find the parameters j_{ab}
                c_theta = self.net_list[sys](x[:, past_x_dim:past_x_dim+dim[0]])

                for i in range(dim[0]):
                    J_mat[:, past_x_dim+i, past_x_dim+1+i:past_x_dim+dim[0]] = c_theta[:, 0:dim[0]-(i+1)]
                    J_mat[:, past_x_dim+1+i:past_x_dim+dim[0], past_x_dim+i] = -c_theta[:, 0:dim[0]-(i+1)]
                    c_theta = c_theta[:, dim[0]-(i+1):]
            past_x_dim += dim[0]
        return J_mat


class var_R_net(nn.Module):
    def __init__(self, system_dim, net=feed_forward_nn, net_kwargs={}):
        super().__init__()
        self.system_dim = system_dim
        self.xc_dim = np.sum(system_dim, 0)[0]
        self.net_list = nn.ModuleList()

        # Create separated subnets:
        for sys, dim in enumerate(system_dim):
            nR = dim[0]*dim[0]
            self.net_list.append(net(n_in=dim[0], n_out=nR, **net_kwargs))
    
    def forward(self, x):
        batch_size = x.shape[0]
        past_x_dim = 0
        R_mat = torch.zeros(batch_size, self.xc_dim, self.xc_dim)

        for sys, dim in enumerate(self.system_dim):
            # Apply the subnetworks to find the matrices R_{i, theta_i}(x_i)
            block = self.net_list[sys](x[:, past_x_dim:past_x_dim+dim[0]]).view(batch_size, dim[0], dim[0])

            # Assign the matrices R_{i, theta_i}(x_i) to the correct locations in the composite system matrix R_{c, theta_c}(x_c)
            # To ensure the symmetric, positive semi-definiteness of the final matrix, the square of the blocks is taken
            R_mat[:, past_x_dim:past_x_dim+dim[0], past_x_dim:past_x_dim+dim[0]] = torch.einsum('bik,bjk->bij', block, block)
            past_x_dim += dim[0]
        return R_mat


class var_G_net(nn.Module):
    def __init__(self, system_dim, net=feed_forward_nn, net_kwargs={}):
        super().__init__()
        self.system_dim = system_dim
        self.xc_dim = np.sum(system_dim, 0)[0]
        self.sigc_dim = np.sum(system_dim, 0)[1]
        self.net_list = nn.ModuleList()

        for sys, dim in enumerate(system_dim):
            nG = dim[0]*dim[1]
            self.net_list.append(net(n_in=dim[0], n_out=nG, **net_kwargs))
    
    def forward(self, x):
        batch_size = x.shape[0]
        past_x_dim = 0
        past_u_dim = 0
        G_mat = torch.zeros(batch_size, self.xc_dim, self.sigc_dim)

        for sys, dim in enumerate(self.system_dim):
            # Apply the subnetworks to find the matrices G_{i, theta_i}(x_i)
            block = self.net_list[sys](x[:, past_x_dim:past_x_dim+dim[0]]).view(batch_size, dim[0], dim[1])

            # Assign the matrices G_{i, theta_i}(x_i) to the correct locations in the composite system matrix G_{c, theta_c}(x_c)
            G_mat[:, past_x_dim:past_x_dim+dim[0], past_u_dim:past_u_dim+dim[1]] = block

            past_x_dim += dim[0]
            past_u_dim += dim[1]
        return G_mat
            

class var_H_net(nn.Module):
    def __init__(self, system_dim, net=feed_forward_nn):
        super().__init__()
        self.system_dim = system_dim
        self.xc_dim = np.sum(system_dim, 0)[0]
        self.net_list = nn.ModuleList()

        for sys, dim in enumerate(system_dim):
            self.net_list.append(net(n_in=dim[0], n_out=dim[0]))
    
    def forward(self, x):
        batch_size = x.shape[0]
        past_x_dim = 0
        H_vec = torch.zeros(batch_size, self.xc_dim)

        for sys, dim in enumerate(self.system_dim):
            # Apply the subnetworks to find values for H_{i, theta_i}(x_i)
            vals = self.net_list[sys](x[:, past_x_dim:past_x_dim+dim[0]])

            # Assign the vectors H_{i, theta_i}(x_i) to the correct location in the composite Hamiltonian H_{c, theta_c}(x_c)
            H_vec[:, past_x_dim:past_x_dim+dim[0]] = vals
            past_x_dim += dim[0]
        return H_vec