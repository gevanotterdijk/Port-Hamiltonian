import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import deepSI_lite as dsi
from deepSI_lite.models import Custom_SUBNET_CT
from deepSI_lite.networks import MLP_res_net


### General utility ###
def RK4_multistep_integrator(deriv, dt, x, n_steps=1):
    for _ in range(n_steps):
        k1 = (dt/n_steps) * deriv(x)            # t=0
        k2 = (dt/(2*n_steps)) * deriv(x+k1/2)   # t=dt/2
        k3 = (dt/(2*n_steps)) * deriv(x+k2/2)   # t=dt/2
        k4 = (dt/n_steps) * deriv(x+k3)         # t=dt
        x = x + (k1 + 2*k2 + 2*k3 + k4)/6
    return x


def RK4_multistep_integrator_with_u(deriv, dt, x, u, n_steps=1):
    for _ in range(n_steps):
        k1 = (dt/n_steps) * deriv(x, u)            # t=0
        k2 = (dt/(2*n_steps)) * deriv(x+k1/2, u)   # t=dt/2
        k3 = (dt/(2*n_steps)) * deriv(x+k2/2, u)   # t=dt/2
        k4 = (dt/n_steps) * deriv(x+k3, u)         # t=dt
        x = x + (k1 + 2*k2 + 2*k3 + k4)/6
    return x


def multisine_generator(t, freq_band, amplitude, n_inputs):
    u = torch.zeros(len(t), n_inputs)
    for state in range(n_inputs):
        for freq in freq_band:
            phase = 2*torch.pi*torch.rand(1)
            wave = torch.sin(freq*t + phase)
            u[:, state] = u[:, state] + wave
    u = u*(amplitude/torch.max(u))
    return u


def DK_matrix_form(vals):
    dim = len(vals)
    mat = torch.zeros(dim, dim)

    for i in range(dim-1):
        mat[i, i] = vals[i]+vals[i+1]
        mat[i, i+1] = mat[i+1, i] = -vals[i+1]
        mat[i+1, i+1] = vals[i+1]
    return mat


def simulate_model(sim_time, model:Custom_SUBNET_CT, x0:torch.FloatTensor, u_ext:torch.FloatTensor):
    s = torch.zeros(len(sim_time), x0.shape[0])     # Empty state tensor
    y = torch.zeros(len(sim_time), u_ext.shape[1])  # Empty output tensor
    x = x0
        
    for step, t_i in enumerate(sim_time):
        s[step, :] = x.detach()
        _, _, G, dHdx = model.get_matrices(x.view(1, -1))
        y[step, :] = torch.einsum('bij, bi -> j', G, dHdx).detach()

        def state_derivative(xnow):
            J, R, G, dHdx = model.get_matrices(xnow.view(1, -1))
            deriv = torch.einsum("bij, bj -> i", J-R, dHdx) + torch.einsum("bij, j -> i", G, u_ext[step, :])
            return deriv
        
        x = RK4_multistep_integrator(deriv=state_derivative, dt=sim_time[1], x=x)
    return s, y


def simulate_model_withP(sim_time, model:Custom_SUBNET_CT, x0:torch.FloatTensor, u_ext:torch.FloatTensor):
    s = torch.zeros(len(sim_time), x0.shape[0])     # Empty state tensor
    y = torch.zeros(len(sim_time), u_ext.shape[1])  # Empty output tensor
    x = x0
        
    for step, t_i in enumerate(sim_time):
        s[step, :] = x.detach()
        _, _, G, dHdx, P = model.get_matrices(x.view(1, -1))            # No need for the complex transposes, as we are only working with real J, R, G, Q, P, S, N estimates
        y[step, :] = torch.einsum('bij, bi -> j', G+P, dHdx).detach()   # Transpose is included in the einsum

        def state_derivative(xnow):
            J, R, G, dHdx, P = model.get_matrices(xnow.view(1, -1))
            deriv = torch.einsum("bij, bj -> i", J-R, dHdx) + torch.einsum("bij, j -> i", G-P, u_ext[step, :])
            return deriv
        
        x = RK4_multistep_integrator(deriv=state_derivative, dt=sim_time[1], x=x)
    return s, y


def plot_simulation(sim_time, true_outputs, sim_outputs, plot_mode="full_sim", title:str="Model simulation"):
    if isinstance(plot_mode, int):
        print(f"Plotting simulation results for state {plot_mode}")
        plt.plot(sim_time, true_outputs[:, plot_mode], label=f"$y_{plot_mode}$")            # True system
        plt.plot(sim_time, sim_outputs[:, plot_mode], label=f"$\\hat{{y}}_{plot_mode}$")    # Model simulation
    elif isinstance(plot_mode, str):
        assert plot_mode=="full_sim" or plot_mode=="error", "Invalid plot_mode argument, please use \"full_sim\" or \"error\"."
        for state in range(true_outputs.shape[1]):
            if plot_mode=="full_sim":
                plt.plot(sim_time, true_outputs[:, state], label=f"$y_{state}$")            # True system
                plt.plot(sim_time, sim_outputs[:, state], label=f"$\\hat{{y}}_{state}$")    # Model simulation
            if plot_mode=="error":
                plt.plot(sim_time, true_outputs[:, state],"k", alpha=0.1, label='_nolegend_')                       # True system
                plt.plot(sim_time, sim_outputs[:, state],"k", alpha=0.1, label='_nolegend_')                        # Model simulation
                plt.plot(sim_time, true_outputs[:, state]-sim_outputs[:, state], label=f"Error $y_{state}$")    # Error f'$q_{i+1}$'
    else:
        print("\033[91m \033[3m Invalid plot_mode specification: \033[0m Please provide the plot_mode argument with a valid specification. \n \
               For plotting the whole system, either use \"full_sim\" or \"error\". \n \
               For specific states, please provide the state number as an integer.")
    # Plot settings
    plt.title(title)
    plt.ylabel("Velocity ($ms^{-1}$)")
    plt.xlabel("Time ($s$)")
    plt.xlim([0, sim_time[-1]])
    plt.legend(loc=1)
    plt.show()


### Blockify functions ###
def blockify_J(system_dim, theta):
    batch_size = theta.shape[0]
    xc_dim = torch.sum(system_dim, 0)[0]
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
    xc_dim = torch.sum(system_dim, 0)[0]
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


def blockify_G(system_dim, theta):
    batch_size = theta.shape[0]
    xc_dim = torch.sum(system_dim, 0)[0]
    sigc_dim = torch.sum(system_dim, 0)[1]
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


def blockify_H(system_dim, theta):
    batch_size = theta.shape[0]
    xc_dim = torch.sum(system_dim, 0)[0]
    past_x_dim = 0

    H_mat = torch.zeros(batch_size, xc_dim)
    for dim in system_dim:
        # Assign values to correct location
        H_mat[:, past_x_dim:past_x_dim+dim[0]] = theta[:, past_x_dim:past_x_dim+dim[0]]

        # Update counters
        past_x_dim += dim[0]
    return H_mat


### General NN structures ###
class feed_forward_nn(nn.Module): # Standard MLP (Same as in deepSI)
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh, initial_output_weight=False):
        super(feed_forward_nn,self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        seq = [nn.Linear(n_in,n_nodes_per_layer), activation()]
        assert n_hidden_layers>0, "feed_forward_nn should only be used for nonlinear neural nets"
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


class simple_NN(nn.Module):
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh, initial_output_is_zero=False):
        super().__init__()
        if n_hidden_layers == 0:    # Does a 0 layer nn make sense in this context?
            self.net = nn.Linear(n_in, n_out) # Do we need an activation function after this?
            if initial_output_is_zero:
                self.net.weight = nn.Parameter(data=torch.zeros_like(self.net.weight))
                self.net.bias = nn.Parameter(data=torch.zeros_like(self.net.bias))
        else:
            seq = [nn.Linear(n_in, n_nodes_per_layer), activation()]
            for i in range(n_hidden_layers-1):
                seq.append(nn.Linear(n_nodes_per_layer, n_nodes_per_layer))
                seq.append(activation())
            seq.append(nn.Linear(n_nodes_per_layer, n_out))
            self.net = nn.Sequential(*seq)
        
            # For the initialisation with a linear estimate, the final output needs to be init at 0
            if initial_output_is_zero:
                with torch.no_grad():
                    self.net[2*n_hidden_layers].weight = nn.Parameter(data=torch.zeros_like(self.net[2*n_hidden_layers].weight))
                    self.net[2*n_hidden_layers].bias = nn.Parameter(data=torch.zeros_like(self.net[2*n_hidden_layers].bias))
    
    def forward(self, x):
        return self.net(x)


### State-independent PHNN subnetworks ###
class constant_J_net(nn.Module):
    def __init__(self, system_dim):
        super().__init__()
        self.system_dim = system_dim
        self.nJ = 0

        for dim in system_dim:
            self.nJ += int((dim[0]**2 - dim[0]) / 2)
        self.J_vals = nn.Parameter(data=torch.rand(self.nJ), requires_grad=True) #TODO: Think about normalization (to fix stuff like R >> J)

    def forward(self, x):
        return blockify_J(self.system_dim, self.J_vals.view(1, -1))


class constant_R_net(nn.Module):
    def __init__(self, system_dim):
        super().__init__()
        self.system_dim = system_dim
        self.nR = 0

        for dim in system_dim:
            self.nR += dim[0]*dim[0]
        self.R_vals = nn.Parameter(data=torch.rand(self.nR), requires_grad=True)  #TODO: Think about normalization (to fix stuff like R >> J)
    
    def forward(self, x):
        return blockify_R(self.system_dim, self.R_vals.view(1, -1))


class constant_G_net(nn.Module):
    def __init__(self, system_dim):
        super().__init__()
        self.system_dim = system_dim
        self.nG = 0

        for dim in system_dim:
            self.nG += dim[0]*dim[1]
        self.G_vals = nn.Parameter(data=torch.rand(self.nG), requires_grad=True) #TODO: Think about normalization (to fix stuff like u >> x)
    
    def forward(self, x):
        return blockify_G(self.system_dim, self.G_vals.view(1, -1))


class constant_H_net(nn.Module):
    def __init__(self, system_dim):
        super().__init__()
        self.system_dim = system_dim
        self.nH = torch.sum(system_dim, 0)[0]
        self.H_vals = nn.Parameter(data=torch.rand(self.nH), requires_grad=True)
    
    def forward(self, x):
        return torch.einsum("i, bi -> bi", self.H_vals, x)


### Variable PHNN subnetworks ###
class var_J_net(nn.Module):
    def __init__(self, system_dim, net=feed_forward_nn, net_kwargs={}):
        super().__init__()
        self.system_dim = system_dim
        self.xc_dim = torch.sum(system_dim, 0)[0]
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
        self.xc_dim = torch.sum(system_dim, 0)[0]
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
        self.xc_dim = torch.sum(system_dim, 0)[0]
        self.sigc_dim = torch.sum(system_dim, 0)[1]
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
    def __init__(self, system_dim, net=feed_forward_nn, net_kwargs={}):
        super().__init__()
        self.system_dim = system_dim
        self.xc_dim = torch.sum(system_dim, 0)[0]
        self.net_list = nn.ModuleList()

        for sys, dim in enumerate(system_dim):
            self.net_list.append(net(n_in=dim[0], n_out=dim[0], **net_kwargs))
    
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
