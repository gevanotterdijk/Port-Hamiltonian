import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import deepSI as dsi
from deepSI.models import Custom_SUBNET_CT
from deepSI.networks import MLP_res_net


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


def cubic_D_matrix_form(x, D_vals, M_mat = None):
    dim = len(D_vals)
    mat = torch.zeros(dim, dim)
    
    # Transform x in case x is not given as qdot
    if M_mat is not None:
        x_tilde = torch.matmul(torch.inverse(M_mat), x[dim:])
        x = x_tilde

    for i in range(dim):
        if i == 0: # if i == 0, no left spring
            mat[i, i] = -D_vals[i]*x[i]**2 - D_vals[i+1]*x[i]**2
            mat[i, i+1] = D_vals[i+1]*(x[i+1]**2 - 3*x[i]*x[i+1] + 3*x[i]**2)
        elif i == dim-1: # if i == dim-1, no right spring
            mat[i, i-1] = D_vals[i]*x[i-1]**2
            mat[i, i] = -D_vals[i]*(x[i]**2 - 3*x[i]*x[i-1] + 3*x[i-1]**2)
        else:
            mat[i, i-1] = D_vals[i]*x[i-1]**2
            mat[i, i] = -D_vals[i]*(x[i]**2 - 3*x[i]*x[i-1] + 3*x[i-1]**2) - D_vals[i+1]*x[i]**2
            mat[i, i+1] = D_vals[i+1]*(x[i+1]**2 - 3*x[i]*x[i+1] + 3*x[i]**2)
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


def timeseries_splitting(data:dict|list, n_past:int, n_future:int, stride:int=1):
    """
    Function taking a dictionary or list of dictionaries containing input and output signals.
    Returns 4 tensors each containing [n_windows, time_steps, n_states], for the past inputs, past outputs, future inputs and future outputs.
    """
    # Exception handling for list of dicts instead of singular dict (i.e. multiple datasets)
    if isinstance(data, list):
        print(f"{len(data)} datasets detected")
        u_past_full = y_past_full = u_future_full = y_future_full = torch.FloatTensor()
        for set in data:
            u_past_set, y_past_set, u_future_set, y_future_set =  timeseries_splitting(set, n_past, n_future, stride)
            u_past_full = torch.concat((u_past_full, u_past_set), dim=0)
            y_past_full = torch.concat((y_past_full, y_past_set), dim=0)
            u_future_full = torch.concat((u_future_full, u_future_set), dim=0)
            y_future_full = torch.concat((y_future_full, y_future_set), dim=0)
        return u_past_full, y_past_full, u_future_full, y_future_full

    assert "inputs" in data.keys()  # Dictionary should contain a set of inputs
    assert "output" in data.keys()  # Dictionary should contain a set of outputs
    assert data["inputs"].shape[0] == data["output"].shape[0] # Input and output should be of same length

    # An output timeseries should be of length: (n_samples-n_past-n_future)/2 + 1
    # How many timeseries should there be?
    n_samples = data["inputs"].shape[0]
    n_windows = int((n_samples-n_past-n_future)/stride + 1)
    
    # We want the tensors in dimension: [n_windows, time_steps, n_states]
    u = data["inputs"]
    u_dim = u.shape[1]
    u_past = torch.as_strided(u[:-n_future, :], size=(n_windows, n_past, u_dim), stride=(stride*u_dim, u_dim, 1))
    u_future = torch.as_strided(u[n_past:, :], size=(n_windows, n_future, u_dim), stride=(stride*u_dim, u_dim, 1))

    y = data["output"]
    y_dim = y.shape[1]
    y_past = torch.as_strided(y[:-n_future, :], size=(n_windows, n_past, y_dim), stride=(stride*y_dim, y_dim, 1))
    y_future = torch.as_strided(y[n_past:, :], size=(n_windows, n_future, y_dim), stride=(stride*y_dim, y_dim, 1))
    return u_past, y_past, u_future, y_future


def custom_data_batcher(*arrays, batch_size):
    n_windows = arrays[0].shape[0]
    assert batch_size <= n_windows, "Batch size can not be larger than dataset size"
    
    while True:
        # Shuffle the dataset in the n_windows dimension (Since all arrays have the same n_windows, we only need to
        shuffled_index = torch.randperm(arrays[0].shape[0])
        start, end = 0, batch_size
        while end <= n_windows:
            batch_array = tuple(array.index_select(0, shuffled_index)[start:end, :, :] for array in arrays)
            yield batch_array
            start, end = start+batch_size, end+batch_size


### Blockify functions ###
def blockify_J(system_dim, theta):
    batch_size = theta.shape[0]
    xc_dim = torch.sum(system_dim, 0)[0]
    past_x_dim, past_vals = 0, 0

    J_mat = torch.zeros(batch_size, xc_dim, xc_dim)
    for dim in system_dim:
        # Specify subsystem dimensions
        nJ = int((dim[0]**2 - dim[0]) / 2)
        block = torch.zeros(batch_size, dim[0], dim[0])

        # Assign vals to correct location
        if nJ > 0:  # Dodge the 1D case
            indu = torch.triu_indices(row=dim[0], col=dim[0], offset=1) # offset=1 keeps the diagonal zeroes
            block[:, indu[0], indu[1]] = theta[:, past_vals:past_vals+nJ]
            J_mat[:, past_x_dim:past_x_dim+dim[0], past_x_dim:past_x_dim+dim[0]] = block - torch.transpose(block, dim0=1, dim1=2)
        
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
class feed_forward_nn(nn.Module): # deprecated MLP (Same as in deepSI)
    def __init__(self, n_in=6, n_out=5, n_nodes=64, n_layers=2, activation=nn.Tanh, initial_output_weight=False):
        super(feed_forward_nn,self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        seq = [nn.Linear(n_in,n_nodes), activation()]
        assert n_layers>0, "feed_forward_nn should only be used for nonlinear neural nets"
        for i in range(n_layers-1):
            seq.append(nn.Linear(n_nodes, n_nodes))
            seq.append(activation())
        seq.append(nn.Linear(n_nodes, n_out))
        self.net = nn.Sequential(*seq)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, val=0)
                
    def forward(self,X):
        return self.net(X)


class simple_NN(nn.Module):
    def __init__(self, n_in=6, n_out=5, n_layers=2, n_nodes=64, activation=nn.Tanh, initial_output_is_zero=False):
        super().__init__()
        if n_layers == 0:    # Does a 0 layer nn make sense in this context?
            self.net = nn.Linear(n_in, n_out) # Do we need an activation function after this?
            if initial_output_is_zero:
                with torch.no_grad():
                    self.net.weight = nn.Parameter(data=torch.zeros_like(self.net.weight))
                    self.net.bias = nn.Parameter(data=torch.zeros_like(self.net.bias))
        else:
            seq = [nn.Linear(n_in, n_nodes), activation()]
            for i in range(n_layers-1):
                seq.append(nn.Linear(n_nodes, n_nodes))
                seq.append(activation())
            seq.append(nn.Linear(n_nodes, n_out))
            self.net = nn.Sequential(*seq)
        
            # For the initialisation with a linear estimate, the final output needs to be init at 0
            if initial_output_is_zero:
                with torch.no_grad():
                    self.net[2*n_layers].weight = nn.Parameter(data=torch.zeros_like(self.net[2*n_layers].weight))
                    self.net[2*n_layers].bias = nn.Parameter(data=torch.zeros_like(self.net[2*n_layers].bias))
    
    def forward(self, x):
        return self.net(x)


class simple_res_NN(nn.Module):
    def __init__(self, n_in=6, n_out=5, n_layers=2, n_nodes=64, activation=nn.Tanh, initial_output_is_zero=False):
        super().__init__()
        if n_layers == 0: # In case there are no hidden layers, the residual and normal FFW networks are simply a linear layer
            self.net = nn.Linear(n_in, n_out)
            if initial_output_is_zero:  # Potentially set the output to 0 for initialisation with an estimate
                with torch.no_grad():
                    self.net.weight = nn.Parameter(data=torch.zeros_like(self.net.weight))
                    self.net.bias = nn.Parameter(data=torch.zeros_like(self.net.bias))
        else: # Otherwise, we leave a linear connection between the input and output to act as a residual
            self.res = nn.Linear(n_in, n_out)
            seq = [nn.Linear(n_in, n_nodes), activation()]
            for i in range(n_layers-1):
                seq.append(nn.Linear(n_nodes, n_nodes))
                seq.append(activation())
            seq.append(nn.Linear(n_nodes, n_out))
            self.net = nn.Sequential(*seq)
            if initial_output_is_zero:  # Potentially set the output to 0 for initialisation with an estimate
                with torch.no_grad():
                    self.net[2*n_layers].weight = nn.Parameter(data=torch.zeros_like(self.net[2*n_layers].weight))
                    self.net[2*n_layers].bias = nn.Parameter(data=torch.zeros_like(self.net[2*n_layers].bias))
                    self.res.weight = nn.Parameter(data=torch.zeros_like(self.res.weight))
                    self.res.bias = nn.Parameter(data=torch.zeros_like(self.res.bias))
    
    def forward(self, x):
        try:
            return self.net(x) + self.res(x)
        except: # exception for the 0 hidden layer case
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
    def __init__(self, system_dim, net=simple_res_NN, net_kwargs={}):
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

        for sys, dim in enumerate(self.system_dim):
            nJ = int((dim[0]**2 - dim[0]) / 2)
            if nJ > 0:  # To avoid the single state system, where J has 0 elements
                # Apply the subnetworks to find the parameters j_{ab}
                c_theta = self.net_list[sys](x[:, past_x_dim:past_x_dim+dim[0]])
                block = torch.zeros(batch_size, dim[0], dim[0])

                indu = torch.triu_indices(row=dim[0], col=dim[0], offset=1) # offset=1 keeps the diagonal zeroes
                block[:, indu[0], indu[1]] = c_theta
                J_mat[:, past_x_dim:past_x_dim+dim[0], past_x_dim:past_x_dim+dim[0]] = block - torch.transpose(block, dim0=1, dim1=2)
            past_x_dim += dim[0]
        return J_mat


class var_R_net(nn.Module):
    def __init__(self, system_dim, net=simple_res_NN, net_kwargs={}):
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
    def __init__(self, system_dim, net=simple_res_NN, net_kwargs={}):
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
    def __init__(self, system_dim, net=simple_res_NN, net_kwargs={}):
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


### Visualization functions
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


def plot_matrix_heatmap(matrix, name=""):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='RdYlGn')
    if torch.max(matrix) <= 100:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, round(matrix[i, j].item(), 2),
                        ha="center", va="center", color="k")
    plt.title("Heatmap of " + name)
    plt.colorbar(im)
    plt.show()
