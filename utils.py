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
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
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
    def __init__(self, system_dim, net=feed_forward_nn):
        super().__init__()
        self.system_dim = system_dim
        self.xc_dim = torch.sum(system_dim, 0)[0]
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


### Custom deepSI_Lite PHNN class ###
class cheat_PHNN(Custom_SUBNET_CT):
    def __init__(self, nu, ny, nx, na, nb, norm=dsi.Norm(0, 1, 0, 1), M=None, D=None, K=None):
        super().__init__()
        self.dim = int(nx/2)
        self.nx = nx
        self.nu = nu
        self.nu, self.ny, self.nx, self.nb, self.na, self.norm = nu, ny, nx, nb, na, norm
        self.M = torch.eye(self.dim) if M is None else M
        self.D = torch.eye(self.dim) if D is None else D
        self.K = torch.eye(self.dim) if K is None else K

        self.encoder = MLP_res_net(input_size = [(nb,nu) , (na,ny)], output_size = nx, n_hidden_layers=1)
        #self.encoder = feed_forward_nn(n_in=(nb*nu+na*ny), n_out=nx)
        #self.enc_net = simple_res_net(n_in=(self.na*self.sigc_dim + self.nb*self.sigc_dim), n_out=self.xc_dim, n_hidden_layers=0) # 0 Hidden layers makes the encoder a linear approximator

    def get_matrices(self, x):
        # Define J
        J = torch.zeros(self.nx, self.nx)
        J[:self.dim, self.dim:] = torch.eye(self.dim)
        J[self.dim:, :self.dim] = -torch.eye(self.dim)
        J = J.expand(x.shape[0], self.nx, self.nx)

        # Define R
        R = torch.zeros(self.nx, self.nx)
        R[self.dim:, self.dim:] = self.D
        R = R.expand(x.shape[0], self.nx, self.nx)

        # Define G
        G = torch.cat((torch.zeros(self.nu, self.nu), torch.eye(self.nu)), dim=0)
        G = G.expand(x.shape[0], self.nx, self.nu)

        # Define dHdx
        Q = torch.zeros(self.nx, self.nx)
        Q[:self.dim, self.dim:] = self.K                           # Potential energy
        Q[self.dim:, :self.dim] = torch.inverse(self.M)            # Kinetic energy
        Qx = torch.einsum("ij, bj -> bi", Q, x)     # First take Q*x
        #H = 0.5*torch.einsum("bj, bi -> b", x, Qx)  # Then take H = 0.5*x*Q*x
        dHdx = Qx
        return J, R, G, dHdx
    
    def get_init_state(self, upast, ypast):
        n_sequences = upast.shape[0]
        data_past = torch.FloatTensor(np.concatenate((upast, ypast), axis=1))
        data_past = data_past.view(n_sequences, (self.nb*self.nu+self.na*self.ny))
        return self.encoder(data_past)

    def forward(self, upast, ypast, ufuture, sampling_time):
        dt = torch.unique(sampling_time)[0] # TODO: Test whether the sampling times are uniform
        x = self.get_init_state(upast, ypast)
        ysim = torch.zeros_like(ufuture)

        for t_i, u in enumerate(ufuture.swapaxes(0, 1)):
            _, _, G, dHdx = self.get_matrices(x)
            yhat = torch.einsum("bij, bi -> bj", G, dHdx)
            ysim[:, t_i, :] = yhat

            def state_deriv(xnow, u):
                J, R, G, dHdx = self.get_matrices(xnow)
                return torch.einsum("bij, bj -> bi", J-R, dHdx) + torch.einsum("bij, bj -> bi", G, u)
            
            x = RK4_multistep_integrator_with_u(state_deriv, dt, x, u)
        return ysim

if __name__ == "__main__":
    import deepSI_lite as dsi
    datasets = torch.load("datasets\\\TEST_DATASET_GENERATED.pt")
    sim_time = torch.linspace(0, 100, 1024)
    x_test = torch.rand(2, 6)
    M_vals = torch.FloatTensor([2, 2, 2])
    D_vals = torch.FloatTensor([0.5, 0.5, 0.5])
    K_vals = torch.FloatTensor([1, 1, 1])

    M = torch.diag(M_vals)
    D = DK_matrix_form(D_vals)
    K = DK_matrix_form(K_vals)

    # Divide the datasets across training, validation and testing datasets
    # The resulting objects are lists of deepSI_lite.Input_output_data objects
    n_train = 5
    n_val = 2
    n_test = 1

    dsi_train = []
    dsi_val = []
    dsi_test = []

    for i, set in enumerate(datasets):
        if i < n_train:
            dsi_train.append(set["dsi_IO"])
        elif i < n_train+n_val:
            dsi_val.append(set["dsi_IO"])
        else:
            dsi_test.append(set["dsi_IO"])

    nx, nb, na = 6, 10, 10
    nu, ny, norm = dsi.get_nu_ny_and_auto_norm(dsi_train)
    print(nu, ny)

    model = cheat_PHNN(nu, ny, nx=nx, nb=nb, na=na, M=M, D=D, K=K)
    upast, ypast, ufuture, sampling_time, yfuture = model.create_arrays(dsi_test[0], T=512)[0]
    # create_arrays returns the tuple (windows, ids)
    #   - windows is again a tuple containing:
    #       0 - torch.FloatTensor: upast
    #       1 - torch.FloatTensor: ypast
    #       2 - torch.FloatTensor: ufuture
    #       3 - torch.FloatTensor: sampling_time
    #       4 - torch.FloatTensor: yfuture
    ysim = model.forward(upast, ypast, ufuture, sampling_time)
    
    cheat_model = cheat_PHNN(nu, ny, nx=nx, nb=nb, na=na, M=M, D=D, K=K)
    cheat_fit = dsi.fit(cheat_model, train=dsi_train, val=dsi_val, n_its=1001, T=128, batch_size=64, val_freq=25)