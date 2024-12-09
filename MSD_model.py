import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from copy import deepcopy

from utils import *
from deepSI_lite.models import past_future_arrays
from deepSI_lite.fitting import data_batcher, compute_NMSE
from deepSI.utils.torch_nets import simple_res_net
from nonlinear_benchmarks import Input_output_data

"""
Goal:   This file is to take the measured system data from MSD-datagen and recover a PH model from it.
Method: Creata PHNN-subnet class, that can be trained in a separate validation file.
"""

class custom_PHNN(nn.Module):
    """
    Custom PHNN setup, based off of the original deepSI implementation. Consists of 4 methods:
    """
    def __init__(self, system_dim, na, nb, dt,
                 Jnet="nonlin", Rnet="nonlin", Gnet="nonlin", Hnet="nonlin"):
        super().__init__()
        self.dt = dt
        self.na, self.nb = na, nb
        self.system_dim = system_dim
        self.xc_dim = torch.sum(system_dim, 0)[0]
        self.sigc_dim = torch.sum(system_dim, 0)[1]

        # Define the matrix NNs
        self.Jnet = var_J_net(system_dim=system_dim) if Jnet=="nonlin" else Jnet
        self.Rnet = var_R_net(system_dim=system_dim) if Rnet=="nonlin" else Rnet
        self.Gnet = var_G_net(system_dim=system_dim) if Gnet=="nonlin" else Gnet
        self.Hnet = var_H_net(system_dim=system_dim) if Hnet=="nonlin" else Hnet # Only constant net that depends on x (quadratic Hamiltonian --> linear dHdx)

        # Define the encoder NN
        self.enc_net = simple_res_net(n_in=self.na*self.sigc_dim+self.nb*self.sigc_dim, n_out=self.xc_dim, n_hidden_layers=2)

    def get_matrices(self, x):
        batch_size = x.shape[0]
        dHdx = torch.empty(batch_size, self.xc_dim)

        with torch.enable_grad():
            if x.requires_grad == False:
                x.requires_grad = True
            H = self.Hnet(x)
            Hsum = H.sum()
            dHdx = torch.autograd.grad(Hsum, x, create_graph=True)[0]

        # TODO: Similar to existing master work!!!
        J = self.Jnet(x)
        R = self.Rnet(x)
        G = self.Gnet(x)
        return J, R, G, dHdx

    def get_init_state(self, u_past, y_past):
        with torch.enable_grad():
            n_sequences = u_past.shape[0]
            data_past = torch.FloatTensor(np.concatenate((u_past, y_past), axis=1))
            data_past = data_past.view(n_sequences, (self.na+self.nb)*self.sigc_dim)
            return self.enc_net(data_past)

    def step(self, x, u):
        _, _, G, dHdx = self.get_matrices(x)
        yhat = torch.einsum("bij, bi -> bj", G, dHdx)

        def state_deriv(xnow):
            J, R, G, dHdx = self.get_matrices(xnow)
            deriv = torch.einsum("bij, bj -> bi", J - R, dHdx) + torch.einsum("bij, bj -> bi", G, u)    # Implicit ZOH, as the input is kept constant over the timestep.
            return deriv
        
        xnext = RK4_multistep_integrator(state_deriv, dt=self.dt, x=x)
        return xnext, yhat
    
    def forward(self, u_past, y_past, u_future):
        T = u_future.shape[1] # Assuming u_future has the shape [batch_size, future steps or T, #inputs]
        y_simulated = torch.zeros_like(u_future)
        x = self.get_init_state(u_past, y_past)

        for step in range(T):
            x, yhat = self.step(x, u_future[:, step, :])
            y_simulated[:, step, :] = yhat
        
        return y_simulated


class linear_PHNN(custom_PHNN):
    """
    Linear PHNN setup, using state-independent J, R, G, with H(x) taken to be quadratic (dHdx(x) linear)
    """
    def __init__(self, system_dim, na, nb, dt,
                 Jnet="con", Rnet="con", Gnet="con", Hnet="con"):
        super().__init__(system_dim, na, nb, dt)

        # Define the matrix NNs
        self.Jnet = constant_J_net(system_dim=system_dim) if Jnet=="con" else Jnet
        self.Rnet = constant_R_net(system_dim=system_dim) if Rnet=="con" else Rnet
        self.Gnet = constant_G_net(system_dim=system_dim) if Gnet=="con" else Gnet
        self.Hnet = constant_H_net(system_dim=system_dim) if Hnet=="con" else Hnet # Only constant net that depends on x (quadratic Hamiltonian --> linear dHdx)

        # Define the encoder NN (0 hidden layers --> linear)
        self.enc_net = simple_res_net(n_in=self.na*self.sigc_dim+self.nb*self.sigc_dim, n_out=self.xc_dim, n_hidden_layers=0)

    def get_matrices(self, x):
        # Direct determination of dHdx
        J = self.Jnet(x)
        R = self.Rnet(x)
        G = self.Gnet(x)
        dHdx = self.Hnet(x)
        return J, R, G, dHdx


class cheat_PHNN(custom_PHNN):
    def __init__(self, system_dim, na, nb, dt, M=None, D=None, K=None):
        super().__init__(system_dim, na, nb, dt)
        self.M = torch.eye(int(self.xc_dim/2)) if M is None else M
        self.D = torch.eye(int(self.xc_dim/2)) if D is None else D
        self.K = torch.eye(int(self.xc_dim/2)) if K is None else K

    def get_matrices(self, x):
        dim = int(self.xc_dim/2)
        bs = x.shape[0]
        
        J = torch.zeros(self.xc_dim, self.xc_dim)
        J[:dim, dim:] = torch.eye(dim)
        J[dim:, :dim] = -torch.eye(dim)
        J = J.expand(bs, self.xc_dim, self.xc_dim)      # Match batch sizes

        R = torch.zeros(self.xc_dim, self.xc_dim)
        R[dim:, dim:] = self.D                          # WATCH OUT!!! R == D, so since we take -R, we also get -D
        R = R.expand(bs, self.xc_dim, self.xc_dim)      # Match batch sizes

        G = torch.cat((torch.zeros(self.sigc_dim, self.sigc_dim), torch.eye(self.sigc_dim)), dim=0) # Might run into issues here when sigc dim is not exactly half xc_dim
        G = G.expand(bs, self.xc_dim, self.sigc_dim)    # Match batch sizes

        Q = torch.zeros(self.xc_dim, self.xc_dim) 
        Q[:dim, :dim] = self.K                          # Potential energy
        Q[dim:, dim:] = self.M/4                        # Kinetic energy
        Qx = torch.einsum("ij, bj -> bi", Q, x)         # First compute Q*x
        #H = 0.5*torch.einsum("bj, bi -> b", x, Qx)     # H = 0.5*x*Q*x
        dHdx = Qx.expand(bs, self.xc_dim)               # dHdx = Q*x
        return J, R, G, dHdx


class combined_PHNN(custom_PHNN):
    def __init__(self, system_dim, na, nb, dt, x0, weight=1, 
                 Jinit=None, Rinit=None, Ginit=None, Qinit=None,
                 Jnet="nonlin", Rnet="nonlin", Gnet="nonlin", Hnet="nonlin"):
        super().__init__(system_dim, na, nb, dt, Jnet, Rnet, Gnet, Hnet)
        self.x0 = x0 # TODO: TEMP SOLUTION!!!
        self.weight = weight #TODO: Potentially learn this scaling factor during training?
        # Takes the matrices from linear estimation as torch.FloatTensor([i, j])
        self.Jinit = torch.ones(self.xc_dim, self.xc_dim) if Jinit is None else Jinit
        self.Rinit = torch.ones(self.xc_dim, self.xc_dim) if Rinit is None else Rinit
        self.Ginit = torch.ones(self.xc_dim, self.sigc_dim) if Ginit is None else Ginit
        self.Qinit = torch.ones(self.xc_dim, self.xc_dim) if Qinit is None else Qinit

    def get_matrices(self, x):
        Jnn, Rnn, Gnn, dHdxnn  = super().get_matrices(x)  
        J = self.Jinit.expand(x.shape[0], -1, -1) + self.weight*Jnn
        R = self.Rinit.expand(x.shape[0], -1, -1) + self.weight*Rnn
        G = self.Ginit.expand(x.shape[0], -1, -1) + self.weight*Gnn
        dHdxinit = torch.einsum("ij, bj -> bi", self.Qinit, x)
        dHdx = dHdxinit + self.weight*dHdxnn
        return J, R, G, dHdx
    
    #def get_init_state(self, var1, var2): # TODO: FIX INITIALISATION!!!
    #    return self.x0


def fit_model(model: nn.Module, train_data:Input_output_data, val_data:torch.FloatTensor, n_its:int, T:int=50, stride:int=1, batch_size:int=256, val_freq:int=100):
    # understand this derivation of the fit_minimal_implementation from deepSI_lite.fitting
    loss_fn = nn.MSELoss() # init loss function

    optimizer = torch.optim.Adam(model.parameters()) # init optimizer

    arrays, indices = past_future_arrays(train_data, na=model.na, nb=model.nb, T=T, add_sampling_time=False, stride=stride) #transform (u, y) into (u1, y1, u2, y2)
    arrays_val, indices_val = past_future_arrays(val_data, na=model.na, nb=model.nb, T=T, add_sampling_time=False, stride=stride) #transform (u, y) into (u1, y1, u2, y2)
    arrays = [a[indices_val] for a in arrays_val] # seems to be the same as the original arrays_val
    itter = data_batcher(*arrays, batch_size=batch_size, indices=indices) # We take a random slice of the data [batch_size x (upast, ypast, ufut, yfut)] per iteration

    best_val, best_model_sd = float('inf'), model.state_dict() # initialize storage
    losses = torch.zeros(n_its)
    val_losses = torch.zeros(n_its)

    for it, batch in zip(tqdm(range(n_its)), itter):
        optimizer.zero_grad()

        ysim = model(batch[0], batch[1], batch[2])
        loss = loss_fn(ysim, batch[3])
        loss_norm_factor = torch.max(torch.abs(batch[3])) - torch.min(torch.abs(batch[3])) #Normalization
        losses[it] = loss.detach().numpy() / loss_norm_factor

        with torch.no_grad():
            ysim_val = model(arrays_val[0], arrays_val[1], arrays_val[2])
            val_loss = loss_fn(ysim_val, arrays_val[3])
            val_loss_norm_factor = torch.max(torch.abs(arrays_val[3])) - torch.min(torch.abs(arrays_val[3]))  #Normalization
            val_losses[it] = val_loss / val_loss_norm_factor
            # Save the best model
            if val_losses[it] < best_val:
                best_val = val_losses[it]
                best_model_sd = model.state_dict()

        loss.backward()
        optimizer.step()

        if it % val_freq == 0:  # Validation step
            if val_losses[it] == best_val:
                print(f'Iteration {it:7,}, with training loss (NRMSE): {losses[it].detach().numpy():.5f} and validation loss (NRMSE): {val_losses[it]:.5f} === NEW BEST VALIDATION!')
            else:
                print(f'Iteration {it:7,}, with training loss (NRMSE): {losses[it].detach().numpy():.5f} and validation loss (NRMSE): {val_losses[it]:.5f}')
    model.load_state_dict(best_model_sd) # Continue with the best state dict
    return losses, val_losses, best_model_sd



if __name__ == "__main__":
    M_vals = torch.FloatTensor([2, 2, 2])
    D_vals = torch.FloatTensor([0.5, 0.5, 0.5])
    K_vals = torch.FloatTensor([1, 1, 1])

    M = torch.diag(M_vals)
    D = DK_matrix_form(D_vals)
    K = DK_matrix_form(K_vals)
    print("========== START ==========")
    datasets = torch.load("datasets/TEST_DATASET_GENERATED.pt")
    sim_time = torch.linspace(0, 100, 1024)
    dt = sim_time[1]-sim_time[0]
    system_dim = torch.IntTensor([[6, 3]])
    x_test = torch.rand(1, 6)
    nf = 600

    data = datasets[-1]
    x0 = data["states"][0, :].view(1, -1)
    u0 = data["inputs"][0, :].view(1, -1)
    # Use the true matrices from cheat_PHNN to init the combined model
    true_model = cheat_PHNN(system_dim, na=10, nb=10, dt=dt, M=M, D=D, K=K)
    true_J, true_R, true_G, _ = true_model.get_matrices(x_test)
    true_Q = torch.zeros(6, 6)
    true_Q[:3, :3] = K
    true_Q[3:, 3:] = M/4

    model = combined_PHNN(system_dim, na=10, nb=10, dt=dt, x0=x0, weight=1e-20,
                          Jinit=true_J[0, :, :], Rinit=true_R[0, :, :], Ginit=true_G[0, :, :], Qinit=true_Q)

    
    # Test run
    u1, y1, u2, y2 = past_future_arrays(data=datasets[1]["dsi_IO"], na=10, nb=10, T=nf, add_sampling_time=False, stride=1)[0]
    ysim = model(u1, y1, u2)
    plt.figure("Figure model before training")
    plt.plot(y2[0, :, :].detach().numpy()) # y2 seems noisy? --> The dsi_IO is taking the noisy_outputs as its observations!
    plt.plot(ysim[0, :, :].detach().numpy())
    plt.legend(["$y_{1}$", "$y_{2}$", "$y_{3}$", "$\\hat{y}_{1}$", "$\\hat{y}_{2}$", "$\\hat{y}_{3}$"], loc=1)
    #plt.show()

    # Check N-step error
    plt.figure()
    plt.plot(sim_time, torch.zeros_like(sim_time), "k--")
    plt.plot(y2[0, :, :]-ysim[0, :, :].detach().numpy()) # Nstep error decreases asymptotically towards noise level, so system is sufficiently excited and mismatch through encoder decreases!

    # Try training the custom function
    fit_model(model, datasets[0]["dsi_IO"], datasets[1]["dsi_IO"], n_its=100, T=25, batch_size=256, val_freq=5)
    ysim_trained = model(u1, y1, u2)
    plt.figure("Figure model after training")
    plt.plot(y2[0, :, :].detach().numpy()) # y2 seems noisy? --> The dsi_IO is taking the noisy_outputs as its observations!
    plt.plot(ysim[0, :, :].detach().numpy())
    plt.legend(["$y_{1}$", "$y_{2}$", "$y_{3}$", "$\\hat{y}_{1}$", "$\\hat{y}_{2}$", "$\\hat{y}_{3}$"], loc=1)
    plt.show()
    print("=========== END ===========")