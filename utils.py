import torch
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
            wave = amplitude*torch.sin(2*torch.pi*freq*t + phase)
            u[:, state] = u[:, state] + wave
    return u