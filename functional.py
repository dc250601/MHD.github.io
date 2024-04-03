import cupy as np

def B_r_f(r):
    return -(r-5)**(2)  + 25

def B_phi_f(r):
    return (r - 5)**(2) - 25

def alpha_f(r):
    alpha_0 = 10.0
    return np.ones_like(r)*alpha_0

def omega_f(r):
    return 10/np.sqrt(1+(r/4)**2)


def V_r_f(r):
    return r*0
def V_z_f(r):
    return r*0
