### The util file for the MHD project

import cupy as np
import matplotlib.pyplot as plt
import os
import sys
import json

def init(path,B_r_f,B_phi_f,alpha_f,omega_f,V_r_f,V_z_f):
    # Open the JSON file containing parameters
    with open('Parameters.json', 'r') as openfile:
        parameters = json.load(openfile)
    
    # Extract parameters from the JSON file
    h_val = parameters["h_val"]
    eta_m = parameters["eta_m"]
    eta_t = parameters["eta_t"]
    r_min = parameters["r_min"]
    t_min = parameters["t_min"]
    r_max = parameters["r_max"]
    t_max = parameters["t_max"]
    d_r = parameters["d_r"]
    d_t = parameters["d_t"]

    # Print out extracted parameters
    print("------------------Parameters------------------")
    print("h_val:", h_val)
    print("eta_m:", eta_m)
    print("eta_t:", eta_t)
    print("r_min:", r_min)
    print("t_min:", t_min)
    print("r_max:", r_max)
    print("t_max:", t_max)
    print("d_r:", d_r)
    print("d_t:", d_t)
    print("------------------------------------------------")
    
    # Create arrays for r, t, dr, and dt
    r = np.arange(r_min, r_max, d_r)
    dr = np.diff(r)
    dr = np.concatenate([np.array([dr[0]]), dr])
    h = h_val * np.ones_like(r)
    t = np.arange(t_min, t_max, d_t)
    dt = np.diff(t, prepend=0.01)
    dt = np.concatenate([np.array([dt[0]]), dt])
    d_t = dt[1]

    # Calculate total diffusivity and other magnetic field parameters
    eta = eta_m + eta_t
    omega = omega_f(r)
    alpha = alpha_f(r)
    V_r = V_r_f(r)
    V_z = V_z_f(r)
    q_omega = -r * np.gradient(omega) / dr
    D = -alpha[0] * q_omega * h[2] ** 3 / eta ** 2
    D = np.average(D)
    B_r = B_r_f(r)
    B_phi = B_phi_f(r)

    # Initialize arrays for magnetic field evolution
    B_r_t = B_r
    Bphi_t = B_phi

    # Define a specific r value
    r_val = 5
    r_index = np.where(np.min(np.abs(r - r_val)) < 1e-5)[0][0]
    
    # Pre-calculating the grad to speed up CUDA
    grad_V_r = np.gradient(V_r) / (dr)
    grad_omega = np.gradient(omega) / dr
    
    return B_r_t,Bphi_t,r,dr,omega,alpha,V_r,V_z,h,eta,dt,d_t,t_min,r_index,grad_omega,grad_V_r,D



def Eqn_Br(Br, Bphi, dr, eta, d_t, r, V_r, V_z, h, alpha):

    diag = ((-V_r/r + V_z/(4*h) - eta/(r**2) - (2*eta)/(dr**2) - (eta*np.pi**2)/(4*h**2))[None, :]*np.eye(Br.shape[0]))

    upper_diag = np.zeros((Br.shape[0], Br.shape[0]))
    upper_diag[:-1, 1:] = ((-V_r/(2*dr) + eta/(dr**2) + eta/(2*r*dr))[None, :]*np.eye(Br.shape[0]))[1:, 1:]

    lower_diag = np.zeros((Br.shape[0], Br.shape[0]))
    lower_diag[1:, :-1] = ((V_r/(2*dr) + eta/(dr**2) - eta/(2*r*dr))[None, :]*np.eye(Br.shape[0]))[:-1, :-1]

    mat2 = ((2*alpha)/(np.pi*h)[None, :]*np.eye(Bphi.shape[0]))

    mat1 = diag + upper_diag + lower_diag
    k = (mat1@Br[:, None])[:, 0] + (mat2@Bphi[:, None])[:, 0]
    k[[0, -1]] = 0
    return k*d_t


# Full Eqn in Bphi
def Eqn_Bphi(Br, Bphi, dr, eta, d_t, r, V_r, V_z, h, alpha, omega, grad_omega, grad_V_r):

    if grad_omega is None:
        grad_omega = np.gradient(omega)/dr
    if grad_V_r is None:
        grad_V_r = np.gradient(V_r)/(dr)


    diag = (-grad_V_r - V_z/(4*h) - (2*eta)/(dr**2) - eta/(r**2) - (eta*np.pi**2)/(4*h**2))[None, :]*np.eye(Bphi.shape[0])

    upper_diag = np.zeros((Bphi.shape[0], Bphi.shape[0]))

    upper_diag[:-1, 1:] = ((-V_r/(2*dr) + eta/(dr**2) + eta/(2*r*dr))[None, :]*np.eye(Bphi.shape[0]))[1:, 1:]

    lower_diag = np.zeros((Br.shape[0], Br.shape[0]))
    lower_diag[1:, :-1] = ((V_r/(2*dr) + eta/(dr**2) - eta/(2*r*dr))[None, :]*np.eye(Bphi.shape[0]))[:-1, :-1]

    mat2 = (-r*grad_omega + (2*alpha)/(np.pi*h))[None, :]*np.eye(Br.shape[0])

    mat1 = diag + upper_diag + lower_diag

    k = (mat1@Bphi[:, None])[:,0] + (mat2@Br[:, None])[:, 0]
    k[[0, -1]] = 0
    return k*d_t


# Runge Kutta Step
def runge_kutta_step(Br, Bphi, dt, eta, r, dr, V_r, V_z, h, alpha, omega, grad_omega = None, grad_V_r = None):

    k1_r = dt * Eqn_Br(Br, Bphi, dr, eta, dt, r, V_r, V_z, h, alpha)
    k1_phi = dt * Eqn_Bphi(Br, Bphi, dr, eta, dt, r, V_r, V_z, h, alpha, omega, grad_omega, grad_V_r)

    k2_r = dt * Eqn_Br(Br + 0.5 * k1_r, Bphi + 0.5 * k1_phi, dr, eta, dt, r, V_r, V_z, h, alpha)
    k2_phi = dt * Eqn_Bphi(Br + 0.5 * k1_r, Bphi + 0.5 * k1_phi, dr, eta, dt, r, V_r, V_z, h, alpha, omega, grad_omega, grad_V_r)

    k3_r = dt * Eqn_Br(Br + 0.5 * k2_r, Bphi + 0.5 * k2_phi, dr, eta, dt, r, V_r, V_z, h, alpha)
    k3_phi = dt * Eqn_Bphi(Br + 0.5 * k2_r, Bphi + 0.5 * k2_phi, dr, eta, dt, r, V_r, V_z, h, alpha, omega, grad_omega, grad_V_r)

    k4_r = dt * Eqn_Br(Br + k3_r, Bphi + k3_phi, dr, eta, dt, r, V_r, V_z, h, alpha)
    k4_phi = dt * Eqn_Bphi(Br + k3_r, Bphi + k3_phi, dr, eta, dt, r, V_r, V_z, h, alpha, omega, grad_omega, grad_V_r)

    return (Br + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6, Bphi + (k1_phi + 2 * k2_phi + 2 * k3_phi + k4_phi) / 6)
