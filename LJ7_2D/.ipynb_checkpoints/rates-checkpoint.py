import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import scipy.stats
pitorch = torch.Tensor([math.pi])
import pickle
import pandas as pd
from pathlib import Path
from LJ7.FEM_TPT import *
import control_traj_LJ7, utils_LJ7
from utils_LJ7 import LJ7_2

def rates_FEMpts(model, beta):
    Free = np.load('./data/LJ7_free_energy_optimal_control.npz')
    free_energy_grid = Free["free_energy_ongrid"]
    free_energy_grid = torch.tensor(free_energy_grid,dtype = torch.float32)

    M_file = np.load('./data/M_diffusion.npz')
    diffusion = M_file['M_diffusion']
    diffusion = torch.tensor(diffusion,dtype = torch.float32)
    
    data_folder = Path('./LJ7')
    pts = np.loadtxt(data_folder/'LJ7_pts_ABellipses.csv', delimiter=',', dtype=float)
    tri = np.loadtxt(data_folder/'LJ7_tri_ABellipses.csv', delimiter=',', dtype=int)
    q = np.loadtxt(data_folder/'LJ7_ABellipses_q.csv', delimiter=',', dtype=float)
    Fpts = np.loadtxt(data_folder/'LJ7_ABellipses_free.csv', delimiter=',', dtype=float)
    Apts = np.loadtxt(data_folder/'LJ7_Apts.csv', delimiter=',', dtype=float)
    Atri = np.loadtxt(data_folder/'LJ7_Atri.csv', delimiter=',', dtype=int)
    Bpts = np.loadtxt(data_folder/'LJ7_Bpts.csv', delimiter=',', dtype=float)
    Btri = np.loadtxt(data_folder/'LJ7_Btri.csv', delimiter=',', dtype=int)
    Fpts_A = np.loadtxt(data_folder/'Fpts_A.csv', delimiter=',', dtype=float)
    Fpts_B = np.loadtxt(data_folder/'Fpts_B.csv', delimiter=',', dtype=float)
    M11pts = np.loadtxt(data_folder/'M11.csv', delimiter=',', dtype=float)
    M12pts = np.loadtxt(data_folder/'M12.csv', delimiter=',', dtype=float)
    M22pts = np.loadtxt(data_folder/'M22.csv', delimiter=',', dtype=float)

    pts_tch = torch.tensor(pts,dtype = torch.float32)

    chia,chib = utils_LJ7.chiAB(pts_tch)
    q_tilde = model(pts_tch)
    Q = utils_LJ7.q_theta(pts_tch,chia,chib,q_tilde)
    # derivQ = torch.autograd.grad(Q,pts_beta10,allow_unused=True, retain_graph=True, \
    #                             grad_outputs = torch.ones_like(Q), create_graph=True)[0]
    Q_minus = torch.tensor(1) - Q

    Z = invariant_pdf(pts,tri,Apts,Atri,Bpts,Btri,Fpts,Fpts_A,Fpts_B,beta)

    Rcurrent, Rrate = reactive_current_and_transition_rate_updated(pts,tri,Fpts,
                                                        M11pts,M12pts,M22pts,beta,
                                                                   Q.detach().numpy().squeeze(),Z)

    rho_AB = probability_reactive(pts,tri,Fpts,beta,Q.detach().numpy().squeeze(),Z)

    rho_A = probability_last_A(pts,tri,Apts,Atri,Fpts,Fpts_A,beta,Q.detach().numpy().squeeze(),Z)

    return rho_A, rho_AB, Rrate


def rates_NNpts(model, beta):
    # load data in CVs
    Free = np.load('./data/LJ7_free_energy_optimal_control.npz')
    free_energy_grid = Free["free_energy_ongrid"]
    free_energy_grid = torch.tensor(free_energy_grid,dtype = torch.float32)

    M_file = np.load('./data/M_diffusion.npz')
    diffusion = M_file['M_diffusion']
    diffusion = torch.tensor(diffusion,dtype = torch.float32)

    test_pts = torch.load('./data/test_pts_beta5.pt')
    
    inA = []
    inB = []
    notinbdry = []
    a = torch.tensor([0.5526,-0.0935])
    b = torch.tensor([0.7184,1.1607])
    Rx = 0.15
    Ry = 0.03
    theta = 5*pitorch/12

    for i in range(test_pts.shape[0]):
        mu2 = test_pts[i,0]
        mu3 = test_pts[i,1]
        distA = torch.norm(test_pts[i,:] - a,2)
        distB = ((mu2 - b[0])*torch.cos(theta)+(mu3 - b[1])*torch.sin(theta))**2/(Rx**2) + \
                    ((mu2 - b[0])*torch.sin(theta)-(mu3 - b[1])*torch.cos(theta))**2/(Ry**2)
        if distA < 0.1**2:
            inA.append(i)
        elif distB < 1.0:
            inB.append(i)
        else:
            notinbdry.append(i)

    Apts = test_pts[inA,:]
    Bpts = test_pts[inB,:]
    interior = test_pts[notinbdry,:]
    interior.requires_grad_(True)
    Dif_intr = diffusion[notinbdry,::]
    FE_intr = free_energy_grid[notinbdry]

    chia,chib = utils_LJ7.chiAB(interior)
    q_tilde = model(interior)
    Q = utils_LJ7.q_theta(interior,chia,chib,q_tilde)
    derivQ = utils_LJ7.deriv_q(Q,interior)

    Z = torch.sum(torch.exp(-beta*free_energy_grid))
    vR = 0
    for i in range(interior.shape[0]):
        Mg = torch.matmul(Dif_intr[i,::],derivQ[i,:])
        vR += torch.matmul(derivQ[i,:],Mg)*torch.exp(-beta*FE_intr[i,0])
    vR = vR/(beta*Z)

    rho_AA = torch.sum(torch.exp(-beta*FE_intr)* (torch.tensor(1) - Q)**2)/Z

    rho_AB = torch.sum(torch.exp(-beta*FE_intr) * (torch.tensor(1) - Q)*Q)/Z
    rho_A = torch.sum(torch.exp(-beta*FE_intr) * (torch.tensor(1) - Q))/Z
    
    return rho_AB.detach().numpy(), vR.detach().numpy()


def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def transition_rate_sample(dt,I, rho_AB):
#     I = I.detach().numpy()
    average_time_steps,left,right = mean_confidence_interval(I)
    average_time = average_time_steps*dt
    nu_AB = rho_AB/average_time
    
    lower = rho_AB/(right*dt)
    upper = rho_AB/(left*dt)
    
    return average_time, (right - average_time_steps)*dt, nu_AB, lower, upper













