import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import scipy.stats
pitorch = torch.Tensor([math.pi])
from pathlib import Path
from distmesh import *
from FEM_TPT import *
from utils import *
import control_traj

def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def rates_FEMpts(model, beta, gamma):
    data_folder = Path('./Duffing_beta'+str(beta))

    pts_file = data_folder/'Duffing_pts_ellip.csv'
    tri_file = data_folder/'Duffing_tri_ellip.csv'
    q_file = str(data_folder)+'/'+'Duffing_committor_beta'+str(beta)+'_ellip.csv'
    qminus_file = str(data_folder)+'/'+'Duffing_backward_committor_beta'+str(beta)+'_ellip.csv'
    Apts_file = data_folder/'Apts_ellip.csv'
    Bpts_file = data_folder/'Bpts_ellip.csv'
    Atri_file = data_folder/'Atri_ellip.csv'
    Btri_file = data_folder/'Btri_ellip.csv'


    data_pts = np.loadtxt(pts_file, delimiter=',', dtype=float)
    data_q = np.loadtxt(q_file, delimiter=',', dtype=float)
    data_qminus = np.loadtxt(qminus_file,delimiter=',', dtype=float)
    data_pts_minus = np.hstack((data_pts[:,0][:,None],-data_pts[:,1][:,None]))
    data_tri = np.loadtxt(tri_file, delimiter=',', dtype=int)
    # NO NEED TO CONVERT TO TORCH TENSOR
    A_pts = np.loadtxt(Apts_file, delimiter=',', dtype=float)
    B_pts = np.loadtxt(Bpts_file, delimiter=',', dtype=float)
    A_tri = np.loadtxt(Atri_file, delimiter=',', dtype=int)
    B_tri = np.loadtxt(Btri_file, delimiter=',', dtype=int)

    A_pts_tensor = torch.tensor(A_pts, dtype = torch.float32)
    B_pts_tensor = torch.tensor(B_pts, dtype = torch.float32)

    pts_beta = torch.tensor(data_pts,dtype = torch.float32)
    pts_beta_minus = torch.tensor(data_pts_minus,dtype = torch.float32)
    q_fem_beta = torch.tensor(data_q[:,2],dtype = torch.float32)
    qminus_fem_beta = torch.tensor(data_qminus[:,2],dtype = torch.float32)

    pts_beta.requires_grad_(True)
    Funs = functions(torch.tensor(-1),torch.tensor(1))
    Q = model(pts_beta)
    Q_minus = torch.tensor(1) - model(pts_beta_minus)

    beta_tensor = torch.tensor(beta)
    U = Funs.funU(pts_beta[:,0]) # potential energy function
    T = Funs.funT(pts_beta[:,1]) # kinetic energy function

    Z = invariant_pdf(data_pts,data_tri,A_pts,A_tri,B_pts,B_tri,fpot_FEM,beta)

    Rcurrent, Rrate = reactive_current_transition_rate_Langevin(data_pts,data_tri,Hamiltonian,divfree_FEM,beta, gamma
                                                                ,Q.detach().numpy(),Q_minus.detach().numpy(),Z)

    rho_A=probability_last_A_Langevin(data_pts,data_tri,A_pts,A_tri,fpot_FEM,beta,Q.detach().numpy(),Q_minus.detach().numpy(),Z)
    rho_AB = probability_reactive_Langevin(data_pts,data_tri,fpot_FEM,beta,Q.detach().numpy(),Q_minus.detach().numpy(),Z)
    
    return rho_A, rho_AB, Rrate

def rates_NNpts(test_pts_file, model, beta, gamma, rx, ry):
    test_pts = torch.load(test_pts_file)
    inA = []
    inB = []
    notinbdry = []
    a = torch.tensor([-1,0])
    b = torch.tensor([1,0])
    for i in range(test_pts.shape[0]):
        x = test_pts[i,0]
        y = test_pts[i,1]
        distA = (x - a[0]).pow(2)/(rx**2) + (y - a[1]).pow(2)/(ry**2)
        distB = (x - b[0]).pow(2)/(rx**2) + (y - b[1]).pow(2)/(ry**2)
        if distA < 1.0:
            inA.append(i)
        elif distB < 1.0:
            inB.append(i)
        else:
            notinbdry.append(i)
    Apts = test_pts[inA,:]
    Bpts = test_pts[inB,:]
    interior = test_pts[notinbdry,:]
    interior.requires_grad = True
    interior_minus = interior*torch.tensor([[1,-1]])

    Hamilton_all = Hamiltonian(test_pts[:,0].detach().numpy(), test_pts[:,1].detach().numpy())
    Hamilton = Hamiltonian(interior[:,0].detach().numpy(), interior[:,1].detach().numpy())

    Z = np.sum(np.exp(-beta*Hamilton_all))

    Q = model(interior)
    derivQ = torch.autograd.grad(Q,interior,allow_unused=True, retain_graph=True, \
                                                 grad_outputs = torch.ones_like(Q), create_graph=True)
    Q_minus = model(interior_minus)

    rho_AB = (np.sum(np.exp(-beta*Hamilton)*(Q.reshape(-1,)*(1-Q_minus).reshape(-1,)).detach().numpy()))/Z

    nu_AB = (np.sum(np.exp(-beta*Hamilton)*(derivQ[0][:,1]**2).detach().numpy())*gamma/beta)/Z
    
    return rho_AB, nu_AB

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

       

    

    



