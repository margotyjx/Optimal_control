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
from distmesh import *
from FEM_TPT import *
import controlled_traj, utils
from utils import Ruggedmueller2

def rates_FEMpts(model, beta):
    data_folder = Path('./data')
    pts = np.loadtxt(data_folder/'mueller_pts.csv', delimiter=',', dtype=float)
    tri = np.loadtxt(data_folder/'mueller_tri.csv', delimiter=',', dtype=int)
    pts_Amesh = np.loadtxt(data_folder/'A_pts.csv', delimiter=',', dtype=float)
    tri_Amesh = np.loadtxt(data_folder/'A_tri.csv', delimiter=',', dtype=int)
    pts_Bmesh = np.loadtxt(data_folder/'B_pts.csv', delimiter=',', dtype=float)
    tri_Bmesh = np.loadtxt(data_folder/'B_tri.csv', delimiter=',', dtype=int)

    Z = invariant_pdf(pts,tri,pts_Amesh,tri_Amesh,pts_Bmesh,tri_Bmesh,utils.mueller,beta)

    pts_tensor = torch.tensor(pts,dtype=torch.float32)
    chia,chib = utils.chiAB(pts_tensor)
    q_tilde = model(pts_tensor)
    q = utils.q_theta(pts_tensor,chia,chib,q_tilde).detach().numpy()
    rho_AB_10 = probability_reactive(pts,tri,utils.mueller,beta,q,Z)
    rho_AA_10 = probability_last_A_return_A(pts,tri,pts_Amesh,tri_Amesh,utils.mueller,beta,q,Z)
    vR_10 = reactive_current_and_transition_rate(pts,tri,utils.mueller,beta,q,Z)
    
    return rho_AA_10.squeeze(), rho_AB_10.squeeze(), vR_10.squeeze()

def rates_NNpts(test_pts, model, betap, beta):
#     test_pts = torch.load(test_pts_file)
    inA = []
    inB = []
    notinbdry = []
    a = torch.tensor([-0.558,1.441])
    b = torch.tensor([0.623,0.028])
    for i in range(test_pts.shape[0]):
        distA = torch.norm(test_pts[i,:] - a,2)
        distB = torch.norm(test_pts[i,:] - b,2)
        if distA < 0.1**2:
            inA.append(i)
        elif distB < 0.1**2:
            inB.append(i)
        else:
            notinbdry.append(i)
            
    Apts = test_pts[inA,:]
    Bpts = test_pts[inB,:]
    interior = test_pts[notinbdry,:]
    interior.requires_grad = True

    U_mueller_all = utils.funU(test_pts[:,0], test_pts[:,1])
    U_mueller = utils.funU(interior[:,0], interior[:,1])

    Z = torch.sum(torch.exp(-(beta - betap)*U_mueller_all))

    chia,chib = utils.chiAB(interior)
    q_tilde = model(interior)
    q = utils.q_theta(interior,chia,chib,q_tilde)
    derivQ = torch.autograd.grad(q,interior,allow_unused=True, retain_graph=True, \
                                                 grad_outputs = torch.ones_like(q), create_graph=True)


    rho_AB_10 = (torch.sum(torch.exp(-(beta - betap)*U_mueller)*q.reshape(-1,)*(1-q).reshape(-1,))/Z).detach().numpy()

    nu_AB_10 = (torch.sum(torch.exp(-(beta - betap)*U_mueller)*torch.norm(derivQ[0],dim=1)**2)*(1/beta)/Z).detach().numpy()

    
    return rho_AB_10, nu_AB_10

def rates_uniformpts(test_pts, model, beta):
#     test_pts = torch.load(test_pts_file)
    inA = []
    inB = []
    notinbdry = []
    a = torch.tensor([-0.558,1.441])
    b = torch.tensor([0.623,0.028])
    for i in range(test_pts.shape[0]):
        distA = torch.norm(test_pts[i,:] - a,2)
        distB = torch.norm(test_pts[i,:] - b,2)
        if distA < 0.1**2:
            inA.append(i)
        elif distB < 0.1**2:
            inB.append(i)
        else:
            notinbdry.append(i)
            
    Apts = test_pts[inA,:]
    Bpts = test_pts[inB,:]
    interior = test_pts[notinbdry,:]
    interior.requires_grad = True

    U_mueller_all = utils.funU(test_pts[:,0], test_pts[:,1])
    U_mueller = utils.funU(interior[:,0], interior[:,1])

    Z = torch.sum(torch.exp(-beta*U_mueller_all))

    chia,chib = utils.chiAB(interior)
    q_tilde = model(interior)
    q = utils.q_theta(interior,chia,chib,q_tilde)
    derivQ = torch.autograd.grad(q,interior,allow_unused=True, retain_graph=True, \
                                                 grad_outputs = torch.ones_like(q), create_graph=True)


    rho_AB_10 = (torch.sum(torch.exp(-beta*U_mueller)*q.reshape(-1,)*(1-q).reshape(-1,))/Z).detach().numpy()

    nu_AB_10 = (torch.sum(torch.exp(-beta*U_mueller)*torch.norm(derivQ[0],dim=1)**2)*(1/beta)/Z).detach().numpy()

    
    return rho_AB_10, nu_AB_10


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

    

    
