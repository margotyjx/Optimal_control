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


def chiAB(X):
    a = torch.tensor([0.5526,-0.0935])
    b = torch.tensor([0.7184,1.1607])
    rA = torch.tensor(0.1034)
    rB = torch.tensor(0.07275)
    m = nn.Tanh()
    sizex, nothing = X.shape
    chiA = 0.5 - 0.5*m(1000*((((X - a).pow(2)).sum(dim = 1).reshape(sizex,1))-(rA + torch.tensor(0.02)).pow(2)))                     
    chiB = 0.5 - 0.5*m(1000*((((X - b).pow(2)).sum(dim = 1).reshape(sizex,1))-(rB + torch.tensor(0.02)).pow(2)))       
                             
    return chiA, chiB

def q_theta(X,chiA,chiB,q_tilde):
    Q = (torch.tensor([1]) - chiA)*(q_tilde*(torch.tensor([1]) - chiB)+chiB)
    return Q

class LJ7_2(nn.Module):
    """Feedfoward neural network with 2 hidden layer"""
    def __init__(self, in_size, hidden_size,hidden_size2, out_size):
        super().__init__()
        # 1st hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # 2nd hidden layer
        self.linear2 = nn.Linear(hidden_size,hidden_size2)
        # output layer
        self.linear3 = nn.Linear(hidden_size2, out_size)
        
    def forward(self, xb):
        # Get information from the data
#         xb = torch.cat((torch.sin(xb),torch.cos(xb)),dim = 1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        tanhf = nn.Tanh()
        out = tanhf(out)
        # Get predictions using output layer
        out = self.linear2(out)
        # apply activation function again
        out = tanhf(out)
        # last hidden layer 
        out = self.linear3(out)
        #sigmoid function
        out = torch.sigmoid(out)
        return out


def LJpot(x): # Lennard-Jones potential, x is the position of each particles
    Na = x.shape[1] # x has shape [2,7] 
    r2 = torch.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2
        r2[k,k] = 1
    er6 = torch.div(torch.ones_like(r2),r2**3) 
    L = (er6-torch.tensor(1))*er6
    V = 2*torch.sum(L) 
    return V

def LJgrad(x):
    Na = x.shape[1]
    r2 = torch.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2
        r2[k,k] = 1
    r6 = r2**3
    L = -6*torch.div((2*torch.div(torch.ones_like(r2),r6)-1),(r2*r6)) # use r2 as variable instead of r
    g = torch.zeros_like(x)
    for k in range(Na):
        Lk = L[:,k]
        g[0,k] = torch.sum((x[0,k] - x[0,:])*Lk)
        g[1,k] = torch.sum((x[1,k] - x[1,:])*Lk)
    g = 4*g 
    return g

# +
def LJpot_np(x): # Lennard-Jones potential, x is the position of each particles
    Na = np.size(x,axis = 1) # x has shape [2,7] 
    r2 = np.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2
        r2[k,k] = 1
    er6 = np.divide(np.ones_like(r2),r2**3) 
    L = (er6-1)*er6
    V = 2*np.sum(L) 
    return V

#dV/dx_i = 4*sum_{i\neq j}(-12r_{ij}^{-13} + 6r_{ij}^{-7})*(x_i/r_{ij})
def LJgrad_np(x):
    Na = np.size(x,axis = 1)
    r2 = np.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2
        r2[k,k] = 1
    r6 = r2**3
    L = -6*np.divide((2*np.divide(np.ones_like(r2),r6)-1),(r2*r6)) # use r2 as variable instead of r
    g = np.zeros_like(x)
    for k in range(Na):
        Lk = L[:,k]
        g[0,k] = np.sum((x[0,k] - x[0,:])*Lk)
        g[1,k] = np.sum((x[1,k] - x[1,:])*Lk)
    g = 4*g 
    return g


# -

def C(x):
    Na = x.shape[1] # x has shape [2,7]
    C = torch.zeros(Na)
    r2 = torch.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2
        r2[k,k] = 0
    for i in range(Na):
        ci = torch.div(torch.ones_like(r2[i,:]) - (r2[i,:]/2.25)**4, torch.ones_like(r2[i,:]) - (r2[i,:]/2.25)**8)
        ci_sum = torch.sum(ci) - torch.tensor(1)
        C[i] = ci_sum
    return C

def mu2n3(x):
    C_list = C(x)
    ave_C = torch.mean(C_list)
    mu2 = torch.mean((C_list - ave_C)**2)
    mu3 = torch.mean((C_list - ave_C)**3)
    return mu2, mu3

def deriv_mu(mu2,mu3,x):
    derivmu2 = torch.autograd.grad(mu2,x,allow_unused=True, retain_graph=True, \
                                             grad_outputs = torch.ones_like(mu2), create_graph=True)
    derivmu3 = torch.autograd.grad(mu3,x,allow_unused=True, retain_graph=True, \
                                             grad_outputs = torch.ones_like(mu3), create_graph=True)
    return derivmu2,derivmu3

def deriv_q(Q,x):
    derivQ = torch.autograd.grad(Q,x,allow_unused=True, retain_graph=True, \
                                             grad_outputs = torch.ones_like(Q), create_graph=True)
    return derivQ[0]

def MALAstep(x,pot_x,grad_x,fpot,fgrad,beta,dt):
    std = torch.sqrt(2*dt/beta)    
    w = np.random.normal(0.0,std,np.shape(x))
    y = x - dt*grad_x + torch.tensor(w)
    pot_y = fpot(y)
    grad_y = fgrad(y)
    qxy =  torch.sum(torch.tensor(w)**2)  #||w||^2
    qyx = torch.sum((x - y + dt*grad_y)**2) # ||x-y+dt*grad V(y)||
    alpha = torch.exp(-beta*(pot_y-pot_x+(qyx-qxy)*0.25/dt))
    if alpha < 1: 
        x = y
        pot_x = pot_y
        grad_x = grad_y
        # print("ACCEPT: alpha = ",alpha)
    else:    
        eta = np.random.uniform(0.0,1.0,(1,))
        if eta < alpha.detach().numpy(): # accept move 
            x = y
            pot_x = pot_y
            grad_x = grad_y
            # print("ACCEPT: alpha = ",alpha," eta = ",eta)
        else:
            pass
#             print("REJECT: alpha = ",alpha," eta = ",eta)
    return x,pot_x,grad_x    


def biased_MALAstep(x,pot_x,grad_x,q,grad_q,fpot,fgrad,beta,dt):
    std = torch.sqrt(2*dt/beta)    
    w = np.random.normal(0.0,std,np.shape(x))
    """
    When the point is too close to A, derivQ = 0, Q = 0 return None
    """
    control = torch.tensor(2)/beta*(grad_q/q)
    y = x - (grad_x - control)*dt + torch.tensor(w)
    pot_y = fpot(y)
    grad_y = fgrad(y)
    qxy =  torch.sum(torch.tensor(w)**2)  #||w||^2
    qyx = torch.sum((x - y + dt*grad_y)**2) # ||x-y+dt*grad V(y)||
    alpha = torch.exp(-beta*(pot_y-pot_x+(qyx-qxy)*0.25/dt))
#     x = y
#     pot_x = pot_y
#     grad_x = grad_y
    if alpha >= 1: # accept move 
        x = y
        pot_x = pot_y
        grad_x = grad_y
    else:    
        eta = np.random.uniform(0.0,1.0,(1,))
        if eta < alpha.detach().numpy(): # accept move 
            x = y
            pot_x = pot_y
            grad_x = grad_y
            # print("ACCEPT: alpha = ",alpha," eta = ",eta)
        else:
            pass
    return x,pot_x,grad_x   
