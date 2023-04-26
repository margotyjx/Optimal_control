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

def dU(x,y):
    a = torch.tensor([-1,-1,-6.5,0.7])
    b = torch.tensor([0,0,11,0.6])
    c = torch.tensor([-10,-10,-6.5,0.7])
    D = torch.tensor([-200,-100,-170,15])
    X = torch.tensor([1,0,-0.5,-1])
    Y = torch.tensor([0,0.5,1.5,1])
    gamma = torch.tensor([9])
    k = torch.tensor([5])
  
    fx1 = D[0]*torch.exp(a[0]*((x-X[0]).pow(2)) + b[0]*(x-X[0])*(y-Y[0]) + c[0]*(y.pow(2)))
    fx2 = D[1]*torch.exp(a[1]*((x-X[1]).pow(2)) + b[1]*(x-X[1])*(y-Y[1]) + c[1]*((y-Y[1]).pow(2)))
    fx3 = D[2]*torch.exp(a[2]*((x-X[2]).pow(2)) + b[2]*(x-X[2])*(y-Y[2]) + c[2]*((y-Y[2]).pow(2)))
    fx4 = D[3]*torch.exp(a[3]*((x-X[3]).pow(2)) + b[3]*(x-X[3])*(y-Y[3]) + c[3]*((y-Y[3]).pow(2)))
    extra = gamma*torch.sin(2*k*pitorch*x)*torch.sin(2*k*pitorch*y)
    
    extrapx = gamma*torch.cos(2*k*pitorch*x)*torch.sin(2*k*pitorch*y)*2*pitorch*k
    extrapy = gamma*torch.cos(2*k*pitorch*y)*torch.sin(2*k*pitorch*x)*2*pitorch*k
    
    dUx = fx1*(2*a[0]*(x-X[0])+b[0]*(y-Y[0])) + fx2*(2*a[1]*(x-X[1])+b[1]*(y-Y[1])) + fx3*(2*a[2]*(x-X[2])+b[2]*(y-Y[2])) + fx4*(2*a[3]*(x-X[3])+b[3]*(y-Y[3]))
    
    dUy = fx1*(2*c[0]*y+b[0]*(x-X[0])) + fx2*(2*c[1]*(y-Y[1])+b[1]*(x-X[1])) + fx3*(2*c[2]*(y-Y[2])+b[2]*(x-X[2])) + fx4*(2*c[3]*(y-Y[3])+b[3]*(x-X[3]))
    
    return dUx, dUy


def funU(x,y):
    a = torch.tensor([-1,-1,-6.5,0.7])
    b = torch.tensor([0,0,11,0.6])
    c = torch.tensor([-10,-10,-6.5,0.7])
    D = torch.tensor([-200,-100,-170,15])
    X = torch.tensor([1,0,-0.5,-1])
    Y = torch.tensor([0,0.5,1.5,1])
    gamma = torch.tensor([9])
    k = torch.tensor([5])
  
    fx1 = D[0]*torch.exp(a[0]*((x-X[0]).pow(2)) + b[0]*(x-X[0])*(y-Y[0]) + c[0]*(y.pow(2)))
    fx2 = D[1]*torch.exp(a[1]*((x-X[1]).pow(2)) + b[1]*(x-X[1])*(y-Y[1]) + c[1]*((y-Y[1]).pow(2)))
    fx3 = D[2]*torch.exp(a[2]*((x-X[2]).pow(2)) + b[2]*(x-X[2])*(y-Y[2]) + c[2]*((y-Y[2]).pow(2)))
    fx4 = D[3]*torch.exp(a[3]*((x-X[3]).pow(2)) + b[3]*(x-X[3])*(y-Y[3]) + c[3]*((y-Y[3]).pow(2)))
    extra = gamma*torch.sin(2*k*pitorch*x)*torch.sin(2*k*pitorch*y)
    
    U = fx1+fx2+fx3+fx4
#     U = U+extra
    
    return U

def mueller(x):
    a = np.array([-1,-1,-6.5,0.7])
    b = np.array([0,0,11,0.6])
    c = np.array([-10,-10,-6.5,0.7])
    D = np.array([-200,-100,-170,15])
    X = np.array([1,0,-0.5,-1])
    Y = np.array([0,0.5,1.5,1])
    # gamma = torch.tensor([9])
    # k = torch.tensor([5])
  
    fx1 = D[0]*np.exp(a[0]*((x[:,0]-X[0])**(2)) + b[0]*(x[:,0]-X[0])*(x[:,1]-Y[0]) + c[0]*((x[:,1]-Y[0])**(2)))
    fx2 = D[1]*np.exp(a[1]*((x[:,0]-X[1])**(2)) + b[1]*(x[:,0]-X[1])*(x[:,1]-Y[1]) + c[1]*((x[:,1]-Y[1])**(2)))
    fx3 = D[2]*np.exp(a[2]*((x[:,0]-X[2])**(2)) + b[2]*(x[:,0]-X[2])*(x[:,1]-Y[2]) + c[2]*((x[:,1]-Y[2])**(2)))
    fx4 = D[3]*np.exp(a[3]*((x[:,0]-X[3])**(2)) + b[3]*(x[:,0]-X[3])*(x[:,1]-Y[3]) + c[3]*((x[:,1]-Y[3])**(2)))
    # extra = gamma*torch.sin(2*k*pitorch*x)*torch.sin(2*k*pitorch*y)
    
    U = fx1+fx2+fx3+fx4
#     U = U+extra
    return U

def dU_control(beta,q,gradq):
    control_x = torch.tensor(2)/beta*(gradq[:,0]/q)
    control_y = torch.tensor(2)/beta*(gradq[:,1]/q)
    
    return control_x,control_y

class Ruggedmueller2(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # output layer
        self.linear3 = nn.Linear(hidden_size, out_size)
        
    def forward(self, xb):
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        tanhf = nn.Tanh()
        out = tanhf(out)
        # Get predictions using output layer
        out = self.linear2(out)
        out = tanhf(out)
        out = self.linear3(out)
        # apply activation function again
        out = torch.sigmoid(out)
        return out

def chiAB(X):
    a = torch.tensor([-0.558,1.441])
    b = torch.tensor([0.623,0.028])
    r = torch.tensor(0.1)
    m = nn.Tanh()
    sizex, nothing = X.shape
    chiA = 0.5 - 0.5*m(1000*((((X - a).pow(2)).sum(dim = 1).reshape(sizex,1))-(r + torch.tensor(0.02)).pow(2)))                     
    chiB = 0.5 - 0.5*m(1000*((((X - b).pow(2)).sum(dim = 1).reshape(sizex,1))-(r + torch.tensor(0.02)).pow(2)))       
                             
    return chiA, chiB
def q_theta(X,chiA,chiB,q_tilde):
    Q = (torch.tensor([1]) - chiA)*(q_tilde*(torch.tensor([1]) - chiB)+chiB)
    return Q
