import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from numpy.random import choice
from pathlib import Path
import scipy.interpolate
import scipy.stats
pitorch = torch.Tensor([math.pi])

def dU_tensor(x):
    beta = torch.tensor([[-1]])
    alpha = torch.tensor([[1]])
    dUx = x*(beta + alpha * x**2)
    
    return dUx

def dU_control_tensor(beta,gamma,q,gradq):
    control = torch.tensor(2)*gamma/beta*(gradq[:,1]/q)
    
    return control


class functions:
    def __init__(self,beta,alpha):
        self.beta = beta
        self.alpha = alpha
        
    def funU(self,x):
        U = 0.5*self.beta*(x**2) + 1/4*self.alpha*(x**4)

        return U

    def funT(self,p):
        T = 0.5*(p**2)
        return T

    def dU(self,x):
        dUx = x*(self.beta + self.alpha * (x**2))

        return dUx
    
def Hamiltonian(x,y):
    return 0.5*y**2 + 0.25*x**4 - 0.5*x**2
def fpot_FEM(x):
    return Hamiltonian(x[:,0],x[:,1])
def divfree_FEM(x,y):
    f1 = y
    f2 = -x**3 + x
    return f1,f2

class Model1(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)
        
    def forward(self, xb):
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        tanhf = nn.Tanh()
        out = tanhf(out)
        # Get predictions using output layer
        out = self.linear2(out)
        # apply activation function again
        out = torch.sigmoid(out)
        return out
    
def data_process(train_data):
    """
    boundary points of A will be denoted as 1, boundary points of B will be 2, non boundary points will be 0
    """
    category = np.zeros(train_data.shape[0])

    for i in range(train_data.shape[0]):
        a = torch.tensor([-1,0])
        b = torch.tensor([1,0])
        rx = 0.3
        ry = 0.4
        curr_xy = train_data[i,:]
        distB = (curr_xy - b).pow(2)/torch.tensor([0.3**2, 0.4**2])
        distA = (curr_xy - a).pow(2)/torch.tensor([0.3**2, 0.4**2])

        if distA.sum() <= 1:
            category[i] = 1
        elif distB.sum() <= 1:
            category[i] = 2
            
    Aset = np.argwhere(category == 1)
    Bset = np.argwhere(category == 2)
    inset = np.argwhere(category == 0)
    train_Aset = train_data[Aset[:,0],:]
    train_Bset = train_data[Bset[:,0],:]
    train_inset = train_data[inset[:,0],:]
    torch.save(train_Aset,'./data/trainA.pt')
    torch.save(train_Bset,'./data/trainB.pt')
    torch.save(train_inset,'./data/train_in.pt')
    