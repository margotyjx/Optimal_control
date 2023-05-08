import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy 
import math
import torch
import torch.onnx as onnx
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import *

def main():
    # import training data from files

    # data_type = 'EM'
    data_type = 'uniform'
    if data_type == 'EM':
        train_data = torch.load('./data/train_data_beta20.pt')
        train_data.requires_grad = True
    elif data_type == 'uniform':
        x = np.linspace(-2.5,2.5,400)
        y = np.linspace(-2,2,400)
        x_train, y_train = np.meshgrid(x,y)
        train_data = torch.tensor(np.hstack((x_train.ravel()[:,None],y_train.ravel()[:,None])),\
                             dtype = torch.float32)
        torch.save(train_data, './data/train_data_uniform.pt')
        train_data.requires_grad = True
    else:
        raise NameError('EM or uniform')
        
        
    data_process(train_data)
    
    train_Aset = torch.load('./data/trainA.pt')
    train_Bset = torch.load('./data/trainB.pt')
    train_inset = torch.load('./data/train_in.pt')
    train_Aset.requires_grad = True
    train_Bset.requires_grad = True
    train_inset.requires_grad = True
    
    ## initialization
    input_size = 2
    output_size = 1
    N_neurons = 40
    torch.manual_seed(1)
    model = Model1(input_size,N_neurons,output_size)
    
    size1,size2 = train_inset.shape
    rhs = torch.zeros(size1,)
    rhsA = torch.zeros(train_Aset.shape[0],1)
    rhsB = torch.ones(train_Bset.shape[0],1)
    train_ds = TensorDataset(train_inset,rhs)
    batch_size = int(size1/50) # the batch size is the size of the training data
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    loss_fn = nn.MSELoss()

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    optimizer1 = optim.Adam(model.parameters(), lr=1e-3)
    
    beta = torch.tensor(-1)
    alpha = torch.tensor(1)
    gamma = torch.tensor(0.5)
    Funs = functions(beta,alpha)
    betaT = torch.tensor(20)
    
    for epoch in range(500):
        for X,y in train_dl:
            #X[:,0] are positions, X[:,1] are momentum

            optimizer1.zero_grad()

            Q = model(X)
            QA = model(train_Aset)
            QB = model(train_Bset)
            U = Funs.funU(X[:,0]) # potential energy function
            T = Funs.funT(X[:,1]) # kinetic energy function
            gradU = Funs.dU(X[:,0])


            derivQ = torch.autograd.grad(Q,X,allow_unused=True, retain_graph=True, 
                                         grad_outputs = torch.ones_like(Q), create_graph=True)
            dQ = derivQ[0]

            deriv_yx_yy = torch.autograd.grad(dQ[:,1], X,allow_unused=True,grad_outputs=torch.ones_like(dQ[:,0]), \
            retain_graph=True, create_graph=True)
            dqqQ = deriv_yx_yy[0][:,1]

            Lq = X[:,1]*dQ[:,0] - gradU*dQ[:,1] - gamma * X[:,1]*dQ[:,1] + gamma/betaT*dqqQ

            lossin = loss_fn(Lq,y)
            lossA = loss_fn(QA,rhsA)
            lossB = loss_fn(QB,rhsB)

            loss = lossin + 0.5*lossA + 0.5*lossB
            loss.backward()
            optimizer1.step()
        if epoch%25 == 0:
            print('Epoch: {}, Loss interior pts: {:.4f}, Loss on A: {:.4f}, Loss on B: {:.4f}'.format(
            epoch, lossin, lossA, lossB))
    
    
    
    torch.save(model,'Duffiing_gamma0.5_beta'+str(betaT.item)+'_PINN.pt')
    
    
    
    
    
if __name__ == "__main__":
    main()