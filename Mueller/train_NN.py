import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.onnx as onnx
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd

import math
pitorch = torch.Tensor([math.pi])

import controlled_traj, utils, rates
from utils import Ruggedmueller2

def main():
    # load data
    train = torch.tensor([])

    df = pd.read_excel('./data/traindata_meta_delta-net_T10.xlsx')

    X_train = pd.DataFrame(df['X'])
    Y_train = pd.DataFrame(df['Y'])

    X_train = torch.tensor(X_train.values,dtype=torch.float32)
    Y_train = torch.tensor(Y_train.values,dtype=torch.float32)

    train_data = torch.cat((X_train, Y_train), 1)
    train_data.requires_grad_(True)
    
    #initialization
    input_size = 2
    output_size = 1
    N_neurons = [10,40]
    model = Ruggedmueller2(input_size,N_neurons[1],output_size)
    
    size1,size2 = train_data.shape
    rhs = torch.zeros(size1,)
    train_ds = TensorDataset(train_data,rhs)
    batch_size = int(size1/125) # the batch size is the size of the training data


    loss_fn = nn.L1Loss()

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    optimizer1 = optim.Adam(model.parameters(), lr=1e-4)

    beta = torch.tensor(1/10)
    
    for epoch in range(1000):
        for X,y in train_dl:
            optimizer1.zero_grad()
            chia,chib = utils.chiAB(X)
            q_tilde = model(X)
            Q = utils.q_theta(X,chia,chib,q_tilde)
            U = utils.funU(X[:,0],X[:,1])
            derivQ = torch.autograd.grad(Q,X,allow_unused=True, retain_graph=True, 
                                         grad_outputs = torch.ones_like(Q),create_graph=True)
            output = (torch.norm(derivQ[0],dim=1)**2)*torch.exp(-beta*U)
            loss = loss_fn(output, y)

            loss.backward()
            optimizer1.step()

        if epoch % 100 == 1:
            print('loss at epoch {} is {:.4f}'.format(epoch, loss))

    # torch.save(model, 'd-net_T10_2hidden40.pt')
    
    
if __name__ == "__main__":
    main()