import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import scipy.interpolate
import utils_LJ7
from utils_LJ7 import LJ7_2

import math
pitorch = torch.Tensor([math.pi])

def main():
    # load data
    data_folder = Path('./LJ7')
    fname = data_folder/"LJ7_traj.npz"
    inData = np.load(fname)
    data = inData["data"]
    print(f"Shape of trajectory data:{data.shape}")
    N = data.shape[1]
    data = torch.tensor(data,dtype=torch.float32)
    train_data = torch.cat((data[0,:].reshape(N,1),data[1,:].reshape(N,1)),dim = 1)
    train_data.requires_grad_(True)


    # Load Diffusions
    fname = data_folder/"LJ7_traj_diffusions.npz"
    inData = np.load(fname)
    diffusions = inData["diffusions"]
    print(f"Shape of diffusion data:{diffusions.shape}")
    diffusions_phipsi = torch.tensor(diffusions,dtype=torch.float32)
    
    # initialization
    input_size = 2
    output_size = 1
    model = LJ7_2(input_size,10,10,output_size)
    
    size1,size2 = train_data.shape
    rhs = torch.zeros(size1,)
    train_ds = TensorDataset(train_data,rhs,diffusions_phipsi)
    batch_size = int(size1/125)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    loss_fn = nn.L1Loss()
    optimizer1 = optim.Adam(model.parameters(), lr=1e-3)
    scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[500,1000], gamma=0.5)

    beta = 5
    
    for epoch in range(1000):
        for X,y,diffusion_map in train_dl:
            optimizer1.zero_grad()

            chia,chib = utils_LJ7.chiAB(X)
            q_tilde = utils_LJ7.model(X)
            Q = utils_LJ7.q_theta(X,chia,chib,q_tilde)

            derivQ = torch.autograd.grad(Q,X,allow_unused=True, retain_graph=True, 
                                         grad_outputs = torch.ones_like(Q), create_graph=True)
            product = torch.tensor([])

            for i in range(batch_size):
                product_new = derivQ[0][i]@diffusion_map[i]@derivQ[0][i].reshape(2,1)
                product = torch.cat((product,product_new.reshape(1,1)), dim = 0)

            output = product.reshape(batch_size)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer1.step()

        if epoch%25 == 0:
            print('Epoch: {}, Loss : {:.4f}'.format(epoch, loss))
            
    # torch.save(model,'LJ7_2hidden_LJ7.pt')
    
    
if __name__ == "__main__":
    main()
