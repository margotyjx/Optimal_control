import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import math
pitorch = torch.Tensor([math.pi])
import pickle
import pandas as pd
from pathlib import Path
import utils
from utils import Ruggedmueller2


def get_init_pt(delta_t, size, model, beta_np = 1/10):
    circle_x = -0.558
    circle_y = 1.441
    circle_r = 0.1

    # random angle
    alpha = 2 * math.pi * np.linspace(0,1,500)
    r = circle_r + delta_t
    x = r * np.cos(alpha) + circle_x
    y = r * np.sin(alpha) + circle_y

    normal_vec = np.vstack((-np.cos(alpha),-np.sin(alpha))).T

    pts = torch.tensor(np.vstack((x,y)).T, dtype = torch.float32)
    pts.requires_grad = True

    chia,chib = utils.chiAB(pts)
    q_tilde = model(pts)
    Q = utils.q_theta(pts,chia,chib,q_tilde)
    derivQ = torch.autograd.grad(Q,pts,allow_unused=True, retain_graph=True, \
                                 grad_outputs = torch.ones_like(Q), create_graph=True)
    U_mueller = utils.funU(pts[:,0],pts[:,1])

    inner_prod = np.sum(normal_vec*derivQ[0].detach().numpy(),axis=1)
    prob_init = np.exp(-beta_np*U_mueller.detach().numpy())*np.abs(inner_prod)
    prob_init = prob_init/np.sum(prob_init)
    
    plt.scatter(x, y, c = prob_init)
    plt.colorbar()
    
    x = np.linspace(-1.5,1,1000)
    y = np.linspace(-1.5,2.5,1000)
    xx,yy = np.meshgrid(x,y)
    A = utils.funU(torch.tensor(xx),torch.tensor(yy))

    plt.contour(xx,yy,A.detach().numpy(),colors='gray',levels = np.arange(-200,100,20), linewidths = 1.5, alpha = 0.5)
    plt.xlim([-1,0.0])
    plt.ylim([1,2])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.savefig('./data/weight_init.pdf')
    plt.show()
    
    init_pt = np.random.choice(pts.shape[0],size,p = prob_init)
    
    pts.requires_grad = False
    
    return pts[init_pt,:]


def running_traj(Time,M,beta,delt,model):
    print(Time)
    rsquare = 0.1**2
#     x0 = torch.tensor([[-0.65]])
#     y0 = torch.tensor([[1.441]])
    I = (Time+1)*np.ones(M)
    
    Pts = get_init_pt(0.001, M, model)
    
    for m in range(M):
        
        newX = Pts[m,0].reshape(1,1)
        newY = Pts[m,1].reshape(1,1)

        w = torch.randn(Time,2)
        w = torch.sqrt(delt)*w
        

        flag = 0

        for i in range(Time):

            b = torch.tensor([0.623,0.028])

            newR = torch.cat((newX, newY), 1)

            newR.requires_grad_(True)

            dUx,dUy = utils.dU(newX,newY)
            chia,chib = utils.chiAB(newR)
            q_tilde = model(newR)
            Q = utils.q_theta(newR,chia,chib,q_tilde)
            derivQ = torch.autograd.grad(Q,newR,allow_unused=True, retain_graph=True, \
                                             grad_outputs = torch.ones_like(Q), create_graph=True)

            dU_control_x, dU_control_y = utils.dU_control(beta,Q,derivQ[0])

            newX = newX - (dUx - dU_control_x)*delt + torch.sqrt(2*beta**(-1))*w[i,0]
            newY = newY - (dUy - dU_control_y)*delt + torch.sqrt(2*beta**(-1))*w[i,1]


            # from the time when the trajectory hit B, values in list chi_B will become one
            if flag == 0:
                distB = (newX - b[0]).pow(2) + (newY - b[1]).pow(2)
                if distB <= rsquare:
                    flag = 1
                    I[m] = i 
                    print('we break at: ', i)
                    break
    return I 


def plot_traj(Time,M,beta,delt,model):
    print(Time)
    a = torch.tensor([-0.558,1.441])
    
    I = (Time+100)*torch.ones(M)
    Pts = get_init_pt(0.001, M, model)
    
    X_save = []
    Y_save = []
    X_orig_save = []
    Y_orig_save = []
    for m in range(M):
        x0 = Pts[m,0].reshape(1,1)
        y0 = Pts[m,1].reshape(1,1)
        
        newX = x0
        newY = y0
        
        newX_orig = x0
        newY_orig = y0

        w = torch.randn(Time,2)
        w = torch.sqrt(delt)*w
        
        flag = 0
        
        X = torch.tensor([])
        Y = torch.tensor([])
        
        X = torch.cat((X,x0), 0)
        Y = torch.cat((Y,y0), 0)
        
        X_orig = torch.tensor([])
        Y_orig = torch.tensor([])
        
        X_orig = torch.cat((X_orig,x0), 0)
        Y_orig = torch.cat((Y_orig,y0), 0)

        for i in range(Time):

            b = torch.tensor([0.623,0.028])

            newR = torch.cat((newX, newY), 1)

            newR.requires_grad_(True)

            dUx,dUy = utils.dU(newX,newY)
            chia,chib = utils.chiAB(newR)
            q_tilde = model(newR)
            Q = utils.q_theta(newR,chia,chib,q_tilde)
            derivQ = torch.autograd.grad(Q,newR,allow_unused=True, retain_graph=True, \
                                             grad_outputs = torch.ones_like(Q), create_graph=True)

            dU_control_x, dU_control_y = utils.dU_control(beta,Q,derivQ[0])

            newX = newX - (dUx - dU_control_x)*delt + torch.sqrt(2*beta**(-1))*w[i,0]
            newY = newY - (dUy - dU_control_y)*delt + torch.sqrt(2*beta**(-1))*w[i,1]
            
            X = torch.cat((X,newX), 0)
            Y = torch.cat((Y,newY), 0)
            
            dUx_orig,dUy_orig = utils.dU(newX_orig,newY_orig)
            
            newX_orig = newX_orig - dUx_orig*delt + torch.sqrt(2*beta**(-1))*w[i,0]
            newY_orig = newY_orig - dUy_orig*delt + torch.sqrt(2*beta**(-1))*w[i,1]
            
            X_orig = torch.cat((X_orig,newX_orig), 0)
            Y_orig = torch.cat((Y_orig,newY_orig), 0)
        
        
        X_save.append(X)
        Y_save.append(Y)
        X_orig_save.append(X_orig)
        Y_orig_save.append(Y_orig)
        if m == 0:
            plt.figure(1)
            plt.scatter(X.detach().numpy(),Y.detach().numpy(),color = 'b', s = 2.5,label = 'trajectory 1')
            plt.figure(2)
            plt.scatter(X_orig.detach().numpy(),Y_orig.detach().numpy(),color = 'b', s = 1,label = 'trajectory 1')
        if m == 1:
            plt.figure(1)
            plt.scatter(X.detach().numpy(),Y.detach().numpy(),color = 'g', s = 2.5,label = 'trajectory 2')
            plt.figure(2)
            plt.scatter(X_orig.detach().numpy(),Y_orig.detach().numpy(),color = 'g', s = 1,label = 'trajectory 2')
        if m == 2:
            plt.figure(1)
            plt.scatter(X.detach().numpy(),Y.detach().numpy(),color = 'orange', s = 2.5,label = 'trajectory 3')
            plt.figure(2)
            plt.scatter(X_orig.detach().numpy(),Y_orig.detach().numpy(),color = 'orange', s = 1,label = 'trajectory 3')
            
    x = np.linspace(-1.5,1,1000)
    y = np.linspace(-1.5,2.5,1000)
    xx,yy = np.meshgrid(x,y)
    A = utils.funU(torch.tensor(xx),torch.tensor(yy))
    
    t = np.linspace(0,2*np.pi,200)
    plt.figure(1)
    plt.scatter(a[0],a[1])
    plt.scatter(b[0],b[1])
    plt.contour(xx,yy,A.detach().numpy(),colors='black',levels = np.arange(-200,100,20), linewidths = 1)
    plt.plot(a[0]+0.1*np.cos(t),a[1]+0.1*np.sin(t))
    plt.plot(b[0]+0.1*np.cos(t),b[1]+0.1*np.sin(t))
    plt.xlim([-1.5,1])
    plt.ylim([-0.5,2])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    ax = plt.gca()
    ax.set_aspect(1)
#     plt.rcParams["figure.figsize"] = (5,5)
    plt.savefig('./data/controlled_mueller_T10.pdf')
    
    plt.figure(2)
    plt.scatter(a[0],a[1])
    plt.scatter(b[0],b[1])
    plt.xlim([-1.5,1])
    plt.ylim([-0.5,2])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.plot(a[0]+0.1*np.cos(t),a[1]+0.1*np.sin(t))
    plt.plot(b[0]+0.1*np.cos(t),b[1]+0.1*np.sin(t))
    plt.contour(xx,yy,A.detach().numpy(),colors='black',levels = np.arange(-200,100,20), linewidths = 1)
    plt.legend()
    ax = plt.gca()
    ax.set_aspect(1)
#     plt.rcParams["figure.figsize"] = (5,5)
    plt.savefig('./data/uncontrolled_mueller_T10.pdf')
    
    plt.show()
    
    return X_save, Y_save, X_orig_save, Y_orig_save

    



    

    

