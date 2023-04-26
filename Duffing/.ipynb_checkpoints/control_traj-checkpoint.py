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
from utils import *

def get_init_pt_ellipse(delta_t, size, model, beta_np, rx, ry):
    circle_x = -1
    circle_y = 0
    rx = rx + delta_t
    ry = ry+ delta_t
    circle_r = 0.1
    
    alpha = 2 * np.pi * np.linspace(0,1,500)
    
    x = rx*np.cos(alpha)+circle_x
    y = ry*np.sin(alpha)+ circle_y

    normal_vec = np.vstack((-np.cos(alpha)*ry,-np.sin(alpha)*rx)).T
    normal_vec = normal_vec/np.linalg.norm(normal_vec, axis=1).reshape(-1,1)

    pts = torch.tensor(np.vstack((x,y)).T, dtype = torch.float32)
    pts.requires_grad = True

    Q = model(pts)
    derivQ = torch.autograd.grad(Q,pts,allow_unused=True, retain_graph=True, \
                                 grad_outputs = torch.ones_like(Q), create_graph=True)

    Hamilton = Hamiltonian(pts[:,0].detach().numpy(),pts[:,1].detach().numpy())

    inner_prod = (normal_vec*derivQ[0].detach().numpy())[:,1]
    prob_init = np.exp(-beta_np*Hamilton)*np.abs(inner_prod)
    prob_init = prob_init/np.sum(prob_init)
    
    plt.scatter(x, y, c = prob_init)
    plt.colorbar()
    
    x = np.linspace(-2,0,1000)
    y = np.linspace(-1,1,1000)
    xx,yy = np.meshgrid(x,y)
    A = Hamiltonian(xx,yy)

    plt.contour(xx,yy,A,colors='gray', levels = 10,linewidths = 1.5, alpha = 0.5)
    plt.xlim([-2,0.0])
    plt.ylim([-1,1])
    plt.xlabel('$x$')
    plt.ylabel('$p$')
    plt.savefig('./data/weight_init.pdf')
    plt.show()
        
    init_pt = np.random.choice(pts.shape[0],size,p = prob_init)
    
    pts.requires_grad = False
    
    return pts[init_pt,:]


def running_traj(Time,M,beta,gamma,delt,model, rx = 0.3, ry = 0.4):
    I = (Time+1)*np.ones(M)
    a = torch.tensor([-1,0])
    b = torch.tensor([1,0])
    
    
    Pts = get_init_pt_ellipse(0.0001, M, model, beta.item(), rx, ry)
    
    for m in range(M):
        
        newX = Pts[m,0].reshape(1,1)
        newY = Pts[m,1].reshape(1,1)

        w = torch.randn(Time)
        w = torch.sqrt(delt)*w
        
        
        inA = False
        lastA = 0

        for i in range(Time):

            newR = torch.cat((newX, newY), 1)

            newR.requires_grad_(True)

            dUx = dU_tensor(newX)
            Q = model(newR)
            derivQ = torch.autograd.grad(Q,newR,allow_unused=True, retain_graph=True, \
                                    grad_outputs = torch.ones_like(Q), create_graph=True)[0]

            dU_control_y = dU_control_tensor(beta,gamma,Q,derivQ)

            newX = newX + newY * delt
            newY = newY - (gamma*newY + dUx - dU_control_y)*delt + torch.sqrt(2*gamma/beta)*w[i]
            
            
            distA = (newX - a[0]).pow(2)/(rx**2) + (newY - a[1]).pow(2)/(ry**2)
            if distA <= 1.0:
                inA = True
            else:
                if inA == True:
                    lastA = i
                inA = False
            if inA == False:
                distB = (newX - b[0]).pow(2)/(rx**2) + (newY - b[1]).pow(2)/(ry**2)
                if distB <= 1.0:
                    I[m] = i - lastA
                    print('TAB: ', I[m])

                    break
            
    return I


def plot_traj(Time,M,beta,gamma,delt, model, rx = 0.3, ry = 0.4):
    beta_np = beta.item()
    I = (Time+1)*np.ones(M)
    a = torch.tensor([-1,0])
    b = torch.tensor([1,0])
    
    Pts = get_init_pt_ellipse(0.0001, M, model, beta.item(), rx, ry)
    
    for m in range(M):
        newX = Pts[m,0].reshape(1,1)
        newY = Pts[m,1].reshape(1,1)
        newX_orig = Pts[m,0].reshape(1,1)
        newY_orig = Pts[m,1].reshape(1,1)

        w = torch.randn(Time)
        w = torch.sqrt(delt)*w
        
        X = torch.tensor([])
        Y = torch.tensor([])
        
        X = torch.cat((X,newX), 0)
        Y = torch.cat((Y,newY), 0)
        
        X_orig = torch.tensor([])
        Y_orig = torch.tensor([])
        
        X_orig = torch.cat((X_orig,newX_orig), 0)
        Y_orig = torch.cat((Y_orig,newY_orig), 0)
        

        for i in range(Time):

            b = torch.tensor([1,0])

            newR = torch.cat((newX, newY), 1)

            newR.requires_grad_(True)

            dUx = dU_tensor(newX)
            Q = model(newR)
            derivQ = torch.autograd.grad(Q,newR,allow_unused=True, retain_graph=True, \
                                    grad_outputs = torch.ones_like(Q), create_graph=True)[0]

            dU_control_y = dU_control_tensor(beta,gamma,Q,derivQ)

            newX = newX + newY * delt
            newY = newY - (gamma*newY + dUx - dU_control_y)*delt + torch.sqrt(2*gamma/beta)*w[i]
            
            X = torch.cat((X,newX), 0)
            Y = torch.cat((Y,newY), 0)
            
            dUx_orig = dU_tensor(newX_orig)
            
            newX_orig = newX_orig + newY_orig * delt
            newY_orig = newY_orig - (gamma*newY_orig + dUx_orig)*delt + torch.sqrt(2*gamma/beta)*w[i]
            
            X_orig = torch.cat((X_orig,newX_orig), 0)
            Y_orig = torch.cat((Y_orig,newY_orig), 0)
            
            
        if m == 0:
            plt.figure(1)
            plt.scatter(X.detach().numpy(),Y.detach().numpy(),color = 'b', s = 1,label = 'trajectory 1')
            plt.figure(2)
            plt.scatter(X_orig.detach().numpy(),Y_orig.detach().numpy(),color = 'b', s = 1,label = 'trajectory 1')
        if m == 1:
            plt.figure(1)
            plt.scatter(X.detach().numpy(),Y.detach().numpy(),color = 'g', s = 1,label = 'trajectory 2')
            plt.figure(2)
            plt.scatter(X_orig.detach().numpy(),Y_orig.detach().numpy(),color = 'g', s = 1,label = 'trajectory 2')
        if m == 2:
            plt.figure(1)
            plt.scatter(X.detach().numpy(),Y.detach().numpy(),color = 'orange', s = 1,label = 'trajectory 3')
            plt.figure(2)
            plt.scatter(X_orig.detach().numpy(),Y_orig.detach().numpy(),color = 'orange', s = 1,label = 'trajectory 3')
    
    t = np.linspace(0,2*np.pi,200)
    x = np.linspace(-2,2,1000)
    y = np.linspace(-2,2,1000)
    xx,yy = np.meshgrid(x,y)
    A = Hamiltonian(xx,yy)
    
    plt.figure(1)
    plt.scatter(a[0],a[1])
    plt.scatter(b[0],b[1])
    Rx = 0.3
    Ry = 0.4
    a = [-1,0]
    b = [1,0]
    plt.plot(Rx*np.cos(t)+a[0],Ry*np.sin(t)+a[1])
    plt.plot(Rx*np.cos(t)+b[0],Ry*np.sin(t)+b[1])
    plt.contour(xx,yy,A,colors='gray', levels = 15,linewidths = 1.5, alpha = 0.5)
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$p$')
    plt.rcParams["figure.figsize"] = (5,5)
    ax = plt.gca()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_aspect(1)
    plt.savefig('./data/controlled_duffing_beta'+str(beta_np)+'_ellip.pdf')
    
    plt.figure(2)
    plt.scatter(a[0],a[1])
    plt.scatter(b[0],b[1])
    
    Rx = 0.3
    Ry = 0.4
    a = [-1,0]
    b = [1,0]
    plt.plot(Rx*np.cos(t)+a[0],Ry*np.sin(t)+a[1])
    plt.plot(Rx*np.cos(t)+b[0],Ry*np.sin(t)+b[1])
    plt.contour(xx,yy,A,colors='gray', levels = 15,linewidths = 1.5, alpha = 0.5)
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$p$')
    plt.rcParams["figure.figsize"] = (5,5)
    ax = plt.gca()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_aspect(1)
    plt.savefig('./data/uncontrolled_duffing_beta'+str(beta_np)+'_ellip.pdf')

                
