import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import scipy.interpolate
import scipy.stats
import math
import utils_LJ7
from utils_LJ7 import LJ7_2
pitorch = torch.Tensor([math.pi])

# +
def get_init_pt(delta_t, size, model, X0, beta_np = 5):
    # X_0 contains all initial points
    fname = "./LJ7/LJ7_free_energy_grid.npz"
    inData = np.load(fname)
    free_energy = inData["free_energy"]
    nx = inData["nx"]
    ny = inData["ny"]
    xmin = inData["xmin"]
    xmax = inData["xmax"]
    ymin = inData["ymin"]
    ymax = inData["ymax"]

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y)

    FE = scipy.interpolate.RegularGridInterpolator((x,y),np.transpose(free_energy)) #, method = "linear",bounds_error=False, fill_value = 1.8)

    circle_x = 0.5526
    circle_y = -0.0935
    circle_r = 0.1

    # compute angle
    MU2 = []
    MU3 = []
    for i in range(X0.shape[-1]):
        x = X0[:,:,i]
        mu2, mu3 = utils_LJ7.mu2n3(x)
        MU2.append(mu2)
        MU3.append(mu3)
        
    Mu = torch.tensor(np.vstack((MU2, MU3)).T, dtype = torch.float32)

    cos_alpha = np.array(MU2) - circle_x
    sin_alpha = np.array(MU3) - circle_y

    normal_vec = np.vstack((-cos_alpha,-sin_alpha)).T

    # X0.requires_grad = True

    Mu.requires_grad = True

    chia,chib = utils_LJ7.chiAB(Mu)
    q_tilde = model(Mu)
    Q = utils_LJ7.q_theta(Mu,chia,chib,q_tilde)
    derivQ = torch.autograd.grad(Q,Mu,allow_unused=True, retain_graph=True, \
                             grad_outputs = torch.ones_like(Q), create_graph=True)

#     pot_x = utils_LJ7.LJpot(x)
    Fmu = FE(Mu.detach().numpy())

    inner_prod = np.sum(normal_vec*derivQ[0].detach().numpy(),axis=1)
#     prob_init = np.exp(-beta_np*pot_x.detach().numpy())*np.abs(inner_prod)
    prob_init = np.exp(-beta_np*Fmu)*np.abs(inner_prod)
    prob_init = prob_init/np.sum(prob_init)
    
    plt.scatter(Mu[:,0].detach().numpy(), Mu[:,1].detach().numpy(), c = prob_init)
    plt.colorbar()
    plt.contour(xx, yy, free_energy, levels=15,colors = 'gray', alpha = 0.5)
    plt.xlim([0.3,0.8])
    plt.ylim([-0.4,0.2])
    plt.xlabel('$\mu_2$')
    plt.ylabel('$\mu_3$')
    plt.savefig('./data/weight_init.pdf')
#     plt.show()

    init_pt = np.random.choice(Mu.shape[0],size,p = prob_init)

    
    return X0[:,:,init_pt]


# +
def running_traj(save_file, Time,M,beta,dt,model,Abdry):
    I = (Time+1)*np.ones(M)
    Rx = 0.15
    Ry = 0.03
    theta = 5*pitorch/12
    X0 = get_init_pt(0.001, M, model, Abdry, beta_np = 5)
    
    a = [0.5526,-0.0935]
    for m in range(M):
        print('iteration ',m)
        X = X0[:,:,m]
        pot_x = utils_LJ7.LJpot(X)
        grad_x = utils_LJ7.LJgrad(X)
        X.requires_grad_(True)
        mu2, mu3 = utils_LJ7.mu2n3(X) # collective variables
        flag = 0
        Mu2i = torch.tensor([]) 
        Mu3i = torch.tensor([])
        for i in range(Time):
            b = torch.tensor([0.7184,1.1607])
            
            Mu = torch.cat((mu2.reshape(1,1),mu3.reshape(1,1)),1)

            chia,chib = utils_LJ7.chiAB(Mu)
            q_tilde = model(Mu)
            Q = utils_LJ7.q_theta(Mu,chia,chib,q_tilde)
            derivQ = utils_LJ7.deriv_q(Q,X)
#             print(Q)
            # here we calculate new_x, with its potential and gradient calculated
            X,pot_x,grad_x = utils_LJ7.biased_MALAstep(X,pot_x,grad_x,Q,derivQ,utils_LJ7.LJpot,utils_LJ7.LJgrad,beta,dt)
#             X,pot_x,grad_x = biased_MALAstep_reflect(X,pot_x,grad_x,Q,derivQ,LJpot,LJgrad,beta,dt)
            # in collective variables
            X.requires_grad_(True)
            mu2, mu3 = utils_LJ7.mu2n3(X)
            

            # from the time when the trajectory hit B, values in list chi_B will become one
            if flag == 0:
                distB = ((mu2 - b[0])*torch.cos(theta)+(mu3 - b[1])*torch.sin(theta))**2/(Rx**2) + \
                ((mu2 - b[0])*torch.sin(theta)-(mu3 - b[1])*torch.cos(theta))**2/(Ry**2)
                
                if distB <= torch.tensor(1):
                    flag = 1
                    I[m] = i 
                    print('iteration: {}, tau_AB: {}'.format(m, i))
                    break
                    
        np.save(save_file,I)
        
#     return I
# -

def plot_traj(Time,M,beta,dt,model,Abdry):
    I = (Time+100)*np.ones(M)
    Rx = 0.15
    Ry = 0.03
    theta = 5*np.pi/12
    std = np.sqrt(2*dt/beta) 
    a = [0.5526,-0.0935]
    X0 = get_init_pt(0.001, M, model, Abdry, beta_np = 5)
    for m in range(M):
        print('iteration ',m)
        X_orig = X0[:,:,m]
        X = X0[:,:,m]
        pot_x = utils_LJ7.LJpot(X)
        grad_x = utils_LJ7.LJgrad(X)
        
        pot_x_orig = pot_x
        grad_x_orig = grad_x
        
        X.requires_grad_(True)
        mu2, mu3 = utils_LJ7.mu2n3(X) # collective variables
        flag = 0
        
        Mu2i = torch.tensor([]) 
        Mu3i = torch.tensor([])
        
        Mu2i_orig = torch.tensor([]) 
        Mu3i_orig = torch.tensor([])
        
        Mu2i = torch.cat((Mu2i,mu2.reshape(1,1)), 0)
        Mu3i = torch.cat((Mu3i,mu3.reshape(1,1)), 0)
        
        Mu2i_orig = torch.cat((Mu2i_orig,mu2.reshape(1,1)), 0)
        Mu3i_orig = torch.cat((Mu3i_orig,mu3.reshape(1,1)), 0)
        
        for i in range(Time):
            b = [0.7184,1.1607]
            w = np.random.normal(0.0,std,np.shape(X))
            
            Mu = torch.cat((mu2.reshape(1,1),mu3.reshape(1,1)),1)

            chia,chib = utils_LJ7.chiAB(Mu)
            q_tilde = model(Mu)
            Q = utils_LJ7.q_theta(Mu,chia,chib,q_tilde)
            derivQ = utils_LJ7.deriv_q(Q,X)
            # here we calculate new_x, with its potential and gradient calculated
            X,pot_x,grad_x = utils_LJ7.biased_MALAstep(X,pot_x,grad_x,Q,derivQ,utils_LJ7.LJpot,
                                                       utils_LJ7.LJgrad,beta,dt)
            # in collective variables
            X.requires_grad_(True)
            mu2, mu3 = utils_LJ7.mu2n3(X)
            
            Mu2i = torch.cat((Mu2i,mu2.reshape(1,1)), 0)
            Mu3i = torch.cat((Mu3i,mu3.reshape(1,1)), 0)
            
            X_orig, pot_x_orig, grad_x_orig = utils_LJ7.MALAstep(X_orig,pot_x_orig,grad_x_orig,utils_LJ7.LJpot,
                                                                 utils_LJ7.LJgrad,beta,dt)
            mu2_orig, mu3_orig = utils_LJ7.mu2n3(X_orig)
            
            Mu2i_orig = torch.cat((Mu2i_orig,mu2_orig.reshape(1,1)), 0)
            Mu3i_orig = torch.cat((Mu3i_orig,mu3_orig.reshape(1,1)), 0)
        
        if m == 0:
            plt.figure(1)
            plt.scatter(Mu2i.detach().numpy(),Mu3i.detach().numpy(),color = 'b', s = 1,label = 'trajectory 1')
            plt.figure(2)
            plt.scatter(Mu2i_orig.detach().numpy(),Mu3i_orig.detach().numpy(),color = 'b', s = 1,label = 'trajectory 1')
        if m == 1:
            plt.figure(1)
            plt.scatter(Mu2i.detach().numpy(),Mu3i.detach().numpy(),color = 'g', s = 1,label = 'trajectory 2')
            plt.figure(2)
            plt.scatter(Mu2i_orig.detach().numpy(),Mu3i_orig.detach().numpy(),color = 'g', s = 1,label = 'trajectory 2')
        if m == 2:
            plt.figure(1)
            plt.scatter(Mu2i.detach().numpy(),Mu3i.detach().numpy(),color = 'orange', s = 1,label = 'trajectory 3')
            plt.figure(2)
            plt.scatter(Mu2i_orig.detach().numpy(),Mu3i_orig.detach().numpy(),color = 'orange', s = 1,label = 'trajectory 3')
    
    t = np.linspace(0,2*np.pi,200)
    plt.figure(1)
    plt.scatter(a[0],a[1])
    plt.scatter(b[0],b[1])
    plt.plot(Rx*np.cos(t)*np.cos(theta) - Ry*np.sin(t)*np.sin(theta)+b[0],
         Rx*np.cos(t)*np.sin(theta) + Ry*np.sin(t)*np.cos(theta)+b[1])
    plt.plot(a[0]+0.1034*np.cos(t),a[1]+0.1034*np.sin(t))
    plt.xlim([0.2,1.2])
    plt.ylim([-0.5,1.7])
    plt.xlabel('$\mu_2$')
    plt.ylabel('$\mu_3$')
    plt.legend()
#     plt.savefig('controlled_LJ7_beta5.pdf')
    
    plt.figure(2)
    plt.scatter(a[0],a[1])
    plt.scatter(b[0],b[1])
    plt.plot(Rx*np.cos(t)*np.cos(theta) - Ry*np.sin(t)*np.sin(theta)+b[0],
         Rx*np.cos(t)*np.sin(theta) + Ry*np.sin(t)*np.cos(theta)+b[1])
    plt.plot(a[0]+0.1034*np.cos(t),a[1]+0.1034*np.sin(t))
    plt.xlim([0.2,1.2])
    plt.ylim([-0.5,1.7])
    plt.xlabel('$\mu_2$')
    plt.ylabel('$\mu_3$')
    plt.legend()
#     plt.savefig('uncontrolled_LJ7_beta5.pdf')
    
    plt.show()
                    
    return Mu2i,Mu3i,Mu2i_orig,Mu3i_orig





