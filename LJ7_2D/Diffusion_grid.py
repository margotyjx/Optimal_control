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



def main():
    # Import initial points
    fname = data_folder/"LJ7_traj.npz"
    inData = np.load(fname)
    data = inData["data"]
    N = data.shape[1]

    fname = data_folder/"LJ7_traj_diffusions.npz"
    inData = np.load(fname)
    diffusions = inData["diffusions"]
    
    test_pts = torch.load('./data/test_pts_beta5.pt')
    
    M11grid = scipy.interpolate.griddata(np.transpose(data),diffusions[:,0,0],(xx,yy),method = 'linear')
    M12grid = scipy.interpolate.griddata(np.transpose(data),diffusions[:,0,1],(xx,yy),method = 'linear')
    M22grid = scipy.interpolate.griddata(np.transpose(data),diffusions[:,1,1],(xx,yy),method = 'linear')

    ind_bad = check_inf_nan(diffusions[:,0,0])
    ind_bad = check_inf_nan(diffusions[:,0,1])
    ind_bad = check_inf_nan(diffusions[:,1,1])

    # Mijgrid are matrix enties linearly interpolated onto regular grid
    # create linear interpolator functions
    M11fun = scipy.interpolate.RegularGridInterpolator((x,y),np.transpose(M11grid))
    M12fun = scipy.interpolate.RegularGridInterpolator((x,y),np.transpose(M12grid))
    M22fun = scipy.interpolate.RegularGridInterpolator((x,y),np.transpose(M22grid))
    # evaluate linear interpolator functions at the FEM mesh points
    M11pts = M11fun(new_grid)
    M12pts = M12fun(new_grid)
    M22pts = M22fun(new_grid)

    # at some points, the linear interpolator fails
    # We create a nearest neighbor interpolator for these bad points
    ind_bad = check_inf_nan(M11pts)
    M11fun_NN = scipy.interpolate.NearestNDInterpolator(np.transpose(data), diffusions[:,0,0])
    M11pts[ind_bad] = M11fun_NN(new_grid[ind_bad,:])

    ind_bad = check_inf_nan(M12pts)
    M12fun_NN = scipy.interpolate.NearestNDInterpolator(np.transpose(data), diffusions[:,0,1])
    M12pts[ind_bad] = M12fun_NN(new_grid[ind_bad,:])

    ind_bad = check_inf_nan(M22pts)
    M22fun_NN = scipy.interpolate.NearestNDInterpolator(np.transpose(data), diffusions[:,1,1])
    M22pts[ind_bad] = M22fun_NN(new_grid[ind_bad,:])

    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    plt.subplot(1,3,1)
    plt.scatter(new_grid[:,0], new_grid[:,1],c = M11pts)
    plt.colorbar(label="M11", orientation="horizontal")
    ind_bad = check_inf_nan(M11pts)
    plt.scatter(new_grid[ind_bad,0], new_grid[ind_bad,1],s = 0.1)
    plt.subplot(1,3,2)
    plt.scatter(new_grid[:,0], new_grid[:,1],c = M12pts)
    plt.colorbar(label="M12", orientation="horizontal")
    ind_bad = check_inf_nan(M12pts)
    plt.scatter(new_grid[ind_bad,0], new_grid[ind_bad,1],s = 0.1)
    plt.subplot(1,3,3)
    plt.scatter(new_grid[:,0], new_grid[:,1],c = M22pts)
    plt.colorbar(label="M22", orientation="horizontal")
    ind_bad = check_inf_nan(M22pts)
    plt.scatter(new_grid[ind_bad,0], new_grid[ind_bad,1],s = 0.1)
    plt.savefig('LJ72D_Mmatrix.pdf')

    
    M_flat = np.hstack((M11pts[:,None],M12pts[:,None],M12pts[:,None],M22pts[:,None]))
    M_diffusion = M_flat.reshape((4232,2,2))
    np.savez('./data/M_diffusion.npz',M_diffusion = M_diffusion)
    
    
    
    
if __name__== "__main__":
    main()