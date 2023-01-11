import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.interpolate
import math
import time


# region
def LJpot(x): # Lennard-Jones potential, x is the position of each particles
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
def LJgrad(x):
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


def MALAstep(x,pot_x,grad_x,fpot,fgrad,beta,dt,std):
#     std = sqrt(2*dt/beta)
    w = np.random.normal(0.0,std,np.shape(x))
    y = x - dt*grad_x + w
    pot_y = fpot(y)
    grad_y = fgrad(y)
    qxy =  np.sum(w**2)  #||w||^2
    qyx = np.sum((x - y + dt*grad_y)**2) # ||x-y+dt*grad V(y)||
    alpha = np.exp(-beta*(pot_y-pot_x+(qyx-qxy)*0.25/dt))
    if alpha < 1: # accept move # are we actually mean when alpha >= 1?
        x = y
        pot_x = pot_y
        grad_x = grad_y
        # print("ACCEPT: alpha = ",alpha)
    else:
        eta = np.random.uniform(0.0,1.0,(1,))
        if eta < alpha: # accept move
            x = y
            pot_x = pot_y
            grad_x = grad_y
            # print("ACCEPT: alpha = ",alpha," eta = ",eta)
        else:
            pass
#             print("REJECT: alpha = ",alpha," eta = ",eta)
    return x,pot_x,grad_x

def C(x):
    Na = np.size(x,axis = 1) # x has shape [2,7]
    C = np.zeros(Na)
    r2 = np.zeros((Na,Na)) # matrix of distances squared
    for k in range(Na):
        r2[k,:] = (x[0,:]-x[0,k])**2 + (x[1,:]-x[1,k])**2
        r2[k,k] = 0
    for i in range(Na):
        ci = np.divide(np.ones_like(r2[i,:]) - (r2[i,:]/2.25)**4, np.ones_like(r2[i,:]) - (r2[i,:]/2.25)**8)
        ci_sum = np.sum(np.delete(ci,i))
        C[i] = ci_sum
    return C

def mu2n3(x):
    C_list = C(x)
    ave_C = np.average(C_list)
    mu2 = np.average((C_list - ave_C)**2)
    mu3 = np.average((C_list - ave_C)**3)
    return mu2, mu3
# endregion


def running_traj(Time,M,beta,dt,X0):
    Count = np.zeros(M) # number of times we hit B
    Rx = 0.15
    Ry = 0.03
    theta = 5*np.pi/12
    rA = 0.072**2

    a = [0.5526,-0.0935]
    std = np.sqrt(2*dt/beta)

    for m in range(M):
        X = X0[:,:,m]
        pot_x = LJpot(X)
        grad_x = LJgrad(X)
        mu2, mu3 = mu2n3(X) # collective variables

        leftA = 1

        Mu2i = np.array([])
        Mu3i = np.array([])

        for i in range(Time):
            b = [0.7184,1.1607]

            X,pot_x,grad_x = MALAstep(X,pot_x,grad_x,LJpot,LJgrad,beta,dt,std)
            # in collective variables
            mu2, mu3 = mu2n3(X)

            Mu2i = np.append(Mu2i,mu2)
            Mu3i = np.append(Mu3i,mu3)

            """
            when the leftA is True(1), it means the trajectory last hit A; when leftA is False(0),
            the trajectory last hit B.
            """
            if leftA == 1: #last hit A
                distB = ((mu2 - b[0])*np.cos(theta)+(mu3 - b[1])*np.sin(theta))**2/(Rx**2) + \
                        ((mu2 - b[0])*np.sin(theta)-(mu3 - b[1])*np.cos(theta))**2/(Ry**2)

                if distB <= 1.0:
                    leftA = 0
                    Count[m] += 1
            else: # last hit B
                distA = (mu2 - a[0])**2 + (mu3 - a[1])**2
                if distA <= rA:
                    leftA = 1

        plt.scatter(Mu2i,Mu3i,s = 0.1)
        plt.scatter(a[0],a[1])
        plt.scatter(b[0],b[1])
        plt.xlim([0.2,1.2])
        plt.ylim([-0.5,1.5])
        plt.show()

    return Count


def running_traj_TAB(num_steps,M,beta,dt,X0):
    file = open("out.txt", "a")
    file.write("dt = %s \n" % dt)
    file.write("M = %s \n" % M)
    file.write("beta = %s \n" % beta)
    file.write("Tab=")
    Count = np.zeros(M) # number of times we hit B
    Rx = 0.15
    Ry = 0.03
    theta = 5*np.pi/12
    rA = 0.072**2

    a = [0.5526,-0.0935]
    std = np.sqrt(2*dt/beta)

    TAB = np.zeros(M)

    for m in range(M):
        X = X0[:,:,m]
        pot_x = LJpot(X)
        grad_x = LJgrad(X)
        mu2, mu3 = mu2n3(X) # collective variables

        lastA = 0 # last time we left A
        inA = True # flag whether we are in region A
        print(f"Starting simulation {m}")
        file.write(f"Starting simulation {m} \n")
        for i in range(num_steps):
            b = [0.7184,1.1607]

            X,pot_x,grad_x = MALAstep(X,pot_x,grad_x,LJpot,LJgrad,beta,dt,std)
            # in collective variables
            mu2, mu3 = mu2n3(X)

            distA = (mu2 - a[0])**2 + (mu3 - a[1])**2
            if distA <= rA:
                inA = True
            else:
                if inA == True:
                    lastA = i
                inA = False

            if inA == False:
                distB = ((mu2 - b[0])*np.cos(theta)+(mu3 - b[1])*np.sin(theta))**2/(Rx**2) + \
                        ((mu2 - b[0])*np.sin(theta)-(mu3 - b[1])*np.cos(theta))**2/(Ry**2)
                if distB <= 1.0:
                    TAB[m] = i - lastA
                    print('TAB: %s' % TAB[m])
                    file.write("%s, " % TAB[m])
                    break
    file.write("\n")
    file.close()
    return TAB

def main():
    # Import initial points
    Data = np.load("Mala_Boundary_Samples.npz")
    Abdry = Data['ABord']
    Abdry_reshaped = np.transpose(Abdry.reshape((7,2,200)),(1,0,2))

    # region
    dt = 5e-5
    beta = 5
    M = 200
    Length = 10000000000

    # randomized initial points
    init = np.random.randint(200, size=M)
    X0 = Abdry_reshaped[:,:,init]
    # endregion

    init = time.time()
    TAB = running_traj_TAB(Length,M,beta,dt,X0)
    final = time.time()
    WallTime = final - init

    file = open("out.txt", "a")
    file.write("WallTime = %s \n" % WallTime)
    file.write("TAB= %s" % TAB)
    file.write("\n")
    file.close()

    # np.savez('Count_original.npz',Count = Count)
    np.savez('TAB_original.npz',TAB = TAB)

if __name__== "__main__":
    main()
