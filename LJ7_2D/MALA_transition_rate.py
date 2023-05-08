import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.interpolate
import math
import pickle
import scipy.stats
from os.path import exists


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
    if alpha < 1: 
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


def running_traj_TAB(Time,M,beta,dt,X0):
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
        inA = False # flag whether we are in region A


        for i in range(Time):
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
                    print('TAB: ', TAB[m])

                    break
    
        
    return TAB

def running_traj_alldata(Time,M,beta,dt,X0):
    Count = [] # number of times we hit B
    TAB = []
    TA = []
    TB = []
    Rx = 0.15
    Ry = 0.03
    theta = 5*np.pi/12
    rA = 0.1**2
    
    a = [0.5526,-0.0935]
    std = np.sqrt(2*dt/beta)
    
    for m in range(M):
        tab = []
        ta = []
        tb = []
        c = 0
        
        X = X0[:,:,m]
        pot_x = LJpot(X)
        grad_x = LJgrad(X)
        mu2, mu3 = mu2n3(X) # collective variables
        
        leftA = 1 # state of the trajectory 
        lastA = 0 # time we last visited A
        firstA = 0 # first time we enter A after leaving B
        firstB = 0 # first time we enter B after leaving A
        
        for i in range(Time):
            b = [0.7184,1.1607]

            X,pot_x,grad_x = MALAstep(X,pot_x,grad_x,LJpot,LJgrad,beta,dt,std)
            # in collective variables
            mu2, mu3 = mu2n3(X)

            """
            when the leftA is 1, it means the trajectory last hit A; when leftA 2, it means the trajectory is in A; 
            when leftA is 0, the trajectory last hit B. 
            """
            if leftA == 1: #last hit A
                distA = (mu2 - a[0])**2 + (mu3 - a[1])**2
                if distA <= rA:
                    leftA = 2
                else: # didn't return to A but last hit A
                    distB = ((mu2 - b[0])*np.cos(theta)+(mu3 - b[1])*np.sin(theta))**2/(Rx**2) + \
                            ((mu2 - b[0])*np.sin(theta)-(mu3 - b[1])*np.cos(theta))**2/(Ry**2)

                    if distB <= 1.0:
                        leftA = 0
                        c += 1
                        tab.append(i - lastA)
                        firstB = i #first time hitting B after leaving A
                        ta.append(i - firstA)
                        
            elif leftA == 2: # currently in A
                distA = (mu2 - a[0])**2 + (mu3 - a[1])**2
                if distA <= rA:
                    leftA = 2
                else:
                    leftA = 1
                    lastA = i
            else: # last hit B
                distA = (mu2 - a[0])**2 + (mu3 - a[1])**2
                if distA <= rA:
                    leftA = 1
                    lastA = i
                    tb.append(i - firstB)
                    firstA = i # first time hitting A after leaving B
        # store the number of counts
        if exists("LJ7_count_new.pickle"):
            with open("LJ7_count_new.pickle", "rb") as f:
                Count = pickle.load(f) 
        Count.append(c)
        with open("LJ7_count_new.pickle", "wb") as f:
            pickle.dump(Count, f)
            
        
        # store the transition time
        if exists("LJ7_TAB_new.pickle"):
            with open("LJ7_TAB_new.pickle", "rb") as f:
                TAB = pickle.load(f) 
        TAB.append(tab)
        with open("LJ7_TAB_new.pickle", "wb") as f:
            pickle.dump(TAB, f) 
        
        # store TA
        if exists("LJ7_TA_new.pickle"):
            with open("LJ7_TA_new.pickle", "rb") as f:
                TA = pickle.load(f) 
        TA.append(ta)
        with open("LJ7_TA_new.pickle", "wb") as f:
            pickle.dump(TA, f)
        
        #store TB
        if exists("LJ7_TB_new.pickle"):
            with open("LJ7_TB_new.pickle", "rb") as f:
                TB = pickle.load(f) 
        TB.append(tb)
        with open("LJ7_TB_new.pickle", "wb") as f:
            pickle.dump(TB, f)

# Import initial points
Data = np.load("./data/Mala Boundary Samples re-1.npz")
Abdry = Data['ABord']
Abdry_reshaped = np.transpose(Abdry.reshape((7,2,200)),(1,0,2))

# region
dt =5e-5
beta = 5
M = 5 # number of runs
Length = 100000000 # 1e8 time steps 

# randomized initial points
init = np.random.randint(200, size=M)
X0 = Abdry_reshaped[:,:,init]
# endregion

# region
# Count = running_traj(1500,M,beta,dt,X0)
# TAB = running_traj_TAB(150000,M,beta,dt,X0)
# running_traj_alldata(Length,M,beta,dt,X0)
# endregion

# region
# TAB_file = np.load('MALA_data/TAB_Nov20th.npz')
# # TAB_file = np.load('MALA_data/TAB_Nov24th.npz')
# TAB = TAB_file['TAB']
# endregion

# region
with open("./data/LJ7_count_new.pickle", "rb") as f:
    Count = pickle.load(f) 
with open("./data/LJ7_TAB_new.pickle", "rb") as f:
    TAB = pickle.load(f) 
with open("./data/LJ7_TA_new.pickle", "rb") as f:
    TA = pickle.load(f) 
with open("./data/LJ7_TB_new.pickle", "rb") as f:
    TB = pickle.load(f) 
    
with open("./data/LJ7_count_new (1).pickle", "rb") as f:
    Count2 = pickle.load(f) 
with open("./data/LJ7_TAB_new (1).pickle", "rb") as f:
    TAB2 = pickle.load(f) 
with open("./data/LJ7_TA_new (1).pickle", "rb") as f:
    TA2 = pickle.load(f) 
with open("./data/LJ7_TB_new (1).pickle", "rb") as f:
    TB2 = pickle.load(f) 
# endregion

# region
Count = np.append(Count,Count2)
M = 10
E_TAB = np.zeros(M)
TAi = np.zeros(M)
TABi = np.zeros(M)
all_TAB = 0
all_TA = 0
all_TB = 0
for i in range(M):
    if i < 5:
        E_TAB[i] = np.mean(TAB[i]) 
        TAi[i] = np.sum(TA[i]) 
        TABi[i] = np.sum(TAB[i])
        all_TAB += np.sum(TAB[i])
        all_TA += np.sum(TA[i])
        all_TB += np.sum(TB[i])
    else:
        j = i - 5
        E_TAB[i] = np.mean(TAB2[j]) 
        TAi[i] = np.sum(TA2[j]) 
        TABi[i] = np.sum(TAB2[j])
        all_TAB += np.sum(TAB2[j])
        all_TA += np.sum(TA2[j])
        all_TB += np.sum(TB2[j])
        
total_count = np.sum(Count)
total_time = Length*M*dt
# endregion

print('Transition rates are: {}, expected travel times are: {}'.format(np.array(Count)/(Length*dt), E_TAB*dt))
print('Averaged transition rates: {} and averaged E_TAB is {}'.format(total_count/total_time, np.mean(E_TAB)*dt))
print('rho_A are: {}, rho_AB are: {}'.format(TAi/Length,TABi/Length))
print('rho_A is ', all_TA*dt/total_time)
print('rho_AB is', all_TAB*dt/total_time)

obj = np.array(Count)/(Length*dt)
n = len(obj)
m, se = np.mean(obj), scipy.stats.sem(obj)
h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
print('mean transition rate is {}, confidence interval is: [{},{}], with standard error\
       {}'.format(m,m-h,m+h,se))

obj = TAi/Length
n = len(obj)
m, se = np.mean(obj), scipy.stats.sem(obj)
h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
print('expected rho_A is {}, confidence interval is: [{},{}], with standard error\
       {}'.format(m,m-h,m+h,se))

obj = TABi/Length
n = len(obj)
m, se = np.mean(obj), scipy.stats.sem(obj)
h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
print('expected rho_AB is {}, confidence interval is: [{},{}], with standard error\
       {}'.format(m,m-h,m+h,se))

obj = E_TAB*dt
n = len(obj)
m, se = np.mean(obj), scipy.stats.sem(obj)
h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
print('expected transition time is {}, confidence interval is: [{},{}], with standard error\
       {}'.format(m,m-h,m+h,se))

print(Count)


