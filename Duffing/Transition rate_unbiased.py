import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.interpolate
import math
from os.path import exists
import scipy.stats
import pickle


# region
def dU(x):
    dUx = x*(-1+ x**2)
    
    return dUx

def get_init_pt(circle_r):
    # center of the circle (x, y)
    circle_x = -1
    circle_y = 0

    # random angle
    alpha = 2 * math.pi * np.random.random()
    # random radius
    r = circle_r * math.sqrt(np.random.random())
    # calculating coordinates
    x = r * math.cos(alpha) + circle_x
    y = r * math.sin(alpha) + circle_y
    
    return x,y
# endregion


# region
def running_traj(Time,M,beta,gamma,delt):
    Count = np.zeros(M) # number of times we hit B
    TAB = []
    rsq = 1**2
    std = np.sqrt(2*delt)
    a = [-1,0]
    rx = 0.3
    ry = 0.4
    
    for m in range(M):
        tab = np.array([]) 
        newX,newY = get_init_pt(0.3)
        
        leftA = 1 # state of the trajectory 
        lastA = 0 # time we last visited A

#         X =np.array([]) 
#         Y =np.array([]) 

        for i in range(Time):
            b = [1,0]
            
            dUx = dU(newX)
            newX = newX + newY * delt
            newY = newY - (gamma*newY + dUx)*delt + np.sqrt(gamma/beta)*np.random.normal(0.0,std,np.shape(newY))

#             X = np.append(X,newX)
#             Y = np.append(Y,newY)

            """
            when the leftA is 1, it means the trajectory last hit A; when leftA 2, it means the trajectory is in A; 
            when leftA is 0, the trajectory last hit B. 
            """
#             distA = (newX - a[0])**2 + (newY - a[1])**2
            distA = (newX - a[0])**2/(rx**2) + (newY - a[1])**2/(ry**2)
            if leftA == 1: #last hit A
                if distA <= rsq:
                    leftA = 2
                else: # didn't return to A but last hit A
                    distB = (newX - b[0])**2/(rx**2) + (newY - b[1])**2/(ry**2)
#                     distB = (newX - b[0])**2 + (newY - b[1])**2

                    if distB <= rsq:
                        leftA = 0
                        Count[m] += 1
                        tab = np.append(tab,i - lastA)
            elif leftA == 2: # currently in A
                if distA <= rsq:
                    leftA = 2
                else:
                    leftA = 1
                    lastA = i
            else: # last hit B
                if distA <= rsq:
                    leftA = 1
                    lastA = i
        TAB.append(tab)
        
    return Count, TAB
# endregion


def running_traj_alldata(Time,M,beta,gamma,delt):
    Count = [] # number of times we hit B
    TAB = []
    TA = []
    TB = []
    rsq = 1
    std = np.sqrt(2*delt)
    a = [-1,0]
    rx = 0.3
    ry = 0.4
    
    for m in range(M):
        tab = []
        ta = []
        tb = []
        c = 0
        
        newX,newY = get_init_pt(0.3)
        
        leftA = 1 # state of the trajectory 
        lastA = 0 # time we last visited A
        firstA = 0 # first time we enter A after leaving B
        firstB = 0 # first time we enter B after leaving A
        
        for i in range(Time):
            b = [1,0]
            
            dUx = dU(newX)
            newX = newX + newY * delt
            newY = newY - (gamma*newY + dUx)*delt + np.sqrt(gamma/beta)*np.random.normal(0.0,std,np.shape(newY))

            """
            when the leftA is 1, it means the trajectory last hit A; when leftA 2, it means the trajectory is in A; 
            when leftA is 0, the trajectory last hit B. 
            """
            if leftA == 1: #last hit A
                distA = (newX - a[0])**2/(rx**2) + (newY - a[1])**2/(ry**2)
                if distA <= rsq:
                    leftA = 2
                else: # didn't return to A but last hit A
                    distB = (newX - b[0])**2/(rx**2) + (newY - b[1])**2/(ry**2)

                    if distB <= rsq:
                        leftA = 0
                        c += 1
                        tab.append(i - lastA)
                        firstB = i #first time hitting B after leaving A
                        ta.append(i - firstA)
                        
            elif leftA == 2: # currently in A
                distA = (newX - a[0])**2/(rx**2) + (newY - a[1])**2/(ry**2)
                if distA <= rsq:
                    leftA = 2
                else:
                    leftA = 1
                    lastA = i
            else: # last hit B
                distA = (newX - a[0])**2/(rx**2) + (newY - a[1])**2/(ry**2)
                if distA <= rsq:
                    leftA = 1
                    lastA = i
                    tb.append(i - firstB)
                    firstA = i # first time hitting A after leaving B
        # store the number of counts
        if exists("Dff_beta10_count.pickle"):
            with open("Dff_beta10_count.pickle", "rb") as f:
                Count = pickle.load(f) 
        Count.append(c)
        with open("Dff_beta10_count.pickle", "wb") as f:
            pickle.dump(Count, f)
            
        
        # store the transition time
        if exists("Dff_beta10_TAB.pickle"):
            with open("Dff_beta10_TAB.pickle", "rb") as f:
                TAB = pickle.load(f) 
        TAB.append(tab)
        with open("Dff_beta10_TAB.pickle", "wb") as f:
            pickle.dump(TAB, f) 
        
        # store TA
        if exists("Dff_beta10_TA.pickle"):
            with open("Dff_beta10_TA.pickle", "rb") as f:
                TA = pickle.load(f) 
        TA.append(ta)
        with open("Dff_beta10_TA.pickle", "wb") as f:
            pickle.dump(TA, f)
        
        #store TB
        if exists("Dff_beta10_TB.pickle"):
            with open("Dff_beta10_TB.pickle", "rb") as f:
                TB = pickle.load(f) 
        TB.append(tb)
        with open("Dff_beta10_TB.pickle", "wb") as f:
            pickle.dump(TB, f)

dt =0.005
beta = 10
M = 1 # number of runs
Length = 10000000
gamma = 0.5

# region
# running_traj_alldata(Length,M,beta,gamma,dt)
# endregion

with open("./data/Dff_beta10_count.pickle", "rb") as f:
    Count = pickle.load(f) 
with open("./data/Dff_beta10_TAB.pickle", "rb") as f:
    TAB = pickle.load(f) 
with open("./data/Dff_beta10_TA.pickle", "rb") as f:
    TA = pickle.load(f) 
with open("./data/Dff_beta10_TB.pickle", "rb") as f:
    TB = pickle.load(f) 

# region
M = len(Count)
print(M)
E_TAB = np.zeros(M)
TAi = np.zeros(M)
TABi = np.zeros(M)
all_TAB = 0
all_TA = 0
all_TB = 0
for i in range(M):
    E_TAB[i] = np.mean(TAB[i]) 
    TAi[i] = np.sum(TA[i]) 
    TABi[i] = np.sum(TAB[i])
    all_TAB += np.sum(TAB[i])
    all_TA += np.sum(TA[i])
    all_TB += np.sum(TB[i])
    
total_count = np.sum(Count)
total_time = Length*M*dt
# endregion

# region
print('Transition rates are: {}, expected travel times are: {}'.format(np.array(Count)/(Length*dt), 
                                                                       E_TAB*dt))
print('Averaged transition rates: {} and averaged E_TAB is {}'.format(round(total_count/total_time,4), round(np.mean(E_TAB)*dt,4)))
print('rho_A are: {}, rho_AB are: {}'.format(TAi/Length,TABi/Length))
print('rho_A is ', round(all_TA*dt/total_time,4))
print('rho_AB is', round(all_TAB*dt/total_time,4))

obj = np.array(Count)/(Length*dt)
n = len(obj)
m, se = np.mean(obj), scipy.stats.sem(obj)
h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
print('mean transition rate is {}, confidence interval is: [{},{}], with standard error\
       {}'.format(m,round(m-h,7),round(m+h,7),round(se,7)))

obj = TAi/Length
n = len(obj)
m, se = np.mean(obj), scipy.stats.sem(obj)
h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
print('expected rho_A is {}, confidence interval is: [{},{}], with standard error\
       {}'.format(round(m,4),round(m-h,4),round(m+h,4),round(se,4)))

obj = TABi/Length
n = len(obj)
m, se = np.mean(obj), scipy.stats.sem(obj)
h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
print('expected rho_AB is {}, confidence interval is: [{},{}], with standard error\
       {}'.format(round(m,4),round(m-h,4),round(m+h,4),round(se,4)))

obj = E_TAB*dt
n = len(obj)
m, se = np.mean(obj), scipy.stats.sem(obj)
h = se * scipy.stats.t.ppf((1 + 0.95) / 2., n-1)
print('expected transition time is {}, confidence interval is: [{},{}], with standard error\
       {}'.format(round(m,4),round(m-h,4),round(m+h,4),round(se,4)))
print(h)
# endregion




