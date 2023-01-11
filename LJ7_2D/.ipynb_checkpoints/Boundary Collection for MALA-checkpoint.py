import numpy as np
from tqdm import tqdm
from celluloid import Camera
import matplotlib.pyplot as plt
from matplotlib import cm


inData = np.load("LJ7_data.npz")
traj = inData["traj"]
start = traj[:,265]
ring = np.linspace(0,2*np.pi,200)

print(start)

# region A and B
Ca = [0.5526,-0.0935]
Cb = [0.7184,1.1607]
Cra = .05
Crb = .05


temp = .2 
gam = 400 # friction?
m = 1
tstep = .01
A = 2 # coefficient a
sig = 1
variance = (gam*temp*m*2)/tstep
stdev = np.sqrt(variance)
Uspring = 200
kspring = 200
wallsize = 2
itera = 50000


Xs = [0, 2, 4, 6, 8, 10, 12]
Ys = [1, 3, 5, 7, 9, 11, 13]

# derivative of the V_pair in the first dimension of xi
def VLJpair0(X,Y):
    dist = np.sqrt((X[0]-Y[0])**2+(X[1]-Y[1])**2) 
    return 4*A*(12*((sig**12)/(dist**13))-6*((sig**6)/(dist**7)))*((X[0]-Y[0]))/(dist)

# derivative of the V_pair in the second dimension of xi
def VLJpair1(X,Y):
    dist = np.sqrt((X[0]-Y[0])**2+(X[1]-Y[1])**2)
    return 4*A*(12*((sig**12)/(dist**13))-6*((sig**6)/(dist**7)))*((X[1]-Y[1]))/(dist)

"""
calculate the gradient of potential on the configuration on dex-th particle at time t on the first dimension
"""
def VLJatom0(X,t,dex):
    Force = 0
    cmass = np.zeros(2)
    cmass[0] = (1/7)*sum(X[Xs,t]) # average of the first dimension at time t
    cmass[1] = (1/7)*sum(X[Ys,t]) # average of the second dimension at time t
    # distance of the dex-th particle to the center of the mass
    cdist = np.sqrt((X[2*dex,t]-cmass[0])**2+(X[2*dex+1,t]-cmass[1])**2) 
    for i in range(7):
        if i != dex:
            Force += VLJpair0([X[2*dex,t],X[2*dex+1,t]],[X[2*i,t],X[2*i+1,t]])
    if cdist > wallsize:
        # if too far subtract the gradient
        Force -= 2*kspring*(cdist-(wallsize))*(X[2*dex,t]-cmass[0])/cdist
    return Force

"""
calculate the force on the configuration on dex-th particle at time t on the second dimension
"""
def VLJatom1(X,t,dex):
    Force = 0
    cmass = np.zeros(2)
    cmass[0] = (1/7)*sum(X[Xs,t])
    cmass[1] = (1/7)*sum(X[Ys,t])
    cdist = np.sqrt((X[2*dex,t]-cmass[0])**2+(X[2*dex+1,t]-cmass[1])**2)
    for i in range(7):
        if i != dex:
            Force += VLJpair1([X[2*dex,t],X[2*dex+1,t]],[X[2*i,t],X[2*i+1,t]])
    if cdist > wallsize:
        Force -= 2*kspring*(cdist-(wallsize))*(X[2*dex+1,t]-cmass[1])/cdist
    return Force

# full langevin dynamics?
def velocity(X,V,U,t):
    # V + [x(t)-x(t-1)]/dt * (1 - 0.5gamma*dt)/(1 + 0.5gamma*dt) + (dt/m)*(U + wt)/(1+0.5gamma*dt)
    return V + ((X[:,t]-X[:,t-1])/tstep) * (1-(.5*gam*tstep))/(1+(.5*gam*tstep))\
        + (tstep/m)*(U+randoms[:,t])/(1+(.5*gam*tstep))

def posit(X,V,t):
    return X[:,t]+tstep*V

# #####Collective Variable Functions######

# coordinate number
def coordnum(X,k):
    cnum = 0
    for i in range(7):
        if i != k:
            dist = np.sqrt((X[2*k]-X[2*i])**2+(X[2*k+1]-X[2*i+1])**2)
            cnum += ((1-(dist/(1.5*sig))**8))/((1-(dist/(1.5*sig))**16))
    return cnum

# \Bar{c}(X)
def avcnum(X):
    avgnum = 0
    for k in range(7):
        avgnum += coordnum(X,k)
    return (1/7)*avgnum

def Mu2(X,c):
    mu2 = 0
    for k in range(7):
        mu2 += (coordnum(X,k)-c)**2
    return (1/7)*mu2

def Mu3(X,c):
    mu3 = 0
    for k in range(7):
        mu3 += (coordnum(X,k)-c)**3
    return (1/7)*mu3


### Derivatives of Collective Variable Functions#####
def dcij0(X,Y):
    dist = np.sqrt((X[0]-Y[0])**2+(X[1]-Y[1])**2)
    return (8*(dist/(1.5*sig))**7*((1-(dist/(1.5*sig))**16))\
            -16*(dist/(1.5*sig))**15*((1-(dist/(1.5*sig))**8)))\
            /(((1-(dist/(1.5*sig))**16))**2)*(X[0]-Y[0])/(dist)

def dcij1(X,Y):
    dist = np.sqrt((X[0]-Y[0])**2+(X[1]-Y[1])**2)
    return (8*(dist/(1.5*sig))**7*((1-(dist/(1.5*sig))**16))\
            -16*(dist/(1.5*sig))**15*((1-(dist/(1.5*sig))**8)))\
            /(((1-(dist/(1.5*sig))**16))**2)*(X[1]-Y[1])/(dist)

# \sum_{j = 1}^7 dcij0
def dci0(X,k):
    dci = 0
    for i in range(7):
        if i != k:
            dci += dcij0([X[2*k],X[2*k+1]],[X[2*i],X[2*i+1]])
    return dci

# \sum_{j = 1}^7 dcij1
def dci1(X,k):
    dci = 0
    for i in range(7):
        if i != k:
            dci += dcij1([X[2*k],X[2*k+1]],[X[2*i],X[2*i+1]])
    return dci

def davcnum0(X,k):
    return (2/7)*(dci0(X,k))

def davcnum1(X,k):
    return (2/7)*(dci1(X,k))

def dMu2(X,C):
    dMu2 = np.zeros((14,1))
    for i in range(7):
        for j in range(7):
            if i != j:
                dMu2[2*i] += (coordnum(X,j)-C)*\
                        (dcij0([X[2*i],X[2*i+1]],[X[2*j],X[2*j+1]])-davcnum0(X,i))
                dMu2[2*i+1] += (coordnum(X,j)-C)*\
                        (dcij1([X[2*i],X[2*i+1]],[X[2*j],X[2*j+1]])-davcnum1(X,i))
        dMu2[2*i] += (coordnum(X,i)-C)*(dci0(X,i)-davcnum0(X,i))
        dMu2[2*i+1] += (coordnum(X,i)-C)*(dci1(X,i)-davcnum1(X,i))
    return (2/7)*dMu2

def dMu3(X,C):
    dMu3 = np.zeros((14,1))
    for i in range(7):
        for j in range(7):
            if i != j:
                dMu3[2*i] += ((coordnum(X,j)-C)**2)*\
                        (dcij0([X[2*i],X[2*i+1]],[X[2*j],X[2*j+1]])-davcnum0(X,i))
                dMu3[2*i+1] += ((coordnum(X,j)-C)**2)*\
                        (dcij1([X[2*i],X[2*i+1]],[X[2*j],X[2*j+1]])-davcnum1(X,i))
        dMu3[2*i] += ((coordnum(X,i)-C)**2)*(dci0(X,i)-davcnum0(X,i))
        dMu3[2*i+1] += ((coordnum(X,i)-C)**2)*(dci1(X,i)-davcnum1(X,i))
    return (3/7)*dMu3

# +
#### Umbrella Sampling #############

# gradient of V_b for mu2
def BrellaMu2(X,C,OneZ):
    return 2*Uspring*(Mu2(X,C)-OneZ)*dMu2(X,C)


# -

def BrellaMu3(X,C,TwoZ):
    return 2*Uspring*(Mu3(X,C)-TwoZ)*dMu3(X,C)


veloc = np.zeros((14,itera+1))
simtraj = np.zeros((14,itera+1))
potatom = np.zeros((14,itera))
randoms = np.random.normal(0,stdev,(14,itera))

simtraj[:,0] = start
for i in range(7):
    potatom[2*i,0] = VLJatom0(simtraj,0,i)
    potatom[2*i+1,0] = VLJatom1(simtraj,0,i)

veloc[:,0] = (tstep/m)*(potatom[:,0]+randoms[:,0])/(1+(.5*gam*tstep))
simtraj[:,1] = simtraj[:,0] + tstep*veloc[:,0]    

ABord = np.zeros((14,200))


for w in tqdm(range(200)):
    # target coordination
    OneZ = (Cra+.005)*np.cos(ring[w])+Ca[0]
    TwoZ = (Cra+.005)*np.sin(ring[w])+Ca[1]
    for k in (range(1,itera)):
        C = avcnum(simtraj[:,k]) # average of coordination number
        m2 = Mu2(simtraj[:,k],C)
        m3 = Mu3(simtraj[:,k],C)
        Stomp = np.abs(TwoZ-m3)
        Tromp = np.abs(OneZ-m2)
        if Stomp < .001:
            if Tromp < .001:
                ABord[:,w] = simtraj[:,k]
                break
        for i in range(7): # update the trajectory
            potatom[2*i,k-1] = VLJatom0(simtraj,k,i)
            potatom[2*i+1,k-1] = VLJatom1(simtraj,k,i) # \nabla U on the configuration on dex-th particle at time t
        Job = BrellaMu3(simtraj[:,k],C,TwoZ) + BrellaMu2(simtraj[:,k],C,OneZ) # total bias added
        veloc[:,k] = velocity(simtraj,veloc[:,k-1],potatom[:,k-1],k)
        veloc[:,k] += Job[:,0]*tstep
        simtraj[:,k+1] = posit(simtraj,veloc[:,k],k) 


 np.savez('Mala Boundary Samples.npz', ABord = ABord)


# +
Data = np.load("Mala Boundary Samples.npz")
Abdry = Data['ABord']

print(Abdry.shape)

# -

C = avcnum(Abdry)
mu2 = Mu2(Abdry,C)
mu3 = Mu3(Abdry,C)
plt.scatter(mu2,mu3,s = 0.1)
plt.scatter(Ca[0],Ca[1])
plt.scatter(Cb[0],Cb[1])
plt.xlim([0.2,1.2])
plt.ylim([-0.5,1.5])


