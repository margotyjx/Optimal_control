# -*- coding: utf-8 -*-
# This code is a python adaptation of the code in the paper:
# Title: Remarks around 50 lines of Matlab: short finite element implementation
# Authors: Jochen Alberty, Carsten Carstensen and Stefan A. Funken
# Journal: Numerical Algorithms 20 (1999) 117–137

# Solves the committor problem using the FEM for the case of anisotropic and position-dependent diffusion
# grad * (exp(-beta V(x,y) M(x,y)*grad q(x,y)) = 0, q(bdry A) = 0, q(bdry B) = 1
# the potential used is the Face potential defined in this cell

import numpy as np
import math
import matplotlib.pyplot as plt
import csv 
import scipy
from scipy.sparse import csr_matrix


def put_pts_on_circle(xc,yc,r,n):
    t = np.linspace(0,math.pi*2,n+1)
    pts = np.zeros((n,2))
    pts[:,0] = xc+r*np.cos(t[0:n])
    pts[:,1] = yc+r*np.sin(t[0:n])
    return pts

def put_pts_on_ellipses(xc,yc,rx,ry,theta,n):
    t = np.linspace(0,math.pi*2,n+1)
    pts = np.zeros((n,2))
    pts[:,0] = rx*np.cos(t[0:n])*np.cos(theta) - ry*np.sin(t[0:n])*np.sin(theta)+xc
    pts[:,1] = rx*np.cos(t[0:n])*np.sin(theta) + ry*np.sin(t[0:n])*np.cos(theta)+yc
    return pts


def reparametrization(path,h):
    dp = path - np.roll(path,1,axis = 0);
    dp[0,:] = 0;
    dl = np.sqrt(np.sum(dp**2,axis=1));
    lp = np.cumsum(dl);
    len = lp[-1];
    lp = lp/len; # normalize
    npath = int(round(len/h));
    g1 = np.linspace(0,1,npath)
    path_x = np.interp(g1,lp,path[:,0])
    path_y = np.interp(g1,lp,path[:,1])
    path = np.zeros((npath,2))
    path[:,0] = path_x
    path[:,1] = path_y
    return path

def find_ABbdry_pts(pts,xc,yc,r,h0):
    ind = np.argwhere(np.sqrt((pts[:,0]-xc)**2+(pts[:,1]-yc)**2)-r < h0*1e-2)
    Nind = np.size(ind)
    ind = np.reshape(ind,(Nind,))
    return Nind,ind

def stima3(verts,Mmatr):
    Aux = np.ones((3,3))
    Aux[1:3,:] = np.transpose(verts)
    rhs = np.zeros((3,2))
    rhs[1,0] = 1
    rhs[2,1] = 1
    G = np.zeros((3,2))
    G[:,0] = np.linalg.solve(Aux,rhs[:,0])
    G[:,1] = np.linalg.solve(Aux,rhs[:,1])
    M = 0.5*np.linalg.det(Aux)*np.matmul(np.matmul(G,Mmatr),np.transpose(G))
    return M

def FEM_committor_solver(pts,tri,Aind,Bind,Fpts,M11pts,M12pts,M22pts,beta):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    Dir_bdry = np.hstack((Aind,Bind))
    free_nodes = np.setdiff1d(np.arange(0,Npts,1),Dir_bdry,assume_unique=True)

    A = csr_matrix((Npts,Npts), dtype = np.float).toarray()
    b = np.zeros((Npts,1))
    q = np.zeros((Npts,1))
    q[Bind] = 1

    # stiffness matrix
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:] # vertices of mesh triangle
        Fmid = np.sum(Fpts[ind])/3
        M11mid = np.sum(M11pts[ind])/3
        M12mid = np.sum(M12pts[ind])/3
        M22mid = np.sum(M22pts[ind])/3
        Mmatr = np.array([[M11mid,M12mid],[M12mid,M22mid]])
        fac = np.exp(-beta*Fmid)
        indt = np.array(ind)[:,None]
        A[indt,ind] = A[indt,ind] + stima3(verts,Mmatr)*fac

    # load vector
    b = b - np.matmul(A,q)

    # solve for committor
    free_nodes_t = np.array(free_nodes)[:,None]
    q[free_nodes] = scipy.linalg.solve(A[free_nodes_t,free_nodes],b[free_nodes])
    q = np.reshape(q,(Npts,))
    return q

def reactive_current_and_transition_rate(pts,tri,Fpts,M11pts,M12pts,M22pts,beta,q):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    # find the reactive current and the transition rate
    Rcurrent = np.zeros((Ntri,2)) # reactive current at the centers of mesh triangles
    Rrate = 0
    Z = 0
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:]
        qtri = q[ind]
        a = np.array([[verts[1,0]-verts[0,0],verts[1,1]-verts[0,1]],[verts[2,0]-verts[0,0],verts[2,1]-verts[0,1]]])
        b = np.array([qtri[1]-qtri[0],qtri[2]-qtri[0]])
        g = np.linalg.solve(a,b)
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))              
        Fmid = np.sum(Fpts[ind])/3
        M11mid = np.sum(M11pts[ind])/3
        M12mid = np.sum(M12pts[ind])/3
        M22mid = np.sum(M22pts[ind])/3
        Mmatr = np.array([[M11mid,M12mid],[M12mid,M22mid]])
        mu = np.exp(-beta*Fmid)
        Z = Z + tri_area*mu
        Mg = np.matmul(Mmatr,g)
        gMg = np.matmul(np.transpose(g),Mg)
        Rcurrent[j,:] = mu*Mg
        Rrate = Rrate + gMg*mu*tri_area                     
    Rrate = Rrate/(Z*beta)
    Rcurrent = Rcurrent/(Z*beta) 
    # map reactive current on vertices
    Rcurrent_verts = np.zeros((Npts,2))
    tcount = np.zeros((Npts,1)) # the number of triangles adjacent to each vertex
    for j in range(Ntri):
        indt = np.array(tri[j,:])[:,None]    
        Rcurrent_verts[indt,:] = Rcurrent_verts[indt,:] + Rcurrent[j,:]
        tcount[indt] = tcount[indt] + 1
    Rcurrent_verts = Rcurrent_verts/np.concatenate((tcount,tcount),axis = 1)
    return Rcurrent_verts, Rrate

def mean_force(pts,tri,Fpts):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    # find the reactive current and the transition rate
    MeanForce = np.zeros((Ntri,2)) # reactive current at the centers of mesh triangles
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:]
        Ftri = Fpts[ind]
        a = np.array([[verts[1,0]-verts[0,0],verts[1,1]-verts[0,1]],[verts[2,0]-verts[0,0],verts[2,1]-verts[0,1]]])
        b = np.array([Ftri[1]-Ftri[0],Ftri[2]-Ftri[0]])
        g = np.linalg.solve(a,b)
        MeanForce[j,:] = g
    # map reactive current on vertices
    MeanForce_verts = np.zeros((Npts,2))
    tcount = np.zeros((Npts,1)) # the number of triangles adjacent to each vertex
    for j in range(Ntri):
        indt = np.array(tri[j,:])[:,None]    
        MeanForce_verts[indt,:] = MeanForce_verts[indt,:] + MeanForce[j,:]
        tcount[indt] = tcount[indt] + 1
    MeanForce_verts = MeanForce_verts/np.concatenate((tcount,tcount),axis = 1)
    return MeanForce_verts


def invariant_pdf(pts,tri,pts_Amesh,tri_Amesh,pts_Bmesh,tri_Bmesh,Fpts,Fpts_A, Fpts_B,beta):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    Npts_Amesh = np.size(pts_Amesh,axis=0)
    Ntri_Amesh = np.size(tri_Amesh,axis=0)
    Npts_Bmesh = np.size(pts_Bmesh,axis=0)
    Ntri_Bmesh = np.size(tri_Bmesh,axis=0)

    # find the reactive current and the transition rate
    Z = 0
    prob = 0
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:]
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux)) 
        Fmid = np.sum(Fpts[ind])/3
        mu = np.exp(-beta*Fmid)
        Z = Z + tri_area*mu
    for j in range(Ntri_Amesh):
        ind = tri_Amesh[j,:]
        verts = pts_Amesh[ind,:]
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))  
        Fmid = np.sum(Fpts_A[ind])/3
        mu = np.exp(-beta*Fmid)
        Z = Z + tri_area*mu
    for j in range(Ntri_Bmesh):
        ind = tri_Bmesh[j,:]
        verts = pts_Bmesh[ind,:]
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))              
        Fmid = np.sum(Fpts_B[ind])/3
        mu = np.exp(-beta*Fmid)
        Z = Z + tri_area*mu
    return Z


def reactive_current_and_transition_rate_updated(pts,tri,Fpts,M11pts,M12pts,M22pts,beta,q,Z):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    # find the reactive current and the transition rate
    Rcurrent = np.zeros((Ntri,2)) # reactive current at the centers of mesh triangles
    Rrate = 0
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:]
        qtri = q[ind]
        a = np.array([[verts[1,0]-verts[0,0],verts[1,1]-verts[0,1]],[verts[2,0]-verts[0,0],verts[2,1]-verts[0,1]]])
        b = np.array([qtri[1]-qtri[0],qtri[2]-qtri[0]])
        g = np.linalg.solve(a,b)
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))              
        Fmid = np.sum(Fpts[ind])/3
        M11mid = np.sum(M11pts[ind])/3
        M12mid = np.sum(M12pts[ind])/3
        M22mid = np.sum(M22pts[ind])/3
        Mmatr = np.array([[M11mid,M12mid],[M12mid,M22mid]])
        mu = np.exp(-beta*Fmid)
        Mg = np.matmul(Mmatr,g)
        gMg = np.matmul(np.transpose(g),Mg)
#         Rcurrent[j,:] = mu*Mg
        Rrate = Rrate + gMg*mu*tri_area                     
    Rrate = Rrate/(Z*beta)
    Rcurrent = Rcurrent/(Z*beta) 
    # map reactive current on vertices
    Rcurrent_verts = np.zeros((Npts,2))
    tcount = np.zeros((Npts,1)) # the number of triangles adjacent to each vertex
    for j in range(Ntri):
        indt = np.array(tri[j,:])[:,None]    
        Rcurrent_verts[indt,:] = Rcurrent_verts[indt,:] + Rcurrent[j,:]
        tcount[indt] = tcount[indt] + 1
    Rcurrent_verts = Rcurrent_verts/np.concatenate((tcount,tcount),axis = 1)
    return Rcurrent_verts, Rrate


def probability_reactive(pts,tri,Fpts,beta,q,Z):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    # find the reactive current and the transition rate
    prob = 0
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:]
        qtri = q[ind]
        qmid = np.sum(qtri)/3
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))              
        Fmid = np.sum(Fpts[ind])/3
        mu = np.exp(-beta*Fmid)
        prob = prob + tri_area*mu*qmid*(1-qmid)
    prob = prob/Z
    return prob


def probability_last_A(pts,tri,pts_Amesh,tri_Amesh,Fpts,Fpts_A,beta,q,Z):
    Npts = np.size(pts,axis=0)
    Ntri = np.size(tri,axis=0)
    Npts_Amesh = np.size(pts_Amesh,axis=0)
    Ntri_Amesh = np.size(tri_Amesh,axis=0)

    # find the reactive current and the transition rate
    prob = 0
    for j in range(Ntri):
        ind = tri[j,:]
        verts = pts[ind,:]
        qtri = q[ind]
        qmid = np.sum(qtri)/3
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux)) 
        Fmid = np.sum(Fpts[ind])/3
        mu = np.exp(-beta*Fmid)
        prob = prob + tri_area*mu*(1-qmid)
    for j in range(Ntri_Amesh):
        ind = tri_Amesh[j,:]
        verts = pts_Amesh[ind,:]
        Aux = np.ones((3,3))
        Aux[1:3,:] = np.transpose(verts)
        tri_area = 0.5*np.absolute(np.linalg.det(Aux))  
        Fmid = np.sum(Fpts_A[ind])/3
        mu = np.exp(-beta*Fmid)
        prob = prob + tri_area*mu
        
    prob = prob/Z
    return prob
