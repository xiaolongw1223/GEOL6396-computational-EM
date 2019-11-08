#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 16:38:53 2019

@author: wxl
"""

import numpy as np
import scipy.sparse as sp
import time

norm = np.linalg.norm


# =============================================================================
#
# Author: Xiaolong Wei, 10/26/2019, at University of Houston 
#
# Based on the Dr.Jiajia Sun's matlab code.
#
# This package is flexible and it can be applied to 
# any geophysical inversion problem -- 
#
# Mesh (finished)
# Discretize (finished)
# 
# =============================================================================


class Mesh(object):
    
    '''
    
    In this mesh class, we can do:
        
    1. calcualte width of cells in x, y, z directions. eg: Mesh.WCx
    2. calculate center of cells in x, y, z directions. eg: Mesh.CCx
    3. calculate number of cells in x, y, z directions. eg: Mesh.nCx
    4. calculate total number of cells. eg: Mesh.nC
    5. calculate mod in x, y, z directions. eg: Mesh.Modx
    
    '''


    def __init__(self, listx, listy, listz, **kwargs):
        
        '''
        
        Here, we use right hand coordinate system:
        x: east
        y: north
        z: depth
        
        
        
        
        listx = [ [CoreModelVectorX], Lx, Px ]
        
        where, 
        CoreModelVectorX : core mesh area in x direction
        Lx: maximum length in east
        Px: power number of padding cells (exponent)
        
        
        
        
        
        listy, listz is totally same with listx
        
        
        
        '''

        self.CoreModelVectorX = listx[0]
        self.CoreModelVectorY = listy[0]
        self.CoreModelVectorZ = listz[0]

        self.Lx = listx[1]
        self.Ly = listy[1]
        self.Lz = listy[1]
        
        self.Px = listx[-1]
        self.Py = listy[-1]
        self.Pz = listy[-1]
        

    @ property
    def WCx(self):
        '''width of cells in x direction'''
        
        if isinstance(self.CoreModelVectorX, (np.ndarray)):
            T = self.CoreModelVectorX.tolist()
        
        
        assert isinstance(T, (list)), "the core vector should be a list"
        
        
        while 1:
            T.insert(0, T[0] * self.Px)
            T.insert(len(T), T[-1] * self.Px)

            if np.sum(T) > self.Lx:
                break   
        
        return np.array(T).flatten('F')
    
    @ property
    def WCy(self):
        '''width of cells in y direction'''
        
        if isinstance(self.CoreModelVectorY, (np.ndarray)):
            T = self.CoreModelVectorY.tolist()
        
        
        assert isinstance(T, (list)), "the core vector should be a list"
        
        
        while 1:
          
            T.insert(0, T[0] * self.Py)
            T.insert(len(T), T[-1] * self.Py)

            if np.sum(T) > self.Ly:
                break   
        
        return np.array(T).flatten('F')
    
    @ property
    def WCz(self):
        '''width of cells in z direction'''
        
        if isinstance(self.CoreModelVectorZ, (np.ndarray)):
            T = self.CoreModelVectorZ.tolist()
        
        
        assert isinstance(T, (list)), "the core vector should be a list"
        
        
        while 1:
          
            T.insert(len(T), T[-1] * self.Pz)

            if np.sum(T) > self.Lz:
                break   
        
        return np.array(T).flatten('F')
    
    
    @ property
    def nCx(self):
        '''number of cells in x direction'''
        
        return len(self.WCx)
    
    @ property
    def nCy(self):
        '''number of cells in y direction'''
        
        return len(self.WCy)
    
    @ property
    def nCz(self):
        '''number of cells in z direction'''
        
        return len(self.WCz)
    
    @ property
    def nC(self):
        
        return self.nCx * self.nCy * self.nCz
    
    @ property
    def Modx(self):

        xmod = []
        xmod.append(0)

        for i in range(self.nCx):

            xmod.append(xmod[i] + self.WCx[i])

        return np.array(xmod)
    
    
    @ property
    def Mody(self):
        '''center point of cells in x direction'''
        
        ymod = []
        ymod.append(0)
        
        for i in range(self.nCy):
            
            ymod.append(ymod[i] + self.WCy[i])
        
        return np.array(ymod)
        
        
    @ property
    def Modz(self):
        '''center point of cells in x direction'''
        
        zmod = []
        zmod.append(0)
        
        for i in range(self.nCz):
            
            zmod.append(zmod[i] + self.WCz[i])

        return np.array(zmod)
    
    @ property
    def CCx(self):
        '''center point of cells in x direction'''
        
        xmid = []
        for i in range(self.nCx):
            
            xmid.append(0.5 * (self.Modx[i] + self.Modx[i+1]))
        
        return np.array(xmid)
    
    
    @ property
    def CCy(self):
        '''center point of cells in x direction'''
        
        ymid = []
        for i in range(self.nCy):
            
            ymid.append(0.5 * (self.Mody[i] + self.Mody[i+1]))
        
        return np.array(ymid)
        
        
    @ property
    def CCz(self):
        '''center point of cells in x direction'''
        
        zmid = []
        for i in range(self.nCz):

            zmid.append(0.5 * (self.Modz[i] + self.Modz[i+1]))
        
        return np.array(zmid)
        
        
        
        

    
class Discretize(Mesh): 
    '''
    This class includes:
    
    1. getMeshGeometry functioin: volumn of cells, and area of faces.
    
    2. finite difference matrix in x, y, z directions.
    
    '''
    def __init__(self, listx, listy, listz, **kwargs):
    
        self.as_super = super(Discretize, self)
        self.as_super.__init__(listx, listy, listz, **kwargs) # pass the mesh to the class Mesh
    
    
    @ property
    def getMeshGeometry(self):
        
        '''
        V is a diagonal matrix that has the cell volumes on its diagonal, 
        F is also a diagonal matrix with the areas of all faces on its diagonal. 
        L is also a diagonal matrix with the length of all edges on its diagonal.
        
        '''
        h1 = self.WCz
        h2 = self.WCx
        h3 = self.WCy
        
        n1 = len(h1) 
        n2 = len(h2) 
        n3 = len(h3)
        
        V = sp.kron(sp.diags(h3), sp.kron(sp.diags(h2), sp.diags(h1)))
        
        F1 = sp.kron(sp.diags(h3), sp.kron(sp.diags(h2), sp.eye(n1+1)))
        F2 = sp.kron(sp.diags(h3), sp.kron(sp.eye(n2+1), sp.diags(h1)))
        F3 = sp.kron(sp.eye(n3+1), sp.kron(sp.diags(h2), sp.diags(h1)))
        
        # extract diag elements and make a long vector
        F = np.r_[np.diag(F1.A), np.diag(F2.A), np.diag(F3.A)]
        F = sp.diags(F)
    
        
        L1 = sp.kron(sp.eye(n3+1), sp.kron(sp.eye(n2+1), sp.diags(h1)))
        L2 = sp.kron(sp.eye(n3+1), sp.kron(sp.diags(h2), sp.eye(n1+1)))
        L3 = sp.kron(sp.diags(h3), sp.kron(sp.eye(n2+1), sp.eye(n1+1)))
        
        L = np.r_[np.diag(L1.A), np.diag(L2.A), np.diag(L3.A)]
        L = sp.diags(L)
        
        return [V, F, L]
        
  
    
    def Diff(self, direc = 'x'):
        
        '''
        In forward finite difference matrix:
        
        X = east = 2
        Y = north = 3
        Z = depth = 1
        
        I is identity matrix
        
        D is 1D finite difference matrix
        
        '''
        Directions = ['x', 'y', 'z']
        
        self.direc = direc
        assert direc in Directions,  "direction not valid!"
        
        
        Iz = sp.eye(self.nCz)
        Ix = sp.eye(self.nCx)
        Iy = sp.eye(self.nCy)
        
        
        if direc == 'x':
            e = np.ones(self.nCx+1)
            Dx = sp.spdiags([-e, e], [0, 1], self.nCx, self.nCx+1)
            D = sp.kron(sp.kron(Iy, Dx), Iz)
        
        
        if direc == 'y':
            e = np.ones(self.nCy+1)
            Dy = sp.spdiags([-e, e], [0, 1], self.nCy, self.nCy+1)
            D = sp.kron(sp.kron(Dy, Ix), Iz)
        
        
        if direc == 'z':
            e = np.ones(self.nCz+1)
            Dz = sp.spdiags([-e, e], [0, 1], self.nCz, self.nCz+1)
            D = sp.kron(sp.kron(Iy, Ix), Dz)
        
        return D
    
    
    
    
    def Diff2(self, direc = 'x'):
        
        '''
        In forward finite difference matrix:
        
        X = east = 2
        Y = north = 3
        Z = depth = 1
        
        I is identity matrix
        
        D is 1D finite difference matrix
        
        '''
        Directions = ['x', 'y', 'z']
        
        self.direc = direc
        assert direc in Directions,  "direction not valid!"
        
        
        Iz = sp.eye(self.nCz)
        Ix = sp.eye(self.nCx)
        Iy = sp.eye(self.nCy)
        
        
        if direc == 'x':
            e = np.ones(self.nCx+1) * 0.5
            Dx = sp.spdiags([e, e], [0, 1], self.nCx, self.nCx+1)
            D = sp.kron(sp.kron(Iy, Dx), Iz)
        
        
        if direc == 'y':
            e = np.ones(self.nCy+1) * 0.5
            Dy = sp.spdiags([e, e], [0, 1], self.nCy, self.nCy+1)
            D = sp.kron(sp.kron(Dy, Ix), Iz)
        
        
        if direc == 'z':
            e = np.ones(self.nCz+1)* 0.5
            Dz = sp.spdiags([e, e], [0, 1], self.nCz, self.nCz+1)
            D = sp.kron(sp.kron(Iy, Ix), Dz)
        
        return D






class Utils(Mesh):
    
    '''
    This class includes:
        
    1. SourceCorrection function
    2. Conjugate gradient
    
    '''
    def __init__(self, listx, listy, listz, **kwargs):
        
        self.as_super = super(Utils, self)
        self.as_super.__init__(listx, listy, listz, **kwargs) # pass the mesh to the class Mesh
    

        
    def CG(self, A, b):
        
        t = time.time()
        
        tol = 1e-3
        maxIter = 5000
        x = 0 * b
        pvec = A.dot(x)
        
        r = b - pvec
        p = r
        rhop = r.T.dot(r)
        
        bnorm = norm(b)
        
        for i in range(maxIter):
            
            q = A.dot(p)
            alpha = rhop / (p.T.dot(q))
            x = x + alpha * p
            r = r - alpha * q
            
            rcg = norm(r)/bnorm
            if rcg < tol:
                break
            
            rho = r.T.dot(r)
            p = r + (rho/rhop) * p
            
            rhop = rho
        
        print("CG solver time", time.time() - t)    
        return x
        
     
            
    '''This function code is borrowed from Felicia'''
    def SourceCorrection(self, A, sigma_H, I, PositiveElectrode_Loc, NegativeElectrode_Loc):
        
        rho = 1 / sigma_H

        xmod = self.Modx
        ymod = self.Mody
        zmod = self.Modz
    
        phi_true = np.zeros((self.nCz, self.nCx, self.nCy))
        phi_true_vec = np.zeros(self.nCz * self.nCx * self.nCy)

        for i in range(self.nCy):
            
            xloc = (xmod[i]+ xmod[i+1])/2.
            
            for j in range(self.nCx):
                
                yloc = (ymod[j] + ymod[j+1])/2.
                
                for k in range(self.nCz):         
                    
                    zloc = (zmod[k] + zmod[k+1])/2.;
                    CellLoc = np.transpose([zloc,yloc,xloc])# the coordinates of the center of a cell
                    
                    distanceA = np.linalg.norm(CellLoc - PositiveElectrode_Loc)
                    distanceB = np.linalg.norm(CellLoc - NegativeElectrode_Loc)
                    
                    phi_true[k,j,i] = rho*I*(1./distanceA - 1./distanceB)/(2*np.pi)

        phi_true_vec = phi_true.flatten('F')
        q_corr = A*phi_true_vec
    
    
    
        return(q_corr)
        
            
        
        
        
        
