#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 16:38:53 2019

@author: Xiaolong Wei
"""

import numpy as np
import scipy.sparse as sp
import time
from tools import sav, sdiff, speye, kron3

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

    1. calcualte width of cells in x, y, z directions. eg: Mesh.wcx
    2. calculate center of cells in x, y, z directions. eg: Mesh.ccx
    3. calculate number of cells in x, y, z directions. eg: Mesh.ncx
    4. calculate total number of cells. eg: Mesh.nc
    5. calculate mod in x, y, z directions. eg: Mesh.modx

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
    def wcx(self):
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
    def wcy(self):
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
    def wcz(self):
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
    def ncx(self):
        '''number of cells in x direction'''

        return len(self.wcx)

    @ property
    def ncy(self):
        '''number of cells in y direction'''

        return len(self.wcy)

    @ property
    def ncz(self):
        '''number of cells in z direction'''

        return len(self.wcz)

    @ property
    def nc(self):

        return self.ncx * self.ncy * self.ncz

    @ property
    def modx(self):

        xmod = []
        xmod.append(0)

        for i in range(self.ncx):

            xmod.append(xmod[i] + self.wcx[i])

        return np.array(xmod)


    @ property
    def mody(self):
        '''center point of cells in x direction'''

        ymod = []
        ymod.append(0)

        for i in range(self.ncy):

            ymod.append(ymod[i] + self.wcy[i])

        return np.array(ymod)


    @ property
    def modz(self):
        '''center point of cells in x direction'''

        zmod = []
        zmod.append(0)

        for i in range(self.ncz):

            zmod.append(zmod[i] + self.wcz[i])

        return np.array(zmod)

    @ property
    def ccx(self):
        '''center point of cells in x direction'''

        xmid = []
        for i in range(self.ncx):

            xmid.append(0.5 * (self.modx[i] + self.modx[i+1]))

        return np.array(xmid)


    @ property
    def ccy(self):
        '''center point of cells in x direction'''

        ymid = []
        for i in range(self.ncy):

            ymid.append(0.5 * (self.mody[i] + self.mody[i+1]))

        return np.array(ymid)


    @ property
    def ccz(self):
        '''center point of cells in x direction'''

        zmid = []
        for i in range(self.ncz):

            zmid.append(0.5 * (self.modz[i] + self.modz[i+1]))

        return np.array(zmid)

    @ property
    def nnx(self):
        '''number of nodes in x direction'''

        return len(self.wcx) + 1

    @ property
    def nny(self):
        '''number of nodes in y direction'''

        return len(self.wcy) + 1

    @ property
    def nnz(self):
        '''number of nodes in z direction'''

        return len(self.wcz) + 1


    @ property
    def nn(self):
        '''number of nodes in z direction'''

        return self.nnx * self.nny * self.nnz


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
        h1 = self.wcz
        h2 = self.wcx
        h3 = self.wcy

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


        Iz = sp.eye(self.ncz)
        Ix = sp.eye(self.ncx)
        Iy = sp.eye(self.ncy)


        if direc == 'x':
            e = np.ones(self.ncx+1)
            Dx = sp.spdiags([-e, e], [0, 1], self.ncx, self.ncx+1)
            D = sp.kron(sp.kron(Iy, Dx), Iz)


        if direc == 'y':
            e = np.ones(self.ncy+1)
            Dy = sp.spdiags([-e, e], [0, 1], self.ncy, self.ncy+1)
            D = sp.kron(sp.kron(Dy, Ix), Iz)


        if direc == 'z':
            e = np.ones(self.ncz+1)
            Dz = sp.spdiags([-e, e], [0, 1], self.ncz, self.ncz+1)
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


        Iz = sp.eye(self.ncz)
        Ix = sp.eye(self.ncx)
        Iy = sp.eye(self.ncy)


        if direc == 'x':
            e = np.ones(self.ncx+1) * 0.5
            Dx = sp.spdiags([e, e], [0, 1], self.ncx, self.ncx+1)
            D = sp.kron(sp.kron(Iy, Dx), Iz)


        if direc == 'y':
            e = np.ones(self.ncy+1) * 0.5
            Dy = sp.spdiags([e, e], [0, 1], self.ncy, self.ncy+1)
            D = sp.kron(sp.kron(Dy, Ix), Iz)


        if direc == 'z':
            e = np.ones(self.ncz+1)* 0.5
            Dz = sp.spdiags([e, e], [0, 1], self.ncz, self.ncz+1)
            D = sp.kron(sp.kron(Iy, Ix), Dz)

        return D


    def getNodalGradientMatrix(self, ncz, ncx, ncy):

        self.n1 = ncz
        self.n2 = ncx
        self.n3 = ncy

        G1 = kron3(speye(self.n3+1), speye(self.n2+1), sdiff(self.n1))
        G2 = kron3(speye(self.n3+1), sdiff(self.n2), speye(self.n1+1))
        G3 = kron3(sdiff(self.n3), speye(self.n2+1), speye(self.n1+1))

        return sp.vstack((G1,
                          G2,
                          G3), format = 'csr'
                )

    def getEdgeToCellCenterMatrix(self, ncz, ncx, ncy):

        self.n1 = ncz
        self.n2 = ncx
        self.n3 = ncy

        G1 = kron3(sav(self.n3), sav(self.n2), speye(self.n1))
        G2 = kron3(sav(self.n3), speye(self.n2), sav(self.n1))
        G3 = kron3(speye(self.n3), sav(self.n2), sav(self.n1))

        return sp.hstack((G1,
                          G2,
                          G3), format = 'csr'
                )

    def getNodalToCellCenterMatrix(self, ncz, ncx, ncy):

        self.n1 = ncz
        self.n2 = ncx
        self.n3 = ncy

        return kron3(sav(self.n3), sav(self.n2), sav(self.n1))


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



    '''This function code is borrowed from Felicia rnurindr@gmail.com'''
    def SourceCorrection(self, A, sigma_H, I, PositiveElectrode_Loc, NegativeElectrode_Loc):

        rho = 1 / sigma_H

        xmod = self.modx
        ymod = self.mody
        zmod = self.modz

        phi_true = np.zeros((self.ncz, self.ncx, self.ncy))
        phi_true_vec = np.zeros(self.ncz * self.ncx * self.ncy)

        for i in range(self.ncy):

            xloc = (xmod[i]+ xmod[i+1])/2.

            for j in range(self.ncx):

                yloc = (ymod[j] + ymod[j+1])/2.

                for k in range(self.ncz):

                    zloc = (zmod[k] + zmod[k+1])/2.;
                    CellLoc = np.transpose([zloc,yloc,xloc])# the coordinates of the center of a cell

                    distanceA = np.linalg.norm(CellLoc - PositiveElectrode_Loc)
                    distanceB = np.linalg.norm(CellLoc - NegativeElectrode_Loc)

                    phi_true[k,j,i] = rho*I*(1./distanceA - 1./distanceB)/(2*np.pi)

        phi_true_vec = phi_true.flatten('F')
        q_corr = A*phi_true_vec

        return(q_corr)



    def SourceCorrection_Node(self, A, sigma_H, I, PositiveElectrode_Loc, NegativeElectrode_Loc):

        rho = 1 / sigma_H

        xmod = self.modx
        ymod = self.mody
        zmod = self.modz

        phi_true = np.zeros((self.ncz+1, self.ncx+1, self.ncy+1))

        for i in range(self.ncy+1):

            for j in range(self.ncx+1):

                for k in range(self.ncz+1):

                    CellLoc = np.transpose([zmod[k],ymod[j],xmod[i]])# the coordinates of the center of a cell

                    distanceA = np.linalg.norm(CellLoc - PositiveElectrode_Loc)
                    distanceB = np.linalg.norm(CellLoc - NegativeElectrode_Loc)

                    phi_true[k,j,i] = rho*I*(1./distanceA - 1./distanceB)/(2*np.pi)

        phi_true_vec = phi_true.flatten('F')
        q_corr = A*phi_true_vec



        return(q_corr)
