#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 21:57:21 2019

@author: Xiaolong Wei
"""


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt


''' frequency at which MT signals are to be computed '''
omega = np.logspace(1, 5, 32)

''' the constant value for magnetic permeability '''
mu = 12.56e-7

''' set up the maximum depth L to be 12000 meters '''
L = 12000

'''
Discretize the entire depth range into a large number of smaller cells
Uniformly discretize the first 6000 meters into 2049 cells
store the thickenss values of all layers (cells) to a column vector h.
'''
h = np.ones(2049)*(6000 / 2049)

'''
Linearly increase the thickness of cells by a factor of 1.1, for cells
whose depths are larger than 6000 meters but no greater than 12,000 meters
'''

for i in range(100000):
    depth = h[-1] * 1.1
    h = np.append(h, depth)
    T_depth = np.sum(h)
    if T_depth > L:
        break

''' total number of cells in the model'''
n = len(h)

'''
assign conductivity values to the cells.
assign 0.01 S/m to cells in the first 200 meters.
assign 0.1 S/m to cells in the next 200 meters.
assign 0.002 S/m to the rest
'''

sig = np.ones(n) * 0.002
idx1 = np.where(np.cumsum(h) > 200)[0] # decide the index of element > 200m
sig[0 : idx1[0]-1] = 0.01
idx2 = np.where(np.cumsum(h) > 400)[0] # decide the index of elelment > 400m
sig[idx1[0]-1 : idx2[0]-1] = 0.1

d = []

for j in range(len(omega)):
    print("Iteration {} ...".format(j))
    '''
    construct G, Linv, Av, Mmu, Msig matrices using spdiags
    '''
    w = sp.diags(np.ones(n) * omega[j] * (1j))

    e = np.ones(n+1) # a list to generate sparse matrix

    Bin = np.array([-e, e])
    D = np.array([0, 1])
    G = sp.spdiags(Bin, D, n, n+1)

    Bin = np.array([0.5*e, 0.5*e])
    D = np.array([0, 1])
    Av = sp.spdiags(Bin, D, n, n+1)

    Linv = sp.diags(1/h)

    Mmu  = sp.diags(h/mu)

    tmp = Av.T.dot(np.multiply(sig, h))
    Msig = sp.diags(tmp)

    '''
    create A matrix
    '''
    A11 = w
    A12 = Linv.dot(G)
    A21 = G.T.dot(Linv).dot(Mmu)
    A22 = Msig
    tmp1 = sp.hstack([A11, A12])
    tmp2 = sp.hstack([A21, A22])
    A =  sp.vstack([tmp1, tmp2]) # construct A matrix

    '''
    Setup boundary conditions b(0) = 1  (using the second method)
    '''
    A = A.A
    b0 = 1 # boundary condition

    bb = A[1:, 0] * (-b0) # Ax = bb
    bb = sp.csr_matrix(bb).T

    A  = A[1:, 1:]
    A = sp.csr_matrix(A)

    '''
    Solve the linear system of equations
    '''
    be = linalg.spsolve(A, bb)

    b = be[0 : n-1]
    e = be[n:]

    '''extract MT data'''
    d = np.append(d, e[0])


fig = plt.figure()
ax1 = plt.subplot(2,1,1)
ax1.semilogx(omega, np.multiply(1/omega * mu, np.abs(d)**2))
ax1.invert_xaxis()
ax1.set_ylabel('Resistivity (Ohm*m)')
ax1.set_title('Nonuniform Earth Model')
#ax1.set_xlabel('Frequency(r/s)')
ax2 = plt.subplot(2,1,2)
ax2.semilogx(omega, 180-np.angle(d)*180/np.pi)
#ax2.set_title('phase')
ax2.set_xlabel('Frequency (r/s)')
ax2.set_ylabel('Phase ($^\circ$)')
ax2.invert_xaxis()
plt.savefig("Nonuniform.png", bbox_inches="tight", dip=300)
