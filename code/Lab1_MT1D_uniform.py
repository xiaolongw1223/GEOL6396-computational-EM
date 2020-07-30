#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:47:25 2019

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

''' the uniform conductivity '''
sig0 = 0.01

''' calculate the skin depth '''
skin = np.sqrt(2 / (omega * mu * sig0))

d = [] # store the data
for j in range(len(omega)):

    print("Iteration {} ...".format(j))

    L = 3 * skin[j] # maximum depth, three times of the skin depth

    '''
    Discretize the entire depth range into a large number of smaller cells
    Uniformly discretize the model into 2049 cells for depth up to twice the skin depth
    store the thickenss values of all layers (cells) to a column vector h.
    '''

    h = np.ones(2049)*(2 * skin[j] / 2049)


    '''
    Linearly increase the thickness of cells by a factor of 1.1, for cells
    whose depths are larger than twice the skin depth but not greater than
    three times the skin depth
    '''
#    depth = 2 * skin[j]
    for i in range(100000):
        depth = h[-1] * 1.1
        h = np.append(h, depth)
        T_depth = np.sum(h)
        if T_depth > L:
            break

    n = len(h) # total number of cells in the model when the frequency is omega(j).
    sig = np.ones(n) * sig0

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


y1 = np.multiply(1/omega * mu, np.abs(d)**2)
y1 = np.round(y1, 5)
y2 = 180-np.angle(d)*180/np.pi
y2 = np.round(y2, 5)

fig = plt.figure()
ax1 = plt.subplot(2,1,1)
ax1.semilogx(omega, y1)
ax1.invert_xaxis()
ax1.set_ylabel('Resistivity (Ohm*m)')
ax1.set_title('Uniform Earth Model')
#ax1.set_xlabel('Frequency(r/s)')
ax2 = plt.subplot(2,1,2)
ax2.semilogx(omega, y2)
#ax2.set_title('phase')
ax2.set_xlabel('Frequency (r/s)')
ax2.set_ylabel('Phase ($^\circ$)')
ax2.invert_xaxis()
plt.savefig("Lab1_uniform.png", bbox_inches="tight", dpi=300)
