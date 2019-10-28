#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:47:25 2019

@author: wxl
"""


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
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
        tmp = h[-1]
        depth = tmp * 1.1
        h = np.append(h, depth)
        T_depth = np.sum(h)
        if T_depth > L:
            break
    
        
    print('h.shape', h.shape)
    
    n = len(h) # total number of cells in the model when the frequency is omega(j).
    sig = np.ones(n) * sig0
    print('sig.shape', sig.shape)

    '''
    construct G, Linv, Av, Mmu, Msig matrices using spdiags
    '''
    w = sp.diags(np.ones(n) * omega[j] * (1j))
    print('w.shape', w.shape)
    
    e = np.ones(n+1) # a list to generate sparse matrix
    
    Bin = np.array([-e, e])
    D = np.array([0, 1])
    G = sp.spdiags(Bin, D, n, n+1)
    print('G.shape', G.shape)

    
    Bin = np.array([0.5*e, 0.5*e])
    D = np.array([0, 1])
    Av = sp.spdiags(Bin, D, n, n+1)
    print('Av.shape', Av.shape)

    
    Linv = sp.diags(1/h)
    print('Linv.shape', Linv.shape)

    Mmu  = sp.diags(h/mu)
    print('Mmu.shape', Mmu.shape)
    
    tmp = Av.T.dot(np.multiply(sig, h))
    Msig = sp.diags(tmp)
    print('Msig.shape', Msig.shape)
    
    '''
    create A matrix
    '''
    A11 = w
    print('A11.shape', A11.shape)

    A12 = Linv.dot(G)
    print('A12.shape', A12.shape)
#    plt.figure()
#    plt.imshow(A12.A)
    
    A21 = G.T.dot(Linv).dot(Mmu)
    print('A21.shape', A21.shape)

    A22 = Msig
    print('A22.shape', A22.shape)

    
    tmp1 = sp.hstack([A11, A12])
    tmp2 = sp.hstack([A21, A22])
    
    A =  sp.vstack([tmp1, tmp2]) # construct A matrix
    print('A.shape', A.shape)

    
    '''
    Setup boundary conditions b(0) = 1  (using the second method)
    '''
    A = A.A
    b0 = 1 # boundary condition
    
    bb = A[1:, 0] * (-b0) # Ax = bb
    bb = sp.csr_matrix(bb).T
    print('bb.shape', bb.shape)
    
    A  = A[1:, 1:]
    A = sp.csr_matrix(A)
    print('A_new.shape', A.shape)

    
    '''
    Solve the linear system of equations
    ''' 

    be = sp.linalg.spsolve(A, bb)

    b = be[0 : n-1]
    e = be[n:]
       
    '''extract MT data'''
    d = np.append(d, e[0])

fig = plt.figure()
ax1 = plt.subplot(2,1,1)
y = np.multiply(1/omega * mu, np.abs(d)**2)
y = np.round(y, 5)
plt.semilogx(omega, y)
ax1.invert_xaxis()
ax1.set_ylabel('Resistivity (Ohm*m)')
ax1.set_title('Uniform Earth Model')
#ax1.set_xlabel('Frequency(r/s)')

ax2 = plt.subplot(2,1,2)
y = 180-np.angle(d)*180/np.pi
y = np.round(y, 5)
plt.semilogx(omega, y)
#ax2.set_title('phase')
ax2.set_xlabel('Frequency (r/s)')
ax2.set_ylabel('Phase ($^\circ$)')
ax2.invert_xaxis()
