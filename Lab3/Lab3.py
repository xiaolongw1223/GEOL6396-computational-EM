#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 11:50:11 2019

@author: wxl
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def kernel_func(x, i):
    
    return np.cos(0.5 * np.pi * x * (i - 1)) * np.exp(-0.25 * x * (i - 1))


data = np.loadtxt('model.txt')
model_true = data[:, 1]
x = data[:, 0]

#---------Task 1: Plot up model and kernel functions---------------------------
plt.figure()
im = plt.plot(x, model_true)
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('True model')

plt.figure()
ax = plt.gca()
for i in range(1, 21):
    g = kernel_func(x, i)
    im = plt.plot(x, g)
ax.set_xlabel('x')  
ax.set_ylabel('kernel value')
ax.set_title('kernel functions for i = 1, 2 ... 20')      


#---------Task 2: Discretization-----------------------------------------------
n = 20 # num of data
m = len(model_true) # num of model
'''dimension of G matrix should be 20 x 100'''
#dx = 0.01
G = np.zeros([n, m])
for i in range(n):
    for j in range(m):
        G[i, j] = kernel_func(x[j], i)

plt.figure()
im = plt.imshow(G)
ax = plt.gca()
ax.set_title('image of kernel matrix')
plt.colorbar(orientation = 'horizontal')


#--------Task 3: Generate clean and noisy data---------------------------------
ind = np.linspace(1, 20, n)
d_clean = G.dot(model_true)
d_noisy = d_clean + np.random.randn(len(d_clean)) * 0.01
plt.figure()
plt.plot(ind, d_clean, 'b', label=r'd_clean')
plt.plot(ind, d_noisy, 'r', label=r'd_noisy')
plt.legend()
plt.title('clean data and noisy data')
plt.ylabel('data value')
plt.xlabel('data index')


#-----Task 4: Understand singular values and singular vectors------------------
u, s, v = np.linalg.svd(G, full_matrices=False)
v = v.T
plt.figure()
plt.plot(s, '*')
plt.title('singular value')
plt.ylabel('singular value')
plt.xlabel('the number of singular value')
plt.yscale('log')

plt.figure()
for i in range(4):
    plt.plot(u[:,i], label = r'i={}'.format(i+1))
plt.title('left singular vector with i = 1, 2, 3, 4')
plt.ylabel('singular vector')
plt.xlabel('x')
plt.legend()

plt.figure()
for i in range(4):
    plt.plot(v[:,i], label = r'i={}'.format(i+1))
plt.title('right singular vector with i = 1, 2, 3, 4')
plt.ylabel('singular vector')
plt.xlabel('x')
plt.legend()

#------Task 5: SVD solution with clean data------------------------------------
rdata_clean = u.T.dot(d_clean)
ratio = np.divide(rdata_clean, s)
plt.figure()
plt.plot(rdata_clean, 'b', label=r'rotated clean data')
plt.plot(s, 'r', label=r'singular value')
plt.plot(ratio, 'g', label =r'ratio') 
plt.legend()
#plt.yscale('log')
plt.title('clean data')

m_inv1 = 0
for i in range(20):
    m_inv1 += (u[:,i].T.dot(d_clean) / s[i]) * v[:,i]

plt.figure()
plt.plot(x, model_true, 'b', label=r'True mdoel')
plt.plot(x, m_inv1, 'r', label=r'SVD inverted model')
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('SVD inverted mdoel from clean data')
plt.legend()


#------Task 6: SVD solution with noisy data -----------------------------------
rdata_noisy = u.T.dot(d_noisy)
ratio = rdata_noisy / s
plt.figure()
plt.plot(rdata_noisy, 'b', label=r'rotated noisy data')
plt.plot(s, 'r', label=r'singular value')
plt.plot(ratio, 'g', label =r'ratio')    
#plt.yscale('log')
plt.legend()
plt.title('noisy data')

rdata_noisy = u.T.dot(d_noisy)
ratio = rdata_noisy / s
plt.figure()
plt.plot(rdata_noisy, 'b', label=r'rotated noisy data')
plt.plot(s, 'r', label=r'singular value')
#plt.plot(ratio, 'g', label =r'ratio')    
#plt.yscale('log')
plt.legend()
plt.title('noisy data')

m_inv1 = 0
M = np.zeros([20, 100])
for i in range(20):
    m_inv1 += (u[:,i].T.dot(d_noisy) / s[i]) * v[:,i]
    M[i, :] = m_inv1
    
plt.figure()
plt.plot(x, model_true, 'b', label=r'True mdoel')
plt.plot(x, m_inv1, 'r', label=r'SVD inverted model')
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('SVD inverted mdoel from noisy data')
plt.legend()

#-------Task 7: Construct Tikhonov curve---------------------------------------
Wd = sp.diags(np.ones(20) * (1/0.01))

phid = []
phim = []
p = np.linspace(1, 20, 20)
for i in range(20):
    phid.append(
            np.linalg.norm(
                    Wd.dot(d_noisy - G.dot(M[i, :]))
                    )
            )

    phim.append(
            np.linalg.norm(M[i, :])
            )

plt.figure()
plt.plot(p, phid)
ax = plt.gca()
ax.set_xlabel('number of singular value (p)')
ax.set_ylabel('phid')
ax.set_title('num of singular value vs. datamisfit')

plt.figure()
plt.plot(p, phim)
ax = plt.gca()
ax.set_xlabel('number of singular value (p)')
ax.set_ylabel('phim')
ax.set_title('num of singular value vs. model norm')
ax.set_yscale('log')
ax.set_xscale('log')

plt.figure()
plt.plot(phid, phim)
ax = plt.gca()
ax.set_xlabel('datamisfit')
ax.set_ylabel('model norm')
ax.set_title('Tikhonov curve')
ax.set_yscale('log')
ax.set_xscale('log')

m_inv1 = 0
for i in range(10):
    m_inv1 += (u[:,i].T.dot(d_noisy) / s[i]) * v[:,i]
    
plt.figure()
plt.plot(x, model_true, 'b', label=r'True mdoel')
plt.plot(x, m_inv1, 'r', label=r'SVD inverted model')
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('The Best SVD inverted mdoel from noisy data')
plt.legend()




