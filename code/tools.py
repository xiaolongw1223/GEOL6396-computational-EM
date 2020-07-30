#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:36:43 2019

@author: wxl
"""

import numpy as np
import scipy.sparse as sp

# =============================================================================
#
# Part of code is borrowed from SimPEG.discretize.utils
#
# This package is created by Xiaolong Wei, 11/08/2019
# University of Houston
# 
# =============================================================================

def mkvc(x, numDims=1):
    """
    
    Creates a vector with the number of dimension specified
    e.g.::
        a = np.array([1, 2, 3])
        mkvc(a, 1).shape
            > (3, )
        mkvc(a, 2).shape
            > (3, 1)
        mkvc(a, 3).shape
            > (3, 1, 1)

    """
    if type(x) == np.matrix:
        x = np.array(x)

    if hasattr(x, 'tovec'):
        x = x.tovec()


    assert isinstance(x, np.ndarray), "Vector must be a numpy array"

    if numDims == 1:
        return x.flatten(order='F')
    elif numDims == 2:
        return x.flatten(order='F')[:, np.newaxis]
    elif numDims == 3:
        return x.flatten(order='F')[:, np.newaxis, np.newaxis]
    
    
def sdiag(h):
    """Sparse diagonal matrix"""

    return sp.spdiags(mkvc(h), 0, h.size, h.size, format="csr")


def sdInv(M):
    "Inverse of a sparse diagonal matrix"
    return sdiag(1.0 / M.diagonal())


def speye(n):
    """Sparse identity"""
    return sp.eye(n, format="csr")


def kron3(A, B, C):
    """
    Three kron prods, convert 2D or 3D to a diag sparse matrix.
    A, B, C = Z, Y, X
    """
    return sp.kron(sp.kron(A, B), C, format="csr")


def outer3(A, B, C):
    '''
    Three ourter prods, convert 2D or 3D to a array
    A, B, C = X, Y, Z
    '''
    return mkvc(np.outer(mkvc(np.outer(A, B)), C))

def sav(n): 
    """
    
    Define 1D averaging operator from faces to cell-centers.
    n is number of cells.
    
    The return value is n by n+1
    
    """
    return sp.spdiags(
        (0.5*np.ones((n+1, 1))*[1, 1]).T, [0, 1], n, n+1,
        format="csr"
    )


def sdiff(n):
    """
    
    Define 1D forward finite difference from faces to cell-centers
    n is number of cells.
    
    The return value is n by n+1
    
    """
    return sp.spdiags(
        (np.ones((n+1, 1))*[-1, 1]).T, [0, 1], n, n+1,
        format="csr"
    )











