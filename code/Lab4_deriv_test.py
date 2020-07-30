#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 16:59:12 2019

@author: wxl
"""
#import scipy.io as sio
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d # unused import
import time
import datetime
import geometry


def run(plotIt=True):

    init_t = time.time()

    hxind = [0.75*np.ones(60), 3*45, 1.2]
    hyind = [0.75*np.ones(60), 3*45, 1.2]
    hzind = [0.75*np.ones(60), 3*45, 1.2]

    Mesh = geometry.Mesh(hxind, hyind, hzind)
    U = geometry.Utils(hxind, hyind, hzind)

    SourceLocation3D_postive_Loc = [0, 64, 76]
    SourceLocation3D_negative_Loc = [0, 87, 76]

    sigma = np.ones([Mesh.ncz, Mesh.ncx, Mesh.ncy]) * 0.01
    sigma[1:16, 39:50, 39:50] = 0.1
    sigma = sigma.flatten('F')
    print('sigma.shape', sigma.shape)

    Discre = geometry.Discretize(hxind, hyind, hzind)
    [V, F, L] = Discre.getMeshGeometry
    print("V.shape", V.shape)
    print("F.shape", F.shape)

    Grad = Discre.getNodalGradientMatrix(Mesh.ncz, Mesh.ncx, Mesh.ncy)
    print('Grad.shape', Grad.shape)

    Ae = Discre.getEdgeToCellCenterMatrix(Mesh.ncz, Mesh.ncx, Mesh.ncy)
    print('Ae.shape', Ae.shape)

    Me = sp.diags(Ae.T.dot(V.dot(sigma)))
    print('Me.shape', Me.shape)

    A = Grad.T.dot(Me.dot(Grad))
    print('A.shape', A.shape)
    print('finishing constructing A matrix ...\n')

    sigma_H = 0.01
    Me_H = sp.diags(Ae.T.dot(V.dot(sigma_H * np.ones(Mesh.nc))))
    print("Me_H.shape", Me_H.shape)

    A_H = Grad.T.dot(Me_H.dot(Grad))
    print("A_H.shape", A_H.shape)
    q_corr = U.SourceCorrection_Node(A_H, sigma_H, 1,
                                          SourceLocation3D_postive_Loc,
                                          SourceLocation3D_negative_Loc)
    print("q_corr.shape", q_corr.shape)
    print('finishing constructing q_corr ...\n')

    u = U.CG(A, q_corr)
    print('finishing calculating potentials ...\n')

    Anc = Discre.getNodalToCellCenterMatrix(Mesh.ncz, Mesh.ncx, Mesh.ncy)

    m = np.log(sigma) # convert sigma into logarithm space

    tmp = sp.diags(Ae.T.dot(V.dot(np.exp(m))))
    dCdu = Grad.T.dot(tmp).dot(Grad)
    print('done dCdu')

    dCdm = Grad.T.dot(sp.diags(Grad.dot(u))).dot(Ae.T.dot(V)).dot(sp.diags(np.exp(m)))
    print('done dCdm')



    '''

    testing dC/du


    '''
    np.random.seed(42)
    t = np.random.randn(Mesh.nn)

    Cum = A.dot(u)

    '''write data'''
    f = open('GradientTest.txt', 'w')
    header = '#     h        diff1       diff2 \n'
    f.write('Gradient test for dC/du ...\n')
    f.write(header)
    f.close


    for i in range(1,11):

        h = 10**(-i)
        diff1 = np.linalg.norm(A.dot(u+h*t) - Cum)

        diff2 = np.linalg.norm(A.dot(u+h*t) - Cum -
                               h*(dCdu).dot(t)
                )



        f = open('GradientTest.txt', 'a')
        f.write(
                '{0:2d} {1:1.4e} {2:1.4e} {3:1.4e}\n'.format(
                        i, h, diff1, diff2
                        )
                )
        f.close()




    '''
    testing dC/dm


    '''
    np.random.seed(44)
    t = np.random.randn(Mesh.nc)
    '''write data'''
    f = open('GradientTest.txt', 'a')
    f.write('Gradient test for dC/dm ...\n')
    header = '#      h         diff3        diff4 \n'
    f.write(header)
    f.close

    for i in range(1,11):

        h = 10**(-i)

        Me_test = sp.diags(Ae.T.dot(V.dot(np.exp(m+h*t)))) # sigma is not logarithm space
        A_test = Grad.T.dot(Me_test.dot(Grad))

        diff3 = np.linalg.norm(A_test.dot(u) - Cum)

        diff4 = np.linalg.norm(A_test.dot(u) - Cum -
                               h*(dCdm).dot(t)
                )



        f = open('GradientTest.txt', 'a')
        f.write(
                '{0:2d} {1:1.4e} {2:1.4e} {3:1.4e}\n'.format(
                        i, h, diff3, diff4
                        )
                )
        f.close()




    '''
    Create the projection matrix that converts the simulate potentials u to
    measured voltages


    '''

    P = np.zeros([3360, Mesh.nc])
    icount = 0
    for j in range(14, 73):
        for k in range(14, 69):
            icount =  icount + 1
            P[icount, 77*88*j + 77*k + 1] = -1
            P[icount, 77*88*j + 77*(k+4) + 1] = 1

    P = sp.csr_matrix(P)
    P = P.dot(Anc)
    print('P.shape', P.shape)
    DCdata = P.dot(u)
    print('done DCdata')




    '''
    TASK 4: Creat two functions that compute Jv and J^T w,
    for any given vectors v and w.


    '''
    def Jv(g, A, v):
        tmp1 = g.dot(v) # vector
        tmp2 = U.CG(A, tmp1) # tmp1 = A * tmp2

        return -P.dot(tmp2)

    def JTw(g, A, w):
        tmp1 = P.T.dot(w)
        tmp2 = U.CG(A.T, tmp1)

        return -g.T.dot(tmp2)





    '''
    TASK 5: Run the adjoint test


    '''
    np.random.seed(48)
    v = np.random.randn(Mesh.nc)
    np.random.seed(56)
    w = np.random.randn(3360)

    Jv = Jv(dCdm, A, v)
    print('done Jv')
    JTw = JTw(dCdm, A, w)
    print('done JTv')

    Diff = np.abs(w.T.dot(Jv) - v.T.dot(JTw))
    print('Adjoint test value is:', Diff)





    if plotIt:

        # figure 1 ----------------
        [xx, yy] = np.meshgrid(Mesh.mody, Mesh.modx)
        zz = np.zeros([len(yy), len(yy[0])])

        fig = plt.figure()
        ax = plt.gca(projection='3d')
        ax.scatter([63, 86], [75, 75], [0, 0], marker='v', c = 'r', linewidth=2)
        ax.plot_surface(xx, yy, zz)
        ax.set_ylabel('Northing (m)')
        ax.set_xlabel('Easting (m)')
        plt.savefig('Lab4_source_location.png', bbox_inches='tight', dpi=300)

        # figure 2 -----------------
        phi3d_center = np.reshape(
            Anc.dot(u),
            (Mesh.ncz, Mesh.ncx, Mesh.ncy), order='F'
            )

        phi3d_surface = np.squeeze(phi3d_center[0, :, :]).T
        extent = [Mesh.ccy[14], Mesh.ccy[74], Mesh.ccx[14], Mesh.ccx[74]]

        fig = plt.figure()
        ax = plt.gca()
        im = ax.imshow(phi3d_surface[14:74, 14:74], cmap ='jet', extent = extent)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_title('Electrical potential at the surface')
        fig.colorbar(im, ax = ax, orientation="vertical")
        plt.savefig('Lab4_potential_surface.png', bbox_inches='tight', dpi=300)


        # figure 2-1 -----------------
        phi3d_slice = np.squeeze(phi3d_center[:, :, 44])
        extent = [Mesh.ccx[14], Mesh.ccx[74], Mesh.ccz[60], Mesh.ccz[0]]

        fig = plt.figure()
        ax = plt.gca()
        im = ax.imshow(phi3d_slice[0:60, 14:74], cmap ='jet', extent = extent)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Depth (m)')
#        ax.invert_yaxis()
        ax.set_title('Electrical potential directly underneath current electrodes')
        fig.colorbar(im, ax = ax, orientation="vertical")
        plt.savefig('Lab4_potential_underneath.png', bbox_inches='tight', dpi=300)


         # figure 3 -----------------

        fig = plt.figure()
        ax = plt.gca()
        im = ax.imshow(np.reshape(DCdata, (56, 60), order = 'F').T, cmap ='jet')
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_title('Measured voltages at the surface')
        fig.colorbar(im, ax = ax, orientation="vertical")
        plt.savefig('Lab4_votage.png', bbox_inches='tight', dpi=300)


    elapse = time.time() - init_t
    print("time: ", str(datetime.timedelta(seconds=elapse)))
if __name__ == '__main__':
    run()
    plt.show()
