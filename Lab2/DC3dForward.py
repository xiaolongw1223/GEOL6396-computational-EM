#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 16:59:12 2019

@author: wxl
"""
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d # unused import
import time
import datetime
import geometry


def run(plotIt=True):

    t = time.time()

    hxind = [0.75*np.ones(60), 3*45, 1.2]
    hyind = [0.75*np.ones(60), 3*45, 1.2]
    hzind = [0.75*np.ones(60), 3*45, 1.2]


    Mesh = geometry.Mesh(hxind, hyind, hzind)
    U = geometry.Utils(hxind, hyind, hzind)

    '''location of positive current electrode: 0 m in depth, 64 m in easting and 76 m in northing.'''
    SourceLocation3D_postive_Loc = [0, 64, 76]

    '''location of negative current electrode: 0 m in depth, 87 m in easting and 76 m in northing.'''
    SourceLocation3D_negative_Loc = [0, 87, 76]


    sigma = np.ones([Mesh.ncz, Mesh.ncx, Mesh.ncy]) * 0.01
    sigma2 = sigma.copy('F')
    sigma2[4:20, 39:50, 39:50] = 0.1

    sigma = sigma.flatten('F')
    sigma2 = sigma2.flatten('F')  # inhomo

    Discre = geometry.Discretize(hxind, hyind, hzind)
    [V, F, L] = Discre.getMeshGeometry
    print("V.shape", V.shape)
    print("F.shape", F.shape)


    Div = sp.hstack([Discre.Diff(direc='z'), Discre.Diff(direc='x'), Discre.Diff(direc='y')])
    print("Div.shape", Div.shape)


    V_inv = sp.diags(1/np.diag(V.A))
    D = V_inv.dot(Div).dot(F)
    print("D.shape", D.shape)


    Afc = sp.hstack([Discre.Diff2(direc='z'), Discre.Diff2(direc='x'), Discre.Diff2(direc='y')])
    print("Afc.shape", Afc.shape)


    Mf = Afc.T.dot(V.dot(1/sigma))
    Mf2 = Afc.T.dot(V.dot(1/sigma2)) # inhomo
    print("Mf.shape", Mf.shape)
    print("Mf.type", type(Mf))


    Mf_inv = sp.diags(1/Mf)
    A = V.dot(D).dot(Mf_inv).dot(D.T).dot(V)
    print("A.shape", A.shape)


    Mf_inv2 = sp.diags(1/Mf2)
    A2 = V.dot(D).dot(Mf_inv2).dot(D.T).dot(V) #inhomo


    sigma_H = 0.01
    Mf_H = sp.diags(Afc.T.dot(V.dot(1/sigma_H * np.ones(Mesh.nc))))
    print("Mf_H.shape", Mf_H.shape)

    Mf_H_inv = sp.diags(1/np.diag(Mf_H.A))
    A_H = V.dot(D).dot(Mf_H_inv.dot(D.T.dot(V)))
    print("A_H.shape", A_H.shape)


    q_corr = U.SourceCorrection(A_H, sigma_H, 1,
                                          SourceLocation3D_postive_Loc,
                                          SourceLocation3D_negative_Loc)
    print("q_corr.shape", q_corr.shape)


    phi = U.CG(A, q_corr)
    phi2 = U.CG(A2, q_corr)
    diff = np.subtract(phi2, phi)


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
        plt.savefig("Lab2_fig_1.png", bbox_inches="tight", dpi=300)


        # figure 2-1 -----------------
        phi3d = np.reshape(phi, [Mesh.ncz, Mesh.ncx, Mesh.ncy], 'F')
        phi3d_surface = np.squeeze(phi3d[0, :, :]).T
        extent = [Mesh.ccy[14], Mesh.ccy[73], Mesh.ccx[14], Mesh.ccx[73]]

        fig = plt.figure()
        ax = plt.gca()
        im = ax.imshow(phi3d_surface[14:73, 14:73], cmap ='jet', extent = extent)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_title('Electrical potential at the surface')
        fig.colorbar(im, ax = ax, orientation="vertical")
        plt.savefig("Lab2_fig_2-1.png", bbox_inches="tight", dpi=300)

        # figure 2-2 -----------------
        phi3d_slice = np.squeeze(phi3d[:, :, 44])
        extent = [Mesh.ccx[14], Mesh.ccx[73], Mesh.ccz[59], Mesh.ccz[0]]

        fig = plt.figure()
        ax = plt.gca()
        im = ax.imshow(phi3d_slice[0:59, 14:73], cmap ='jet', extent = extent)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Depth (m)')
#        ax.invert_yaxis()
        ax.set_title('Electrical potential directly underneath current electrodes')
        fig.colorbar(im, ax = ax, orientation="vertical")
        plt.savefig("Lab2_fig_2-2.png", bbox_inches="tight", dpi=300)


         # figure 3-1 -----------------
        phi3d3 = np.reshape(diff, [Mesh.ncz, Mesh.ncx, Mesh.ncy], 'F')
        phi3d_surface3 = np.squeeze(phi3d3[0, :, :]).T
        extent = [Mesh.ccy[14], Mesh.ccy[73], Mesh.ccx[14], Mesh.ccx[73]]

        fig = plt.figure()
        ax = plt.gca()
        im = ax.imshow(phi3d_surface3[14:73, 14:73], cmap ='jet', extent = extent)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_title('Secondary potential at the surface')
        fig.colorbar(im, ax = ax, orientation="vertical")
        plt.savefig("Lab2_fig_3-1.png", bbox_inches="tight", dpi=300)

        # figure 3-2 -----------------
        phi3d_slice3 = np.squeeze(phi3d3[:, :, 44])
        extent = [Mesh.ccx[14], Mesh.ccx[73], Mesh.ccz[59], Mesh.ccz[0]]

        fig = plt.figure()
        ax = plt.gca()
        im = ax.imshow(phi3d_slice3[0:59, 14:73], cmap ='jet', extent = extent)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('Secondary potential directly underneath current electrodes')
        fig.colorbar(im, ax = ax, orientation="vertical")
        plt.savefig("Lab2_fig_3-2.png", bbox_inches="tight", dpi=300)

    elapse = time.time() - t
    print("time: ", str(datetime.timedelta(seconds = elapse)))
if __name__ == '__main__':
    run()
    plt.show()
