"""
PF: Gravity: Inversion Linear
=============================

Create a synthetic block model and invert
with a compact norm

"""
import numpy as np
import matplotlib.pyplot as plt

from SimPEG import Mesh
from SimPEG import Utils
from SimPEG import Maps
from SimPEG import Regularization
from SimPEG import DataMisfit
from SimPEG import Optimization
from SimPEG import InvProblem
from SimPEG import Directives # commands where stop iteration...
from SimPEG import Inversion
from SimPEG import PF # potential feild


def run(plotIt=True):

    # Create a mesh
    dx = 5.

    hxind = [(dx, 5, -1.3), (dx, 15), (dx, 5, 1.3)] # h1 = [ (cellSize, numPad, increaseFactor),  (cellSize, numCore),  (cellSize, numPad, increaseFactor) ]
    hyind = [(dx, 5, -1.3), (dx, 15), (dx, 5, 1.3)] # we dont expect regular size
    hzind = [(dx, 5, -1.3), (dx, 7), (3.5, 1), (2, 5)]

    mesh = Mesh.TensorMesh([hxind, hyind, hzind], 'CCC') # a 'C' for the center for axiss. A 'N' will make the entire mesh negative, and a '0' (or a 0) will make the mesh start at zero.

    # Get index of the center
    midx = int(mesh.nCx/2) # giving the index of center cell in any depth
    midy = int(mesh.nCy/2)

    # Lets create a simple Gaussian topo and set the active cells
    [xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy) # location of nodas in x and y direction
    zz = -np.exp((xx**2 + yy**2) / 75**2) + mesh.vectorNz[-1] # Gaussain bump
    
    # We would usually load a topofile
    topo = np.c_[Utils.mkvc(xx), Utils.mkvc(yy), Utils.mkvc(zz)]
    print(topo)
    # Go from topo to actv cells
    actv = Utils.surface2ind_topo(mesh, topo, 'N') 
    actv = np.asarray([inds for inds, elem in enumerate(actv, 1) if elem],
                      dtype=int) - 1 # this inds of active cell

    # Create active map to go from reduce space to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100) # -100 means air, if we ues -0.1 system cannot tell the difference bewteern active and inactive cells
    nC = len(actv)

    # Create and array of observation points
    xr = np.linspace(-30., 30., 20) # np.linspace(-30., 30., num = 20)
    yr = np.linspace(-30., 30., 20)
    X, Y = np.meshgrid(xr, yr)

    # Move the observation points 5m above the topo
    Z = -np.exp((X**2 + Y**2) / 75**2) + mesh.vectorNz[-1] + 0.1 # move every thing 0.1 uper, if without up move the gravity will be infinity because the distance is 0

    # Create a MAGsurvey
    rxLoc = np.c_[Utils.mkvc(X.T), Utils.mkvc(Y.T), Utils.mkvc(Z.T)]
    rxLoc = PF.BaseGrav.RxObs(rxLoc) # gravity location observeration
    srcField = PF.BaseGrav.SrcField([rxLoc]) # source location, source fields
    survey = PF.BaseGrav.LinearSurvey(srcField) 

    # We can now create a susceptibility model and generate data
    # Here a simple block in half-space
    model = np.zeros((mesh.nCx, mesh.nCy, mesh.nCz))
    model[(midx-5):(midx-1), (midy-2):(midy+2), -10:-6] = 0.75 # the abnormal value of model is 0.75
    model[(midx+1):(midx+5), (midy-2):(midy+2), -10:-6] = -0.75
    model = Utils.mkvc(model) # convert to a vector
    model = model[actv] # model only with active cells

    # Create active map to go from reduce set to full
    actvMap = Maps.InjectActiveCells(mesh, actv, -100)

    # Create reduced identity map
    idenMap = Maps.IdentityMap(nP=nC) # np means num of active cell, nc means num of cells

    # Create the forward model operator
    prob = PF.Gravity.GravityIntegral(mesh, rhoMap=idenMap, actInd=actv)

    # Pair the survey and problem
    survey.pair(prob)

    # Compute linear forward operator and compute some data
    d = prob.fields(model) # d=Gm

    # Add noise and uncertainties
    # We add some random Gaussian noise (1nT)
    data = d + np.random.randn(len(d))*1e-3 # Transform the unit, depends on G
    wd = np.ones(len(data))*1e-3  # Assign flat uncertainties

    survey.dobs = data # input any data in survey
    survey.std = wd
    survey.mtrue = model

    # Create sensitivity weights from our linear forward operator
    rxLoc = survey.srcField.rxList[0].locs
    wr = np.sum(prob.G**2., axis=0)**0.5 # depth weighting thing, how much contribution to each cell, **0.5 sqrt
    wr = (wr/np.max(wr)) # normalize the max value, having or not doesn't effects the inversion results

    # Create a regularization
    reg = Regularization.Sparse(mesh, indActive=actv, mapping=idenMap)
    reg.cell_weights = wr
    reg.norms = np.c_[1, 0, 0, 0] # weights like alph-s, x, y, z

    # Data misfit function
    dmis = DataMisfit.l2_DataMisfit(survey)
    dmis.W = Utils.sdiag(1/wd)

    # Add directives to the inversion
    opt = Optimization.ProjectedGNCG(maxIter=100, lower=-1., upper=1.,
                                     maxIterLS=20, maxIterCG=10,
                                     tolCG=1e-3) # the value of inversion could be bounded by -1 to 1, so the gravity inversion bevome non-linear problem.
    invProb = InvProblem.BaseInvProblem(dmis, reg, opt)
    betaest = Directives.BetaEstimate_ByEig() # using Eigen value to determine beta

    # Here is where the norms are applied
    # Use pick a treshold parameter empirically based on the distribution of
    # model parameters
    IRLS = Directives.Update_IRLS(
        f_min_change=1e-4, maxIRLSiter=30, coolEpsFact=1.5, beta_tol=1e-1,
    )
    saveDict = Directives.SaveOutputEveryIteration(save_txt=False)
    update_Jacobi = Directives.UpdatePreconditioner()
    inv = Inversion.BaseInversion(
        invProb, directiveList=[IRLS, betaest, update_Jacobi, saveDict]
    )

    # Run the inversion
    m0 = np.ones(nC)*1e-4  # Starting model
    mrec = inv.run(m0)

    if plotIt:
        # Here is the recovered susceptibility model
        ypanel = midx
        zpanel = -7
        m_l2 = actvMap * invProb.l2model
        m_l2[m_l2 == -100] = np.nan

        m_lp = actvMap * mrec
        m_lp[m_lp == -100] = np.nan

        m_true = actvMap * model
        m_true[m_true == -100] = np.nan

        vmin, vmax = mrec.min(), mrec.max()

        # Plot the data
        PF.Gravity.plot_obs_2D(rxLoc, d=data)

        plt.figure()

        # Plot L2 model
        ax = plt.subplot(321)
        mesh.plotSlice(m_l2, ax=ax, normal='Z', ind=zpanel,
                       grid=True, clim=(vmin, vmax))
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCy[ypanel], mesh.vectorCCy[ypanel]]), color='w')
        plt.title('Plan l2-model.')
        plt.gca().set_aspect('equal')
        plt.ylabel('y')
        ax.xaxis.set_visible(False)
        plt.gca().set_aspect('equal', adjustable='box')

        # Vertica section
        ax = plt.subplot(322)
        mesh.plotSlice(m_l2, ax=ax, normal='Y', ind=midx,
                       grid=True, clim=(vmin, vmax))
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCz[zpanel], mesh.vectorCCz[zpanel]]), color='w')
        plt.title('E-W l2-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

        # Plot Lp model
        ax = plt.subplot(323)
        mesh.plotSlice(m_lp, ax=ax, normal='Z', ind=zpanel,
                       grid=True, clim=(vmin, vmax))
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCy[ypanel], mesh.vectorCCy[ypanel]]), color='w')
        plt.title('Plan lp-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')

        # Vertical section
        ax = plt.subplot(324)
        mesh.plotSlice(m_lp, ax=ax, normal='Y', ind=midx,
                       grid=True, clim=(vmin, vmax))
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCz[zpanel], mesh.vectorCCz[zpanel]]), color='w')
        plt.title('E-W lp-model.')
        plt.gca().set_aspect('equal')
        ax.xaxis.set_visible(False)
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

        # Plot True model
        ax = plt.subplot(325)
        mesh.plotSlice(m_true, ax=ax, normal='Z', ind=zpanel,
                       grid=True, clim=(vmin, vmax))
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCy[ypanel], mesh.vectorCCy[ypanel]]), color='w')
        plt.title('Plan true model.')
        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.gca().set_aspect('equal', adjustable='box')

        # Vertical section
        ax = plt.subplot(326)
        mesh.plotSlice(m_true, ax=ax, normal='Y', ind=midx,
                       grid=True, clim=(vmin, vmax))
        plt.plot(([mesh.vectorCCx[0], mesh.vectorCCx[-1]]),
                 ([mesh.vectorCCz[zpanel], mesh.vectorCCz[zpanel]]), color='w')
        plt.title('E-W true model.')
        plt.gca().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.gca().set_aspect('equal', adjustable='box')

        # Plot convergence curves
        fig, axs = plt.figure(), plt.subplot()
        axs.plot(saveDict.phi_d, 'k', lw=2)
        axs.plot(
            np.r_[IRLS.iterStart, IRLS.iterStart],
            np.r_[0, np.max(saveDict.phi_d)], 'k:'
        )

        twin = axs.twinx()
        twin.plot(saveDict.phi_m, 'k--', lw=2)
        axs.text(
            IRLS.iterStart, np.max(saveDict.phi_d)/2.,
            'IRLS Steps', va='bottom', ha='center',
            rotation='vertical', size=12,
            bbox={'facecolor': 'white'}
        )

        axs.set_ylabel('$\phi_d$', size=16, rotation=0)
        axs.set_xlabel('Iterations', size=14)
        twin.set_ylabel('$\phi_m$', size=16, rotation=0)

if __name__ == '__main__':
    run()
    plt.show()
