import numpy as np
import matplotlib.pyplot as plt
from physics import secpera

def mkdir(dirname):
    import os
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

def livefigure(basemesh, b, s, t, fname=None, writehalfar=False):
    xx = basemesh.coordinates.dat.data
    slabel = f's(t,x) at t = {t / secpera:.3f} a'
    plt.figure(figsize=(6.0, 2.0))
    plt.plot(xx / 1.0e3, s.dat.data, color='C1', label=slabel)
    if writehalfar:
        from geometryinit import generategeometry, t0
        _, shalfar = generategeometry(xx, t=t+t0, bed='flat')  # time t after initial time
        plt.plot(xx / 1.0e3, shalfar, '--', color='C1', label=f's_SIA(t,x)')
    plt.plot(xx / 1.0e3, b.dat.data, color='C3', label='b(x)')
    plt.legend(loc='upper left')
    plt.grid(visible=True)
    plt.gca().set_xticklabels([])
    plt.gca().set_ylabel('elevation (m)')
    plt.xlabel('x (km)')
    if fname == None:
        plt.show()
    else:
        plt.savefig(fname)
    plt.close()

def snapsfigure(basemesh, b, snaps, fname=None):
    xx = basemesh.coordinates.dat.data
    plt.figure(figsize=(6.0, 3.0))
    plt.plot(xx / 1.0e3, b.dat.data, 'k--')
    plt.plot(xx / 1.0e3, snaps[0].dat.data, 'k:')
    for j in range(len(snaps) - 1):
        plt.plot(xx / 1.0e3, snaps[j+1].dat.data, 'k')
    plt.axis('off')
    if fname == None:
        plt.show()
    else:
        plt.savefig(fname)
    plt.close()

def badcoercivefigure(dirroot, basemesh, b, r, s, ur, us, tr, ts):
    from matplotlib.gridspec import GridSpec
    import firedrake as fd
    xx = basemesh.coordinates.dat.data_ro
    xxm = (xx[:-1] + xx[1:]) / 2.0 # midpoints; see DG0 spaces below
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(4, 1, figure=fig)
    ax = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2:4, 0]))
    rlabel = f's(t,x) at t = {tr / secpera:.3f} a'
    ax[0].plot(xx / 1.0e3, r.dat.data, color='C1', label=rlabel)
    slabel = f's(t,x) at t = {ts / secpera:.3f} a'
    ax[0].plot(xx / 1.0e3, s.dat.data, ':', color='C1', label=slabel)
    ax[0].plot(xx / 1.0e3, b.dat.data, color='C3', label='b(x)')
    ax[0].legend(loc='upper left')
    ax[0].set_xticklabels([])
    ax[0].set_xlim([1.05 * min(xx) / 1.0e3, 1.05 * max(xx) / 1.0e3])
    ax[0].grid(visible=True)
    ax[0].set_ylabel('elevation (m)')
    DG0bm = fd.FunctionSpace(basemesh, 'DG', 0)
    nr, ns = fd.as_vector([-r.dx(0), fd.Constant(1.0)]), fd.as_vector([-s.dx(0), fd.Constant(1.0)])
    Phir = fd.Function(DG0bm).project(- fd.dot(ur, nr))
    Phis = fd.Function(DG0bm).project(- fd.dot(us, ns))
    Phirlabel = f'Phi(t,x) at t = {tr / secpera:.3f} a'
    ax[1].plot(xxm / 1.0e3, Phir.dat.data * secpera, '+', ms=6.0, color='C2', label=Phirlabel)
    Phislabel = f'Phi(t,x) at t = {ts / secpera:.3f} a'
    ax[1].plot(xxm / 1.0e3, Phis.dat.data * secpera, 'x', ms=6.0, color='C2', label=Phislabel)
    ax[1].legend(loc='upper right')
    ax[1].set_ylabel(r'$\Phi$ (m a-1)')
    ax[1].set_xlim([1.05 * min(xx) / 1.0e3, 1.05 * max(xx) / 1.0e3])
    ax[1].grid(visible=True)
    igDG0 = fd.Function(DG0bm).project(- (fd.dot(ur, nr) - fd.dot(us, ns)) * (r - s))
    igm = igDG0.dat.data_ro
    xxmpos, igpos = xxm[igm > 0.0] / 1.0e3, igm[igm > 0.0]
    xxmneg, igneg = xxm[igm < 0.0] / 1.0e3, igm[igm < 0.0]
    ax[2].semilogy(xxmpos, igpos, '.', color='k')
    ax[2].semilogy(xxmneg, - igneg, '.', color='r')
    ax[2].set_ylabel('integrand (red negative)')
    ax[2].set_xlim([1.05 * min(xxm) / 1.0e3, 1.05 * max(xxm) / 1.0e3])
    ax[2].grid(visible=True)
    if len(igpos) > 0:
        if len(igneg) > 0:
            ymax = max((max(igpos), max(-igneg)))
        else:
            ymax = max(igpos)
    else:
        if len(igneg) > 0:
            ymax = max(-igneg)
        else:
            ymax = 1.0
    ax[2].set_ylim([ymax * 1.0e-5, 1.5 * ymax])
    plt.xlabel('x (km)')
    fname = dirroot + f'badcoercive-{tr / secpera:.3f}-{ts / secpera:.3f}.png'
    plt.savefig(fname)
    plt.close()

def histogramPhirat(dirname, ratlist):
    binsp = 24  # bins on positive side
    rlp = ratlist[ratlist > 0.0]
    assert len(rlp) > 1
    rlzero = ratlist[ratlist == 0.0]
    if len(rlzero) > 0:
        print(f"WARNING: {len(rlzero)} zero ratios detected ... excluded")
    rln = ratlist[ratlist < 0.0]  # may be empty list
    h, edges = np.histogram(np.log(rlp)/np.log(10.0), bins=binsp)
    width = edges[1] - edges[0]  # for equal-width bins
    fig = plt.figure(figsize=(6.0, 4.0))
    ax = plt.gca()
    ax.stairs(h, edges, color='k', label=f"positive ({len(rlp)})")
    if len(rln) == 1:
        print(f"WARNING: undefined behavior since rln has length 1")
    if len(rln) > 0:
        lrln = np.log(-rln)/np.log(10.0)
        # make bins (nearly) same width as for positive
        binsn = int(np.ceil((lrln.max() - lrln.min()) / width))
        hn, edgesn = np.histogram(lrln, bins=binsn)
        ax.stairs(hn, edgesn, color='k', fill=True, label=f"negative ({len(rln)})")
    plt.xlabel(r'log10 $\Phi$ ratios')
    plt.legend()
    fname = dirname + 'Phiratios.png'
    plt.savefig(fname)
    plt.close()
