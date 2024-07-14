_secpera = 31556926.0    # seconds per year

def mkoutdir(dirname):
    import os
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

def livefigure(basemesh, b, s, Phi, t, fname=None, writehalfar=False):
    import matplotlib.pyplot as plt
    xx = basemesh.coordinates.dat.data
    fig, ax = plt.subplots(2, 1)
    slabel = f's(t,x) at t = {t / _secpera:.3f} a'
    ax[0].plot(xx / 1.0e3, s.dat.data, color='C1', label=slabel)
    if writehalfar:
        from icegeometry import geometry, t0
        _, shalfar = geometry(xx, t=t+t0, bed='flat')  # time t after initial time
        ax[0].plot(xx / 1.0e3, shalfar, '--', color='C1', label=f's_SIA(t,x)')
    ax[0].plot(xx / 1.0e3, b.dat.data, color='C3', label='b(x)')
    ax[0].legend(loc='upper left')
    ax[0].set_xticklabels([])
    ax[0].grid(visible=True)
    ax[0].set_ylabel('elevation (m)')
    if Phi is not None:
        ax[1].plot(xx / 1.0e3, Phi.dat.data * _secpera, color='C2', label=r'$\Phi(s)$')
        ax[1].legend(loc='upper right')
        ax[1].set_ylabel(r'$\Phi$ (m a-1)')
        ax[1].grid(visible=True)
    plt.xlabel('x (km)')
    if fname == None:
        plt.show()
    else:
        plt.savefig(fname)
    plt.close()

def badcoercivefigure(basemesh, b, r, s, Phir, Phis, tr, ts):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    xx = basemesh.coordinates.dat.data
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(4, 1, figure=fig)
    ax = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[2:4, 0]))
    rlabel = f's(t,x) at t = {tr / _secpera:.3f} a'
    ax[0].plot(xx / 1.0e3, r.dat.data, color='C1', label=rlabel)
    slabel = f's(t,x) at t = {ts / _secpera:.3f} a'
    ax[0].plot(xx / 1.0e3, s.dat.data, ':', color='C1', label=slabel)
    ax[0].plot(xx / 1.0e3, b.dat.data, color='C3', label='b(x)')
    ax[0].legend(loc='upper left')
    ax[0].set_xticklabels([])
    ax[0].grid(visible=True)
    ax[0].set_ylabel('elevation (m)')
    Phirlabel = f'Phi(t,x) at t = {tr / _secpera:.3f} a'
    ax[1].plot(xx / 1.0e3, Phir.dat.data * _secpera, color='C2', label=Phirlabel)
    Phislabel = f'Phi(t,x) at t = {ts / _secpera:.3f} a'
    ax[1].plot(xx / 1.0e3, Phis.dat.data * _secpera, ':', color='C2', label=Phislabel)
    ax[1].legend(loc='upper right')
    ax[1].set_ylabel(r'$\Phi$ (m a-1)')
    ax[1].grid(visible=True)
    ig = (Phir.dat.data - Phis.dat.data) * (r.dat.data - s.dat.data)  # integrand
    xxpos, igpos = xx[ig > 0.0] / 1.0e3, ig[ig > 0.0]
    xxneg, igneg = xx[ig < 0.0] / 1.0e3, ig[ig < 0.0]
    ax[2].semilogy(xxpos, igpos, '.', color='k')
    ax[2].semilogy(xxneg, - igneg, '.', color='r')
    ax[2].set_ylabel('integrand (red negative)')
    ax[2].grid(visible=True)
    ymax = max((max(igpos),max(-igneg)))
    ax[2].set_ylim([ymax * 1.0e-5, 1.5 * ymax])
    plt.xlabel('x (km)')
    fname = f'badcoercive-{tr / _secpera:.3f}-{ts / _secpera:.3f}.png'
    plt.savefig(fname)
    plt.close()