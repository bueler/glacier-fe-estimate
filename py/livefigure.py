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
    fig, (ax1, ax2) = plt.subplots(2, 1)
    slabel = f's(t,x) at t = {t / _secpera:.3f} a'
    ax1.plot(xx / 1.0e3, s.dat.data, color='C1', label=slabel)
    if writehalfar:
        from icegeometry import geometry, t0
        _, shalfar = geometry(xx, t=t+t0, bed='flat')  # time t after initial time
        ax1.plot(xx / 1.0e3, shalfar, '--', color='C1', label=f's_SIA(t,x)')
    ax1.plot(xx / 1.0e3, b.dat.data, color='C3', label='b(x)')
    ax1.legend(loc='upper left')
    ax1.set_xticklabels([])
    ax1.grid(visible=True)
    ax1.set_ylabel('elevation (m)')
    if Phi is not None:
        ax2.plot(xx / 1.0e3, Phi.dat.data * _secpera, color='C2', label=r'$\Phi(s)$')
        ax2.legend(loc='upper right')
        ax2.set_ylabel(r'$\Phi$ (m a-1)')
        ax2.grid(visible=True)
    plt.xlabel('x (km)')
    if fname == None:
        plt.show()
    else:
        plt.savefig(fname)
    plt.close()

def badcoercivefigure(basemesh, b, r, s, Phir, Phis, tr, ts):
    import matplotlib.pyplot as plt
    xx = basemesh.coordinates.dat.data
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    rlabel = f's(t,x) at t = {tr / _secpera:.3f} a'
    ax1.plot(xx / 1.0e3, r.dat.data, color='C1', label=rlabel)
    slabel = f's(t,x) at t = {ts / _secpera:.3f} a'
    ax1.plot(xx / 1.0e3, s.dat.data, color='C1', label=slabel)
    ax1.plot(xx / 1.0e3, b.dat.data, color='C3', label='b(x)')
    ax1.legend(loc='upper left')
    ax1.set_xticklabels([])
    ax1.grid(visible=True)
    ax1.set_ylabel('elevation (m)')
    ax2.plot(xx / 1.0e3, r.dat.data - s.dat.data, color='C1')
    ax2.set_ylabel('s diff')
    ax2.grid(visible=True)
    Phirlabel = f'Phi(t,x) at t = {tr / _secpera:.3f} a'
    ax3.plot(xx / 1.0e3, Phir.dat.data * _secpera, color='C2', label=Phirlabel)
    Phislabel = f'Phi(t,x) at t = {ts / _secpera:.3f} a'
    ax3.plot(xx / 1.0e3, Phis.dat.data * _secpera, color='C2', label=Phislabel)
    ax3.legend(loc='upper right')
    ax3.set_ylabel(r'$\Phi$ (m a-1)')
    ax3.grid(visible=True)
    ax4.plot(xx / 1.0e3, Phir.dat.data - Phis.dat.data, color='C2')
    ax4.set_ylabel('Phi diff')
    ax4.grid(visible=True)
    plt.xlabel('x (km)')
    fname = f'badcoercive-{tr / _secpera:.3f}-{ts / _secpera:.3f}.png'
    plt.savefig(fname)
    plt.close()
