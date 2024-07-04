import matplotlib.pyplot as plt

_secpera = 31556926.0    # seconds per year

def mkoutdir(dirname):
    import os
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

def livefigure(basemesh, sbm, Phibm, t=None, fname=None):
        xx = basemesh.coordinates.dat.data
        fig, (ax1, ax2) = plt.subplots(2, 1)
        if t == None:
            slabel = 's'
        else:
            slabel = f's at t = {t / _secpera:.3f} a'
        ax1.plot(xx / 1.0e3, sbm.dat.data, color='C1', label=slabel)
        ax1.legend(loc='upper left')
        ax1.set_xticklabels([])
        ax1.grid(visible=True)
        ax1.set_ylabel('elevation (m)')
        if Phibm is not None:
            ax2.plot(xx / 1.0e3, Phibm.dat.data * _secpera, color='C2', label=r'$\Phi(s)$')
            ax2.legend(loc='upper right')
            ax2.set_ylabel(r'$\Phi$ (m a-1)')
            ax2.grid(visible=True)
        plt.xlabel('x (km)')
        if fname == None:
            plt.show()
        else:
            plt.savefig(fname)
        plt.close()
