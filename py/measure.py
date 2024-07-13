'''Measure continuity and coercivity ratios from a list of states.
The list is a list of dictionaries, built by
    slist.append({'t': t,
                  's': s,
                  'us': us,
                  'Phi': Phi})
where s, us, Phi are Firedrake Functions on the basemesh.
'''

# FIXME use H1 norms in the right places, and scale the H1 norm as in the paper

import numpy as np
from firedrake import *
from stokesextruded import printpar

def _us_ratio(slist, k, l):
    # compute the ratio  |ur-us|_L2 / |r-s|_L2
    dus = errornorm(slist[k]['us'], slist[l]['us'], norm_type='L2')
    ds = errornorm(slist[k]['s'], slist[l]['s'], norm_type='L2')
    return dus / ds

def _Phi_ratio(slist, k, l, b, q):
    # compute the ratio  (Phi(r)-Phi(s))[r-s] / |r-s|_H1,  but chop
    # where either thickness r-b, s-b is below threshold
    Hth = 100.0
    rr, ss = slist[k]['s'], slist[l]['s']
    ig = (slist[k]['Phi'] - slist[l]['Phi']) * (rr - ss)
    igcrop = conditional(rr - b > Hth, conditional(ss - b > Hth, ig, 0.0), 0.0)
    dPhi = assemble(igcrop * dx)
    ds = errornorm(rr, ss, norm_type='H1')
    return dPhi / ds**q

def sampleratios(slist, basemesh, b, Nsamples=10, qcoercive=1.5):
    printpar(f'computing ratios from {Nsamples} pair samples from state list ...')
    from random import randrange
    _max_us_rat = -np.Inf
    _min_Phi_rat = np.Inf
    _n = 0
    while _n < Nsamples:
        i1 = randrange(0, len(slist))
        i2 = randrange(0, len(slist))
        if i1 != i2:
            usrat = _us_ratio(slist, i1, i2)
            Phirat = _Phi_ratio(slist, i1, i2, b, qcoercive)
            if Phirat < 0.0:
                printpar(RED % f'{i1},{i2}')
                badcoercivefigure(basemesh,
                                  b,
                                  slist[i1]['s'],
                                  slist[i2]['s'],
                                  slist[i1]['Phi'],
                                  slist[i2]['Phi'],
                                  slist[i1]['t'],
                                  slist[i2]['t'])
            _max_us_rat = max(_max_us_rat, usrat)
            _min_Phi_rat = min(_min_Phi_rat, Phirat)
            _n += 1
    printpar(f'  max continuity ratio |ur-us|_L2/|r-s|_L2:             {_max_us_rat:.3e}')
    printpar(f'  min coercivity ratio (Phi(r)-Phi(s))[r-s]/|r-s|_H1^q: {_min_Phi_rat:.3e}')
