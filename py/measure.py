'''Measure continuity and coercivity ratios from a list of states.
The list is a list of dictionaries, built by
    slist.append({'t': t,
                  's': s,
                  'us': us,
                  'Phi': Phi})
where s, us, Phi are Firedrake Functions on the basemesh.
'''

import numpy as np
from firedrake import *
from stokesextruded import printpar
from figures import badcoercivefigure

def norm_h1sc(v, Lsc):
    '''Scaled H^1 = W^{1,2} norm as in paper, using a characteristic length
    scale Lsc.  Works for both scalar and vector-valued Functions.  Works
    on both base meshes and extruded meshes.  Compare source of norm() at
    https://www.firedrakeproject.org/_modules/firedrake/norms.html.'''
    assert Lsc > 0
    expr = inner(v, v) + Lsc**2 * inner(grad(v), grad(v))
    return assemble(expr * dx)**0.5

def _us_ratio(slist, k, l, Lsc):
    # compute the ratio  |ur-us|_L2 / |r-s|_H1
    assert k != l
    dus = errornorm(slist[k]['us'], slist[l]['us'], norm_type='L2')
    ds = norm_h1sc(slist[k]['s'] - slist[l]['s'], Lsc)
    return dus / ds

def _Phi_ratio(slist, k, l, Lsc, b, q):
    # compute the ratio  (Phi(r)-Phi(s))[r-s] / |r-s|_H1^q,  but chop
    # integrand ig where either thickness (i.e. r-b or s-b) is below threshold
    assert k != l
    Hth = 100.0
    rr, ss = slist[k]['s'], slist[l]['s']
    ig = (slist[k]['Phi'] - slist[l]['Phi']) * (rr - ss)
    igcrop = conditional(rr - b > Hth, conditional(ss - b > Hth, ig, 0.0), 0.0)
    dPhi = assemble(igcrop * dx)
    ds = norm_h1sc(rr - ss, Lsc)
    return dPhi / ds**q

def sampleratios(slist, basemesh, b, N=10, q=2.0, Lsc=100.0e3):
    printpar(f'computing ratios from {N} pair samples from state list ...')
    assert N >= 2
    from random import randrange
    _max_us_rat = -np.Inf
    _min_Phi_rat = np.Inf
    _n = 0
    while _n < N:
        i1 = randrange(0, len(slist))
        i2 = randrange(0, len(slist))
        if i1 != i2:
            usrat = _us_ratio(slist, i1, i2, Lsc)
            Phirat = _Phi_ratio(slist, i1, i2, Lsc, b, q)
            if Phirat < 0.0:
                printpar(RED % f'{i1},{i2}')  # color provided by firedrake logging.py
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
    printpar(f'  max continuity ratio:  {_max_us_rat:.3e}')
    printpar(f'  min coercivity ratio:  {_min_Phi_rat:.3e}')
