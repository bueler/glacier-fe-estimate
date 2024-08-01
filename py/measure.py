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
from geometry import secpera

_Hth = 100.0   # thickness threshold when computing coercivity (Phi) ratios

def norm_h1sc(v, Lsc):
    '''Scaled H^1 = W^{1,2} norm as in paper, using a characteristic length
    scale Lsc.  Works for both scalar and vector-valued Functions.  Works
    on both base meshes and extruded meshes.  Compare source of norm() at
    https://www.firedrakeproject.org/_modules/firedrake/norms.html.'''
    assert Lsc > 0
    expr = inner(v, v) + Lsc**2 * inner(grad(v), grad(v))
    return assemble(expr * dx)**0.5

def geometryreport(basemesh, n, t, s, b, Lsc):
    snorm = norm_h1sc(s, Lsc=Lsc)
    if basemesh.comm.size == 1:
        H = s.dat.data_ro - b.dat.data_ro # numpy array
        x = basemesh.coordinates.dat.data_ro
        width = 0.0
        if len(x[H > 1.0]) > 0:
            width = max(x[H > 1.0]) - min(x[H > 1.0])
        printpar(f't_{n} = {t / secpera:.3f} a:  |s|_H1 = {snorm:.3e},  width = {width / 1000.0:.3f} km')
    else:
        printpar(f't_{n} = {t / secpera:.3f} a:  |s|_H1 = {snorm:.3e}')

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
    r, s = slist[k]['s'], slist[l]['s']
    nr, ns = as_vector([-r.dx(0), Constant(1.0)]), as_vector([-s.dx(0), Constant(1.0)])
    ur, us = slist[k]['us'], slist[l]['us']
    ig = - (dot(ur, nr) - dot(us, ns)) * (r - s)
    igcrop = conditional(r - b > _Hth, conditional(s - b > _Hth, ig, 0.0), 0.0)
    # because of threshhold, igcrop can end up identically zero if rr==b or ss==b
    if norm(igcrop) == 0.0:
        return np.inf  # won't affect min
    dPhi = assemble(igcrop * dx)
    ds = norm_h1sc(r - s, Lsc)
    return dPhi / ds**q

def sampleratios(slist, basemesh, b, N=10, q=2.0, Lsc=100.0e3, aconst=0.0):
    printpar(f'computing ratios from {N} pair samples from state list ...')
    assert N >= 2
    from random import randrange
    _max_us_rat = -np.inf
    _min_Phi_rat = np.inf
    _n = 0
    while _n < N:
        i1 = randrange(0, len(slist))
        i2 = randrange(0, len(slist))
        imatch = (i1 == i2)
        if imatch:
            print('#', end='')
        else:
            smatch = (norm_h1sc(slist[i1]['s'] - slist[i2]['s'], Lsc) == 0.0)
            if smatch:
                print(RED % '!', end='')
        if (not imatch) and (not smatch):
            usrat = _us_ratio(slist, i1, i2, Lsc)
            Phirat = _Phi_ratio(slist, i1, i2, Lsc, b, q)
            if Phirat <= 0.0:
                print(RED % '.', end='')
            else:
                print('.', end='')
            if Phirat <= 0.0:
                printpar(RED % f'{i1},{i2}')  # color provided by firedrake logging.py
                badcoercivefigure(basemesh,
                                  b,
                                  slist[i1]['s'],
                                  slist[i2]['s'],
                                  slist[i1]['us'],
                                  slist[i2]['us'],
                                  slist[i1]['t'],
                                  slist[i2]['t'],
                                  _Hth)
            _max_us_rat = max(_max_us_rat, usrat)
            _min_Phi_rat = min(_min_Phi_rat, Phirat)
            _n += 1
    print()
    printpar(f'  max continuity ratio:  {_max_us_rat:.3e}')
    printpar(f'  min coercivity ratio:  {_min_Phi_rat:.3e}')
    return _max_us_rat, _min_Phi_rat
