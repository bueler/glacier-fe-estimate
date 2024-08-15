'''Measure continuity and coercivity ratios from a list of states.
The list is a list of dictionaries, built by
    slist.append({'t': t,
                  's': s,
                  'us': us})
where s and us are Firedrake Functions on the basemesh.
'''

import numpy as np
from firedrake import *
from stokesextrude import printpar
from figures import badcoercivefigure
from geometry import secpera

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

def _Phi_ratio(slist, k, l, Lsc, b):
    # compute the ratio  (Phi(r)-Phi(s))[r-s] / |r-s|_H1^2
    assert k != l
    r, s = slist[k]['s'], slist[l]['s']
    nr, ns = as_vector([-r.dx(0), Constant(1.0)]), as_vector([-s.dx(0), Constant(1.0)])
    ur, us = slist[k]['us'], slist[l]['us']
    ig = - (dot(ur, nr) - dot(us, ns)) * (r - s)
    dPhi = assemble(ig * dx)
    ds = norm_h1sc(r - s, Lsc)
    return dPhi / ds**2.0

def sampleratios(dirroot, slist, basemesh, b, N=10, Lsc=100.0e3, aconst=0.0):
    printpar(f'computing ratios from {N} pair samples from state list ...')
    assert N >= 2
    from random import randrange
    _max_us_rat = -np.inf
    pairs = []
    Phiratlist = []
    _n = 0
    while _n < N:
        # generate a pair of indices; test if it is new
        i1 = randrange(0, len(slist))
        i2 = randrange(0, len(slist))
        if i1 == i2:
            continue
        ipair = [i1, i2]
        ipair.sort()
        if ipair in pairs:
            continue
        i1, i2 = ipair
        # measure, and bail/report on freak cases
        if norm_h1sc(slist[i1]['s'] - slist[i2]['s'], Lsc) == 0.0:
            print(RED % '!', end='')  # color provided by firedrake logging.py
            continue
        usrat = _us_ratio(slist, i1, i2, Lsc)
        Phirat = _Phi_ratio(slist, i1, i2, Lsc, b)
        if Phirat == np.inf:
            print(RED % '*', end='')
            continue
        # at this point we are actually recording results for this sample pair
        pairs.append(ipair)
        _n += 1
        _max_us_rat = max(_max_us_rat, usrat)
        Phiratlist.append(Phirat)
        # stdout and figure for this pair
        if Phirat < 0.0:
            print(RED % '.', end='')
            printpar(RED % f'{i1},{i2}')
        elif Phirat == 0.0:
            print(BLUE % '.', end='')
            printpar(BLUE % f'{i1},{i2}')
        else:
            print('.', end='')
        if Phirat <= 0.0:
            badcoercivefigure(dirroot,
                              basemesh,
                              b,
                              slist[i1]['s'],
                              slist[i2]['s'],
                              slist[i1]['us'],
                              slist[i2]['us'],
                              slist[i1]['t'],
                              slist[i2]['t'])
    print()
    return _max_us_rat, np.array(Phiratlist)
