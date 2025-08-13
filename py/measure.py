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
from physics import secpera, rho, g, nglen, A3, Phi

def norm_w1r_sc(q, rpow, Lsc):
    '''Scaled W^{1,r} norm as in paper, using a characteristic length
    scale Lsc.  Works for both scalar and vector-valued Functions.  Works
    on both base meshes and extruded meshes.  Compare source of norm() at
    https://www.firedrakeproject.org/_modules/firedrake/norms.html.'''
    assert Lsc > 0
    assert rpow >= 1
    q2 = inner(q, q)
    gq2 = inner(grad(q), grad(q))
    expr = q2**(rpow / 2) + Lsc**rpow * gq2**(rpow / 2)
    return assemble(expr * dx)**(1.0 / rpow)

def geometryreport(basemesh, n, t, s, b, Lsc=1.0, rpow=4.0, eps_H=1.0):
    snorm = norm_w1r_sc(s, rpow, Lsc)
    if basemesh.comm.size == 1:
        H = s.dat.data_ro - b.dat.data_ro # numpy array
        x = basemesh.coordinates.dat.data_ro
        width = 0.0
        if len(x[H > eps_H]) > 0:
            width = max(x[H > eps_H]) - min(x[H > eps_H])
        printpar(f't_{n} = {t / secpera:.3f} a:  |s|_W1{int(rpow)} = {snorm:.3e},  width = {width / 1000.0:.3f} km')
    else:
        printpar(f't_{n} = {t / secpera:.3f} a:  |s|_W1{int(rpow)} = {snorm:.3e}')

def _us_ratio(slist, k, l, rpow, Lsc):
    # compute the ratio  |ur-us|_Lr' / |r-s|_W1r
    assert k != l
    rprime = rpow / (rpow - 1)
    du = slist[k]['us'] - slist[l]['us']  # = u|_r - u|_s
    dunorm = assemble(inner(du, du)**(rprime / 2) * dx)**(1.0 / rprime)
    ds = slist[k]['s'] - slist[l]['s']  # = r - s
    dsnorm = norm_w1r_sc(ds, rpow, Lsc)
    return dunorm / dsnorm

def _Phi_ratio_parts(slist, k, l, rpow, Lsc, b, epsreg=0.0):
    # compute the parts of the ratio
    #   (Phi(r)-Phi(s))[r-s] / |r-s|_W1r^q
    # the parts are:  Phi(r)[r-s], Phi(r)[r-s], |r-s|_W1r
    assert k != l
    r, s = slist[k]['s'], slist[l]['s']
    ur, us = slist[k]['us'], slist[l]['us']
    Phi_r = assemble(Phi(r, ur, r - s, eps=epsreg) * dx)
    Phi_s = assemble(Phi(s, us, r - s, eps=epsreg) * dx)
    dsnorm = norm_w1r_sc(r - s, rpow, Lsc)
    return Phi_r, Phi_s, dsnorm

def sampleratios(dfilename, slist, basemesh, b, N=10, Lsc=100.0e3, aconst=0.0, epsreg=0.0):
    # measure 4-coercivity over W^{1,4}
    rpow = nglen + 1.0
    qpow = rpow  # motivated by coercivity for p-Laplacian implicit step in Bueler 2021
    with open(dfilename, 'w') as dfile:
        dfile.write(f'i1,i2,Phi_r,Phi_s,dsnorm[r={rpow}],Phirat[q={qpow}]\n')
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
        # by-pass extreme cases where norm is zero or ratio is infinity
        if norm_w1r_sc(slist[i1]['s'] - slist[i2]['s'], rpow, Lsc) == 0.0:
            print(RED % '!', end='')  # color provided by firedrake logging.py
            continue
        # measure ratios
        usrat = _us_ratio(slist, i1, i2, rpow, Lsc)
        Phi_r, Phi_s, dsnorm = _Phi_ratio_parts(slist, i1, i2, rpow, Lsc, b, epsreg=epsreg)
        Phirat = (Phi_r - Phi_s) / dsnorm**qpow
        if Phirat == np.inf:
            print(RED % '*', end='')
            continue
        # now we are actually recording results for this sample pair
        with open(dfilename, 'a') as dfile:
            dfile.write(f'{i1},{i2},{Phi_r:.14e},{Phi_s:.14e},{dsnorm:.14e},{Phirat:.14e}\n')
        pairs.append(ipair)
        _n += 1
        _max_us_rat = max(_max_us_rat, usrat)
        Phiratlist.append(Phirat)
        # stdout for this pair
        if Phirat < 0.0:
            print(RED % '.', end='')
            printpar(RED % f'{i1},{i2}')
        elif Phirat == 0.0:
            print(BLUE % '.', end='')
            printpar(BLUE % f'{i1},{i2}')
        else:
            print('.', end='')
    print()
    return _max_us_rat, np.array(Phiratlist)
