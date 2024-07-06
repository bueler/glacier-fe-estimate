# A study using the geometry error bounds which are described in the
# paper.  Constructs 2D glaciers with a Halfar or Halfar-like profile
# over different beds.  Does time-steps using the free-surface
# stabilization algorithm (FSSA) from Lofgren et al 2022.
# The steps are explicit but regarded as approximate solutions of the
# implicit backward-Euler method of the paper.  Each step solves the
# Stokes problem, with the FSSA modification, computes the surface
# motion map Phi(s) = - u|_s . n_s, does the truncated explicit step,
# and evaluates the diagnostic quantities defined in Theorem 6.1.

# TODO:
#   * bed cases
#   * in last frame show Halfar by comparison (if b=0)
#   * evaluate bounds
#   * evaluate ratios

import numpy as np
from firedrake import *
from firedrake.output import VTKFile
from stokesextruded import *
from geometry import geometry
from livefigure import *

secpera = 31556926.0    # seconds per year
bedtypes = ['flat', 'smooth', 'rough']

mx = 201  # odd is slightly better(!) for symmetrical Halfar-on-flat case
mz = 15
Nsteps = 100
dt = 5.0 * secpera
bed = 'flat'

L = 100.0e3             # domain is [-L,L]
Hmin = 20.0             # kludge: fake ice for Stokes solve, where ice-free
fssa = True             # use Lofgren et al (2022) FSSA technique,
theta_fssa = 1.0        #   with this theta value
writediag = True        # write extra diagnostics into .pvd

# physics parameters
g, rho = 9.81, 910.0    # m s-2, kg m-3
nglen = 3.0
A3 = 3.1689e-24         # Pa-3 s-1; EISMINT I value of ice softness
B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness
eps = 0.01
Dtyp = 1.0 / secpera    # 1 a-1
qq = 1.0 / nglen - 1.0

# set up basemesh once
basemesh = IntervalMesh(mx, -L, L)
P1bm = FunctionSpace(basemesh, 'P', 1)
xb = basemesh.coordinates.dat.data_ro
b, sb_initial = geometry(xb, nglen=nglen, bed=bed)
bbm = Function(P1bm, name='bed elevation (m)')
bbm.dat.data[:] = b
sbm = Function(P1bm, name='surface elevation (m)')  # this is the state variable
sbm.dat.data[:] = sb_initial

# set up extruded mesh, but leave z coordinate unfilled
mesh = ExtrudedMesh(basemesh, layers=mz, layer_height=1.0/mz)
P1 = FunctionSpace(mesh, 'P', 1)
P1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
Vcoord = mesh.coordinates.function_space()
x, _ = SpatialCoordinate(mesh)
z_flat = mesh.coordinates.dat.data_ro[:,1].copy()

# set up the Stokes solver on the extruded mesh
se = StokesExtruded(mesh)
se.mixed_TaylorHood()
se.body_force(Constant((0.0, - rho * g)))
se.dirichlet((1,2), Constant((0.0, 0.0)))  # wrong if ice advances to margin
se.dirichlet(('bottom',), Constant((0.0, 0.0)))
params = SolverParams['newton']
params.update(SolverParams['mumps'])
#params.update({'snes_monitor': None})
params.update({'snes_converged_reason': None})
params.update({'snes_atol': 1.0e-1})
params.update({'snes_linesearch_type': 'bt'})  # helps with non-flat beds, it seems

def geometryreport(n, t, sbm):
    snorm = norm(sbm, norm_type='H1')
    if basemesh.comm.size == 1:
        H = sbm.dat.data_ro - b
        wkm = (max(xb[H > 1.0]) - min(xb[H > 1.0])) / 1000.0
        printpar(f't_{n} = {t / secpera:7.3f} a:  |s|_H1 = {snorm:.3e},  width = {wkm:.3f} km')
    else:
        printpar(f't_{n} = {t / secpera:7.3f} a:  |s|_H1 = {snorm:.3e}')

def _D(w):
    return 0.5 * (grad(w) + grad(w).T)

# the weak form for the Stokes problem
def _form_stokes(mesh, se, sbm):
    u, p = split(se.up)
    v, q = TestFunctions(se.Z)
    Du2 = 0.5 * inner(_D(u), _D(u)) + (eps * Dtyp)**2.0
    F = inner(B3 * Du2**(qq / 2.0) * _D(u), _D(v)) * dx(degree=4)
    F -= (p * div(v) + div(u) * q) * dx
    source = inner(se.f_body, v) * dx
    if fssa:
        # see section 4.2 in Lofgren et al
        sR = extend_p1_from_basemesh(mesh, sbm)
        nsR = as_vector([-sR.dx(0), Constant(1.0)])
        nunit = nsR / sqrt(sR.dx(0)**2 + 1.0)
        F -= theta_fssa * dt * inner(u, nunit) * inner(se.f_body, v) * ds_t
        # FIXME SMB a into source
    F -= source
    return F

# generate effective viscosity nu from the velocity solution
def _effective_viscosity(mesh, u):
    Du2 = 0.5 * inner(_D(u), _D(u))
    nu = Function(P1).interpolate(0.5 * B3 * Du2**(qq/2.0))
    nu.rename('nu (unregularized; Pa s)')
    nueps = Function(P1).interpolate(0.5 * B3 * (Du2  + (eps * Dtyp)**2)**(qq/2.0))
    nueps.rename(f'nu (eps={eps:.3f}; Pa s)')
    return nu, nueps

# generate effective viscosity nu from the velocity solution
def _p_hydrostatic(mesh, sR):
    _, z = SpatialCoordinate(mesh)
    phydro = Function(P1).interpolate(rho * g * (sR - z))
    phydro.rename('p_hydro (Pa)')
    return phydro

# time-stepping loop
newcoord = Function(Vcoord)
sR = Function(P1R)
bR = Function(P1R)
bR.dat.data[:] = b
t = 0.0
mkoutdir('result/')
printpar(f'doing N = {Nsteps} steps of dt = {dt/secpera:.3f} a ...')
printpar(f'  solving 2D Stokes + SKE on {mx} x {mz} extruded mesh')
printpar(f'  dimensions: n_u = {se.V.dim()}, n_p = {se.W.dim()}')
outfile = VTKFile("result.pvd")
for n in range(Nsteps):
    # start with reporting
    if n == 0:
        geometryreport(n, t, sbm)
        if basemesh.comm.size == 1:
            livefigure(basemesh, b, sbm, None, t=t, fname=f'result/t{t/secpera:010.3f}.png')

    # set geometry (z coordinate) of extruded mesh
    sR.dat.data[:] = np.maximum(sbm.dat.data_ro, Hmin + b)  # only *here* is the fake ice
    ztmp = Function(P1)
    ztmp.dat.data[:] = z_flat
    newcoord.interpolate(as_vector([x, bR + (sR - bR) * ztmp]))
    mesh.coordinates.assign(newcoord)

    # solve Stokes; this uses se.up as initial iterate
    u, p = se.solve(par=params, F=_form_stokes(mesh, se, sbm))
    printpar(f'  solution norms: |u|_L2 = {norm(u):8.3e},  |p|_L2 = {norm(p):8.3e}')
    u.rename('velocity (m s-1)')
    p.rename('pressure (Pa)')
    if writediag:
        nu, nueps = _effective_viscosity(mesh, u)
        phydro = _p_hydrostatic(mesh, sR)
        pdiff = Function(P1).interpolate(p - phydro)
        pdiff.rename('pdiff = phydro - p (Pa)')
        outfile.write(u, p, nu, nueps, pdiff, time=t)
    else:
        outfile.write(u, p, time=t)

    # compute surface motion map  Phi(s) = - u|_s . n_s   (m s-1)
    ns = as_vector([-sbm.dx(0), Constant(1.0)])
    ubm = trace_vector_to_p2(basemesh, mesh, u)  # surface velocity (m s-1)
    Phibm = Function(P1bm).project(- dot(ubm, ns))  # interpolate bad here (because P2 nodes)

    # explicit step SKE using truncation
    # FIXME make semi-implicit an option
    snew = sbm - dt * Phibm
    sbm.interpolate(conditional(snew < bbm, bbm, snew))
    t += dt

    # end of step reporting
    geometryreport(n+1, t, sbm)
    if basemesh.comm.size == 1:
        livefigure(basemesh, b, sbm, Phibm, t=t, fname=f'result/t{t/secpera:010.3f}.png')

printpar('finished writing to result.pvd')
