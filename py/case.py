# Run one case as specified by runtime options.  Constructs 2D glaciers
# with initial Halfar profiles over different beds.  Does time-steps using
# the free-surface stabilization algorithm (FSSA) from Lofgren et al 2022.
# (The steps are explicit, but may be regarded as approximate solutions of
# the implicit backward-Euler method of the paper.)  Each step solves the
# Stokes problem, with FSSA modification, then computes the surface
# motion map Phi(s) = - u|_s . n_s, and then does the truncated
# explicit step.  Note this runs only in serial.
#
# After activating the Firedrake venv, run as
#   $ python3 study.py MX NSTEPS DT BED
# For example:
#   $ python3 study.py 201 20 1.0 flat
# To write an optional t-dependent image files indo directory do:
#   $ python3 study.py 201 20 1.0 flat result/
# To write an optional t-dependent .pvd file with Stokes results and
# diagnostics, append the filename:
#   $ python3 study.py 201 20 1.0 flat result/ result.pvd

import sys
import numpy as np
from firedrake import *
from firedrake.output import VTKFile
from stokesextruded import StokesExtruded, SolverParams, trace_vector_to_p2, printpar
from geometry import secpera, bedtypes, g, rho, nglen, A3, B3, t0, halfargeometry
from figures import mkoutdir, livefigure, badcoercivefigure
from measure import geometryreport, sampleratios

mx = int(sys.argv[1])
Nsteps = int(sys.argv[2])
dt = float(sys.argv[3]) * secpera
bed = sys.argv[4]
writepng = (len(sys.argv) > 5)
writepvd = (len(sys.argv) > 6)

mz = 15                 # number of cells in each column
Nsamples = 200          # number of samples when evaluating minimal ratios
qcoercive = 2.0         # try this?  justified by scaling argument?

L = 100.0e3             # domain is [-L,L]
Hmin = 20.0             # kludge: insert fake ice for Stokes solve
fssa = True             # use Lofgren et al (2022) FSSA technique
theta_fssa = 1.0        #   with this theta value

# Stokes regularization
eps = 0.01
Dtyp = 1.0 / secpera    # 1 a-1
qq = 1.0 / nglen - 1.0

# set up basemesh once
basemesh = IntervalMesh(mx, -L, L)
P1bm = FunctionSpace(basemesh, 'P', 1)
xbm = basemesh.coordinates.dat.data_ro
assert bed in bedtypes
#print(f"Halfar t0 = {t0 / secpera:.3f} a")
b_np, s_np = halfargeometry(xbm, t=t0, bed=bed)  # get numpy arrays
b = Function(P1bm, name='bed elevation (m)')
b.dat.data[:] = b_np
s = Function(P1bm, name='surface elevation (m)')  # this is the state variable
s.dat.data[:] = s_np

# set up extruded mesh, but leave z coordinate unassigned
mesh = ExtrudedMesh(basemesh, layers=mz, layer_height=1.0/mz)
P1 = FunctionSpace(mesh, 'P', 1)
P1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
Vcoord = mesh.coordinates.function_space()
x, _ = SpatialCoordinate(mesh)
z_flat = mesh.coordinates.dat.data_ro[:,1].copy()  # z coord before assignment

# set up the Stokes solver on the extruded mesh
se = StokesExtruded(mesh)
se.mixed_TaylorHood()
se.body_force(Constant((0.0, - rho * g)))
se.dirichlet((1,2), Constant((0.0, 0.0)))      # problematic if ice advances to margin
se.dirichlet(('bottom',), Constant((0.0, 0.0)))
params = SolverParams['newton']
params.update(SolverParams['mumps'])
#params.update({'snes_monitor': None})
params.update({'snes_converged_reason': None})
params.update({'snes_atol': 1.0e-1})
params.update({'snes_linesearch_type': 'bt'})  # helps with non-flat beds, it seems

def _D(w):
    return 0.5 * (grad(w) + grad(w).T)

# the weak form for the Stokes problem
def _form_stokes(mesh, se, sR):
    u, p = split(se.up)
    v, q = TestFunctions(se.Z)
    Du2 = 0.5 * inner(_D(u), _D(u)) + (eps * Dtyp)**2.0
    F = inner(B3 * Du2**(qq / 2.0) * _D(u), _D(v)) * dx(degree=4)
    F -= (p * div(v) + div(u) * q) * dx
    source = inner(se.f_body, v) * dx
    if fssa:
        # see section 4.2 in Lofgren et al
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
sRfake = Function(P1R)
sR = Function(P1R)
bR = Function(P1R)
bR.dat.data[:] = b.dat.data_ro
t = 0.0
printpar(f'doing N = {Nsteps} steps of dt = {dt/secpera:.3f} a ...')
printpar(f'  solving 2D Stokes + SKE on {mx} x {mz} extruded mesh over {bed} bed')
printpar(f'  dimensions: n_u = {se.V.dim()}, n_p = {se.W.dim()}')
if writepng:
    printpar(f'  creating directory {sys.argv[5]} for image files ...')
    mkoutdir(sys.argv[5])
if writepvd:
    printpar(f'  opening {sys.argv[6]} ...')
    outfile = VTKFile(sys.argv[6])
_slist = []
for n in range(Nsteps):
    # start with reporting
    if n == 0:
        geometryreport(basemesh, 0, t, s, b, L)
        if writepng:
            livefigure(basemesh, b, s, None, t, fname=f'result/t{t/secpera:010.3f}.png')

    # set geometry (z coordinate) of extruded mesh
    sRfake.dat.data[:] = np.maximum(s.dat.data_ro, Hmin + b.dat.data_ro)  # *here* is the fake ice
    ztmp = Function(P1)
    ztmp.dat.data[:] = z_flat
    newcoord.interpolate(as_vector([x, bR + (sRfake - bR) * ztmp]))
    mesh.coordinates.assign(newcoord)

    # solve Stokes on extruded mesh
    # this uses fake ice, and se.up as initial iterate
    u, p = se.solve(par=params, F=_form_stokes(mesh, se, sRfake))
    #printpar(f'  solution norms: |u|_L2 = {norm(u):8.3e},  |p|_L2 = {norm(p):8.3e}')

    # optionally write t-dependent .pvd
    if writepvd:
        u.rename('velocity (m s-1)')
        p.rename('pressure (Pa)')
        nu, nueps = _effective_viscosity(mesh, u)
        phydro = _p_hydrostatic(mesh, sRfake)
        pdiff = Function(P1).interpolate(p - phydro)
        pdiff.rename('pdiff = phydro - p (Pa)')
        outfile.write(u, p, nu, nueps, pdiff, time=t)

    # compute surface motion map  Phi(s) = - u|_s . n_s   (m s-1)
    # this uses true surface elevation s
    ns = as_vector([-s.dx(0), Constant(1.0)])
    ubm = trace_vector_to_p2(basemesh, mesh, u)  # surface velocity (m s-1)
    Phi = Function(P1bm).project(- dot(ubm, ns))  # interpolate() would be bad here (P2 nodes)

    # save surface elevation and velocity into list for ratio evals
    _slist.append({'t': t,
                   's': s.copy(deepcopy=True),
                   'us': ubm.copy(deepcopy=True),
                   'Phi': Phi.copy(deepcopy=True)})

    # explicit step SKE using truncation
    # FIXME include SMB choices
    # FIXME replace with semi-implicit VI solve (explicit as option)
    snew = s - dt * Phi
    s.interpolate(conditional(snew < b, b, snew))
    t += dt

    # end of step reporting
    geometryreport(basemesh, n + 1, t, s, b, L)
    if writepng:
        livefigure(basemesh, b, s, Phi, t, fname=f'result/t{t/secpera:010.3f}.png',
                   writehalfar=(bed == 'flat' and n + 1 == Nsteps))

if writepng:
    printpar(f'finished writing to {sys.argv[5]}')
if writepvd:
    printpar(f'finished writing to {sys.argv[6]}')

sampleratios(_slist, basemesh, b, N=Nsamples, q=qcoercive, Lsc=L)