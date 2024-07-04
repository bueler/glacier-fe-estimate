# A study of the geometry error bounds described in the paper.
# Constructs 2D glaciers with a Halfar or Halfar-like profile
# over different beds.  Does a few time-steps using the free-surface
# stabilization algorithm (FSSA) from Lofgren et al 2022.
# The steps are explicit but evaluated as solutions of the
# implicit backward-Euler method.  Each step solves the Glen-law
# Stokes problem using the FSSA modification, computes the surface
# motion map Phi(s) = - u|_s . n_s, and evaluates the diagnostic
# quantities defined in Theorem 6.1.

# TODO:
#   * explicit step of SKE
#   * time-stepping loop
#   * FSSA
#   * evaluate bounds
#   * evaluate ratios

import numpy as np
from firedrake import *
from firedrake.output import VTKFile
from stokesextruded import *
from initialgeometry import initials

mx = 201
mz = 15

L = 100.0e3             # domain is [-L,L]
R0 = 70.0e3             # Halfar dome radius
H0 = 1200.0             # Halfar dome height

Hmin = 50.0             # kludge

# physics parameters
secpera = 31556926.0    # seconds per year
g, rho = 9.81, 910.0    # m s-2, kg m-3
nglen = 3.0
A3 = 3.1689e-24         # Pa-3 s-1; EISMINT I value of ice softness
B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness
eps = 0.01
Dtyp = 2.0 / secpera    # 2 a-1
qq = 1.0 / nglen - 1.0

def _form_stokes(mesh, se):
    def D(w):
        return 0.5 * (grad(w) + grad(w).T)
    u, p = split(se.up)
    v, q = TestFunctions(se.Z)
    Du2 = 0.5 * inner(D(u), D(u)) + (eps * Dtyp)**2.0
    F = ( inner(B3 * Du2**(qq / 2.0) * D(u), D(v)) \
              - p * div(v) - div(u) * q - inner(se.f_body, v) ) * dx(degree=4)
    return F

basemesh = IntervalMesh(mx, -L, L)
xb = basemesh.coordinates.dat.data_ro
sb = initials(xb, Hmin, nglen=nglen)

# extend sbase, defined on the base mesh, to the extruded mesh using the
#   'R' constant-in-the-vertical space
mesh = ExtrudedMesh(basemesh, layers=mz, layer_height=1.0/mz)
P1R = FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
s = Function(P1R)
s.dat.data[:] = sb
Vcoord = mesh.coordinates.function_space()
x, z = SpatialCoordinate(mesh)
newcoord = Function(Vcoord).interpolate(as_vector([x, s * z]))
mesh.coordinates.assign(newcoord)

se = StokesExtruded(mesh)
se.mixed_TaylorHood()
se.body_force(Constant((0.0, - rho * g)))
se.dirichlet((1,2), Constant((0.0,0.0)))  # wrong if ice advances to margin
se.dirichlet(('bottom',), Constant((0.0,0.0)))

params = SolverParams['newton']
params.update(SolverParams['mumps'])
params.update({'snes_monitor': None,
               'snes_converged_reason': None})

printpar(f'solving 2D Stokes on {mx} x {mz} extruded mesh ...')
n_u, n_p = se.V.dim(), se.W.dim()
printpar(f'  sizes: n_u = {n_u}, n_p = {n_p}')
u, p = se.solve(par=params, F=_form_stokes(mesh, se))
se.savesolution(name='result.pvd')
printpar(f'u, p solution norms = {norm(u):8.3e}, {norm(p):8.3e}')

sbm = trace_scalar_to_p1(basemesh, mesh, z)
sbm.rename('surface elevation (m)')
ns = [-sbm.dx(0), Constant(1.0)]
ubm = trace_vector_to_p2(basemesh, mesh, u)
ubm.rename('surface velocity (m s-1)')

P1bm = FunctionSpace(basemesh, 'CG', 1)
# observation: if interpolate() is used in next line then asymmetry results
Phibm = Function(P1bm).project(- dot(ubm, as_vector(ns)))
Phibm.rename('surface motion map Phi (m s-1)')

if basemesh.comm.size == 1:
    # figure with s(x) and Phi(s)  [serial only]
    xx = basemesh.coordinates.dat.data
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(xx / 1.0e3, sbm.dat.data, color='C1', label='s')
    ax1.legend(loc='upper left')
    ax1.set_xticklabels([])
    ax1.grid(visible=True)
    ax1.set_ylabel('elevation (m)')
    ax2.plot(xx / 1.0e3, Phibm.dat.data * secpera, color='C2', label=r'$\Phi(s)$')
    ax2.legend(loc='upper right')
    ax2.set_ylabel(r'$\Phi$ (m a-1)')
    ax2.grid(visible=True)
    plt.xlabel('x (km)')
    plt.show()
