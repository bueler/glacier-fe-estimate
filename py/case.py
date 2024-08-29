from sys import argv
import numpy as np
from firedrake import *
from firedrake.output import VTKFile
from stokesextrude import StokesExtrude, SolverParams, extend_p1_from_basemesh, trace_vector_to_p2, printpar
from geometry import secpera, bedtypes, g, rho, nglen, A3, B3, t0, halfargeometry
from figures import mkdir, livefigure, snapsfigure, histogramPhirat
from measure import geometryreport, sampleratios

# parameters set at runtime
mx = int(argv[1])              # number of elements in x direction
mz = int(argv[2])              # number of elements in z (vertical) direction
Nsteps = int(argv[3])          # number of time steps
dt = float(argv[4]) * secpera  # dt in years
bed = argv[5]                  # 'flat', 'smooth', 'rough'
ratiosfile = argv[6]           # at the end, append a pair of ratios into this file
writepng = (len(argv) > 7)
if writepng:
    dirroot = argv[7]
writepvd = (len(argv) > 8)
if writepvd:
    pvdroot = argv[8]

# experiment parameters
L = 100.0e3                       # domain is (-L,L)
SMBlist = [0.0, -2.5e-7, 1.0e-7]  # m s-1; values of aconst used in experiments
aposfrac = 0.75                   # fraction of domain on which positive SMB is applied
Nsamples = 1000                   # number of samples when evaluating minimal ratios

# solution method
explicit = False        # defaults to semi-implicit steps
zeroheight = 'indices'  # how should StokesExtrude handle zero-height columns;
                        #   alternative is 'bounds', but it seems to do poorly?
fssa = True             # use Lofgren et al (2022) FSSA technique in Stokes solve
theta_fssa = 1.0        #   with this theta value

# Stokes parameters
qq = 1.0 / nglen - 1.0
Dtyp = 1.0 / secpera      # = 1 a-1; strain rate scale
eps = 0.0001 * Dtyp**2.0  # viscosity regularization

# set up bm = basemesh once
bm = IntervalMesh(mx, -L, L)
P1bm = FunctionSpace(bm, 'P', 1)
xbm = bm.coordinates.dat.data_ro

# bed and initial geometry
assert bed in bedtypes
print(f"Halfar t0 = {t0 / secpera:.3f} a")
b_np, s_initial_np = halfargeometry(xbm, t=t0, bed=bed)  # get numpy arrays
b = Function(P1bm, name='bed elevation (m)')
b.dat.data[:] = b_np
s_initial = Function(P1bm)
s_initial.dat.data[:] = s_initial_np

# surface mass balance (SMB) function
a = Function(P1bm, name='surface mass balance (m s-1)')

# create the extruded mesh, but leave z coordinate at default
se = StokesExtrude(bm, mz=mz, htol=1.0)
P1 = FunctionSpace(se.mesh, 'P', 1)
x, _ = SpatialCoordinate(se.mesh)

# set up Stokes problem
se.mixed_TaylorHood()
#se.mixed_PkDG(kp=0) # = P2xDG0; seems to NOT be better; not sure about P2xDG1
se.body_force(Constant((0.0, - rho * g)))
se.dirichlet((1,2), Constant((0.0, 0.0)))      # consequences if ice advances to margin
se.dirichlet(('bottom',), Constant((0.0, 0.0)))
params = SolverParams['newton']
params.update(SolverParams['mumps'])
#params.update({'snes_monitor': None})
params.update({'snes_converged_reason': None})
params.update({'snes_atol': 1.0e-2})
params.update({'snes_linesearch_type': 'bt'})  # helps with non-flat beds, it seems

def _D(w):
    return 0.5 * (grad(w) + grad(w).T)

# weak form for the Stokes problem
def _form_stokes(se, sR):
    u, p = split(se.up)
    v, q = TestFunctions(se.Z)
    Du2 = 0.5 * inner(_D(u), _D(u)) + eps
    F = inner(B3 * Du2**(qq / 2.0) * _D(u), _D(v)) * dx(degree=4)
    F -= (p * div(v) + div(u) * q) * dx
    source = inner(se.f_body, v) * dx
    if fssa:
        # see section 4.2 in Lofgren et al
        nsR = as_vector([-sR.dx(0), Constant(1.0)])
        nunit = nsR / sqrt(sR.dx(0)**2 + 1.0)
        F -= theta_fssa * dt * inner(u, nunit) * inner(se.f_body, v) * ds_t
        aR = extend_p1_from_basemesh(se.mesh, a)
        zvec = Constant(as_vector([0.0, 1.0]))
        source += theta_fssa * dt * aR * inner(zvec, nunit) * inner(se.f_body, v) * ds_t
    F -= source
    return F

# diagnostic: effective viscosity nu from the velocity solution
def _effective_viscosity(u):
    Du2 = 0.5 * inner(_D(u), _D(u))
    nu = Function(P1).interpolate(0.5 * B3 * Du2**(qq/2.0))
    nu.rename('nu (unregularized; Pa s)')
    nueps = Function(P1).interpolate(0.5 * B3 * (Du2  + (eps * Dtyp)**2)**(qq/2.0))
    nueps.rename(f'nu (eps={eps:.3f}; Pa s)')
    return nu, nueps

# diagnostic: hydrostatic pressure
def _p_hydrostatic(se, sR):
    _, z = SpatialCoordinate(se.mesh)
    phydro = Function(P1).interpolate(rho * g * (sR - z))
    phydro.rename('p_hydro (Pa)')
    return phydro

# initialize surface elevation state variable
s = Function(P1bm, name='surface elevation (m)')
s.interpolate(conditional(s_initial < b, b, s_initial))
ns = as_vector([-s.dx(0), Constant(1.0)])

# set up for *semi-implicit*: implement equation (3.7) and (3.23) in paper,
#     but use old velocity in weak form
if not explicit:
    sibcs = [DirichletBC(P1bm, b.dat.data_ro[0], 1),
             DirichletBC(P1bm, b.dat.data_ro[-1], 2)]
    siv = TestFunction(P1bm)
    sisold = Function(P1bm)
    siP2Vbm = VectorFunctionSpace(bm, 'CG', 2, dim=2)
    siubm = Function(siP2Vbm)
    siparams = {#"snes_monitor": None,
                "snes_converged_reason": None,
                "snes_rtol": 1.0e-6,
                "snes_atol": 1.0e-6,
                "snes_stol": 0.0,
                "snes_type": "vinewtonrsls",
                "snes_vi_zero_tolerance": 1.0e-8,
                "snes_linesearch_type": "basic",
                "snes_max_it": 200,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"}
    # weak form for (3.23), made semi-implicit
    siF = inner(s - dt * dot(siubm, ns) - (sisold + dt * a), siv) * dx
    siproblem = NonlinearVariationalProblem(siF, s, sibcs)
    sisolver = NonlinearVariationalSolver(siproblem, solver_parameters=siparams,
                                          options_prefix="step")
    siub = Function(P1bm).interpolate(Constant(PETSc.INFINITY))

# set up for livefigure() and snapsfigure()
if writepng:
    printpar(f'creating root directory {dirroot} for image files ...')
    mkdir(dirroot)
    snaps = [s.copy(deepcopy=True),]

# outer surface mass balance (SMB) loop
_slist = []
for aconst in SMBlist:
    # describe run
    printpar(f'using aconst = {aconst:.3e} m/s constant value of SMB ...')
    printpar(f'doing N = {Nsteps} steps of dt = {dt/secpera:.3f} a and saving states ...')
    printpar(f'  solving 2D Stokes + SKE on {mx} x {mz} extruded mesh over {bed} bed')
    printpar(f'  dimensions: n_u = {se.V.dim()}, n_p = {se.W.dim()}')
    # set up directory and open file (if wanted)
    afrag = 'aneg' if aconst < 0.0 else ('apos' if aconst > 0.0 else 'azero')
    if writepng:
        outdirname = dirroot + afrag + '/'
        printpar(f'  creating directory {outdirname} for image files ...')
        mkdir(outdirname)
    if writepvd:
        pvdfilename = pvdroot + '_' + afrag + '.pvd'
        printpar(f'  opening {pvdfilename} ...')
        outfile = VTKFile(pvdfilename)
    # reset for time-stepping
    if aconst > 0.0:
        a.dat.data[:] = 0.0
        a.dat.data[abs(xbm) < aposfrac * L] = aconst
    else:
        a.dat.data[:] = aconst
    s.interpolate(conditional(s_initial < b, b, s_initial))
    t = 0.0
    # inner time-stepping loop
    for n in range(Nsteps):
        # start with reporting
        if n == 0:
            geometryreport(bm, 0, t, s, b, L)
            if writepng:
                livefigure(bm, b, s, t, fname=f'{outdirname}t{t/secpera:010.3f}.png')

        # set geometry (z coordinate) of extruded mesh
        se.reset_elevations(b, s)
        P1R = FunctionSpace(se.mesh, 'P', 1, vfamily='R', vdegree=0)
        sR = Function(P1R)
        sR.dat.data[:] = s.dat.data_ro

        # solve Stokes on extruded mesh and extract surface trace
        u, p = se.solve(F=_form_stokes(se, sR),
                        par=params,
                        zeroheight=zeroheight)
        ubm = trace_vector_to_p2(bm, se.mesh, u)  # surface velocity (m s-1)
        #printpar(f'  solution norms: |u|_L2 = {norm(u):8.3e},  |p|_L2 = {norm(p):8.3e}')

        # optionally write t-dependent .pvd with 2D fields
        if writepvd:
            u.rename('velocity (m s-1)')
            p.rename('pressure (Pa)')
            nu, nueps = _effective_viscosity(u)
            phydro = _p_hydrostatic(se, sR)
            pdiff = Function(P1).interpolate(p - phydro)
            pdiff.rename('pdiff = phydro - p (Pa)')
            outfile.write(u, p, nu, nueps, pdiff, time=t)

        # save surface elevation and velocity into list for ratio evals
        _slist.append({'t': t,
                       's': s.copy(deepcopy=True),
                       'us': ubm.copy(deepcopy=True)})

        # time step of VI problem (3.23)
        if explicit:
            # explicit time step, mostly pointwise operation (interpolate and truncate)
            Phi = Function(P1bm).project(- dot(ubm, ns))  # interpolate() would be bad here (P2 nodes)
            snew = s + dt * (a - Phi)
            s.interpolate(conditional(snew < b, b, snew))
            # FIXME consider project() in above line?
        else:
            # semi-implicit: solve VI problem with surface velocity from old surface elevation
            sisold.dat.data[:] = s.dat.data_ro
            siubm.dat.data[:] = ubm.dat.data_ro
            sisolver.solve(bounds=(b, siub))
        t += dt

        # end of step reporting
        geometryreport(bm, n + 1, t, s, b, L)
        if writepng:
            livefigure(bm, b, s, t, fname=f'{outdirname}t{t/secpera:010.3f}.png',
                    writehalfar=(bed == 'flat' and aconst == 0.0 and n + 1 == Nsteps))
            if n + 1 == int(round(0.7 * Nsteps)): # reliable if Nsteps is divisible by 10
                snaps.append(s.copy(deepcopy=True))

    if writepng:
        printpar(f'  finished writing to {outdirname}')
    if writepvd:
        printpar(f'  finished writing to {pvdfilename}')

if writepng:
    snapsname = dirroot + 'snaps.png'
    snapsfigure(bm, b, snaps, fname=snapsname)
    printpar(f'  finished writing to {snapsname}')

# process _slist from all three SMB cases
maxcont, rats = sampleratios(dirroot, _slist, bm, b, N=Nsamples, Lsc=L)
printpar(f'  max continuity ratio:               {maxcont:.3e}')
histogramPhirat(dirroot, rats)
pos = rats[rats > 0.0]
assert len(pos) > 0
pmin = min(pos)
pmed = np.median(pos)
printpar(f'  pos coercivity ratio min:           {pmin:.3e}')
printpar(f'                       median:        {pmed:.3e}')
nonpos = rats[rats <= 0.0]
if len(nonpos) > 0:
    npmin, npmed, npf = min(nonpos), np.median(nonpos), len(nonpos) / len(rats)
    printpar(f'  non-pos coercivity ratio min:       {npmin:.3e}')
    printpar(f'                           median:    {npmed:.3e}')
    printpar(f'                           fraction:  {npf:.4f}')
    with open(ratiosfile, 'a') as rfile:
        rfile.write(f'{maxcont:.3e}, {pmin:.3e}, {pmed:.3e}, {npmin:.3e}, {npmed:.3e}, {npf:.4f}\n')
else:
    with open(ratiosfile, 'a') as rfile:
        rfile.write(f'{maxcont:.3e}, {pmin:.3e}, {pmed:.3e}, N/A, N/A, 0.0000\n')
