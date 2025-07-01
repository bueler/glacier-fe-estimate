import firedrake as fd
from stokesextrude import extend_p1_from_basemesh

# public physics parameters
secpera = 31556926.0    # seconds per year
g, rho = 9.81, 910.0    # m s-2, kg m-3
nglen = 3.0
A3 = 3.1689e-24         # Pa-3 s-1; EISMINT I value of ice softness
B3 = A3**(-1.0/3.0)     # Pa s(1/3);  ice hardness

def _D(w):
    return 0.5 * (fd.grad(w) + fd.grad(w).T)

# weak form for the Stokes problem; se is a StokesExtrude object
def form_stokes(se, sR, pp=(1.0/nglen)+1.0, mu0=0.0, fssa=True, theta_fssa=1.0, dt_fssa=0.0, smb_fssa=None):
    u, p = fd.split(se.up)
    v, q = fd.TestFunctions(se.Z)
    Du2 = 0.5 * fd.inner(_D(u), _D(u)) + mu0
    qqq = (pp - 2.0) / 2.0
    F = fd.inner(B3 * Du2**qqq * _D(u), _D(v)) * fd.dx(degree=4)  # correspondence with paper: nu_p = 0.5 B3
    F -= (p * fd.div(v) + fd.div(u) * q) * fd.dx
    source = fd.inner(se.f_body, v) * fd.dx
    if fssa:
        # see section 4.2 in Lofgren et al
        nsR = fd.as_vector([-sR.dx(0), fd.Constant(1.0)])
        nunit = nsR / fd.sqrt(sR.dx(0)**2 + 1.0)
        F -= theta_fssa * dt_fssa * fd.inner(u, nunit) * fd.inner(se.f_body, v) * fd.ds_t
        aR = extend_p1_from_basemesh(se.mesh, smb_fssa)
        zvec = fd.Constant(fd.as_vector([0.0, 1.0]))
        source += theta_fssa * dt_fssa * aR * fd.inner(zvec, nunit) * fd.inner(se.f_body, v) * fd.ds_t
    F -= source
    return F


# diagnostic: effective viscosity nu from the velocity solution
def effective_viscosity(u, P1, pp=(1.0/nglen)+1.0, mu0=0.0):
    Du2 = 0.5 * fd.inner(_D(u), _D(u))
    qqq = (pp - 2.0) / 2.0
    nu = fd.Function(P1).interpolate(0.5 * B3 * Du2**qqq)   # correspondence with paper: nu_p = 0.5 B3
    nu.rename('nu (unregularized; Pa s)')
    Du2 += mu0
    nueps = fd.Function(P1).interpolate(0.5 * B3 * Du2**qqq)
    nueps.rename(f'nu (mu0={mu0:.3f}; Pa s)')
    return nu, nueps


# diagnostic: hydrostatic pressure
def p_hydrostatic(se, sR, P1):
    _, z = fd.SpatialCoordinate(se.mesh)
    phydro = fd.Function(P1).interpolate(rho * g * (sR - z))
    phydro.rename('p_hydro (Pa)')
    return phydro


# surface motion operator
# if eps>0 then regularized with SIA vertical velocity diffusivity
# returns a UFL expression, so q can be a TestFunction or a Function
def Phi(s, us, q, eps=0.0, H0=1000.0, slopereg=1.0e-8):
    ns = fd.as_vector([-s.dx(0), fd.Constant(1.0)])
    Phi = - fd.dot(us, ns) * q
    if eps > 0.0:
        Gamma = 2.0 * A3 * (rho * g)**nglen / (nglen + 2)
        C = Gamma * H0**(nglen + 1)
        gsnorm = (fd.dot(fd.grad(s), fd.grad(s)) + slopereg)**0.5
        Phi += eps * C * fd.inner(gsnorm**(nglen - 1) * fd.grad(s), fd.grad(q))
    return Phi
