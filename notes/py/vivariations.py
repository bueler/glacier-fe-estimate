from firedrake import *
import matplotlib.pyplot as plt

m = 1000
mesh = IntervalMesh(m, 10.0)
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)

psi = Function(V, name="psi").interpolate(0.2 * sin(0.4 * x[0]*x[0]))
f = Function(V, name="f").interpolate(conditional(x[0] < 2.0, 2.0, -5.0))
u = Function(V, name="u")
v = TestFunction(V)

vip = {"snes_converged_reason": None,
        #"snes_monitor": None,
        #"snes_rtol": 1.0e-6,
        #"snes_atol": 1.0e-6,
        "snes_stol": 0.0,
        "snes_type": "vinewtonrsls",
        "snes_vi_zero_tolerance": 1.0e-12,
        "snes_linesearch_type": "basic",
        "snes_max_it": 200,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"}

def visualize(mesh, u, psi, f, name):
    xx = mesh.coordinates.dat.data
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(10.0, 6.0))
    ax1.plot(xx, u.dat.data, color='C1', linewidth=2.0, label='$u(x)$')
    ax1.plot(xx, psi.dat.data, color='C2', label='$\psi(x)$')
    ax1.set_xlabel(None)
    ax1.set_xticklabels([])
    ax1.set_ylabel('elevation')
    ax1.legend(loc='upper right')
    ax1.grid(visible=True)
    ax1.set_title(name)
    ax2.plot(xx, f.dat.data,  color='C3', label='$f(x)$')
    ax2.set_xlabel('x')
    ax2.set_ylabel('$f(x)$')
    ax2.grid(visible=True)
    fname = None  # "classic.png"
    if fname == None:
        plt.show()
    else:
        plt.savefig(fname)
    plt.close()

casenames = ["classic", "advect"]
Fforms = [inner(grad(u), grad(v)) * dx - f * v * dx,
          0.01 * inner(grad(u), grad(v)) * dx + 4.0 * grad(u - psi)[0] * v * dx - f * v * dx]

bcs = [DirichletBC(V, Constant(1.0), (1,)),
        DirichletBC(V, Constant(0.0), (2,))]
u.interpolate(Constant(1.0))

for k in range(len(casenames)):
    F = Fforms[k]
    viproblem = NonlinearVariationalProblem(F, u, bcs)
    solver = NonlinearVariationalSolver(viproblem, solver_parameters=vip, options_prefix="s")
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    solver.solve(bounds=(psi, ub))
    visualize(mesh, u, psi, f, casenames[k])
