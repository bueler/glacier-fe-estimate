import numpy as np

R0 = 70.0e3             # Halfar dome radius
H0 = 1200.0             # Halfar dome height

# the Halfar time-dependent SIA geometry from
#   * P. Halfar (1981), On the dynamics of the ice sheets,
#     J. Geophys. Res. 86 (C11), 11065--11072
# The solution is evaluated at t = t0.
def initialgeometry(xb, nglen=3.0):
    pp = 1.0 + 1.0 / nglen
    rr = nglen / (2.0 * nglen + 1.0)
    sb = np.zeros(np.shape(xb))
    sb[abs(xb) < R0] = H0 * (1.0 - abs(xb[abs(xb) < R0] / R0)**pp)**rr
    sb[sb < 0.0] = 0.0
    return sb
