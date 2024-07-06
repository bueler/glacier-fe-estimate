# Compute 1D numpy arrays for b(x) and s(x).  Used to set
# initial geometry.

import numpy as np

_R0 = 70.0e3             # Halfar dome radius (m)
_H0 = 1200.0             # Halfar dome height (m)
_len1 = 20.0e3           # wavelength of lowest-frequency bed oscillation
_amp1 = 50.0            # amplitude ...
_len2 = 5.0e3            # ... next-lowest ...
_amp2 = 20.0
_len3 = 1.0e3            # ... next-lowest ...
_amp3 = 10.0

# the Halfar time-dependent SIA geometry from
#   * P. Halfar (1981), On the dynamics of the ice sheets,
#     J. Geophys. Res. 86 (C11), 11065--11072
# The solution is evaluated at t = t0.
def _s_halfar(x, nglen=3.0):
    pp = 1.0 + 1.0 / nglen
    rr = nglen / (2.0 * nglen + 1.0)
    s = np.zeros(np.shape(x))
    s[abs(x) < _R0] = _H0 * (1.0 - abs(x[abs(x) < _R0] / _R0)**pp)**rr
    s[s < 0.0] = 0.0
    return s

def geometry(x, nglen=3.0, bed='flat'):
    b = np.zeros(np.shape(x))
    if bed == 'flat':
        pass
    else:
        b += _amp1 * np.sin(2 * np.pi * x / _len1)
        b += _amp2 * np.cos(2 * np.pi * x / _len2)
        if bed == 'rough':
            b += _amp3 * np.sin(2 * np.pi * x / _len3)
    return b, b + _s_halfar(x, nglen=nglen)
