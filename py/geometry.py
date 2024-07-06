# Compute 1D numpy arrays for b(x) and s(x).  Used to set
# initial geometry.

import numpy as np

bedtypes = ['flat', 'smooth', 'rough']

_R0 = 65.0e3                             # Halfar dome radius (m)
_H0 = 1400.0                             # Halfar dome height (m)
_len = [120.0e3, 20.0e3, 10.0e3, 4.0e3]  # wavelengths of bed oscillations
_amp = [100.0, 40.0, 20.0, 25.0]         # amplitudes ...
_off = [30.0e3, 0.0, -1.0e3, -400.0]     # offsets ...

# the Halfar time-dependent SIA geometry from
#   * P. Halfar (1981), On the dynamics of the ice sheets,
#     J. Geophys. Res. 86 (C11), 11065--11072
# The solution is evaluated at t = t0.
def _s_halfar(x, nglen=3.0):
    pp = 1.0 + 1.0 / nglen
    rr = nglen / (2.0 * nglen + 1.0)
    s = np.zeros(np.shape(x))
    s[abs(x) < _R0] = _H0 * (1.0 - abs(x[abs(x) < _R0] / _R0)**pp)**rr
    s[abs(x) >= _R0] = -10000.0  # needs max with bed below
    return s

def geometry(x, nglen=3.0, bed='flat'):
    assert bed in bedtypes
    b = np.zeros(np.shape(x))
    if bed != 'flat':
        for k in range(3 if bed == 'smooth' else 4):
            b += _amp[k] * np.sin(2 * np.pi * (x + _off[k]) / _len[k])
    return b, np.maximum(b, _s_halfar(x, nglen=nglen))
