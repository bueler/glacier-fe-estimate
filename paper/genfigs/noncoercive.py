import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from writeout import writeout

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
bigfsize=24.0

x = np.arange(7)
y = np.array([1.0, 3.0, 2.0, 3.97, 3.0, 4.0, 2.0])

cs = CubicSpline(x, y, bc_type='natural')
xf = np.linspace(0.2, x.max(), 201)  # also crop left end

b = cs(xf)
sfilled = b.copy()
sfilled[(1.0 < xf) & (xf < 2.5)] = 3.0
sfilled[(3.03 < xf) & (xf < 5.0)] = 4.0

fnames = ['filled', 'icefree']
for j in range(2):
    plt.figure(figsize=(8, 3))
    if j == 0:
        plt.plot(xf, b, 'k--', lw=1.5)
        plt.plot(xf, sfilled, 'k', lw=3.0)
        plt.text(1.8, 3.2, r'$s_1$', fontsize=bigfsize, color='k')
        plt.text(2.3, 2.0, r'$b$', fontsize=bigfsize, color='k')
    else:
        plt.plot(xf, b, 'k', lw=3.0)
        plt.text(1.7, 3.5, r'$s_2=b$', fontsize=bigfsize, color='k')
    plt.xlabel(r'$x$')
    plt.ylabel('elevation')
    plt.axis(False)
    writeout(fnames[j] + '.pdf')
