import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from writeout import writeout

x = np.arange(7)
y = np.array([1.0, 3.0, 2.0, 3.97, 3.0, 4.0, 2.0])

cs = CubicSpline(x, y, bc_type='natural')
xf = np.linspace(x.min(), x.max(), 201)

b = cs(xf)
sfilled = b.copy()
sfilled[(1.0 < xf) & (xf < 2.5)] = 3.0
sfilled[(3.03 < xf) & (xf < 5.0)] = 4.0

fnames = ['filled', 'icefree']
for j in range(2):
    plt.figure(figsize=(8, 6))
    if j == 0:
        plt.plot(xf, b, 'k--', lw=1.5)
        plt.plot(xf, sfilled, 'k', lw=3.0)
    else:
        plt.plot(xf, b, 'k', lw=3.0)
    plt.xlabel(r'$x$')
    plt.ylabel('elevation')
    plt.axis(False)
    writeout(fnames[j] + '.pdf')
