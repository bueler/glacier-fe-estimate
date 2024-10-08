#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

#plt.rcParams["font.family"] = "serif"
#plt.rcParams["mathtext.fontset"] = "cm"
fsize=9.0
bigfsize=18.0
biggerfsize=24.0

x = np.linspace(0.0,1.0,11)

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(x, np.zeros(np.shape(x)), 'k-')
ax.plot(1.0, 0.0, '>k')
ax.text(1.04, 0.0, r'$x = (x_1,x_2)$', size=bigfsize)
ax.plot(np.zeros(np.shape(x)), x, 'k-')
ax.plot(0.0, 1.0, '^k')
ax.text(0.0, 1.05, r'$t$', size=bigfsize)
ofst = -0.12
vfoff = -0.01
ax.plot([-0.02, 0.0], [0.0 - vfoff, 0.0 - vfoff], 'k-', lw=1.0)
ax.text(ofst, 0.00, '2024', size=fsize)
ax.plot([-0.02, 0.0], [0.35 - vfoff, 0.35 - vfoff], 'k-', lw=1.0)
ax.text(ofst, 0.35, '2050', size=fsize)
ax.plot([-0.02, 0.0], [0.95 - vfoff, 0.95 - vfoff], 'k-', lw=1.0)
ax.text(ofst, 0.95, '2100', size=fsize)
plt.axis(False)
#plt.show()
writeout('xtaxes.png')

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(x, np.zeros(np.shape(x)), 'k-')
ax.plot(1.0, 0.0, '>k')
ax.text(1.04, 0.0, r'$x_1$', size=bigfsize)
ax.plot(np.zeros(np.shape(x)), x, 'k-')
ax.plot(0.0, 1.0, '^k')
ax.text(0.0, 1.05, r'$x_2$', size=bigfsize)
ax.plot([0.0, 0.95], [0.95, 0.95], 'k-')
ax.plot([0.95, 0.95], [0.0, 0.95], 'k-')
ax.text(0.8, 0.84, r'$\Omega$', size=biggerfsize)
plt.axis(False)
#plt.show()
writeout('xxaxes.png')