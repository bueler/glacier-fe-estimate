#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

fsize=18.0
bigfsize=24.0

def genbasicfig(xshift=0.0):
    x = np.linspace(0.0,10.0,1001)
    # bed elevation
    b = 0.07*(x-3.0)**2 + 0.2*np.sin(2.0*x) - 0.1
    plt.plot(x + xshift, b, 'k--', lw=2.5)
    # current thickness for Omega^{n-1}
    h0 = 3.0
    L = 3.0
    firstshape = h0*(-0.2 + np.sqrt(np.maximum(0.0,1.0 - (x-5)**2/L**2)))
    thk = np.maximum(0.0, firstshape)
    # surface
    s = b + thk
    offset = 0.1
    plt.plot(x + xshift, s + offset, 'k', lw=3.0)
    return x + xshift, s, b

def drawclimate(x,s):
    plt.text(x[10], s[10]+2.7, r'$a(t,x)$', fontsize=bigfsize, color='k')
    for j in range(10):
        xarr = x[50+100*j]
        if j>0:
            magarr = 0.6*np.sin(np.pi/2 + 0.6*xarr)
        else:
            magarr = 0.05
        plt.arrow(xarr, s.max()+0.3, 0.0, magarr,
                  lw=1.5, head_width=0.1, color='k')

# domain notation figure
plt.figure(figsize=(10,5.5))
x, s, b = genbasicfig()
#plt.text(x[550] - 1.0, b[600] + 0.4 * s[600], r'$\Lambda(t)$',
#         fontsize=bigfsize, color='k')
plt.text(x[550] - 2.0, b[600] + 0.4 * s[600], r'$\mathbf{u}(t,x,z),\,p(t,x,z)$',
         fontsize=bigfsize, color='k')
drawclimate(x,s)
# mark top surface
plt.text(x[150], s[300] + 0.7, r'$z=s(t,x)$', fontsize=bigfsize, color='k')
# mark bottom surface
plt.text(x[650], b[650] - 0.5, r'$z=b(x)$', fontsize=bigfsize, color='k')
# show \Omega
yR = min(b) - 0.5
#plt.plot([min(x),max(x)],[yR,yR],color='k',lw=1.0)
#plt.text(x[875],yR+0.2,r'$\Omega$',fontsize=bigfsize)
#plt.axis([0.0,10.0,yR-0.8,4.5])
plt.axis([0.0,10.0,yR,4.5])
plt.axis('off')
writeout('sdusual.png')
