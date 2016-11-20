# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 11:51:39 2016

@author: Patrick
"""

import numpy as np
import matplotlib.pyplot as plt

def greenGrav(kx, ky):
    k2 = kx**2. + ky**2.
    return -4.*np.pi/(k2)

Ngrid = 256
L = 1.
h = L / Ngrid

runs = 10
m = 100
results = np.zeros((runs * m, 2))

for counter in range(runs):
    
    mesh = np.zeros((Ngrid, Ngrid))
    x_part = np.random.rand() * L
    y_part = np.random.rand() * L
    
    # CIC assignment
    xf = np.floor(x_part / L * Ngrid)
    x_frac = x_part - xf
    yf = np.floor(y_part / L * Ngrid)
    y_frac = y_part - yf
    mesh[xf, yf]     = mesh[xf, yf] + (1 - x_frac) * (1 - y_frac)
    if xf < Ngrid-1:
        mesh[xf + 1, yf] = mesh[xf + 1, yf] + x_frac * (1 - y_frac)
    else:
        mesh[0, yf] = mesh[0, yf] + x_frac * (1. - y_frac)
    if yf < Ngrid-1:    
        mesh[xf, yf + 1.] = mesh[xf, yf + 1] + (1 - x_frac) * y_frac
        if xf < Ngrid-1.:
            mesh[xf + 1, yf + 1] = mesh[xf + 1, yf + 1] + x_frac * y_frac         
    else:
        mesh[xf, 0] = mesh[xf, 0] + (1 - x_frac) * y_frac  
        if xf == Ngrid-1:
            mesh[0, 0] = mesh[0,0] + x_frac * y_frac 
            
    # Fourier transform the density field to obtain potential
    mesh_fourier = np.fft.fft2(mesh)
#    mesh_fourier = np.fft.fftshift(mesh_fourier)
    axis = np.linspace(-Ngrid*(2*np.pi)/L/2., Ngrid*(2*np.pi)/L/2., Ngrid)
    green_fourier = greenGrav(axis[:,None], axis[None, :])
    potential_fourier = green_fourier * mesh_fourier
    potential_fourier = np.fft.ifftshift(potential_fourier)
 #   potential_fourier[0,0] = 0
 #   potential_fourier[0,Ngrid-1] = 0
 #   potential_fourier[Ngrid-1,0] = 0
    potential_fourier[Ngrid-1,Ngrid-1] = 0
    potential = np.fft.ifft2(potential_fourier)
    
    #produce force field using two point finite differencing
    force = np.zeros((Ngrid, Ngrid, 2), dtype=complex)
    for i in range(Ngrid):
        for j in range(Ngrid):
            if (i>0):
                if (i<Ngrid-1):
                    force[i,j,0] = 1/(2.*h) * ( potential[i+1, j] - potential[i-1, j])
                else:
                    force[i,j,0] = 1/(2.*h) * ( potential[0, j] - potential[i-1, j])
            else:
                force[i,j,0] = 1/(2.*h) * ( potential[i+1, j] - potential[Ngrid-1, j])
            if (j>0):
                if (j<Ngrid-1):
                    force[i,j,1] = 1/(2.*h) * ( potential[i, j+1] - potential[i, j-1])
                else:
                    force[i,j,1] = 1/(2.*h) * ( potential[i, 0] - potential[i, j-1])
            else:
                force[i,j,1] = 1/(2.*h) * ( potential[i, j+1] - potential[i, Ngrid-1])    
    force = np.abs(force)*np.sign(np.real(force))
                
    #transform to scaled coordinates of the grid
    rmin = 0.3 * L / Ngrid
    rmax = L / 2 
    samplePos = np.zeros((m, 2))
    for i in range(m):
        p = np.random.rand()
        q = np.random.rand()
        dx = rmin * (rmax / rmin)**p * np.cos(2*np.pi*q)
        dy = rmin * (rmax / rmin)**p * np.sin(2*np.pi*q)
        if x_part + dx < L:
            samplePos[i,0] = x_part + dx
        else:
            samplePos[i,0] = x_part + dx - L
        if y_part + dy < L:
            samplePos[i,1] = y_part + dy
        else:
            samplePos[i,1] = y_part + dy - L 
        results[counter*m + i, 0] = rmin * (rmax / rmin)**p  
    
    #reassign force to position using the same kernel (CIC) to avoid a nonzero self-force
    acc = np.zeros((m,3))
    mass = 1.
    for i in range(m):
        x_sample = samplePos[i,0]
        y_sample = samplePos[i,1]
        xf = np.floor(x_sample * Ngrid / L)
        yf = np.floor(y_sample * Ngrid / L)
        x_frac = x_sample - xf
        y_frac = y_sample - yf
     #   mesh[xf, yf]     = mesh[xf, yf] + (1. - x_frac) * (1. - y_frac)
        for c in range(2):
            if xf < Ngrid-1:
                if yf < Ngrid-1:
                    acc[i, c] = (1 - x_frac)*(1 - y_frac)*force[xf, yf, c] + x_frac*(1 - y_frac)*force[xf+1,yf, c] + \
                        (1 - x_frac)*y_frac*force[xf, yf+1, c] + x_frac*y_frac*force[xf+1, yf+1, c]                
                else:
                    acc[i, c] = (1 - x_frac)*(1 - y_frac)*force[xf, yf, c] + x_frac*(1 - y_frac)*force[xf+1,yf, c] + \
                        (1 - x_frac)*y_frac*force[xf, 0, c] + x_frac*y_frac*force[xf+1, 0, c] 
            else:        
                if yf < Ngrid-1:
                    acc[i, c] = (1 - x_frac)*(1 - y_frac)*force[xf, yf, c] + x_frac*(1 - y_frac)*force[0,yf, c] + \
                        (1 - x_frac)*y_frac*force[xf, yf+1, c] + x_frac*y_frac*force[0, yf+1, c]
                else:
                    acc[i, c] = (1 - x_frac)*(1 - y_frac)*force[xf, yf, c] + x_frac*(1 - y_frac)*force[0, yf, c] + \
                        (1 - x_frac)*y_frac*force[xf, 0, c] + x_frac*y_frac*force[0, 0, c]
        acc[i,2] = np.sqrt(acc[i,0]**2 + acc[i,1]**2) # magnitude of acceleration
    acc = mass * acc
    
    # save results
    results[counter*m:(counter+1)*m, 1] = acc[:,2]



plt.figure(1)
plt.loglog(results[:,0], results[:,1], 'x', label='data')
x = np.linspace(rmin, rmax, 1000)
Newtonr = 2 / x
Newtonr2 = 1 / x**2
Newtonr3 = 1 / x**3
plt.loglog(x, Newtonr, label='2/r')
plt.loglog(x, Newtonr2, label='1/r^2')
plt.loglog(x, Newtonr3, label='1/r^3')
plt.legend()
plt.axvline(L/Ngrid)
    