##########################################################
# Navier-Stokes equation: Cavity Flow   
# du/dt + udu/dx + vdu/dy = -1/rho*dp/dx + vis(d^2u/dx^2 + d^2u/dy^2)
# dv/dt + udv/dx + vdv/dy = -1/rho*dp/dy + vis(d^2v/dx^2 + d^2v/dy^2)
##########################################################

from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
import time

# parameters
nx = 20;    ny = 20;     
nt = 100;   nit = 100;    dt = 0.01;
vis = 0.1;  rho = 1;
dx = 2./(nx-1); dy = 1./(ny-1);

# arrays
x = np.linspace(0,2,nx);
y = np.linspace(0,2,ny);

u = np.zeros((nx,ny), dtype=float);
v = np.zeros((nx,ny), dtype=float);
p = np.zeros((nx,ny), dtype=float);
un = np.zeros((nx,ny), dtype=float);
vn = np.zeros((nx,ny), dtype=float);
pn = np.zeros((nx,ny), dtype=float);
b = np.zeros((nx,ny), dtype=float);

# figure initialization
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x,y)
wframe = ax.plot_wireframe(X,Y,p)

# time iteration
for it in range(nt):
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            b[i,j] = rho*((u[i+1,j]-u[i-1,j])/2/dx + (v[i,j+1]-v[i,j-1])/2/dt)/dt + ((u[i+1,j]-u[i-1,j])/2/dx)**2 + 2*(u[i,j+1]-u[i,j-1])/2/dy*(v[i+1,j]-v[i-1,j])/2/dx + ((v[i,j+1]-v[i,j-1])/2/dy)**2;

    for iit in range(nit):
        pn = p;    
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                p[i,j] = ((pn[i+1,j]+pn[i-1,j])*dy**2 + (pn[i,j+1]+pn[i,j-1])*dx**2 - b[i,j]*dx**2*dy**2)/(dx**2+dy**2)/2;
                         
        p[0,:] = p[1,:];            # dp/dy = 0 at y = 0
        p[-1,:] = p[-2,:];          # dp/dy = 0 at y = 2 
        p[:,0] = p[:,1];            # dp/dx = 0 at x = 0
        p[:,-1] = 0;                # p = 0     at x = 2

    un = u;
    vn = v;
    # space iteration
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            u[i,j] = un[i,j] - un[i,j]*dt/dx*(un[i,j]-un[i-1,j]) - vn[i,j]*dt/dy*(un[i,j]-un[i,j-1]) - 1/rho*(p[i+1,j]-p[i-1,j])*dt/2/dx + vis*dt/dx**2*(un[i+1,j]-2*un[i,j]+un[i-1,j]) + vis*dt/dy**2*(un[i,j+1]-2*un[i,j]+un[i,j-1]);
            v[i,j] = vn[i,j] - un[i,j]*dt/dx*(vn[i,j]-vn[i-1,j]) - vn[i,j]*dt/dy*(vn[i,j]-vn[i,j-1]) - 1/rho*(p[i+1,j]-p[i-1,j])*dt/2/dy + vis*dt/dx**2*(vn[i+1,j]-2*vn[i,j]+vn[i-1,j]) + vis*dt/dy**2*(vn[i,j+1]-2*vn[i,j]+vn[i,j-1]);
    
    u[0,:] = 0; u[nx-1,:] = 0; u[:,1] = 0; u[:,ny-1] = 1;   
    v[0,:] = 0; v[nx-1,:] = 0; v[:,1] = 0; u[:,ny-1] = 0;
             
    # Draw new frame
    if np.floor((iit+1)/10) < np.floor((iit+2)/10): 
        # Remove old line collection before drawing
        oldcol = wframe
        if oldcol is not None:
            ax.collections.remove(oldcol)

        # Draw new wireframe
        wframe = ax.plot_wireframe(X,Y,p)
        plt.draw()
        plt.pause(0.1)
