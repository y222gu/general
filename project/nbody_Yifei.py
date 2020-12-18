import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def greens(n,soft,G,non_periodic):
    # n is the number of grids
    if non_periodic:
        n=2*n
    x = np.arange(n)
    x[n//2:] = x[n//2:]
    xmat,ymat,zmat = np.meshgrid(x, x, x)
    kernel = np.zeros([n,n,n])
    dr=np.sqrt(xmat**2+ymat**2+zmat**2+soft**2)
    kernel=1*G/dr
    return kernel

def density(position,n,m):   
    grid_min = 0
    grid_max = n
    den, edges = np.histogramdd(position,bins=(n,n,n),range=((grid_min, grid_max), (grid_min, grid_max), (grid_min, grid_max)),weights=m)
    mask = np.zeros([n,n,n],dtype='bool')
    return den, mask

def get_pot(kernelft,den,non_periodic):
    
    pot=den.copy()
    if non_periodic:
        pot=np.pad(pot,(0,pot.shape[0]))

    potft=np.fft.rfftn(pot)
    pot=np.fft.irfftn(potft*kernelft)
    
    if len(den.shape)==3:
        pot=pot[:den.shape[0],:den.shape[1],:den.shape[2]]
        return pot
    print("error in get_pot - unexpected number of dimensions")
    assert(1==0)

def get_forces(pot,dx):
    force = np.gradient(pot,dx)
    force_x = np.asarray(force[0])
    force_y = np.asarray(force[1])
    force_z = np.asarray(force[2])
    return force_x,force_y,force_z

def take_step(position,v,dt,n,kernel,m,non_periodic):
    pos=position+0.5*v*dt
    
    if non_periodic==False:
        pos[pos<=1] = n-1
        pos[pos>=n-1]= -1
        
    den = density(pos,n,m)[0]
    pot = get_pot(kernelft,den,non_periodic)
    force_x,force_y,force_z = get_forces(pot,1/n)
    #find the corresponding index in the force matrix for each particle
    bins = np.arange(0,n)
    
    ind_x = np.digitize(position[:,0],bins,right=True)
    ind_y = np.digitize(position[:,1],bins,right=True)
    ind_z = np.digitize(position[:,2],bins,right=True)
    
    fx = np.zeros(position.shape[0])
    fy = np.zeros(position.shape[0])
    fz = np.zeros(position.shape[0])
    
    for i in range(position.shape[0]):
        fx[i]=force_x[ind_x[i]-1,ind_y[i]-1,ind_z[i]-1]
        fy[i]=force_y[ind_x[i]-1,ind_y[i]-1,ind_z[i]-1]
        fz[i]=force_z[ind_x[i]-1,ind_y[i]-1,ind_z[i]-1]
    
    f = np.c_[fx,fy,fz]
    
    #update the velocity and position
    vv=v+0.5*dt*f
    position=position+dt*vv
    v=v+dt*f
    return position,v,pot,f

def get_energy(position,v,pot):
    PE = np.sum(pot)
    vx = v[:,0]
    vy = v[:,1]
    vz = v[:,2]
    KE = np.sum(0.5*(vx**2+vy**2+vz**2)*1)
    energy = PE+KE
    return energy,KE
   
def plot(bin_edges_x, bin_edges_y, x, y, field, fx, fy, plot_scatter=True, plot_quiver=True, grid=True, cmap="viridis"):
    """
    A useful function for plotting parameters for the nbody assignment
    
    Most variables should be self explanatory.
    
    field should be for example the density field, potential field, forces... Whatever
    you think would be helpful to plot :)
    
    (fx, fy) are only for if you want to quiver plot and should be the force for each particle
    arranged in the same order as (x,y)
    """
    fig,ax = plt.subplots(figsize=(10,10), dpi=100)
    minor_ticks_x = bin_edges_x
    minor_ticks_y = bin_edges_y
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.set(xlim=(minor_ticks_x[0], minor_ticks_x[-1]), ylim=(minor_ticks_x[0], minor_ticks_x[-1]))
    if grid:
        ax.grid(which='minor', alpha=0.75, color='xkcd:grey')
    if plot_scatter:
        scat = ax.scatter(x, y, color='k', s=10)
    cax = ax.imshow(
        field,
        # Extent goes (left, right, bottom, top)
        origin="lower",
        extent=(minor_ticks_x[0], minor_ticks_x[-1], minor_ticks_y[0], minor_ticks_y[-1]), 
        cmap=cmap
    )
    if plot_quiver:
        qax = ax.quiver(
            x, 
            y,
            fx, 
            fy,
            units="x",
            width=0.005*len(bin_edges_x),
            color="w"
        )
    
    plt.show()
    

# =============================================================================
# setting parameters        
# =============================================================================

#grid number
n=50
#particle number
N=1000
#mass
m=np.ones(N)
#random scattered initial position
position=n*np.random.rand(N,3)
#starting at rest
v=0*np.random.rand(N,3)
#time-step
oversamp=1
dt=0.1
#softening
soft=0.03
#gravational constant
G=1
#iterations
steps=1000
# =============================================================================
# simulation (periodic)
# =============================================================================
energy_table=np.zeros([steps*oversamp,3])
kernel = greens(n,soft,G,non_periodic=False)
kernelft=np.fft.rfftn(kernel)
# Check if the potential for one particle in 3d is correct

for i in range(steps):
    plt.clf()       
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(position[:,0],position[:,1],position[:,2], marker='o',s=5)
    ax.set_xlim(0,n)
    ax.set_ylim(0,n)
    ax.set_zlim(0,n)
    plt.show()
    
    for j in range(oversamp):
        position,v,pot,f = take_step(position,v,dt,n,kernelft,m,non_periodic=False)
        energy,KE = get_energy(position,v,pot)
    print(str((i+1)*oversamp)+' iterations. The total energy is '+str(energy))
    

