import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def greens(n,soft,G):
    # n is the number of grids
    n = n*2
    x = np.arange(n)
    x[n//2:] = x[n//2:] - n
    xmat,ymat,zmat = np.meshgrid(x, x, x)
    kernel = np.zeros([n,n,n])
    dr=np.sqrt(xmat**2+ymat**2+zmat**2+soft**2)
    kernel=1*G/dr
    return kernel

def density(points,n,m):   
    grid_min = -n/2
    grid_max = n/2
    #weights = np.ones(len(points[0]))
    den, edges = np.histogramdd(points,bins=n,range=((grid_min, grid_max), (grid_min, grid_max), (grid_min, grid_max)),weights=m)
    mask = np.zeros([n,n,n],dtype='bool')
    return den, mask

def get_pot(kernel,den):
    kernelft = np.fft.rfftn(kernel)
    
    pot=den.copy()
    pot=np.pad(pot,(0,pot.shape[0]))

    potft=np.fft.rfftn(pot)
    potft=np.fft.irfftn(potft*kernelft)
    if len(pot.shape)==3:
        pot=pot[:den.shape[0],:den.shape[1],:den.shape[2]]
        return pot
    print("error in rho2pot - unexpected number of dimensions")
    assert(1==0)

def get_forces(pot,dx):
    force = np.gradient(pot,dx)
    force_x = np.asarray(force[0])
    force_y = np.asarray(force[1])
    force_z = np.asarray(force[2])
    return force_x,force_y,force_z

def take_step(position,v,dt,n,kernel,m):
    pos=position+0.5*v*dt
    den = density(pos,n,m)[0]
    pot = get_pot(kernel,den)
    force_x,force_y,force_z = get_forces(pot,1/n)
    #find the corresponding index in the force matrix for each particle
    bins = np.arange(0,n+1)
    
    ind_x = np.digitize(position[:,0],bins,right=True)
    ind_y = np.digitize(position[:,1],bins,right=True)
    ind_z = np.digitize(position[:,2],bins,right=True)
    
    fx = np.zeros(position.shape[0])
    fy = np.zeros(position.shape[0])
    fz = np.zeros(position.shape[0])
    
    for i in range(position.shape[0]):
        fx[i]=force_x[ind_x[i],ind_y[i],ind_z[i]]
        fy[i]=force_y[ind_x[i],ind_y[i],ind_z[i]]
        fz[i]=force_z[ind_x[i],ind_y[i],ind_z[i]]
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
    return energy
    
# =============================================================================
# setting parameters        
# =============================================================================

#grid number
n=256
#particle number
N=100
#mass
m=np.ones(N)
#random scattered initial position
position=n*np.random.rand(N,3)
#starting at rest
v=0*np.random.rand(N,3)
#time-step
oversamp=5
dt=0.01
#softening
soft=0.03
#gravational constant
G=1.0

# =============================================================================
# simulation
# =============================================================================
kernel = greens(n,soft,G)

for i in range(5):
    for j in range(oversamp):
        position,v,pot,f = take_step(position,v,dt,n,kernel,m)
        energy = get_energy(position,v,pot)
        #print(kin,pot,kin-pot/2)
    print(str(i+1)+'iterations. The total energy is'+str(energy))
    plt.clf()       
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(position[:,0],position[:,1],position[:,2], marker='o',s=1)
    plt.show()
    plt.pause(0.1)
