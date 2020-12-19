import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import time
def greens(n,soft,G,non_periodic):
    # n is the number of grids
    if non_periodic:
        n=2*n
    x=np.arange(n)/n
    x[n//2:] = x[n//2:]-n/n
    xmat,ymat,zmat = np.meshgrid(x, x, x)
    kernel = np.zeros([n,n,n])
    dr=np.sqrt(xmat**2+ymat**2+zmat**2+soft**2)
    kernel=1*G/(4*np.pi*dr)
    return kernel

def density(position,n,m):   
    grid_min = 0
    grid_max = n
    den, edges = np.histogramdd(position,bins=(n,n,n),range=((grid_min, grid_max), (grid_min, grid_max), (grid_min, grid_max)),weights=m)
    return den, edges

def get_pot(kernelft,den,non_periodic):
    
    pot=den.copy()
    if non_periodic:
        pot=np.pad(pot,(0,pot.shape[0]))
    
    potft=np.fft.rfftn(pot)
    pot=np.fft.irfftn(potft*kernelft)
    pot=np.fft.fftshift(pot)
    if len(den.shape)==3:
        pot=pot[:den.shape[0],:den.shape[1],:den.shape[2]]
    return pot
    print("error in get_pot - unexpected number of dimensions")
    assert(1==0)


def get_forces(pot,n):
    force = np.gradient(pot,1/n)
    force_x = np.asarray(force[0])   
    force_y = np.asarray(force[1])
    force_z = np.asarray(force[2])

    return force_x,force_y,force_z

def take_step(position,v,dt,n,kernelft,m,non_periodic):
    pos=position+0.5*v*dt

    if non_periodic==False:
        pos[pos<=0] = pos[pos<=0]%n
        pos[pos>=n]= pos[pos>=n]%n

    den = density(pos,n,m)[0]
    pot = get_pot(kernelft,den,non_periodic)
    force_x,force_y,force_z = get_forces(pot,n)
    
    #find the corresponding index in the force matrix for each particle
    bins = np.arange(0,n)
    f = np.zeros(position.shape)
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
    if non_periodic==False:
        position[position<=0] = position[position<=0]%n
        position[position>=n]= position[position>=n]%n
    v=v+dt*f
    return position,v,pot,f

def get_energy(position,v,pot,m):
    PE = -0.5*np.sum(pot)
    vx = v[:,0]
    vy = v[:,1]
    vz = v[:,2]
    KE = np.sum(0.5*(vx**2+vy**2+vz**2)*m)
    energy = PE+KE
    return energy,PE,KE

def k3_mass(den,edges,position,n,soft):
    
    x=np.arange(n)
    xmat,ymat,zmat = np.meshgrid(x, x, x)
    k = np.zeros([n,n,n])
    dr=np.sqrt(xmat**2+ymat**2+zmat**2+soft**2)
    k=1/dr**3
    
    new_den =np.fft.fftshift(np.fft.irfftn(k))
    
    ind_x = np.digitize(position[:,0],edges[0])-1
    ind_y = np.digitize(position[:,1],edges[1])-1
    ind_z = np.digitize(position[:,2],edges[2])-1    
    
    new_m = new_den[ind_x,ind_y,ind_z]
    m_min = np.min(new_m)
    
    return new_m, m_min
   
# =============================================================================
# setting parameters        
# =============================================================================

#grid number
n=50
#particle number
N=100000
#mass
m=np.ones(N)

#random scattered initial position
position=n*np.random.rand(N,3)
#starting at rest
v=0*np.random.rand(N,3)
#time-step
oversamp=20
dt=0.01#0.04/0.01
#softening
soft=0.03
#gravational constant
G=0.001#0.01
#iterations
steps=50
#boundary condition
non_periodic=False

# =============================================================================
# simulation 
# =============================================================================
energy_table=np.zeros([steps*oversamp,3])
kernel = greens(n,soft,G,non_periodic=non_periodic)
kernel=np.fft.fftshift(kernel)
kernelft=np.fft.rfftn(kernel)

#derive masses from a realization of k^-3
den,edges = density(position, n, m)
new_m, m_min = k3_mass(den,edges,position,n,0.5) 
m = new_m/m_min

# Check if the potential for one particle in 3d is correct
x_1=np.zeros(steps*oversamp)
y_1=np.zeros(steps*oversamp)
z_1=np.zeros(steps*oversamp)
x_2=np.zeros(steps*oversamp)
y_2=np.zeros(steps*oversamp)
z_2=np.zeros(steps*oversamp)

t_1 = time.time()
for i in range(steps):
    t_2 = time.time()
    plt.clf()       
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(position[:,0],position[:,1],position[:,2], marker='o',s=0.05)
    ax.set_title(str(i*oversamp)+" steps. Time: "+ str(t_2-t_1))
    ax.set_xlim(0,n)
    ax.set_ylim(0,n)
    ax.set_zlim(0,n)
    plt.show()
    plt.clf() 
    
    for j in range(oversamp):
        position,v,pot,f = take_step(position,v,dt,n,kernelft,m,non_periodic=non_periodic)
        energy,PE,KE = get_energy(position,v,pot,m)
        energy_table[i*oversamp+j]=[energy,PE,KE]
        x_1[i*oversamp+j]=position[0,0]
        y_1[i*oversamp+j]=position[0,1]
        z_1[i*oversamp+j]=position[0,2]
        x_2[i*oversamp+j]=position[1,0]
        y_2[i*oversamp+j]=position[1,1]
        z_2[i*oversamp+j]=position[1,2]
    print(str((i+1)*oversamp)+' iterations. The total energy is '+str(energy))

plt.figure()
plt.plot(energy_table[:,0],'k',label='total energy')
plt.legend()
plt.title('nbody energy tracking')
plt.xlabel('steps')
plt.ylabel('energy')
plt.savefig('nbody energy tracking.png')
plt.show()
np.savetxt('nbody energy.txt',energy_table)

plt.figure()
plt.hist(m,bins=100)
plt.title('particle masses')
plt.xlabel('Mass')
plt.ylabel('Frequency')
plt.savefig('mass histogram.png')
plt.show()