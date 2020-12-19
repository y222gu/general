import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

def greens(n,soft,G,non_periodic):
    # n is the number of grids
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
    mask = np.zeros([n,n,n],dtype='bool')
    return den, mask

def get_pot(kernelft,den,non_periodic):
    
    pot=den.copy()
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
    force_x[abs(force_x)<1e-10]=0
    force_y = np.asarray(force[1])
    force_y[abs(force_y)<1e-10]=0
    force_z = np.asarray(force[2])
    force_z[abs(force_z)<1e-10]=0
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
   
# =============================================================================
# setting parameters        
# =============================================================================
#grid number
n=50
#particle number
N=2
#mass
m=np.ones(N)
#initial position
position = np.zeros((N,3))
position[0,0]=n*0.4
position[1,0]=n*0.6
position[:,1]=n/2
position[:,2]=n/2

#initial velocity
init_v=0.2#0.2
v=np.zeros((N,3))
v[:,1]=0
v[:,0]=0
v[0,2]=-init_v
v[1,2]=init_v
#time-step
oversamp=10
dt=0.1#0.04
#softening
soft=0.3
#gravational constant
G=0.5
#iterations
steps=100
#boundary condition
non_periodic=False
# =============================================================================
# simulation 
# =============================================================================
energy_table=np.zeros([steps*oversamp,3])
kernel = greens(n,soft,G,non_periodic=non_periodic)
kernel=np.fft.fftshift(kernel)
kernelft=np.fft.rfftn(kernel)

# Check if the potential for one particle in 3d is correct
x_1=np.zeros(steps*oversamp)
y_1=np.zeros(steps*oversamp)
z_1=np.zeros(steps*oversamp)
x_2=np.zeros(steps*oversamp)
y_2=np.zeros(steps*oversamp)
z_2=np.zeros(steps*oversamp)
for i in range(steps):
    
    plt.clf()       
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(position[:,0],position[:,1],position[:,2], marker='o',s=15)
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
plt.title('energy tracking')
plt.xlabel('steps')
plt.ylabel('energy')
#plt.savefig('Two orbital total energy tracking.png')
plt.show()
#np.savetxt('Two orbital energy.txt',energy_table)

# Data for a three-dimensional line
plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_1, y_1, z_1, 'r',label='particle 1')
ax.plot3D(x_2, y_2, z_2, 'g',label='particle 2')
ax.set_xlabel('x grid')
ax.set_ylabel('y grid')
ax.set_zlabel('z grid')
ax.set_xlim(n/4, 3*n/4)
ax.set_ylim(n/4, 3*n/4)
ax.set_zlim(n/4, 3*n/4)
ax.legend()
ax.set_title('Orbital path of 2 particles')
#plt.savefig('Two orbital path.png')

         