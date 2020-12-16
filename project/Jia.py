import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import time
import pandas
plt.rcParams['axes.prop_cycle'    ] = plt.cycler('color',['lightseagreen', 'indigo' ])
def greens(n,sr,soft=0.03,G=0.001):
    #get the potential from a particule at (0,0,0)
    #Take G to 1
    x = np.arange(n)/n
    x[n//2:] = x[n//2:] - n/n
    # x= np.arange(-n//2,n//2)/n
    xx,yy,zz = np.meshgrid(x, x, x)
    pot = np.zeros([n,n,n])
    dr=np.sqrt(xx**2+yy**2+zz**2)
    dr[dr<soft] = soft
    pot=1*G/dr
    # pot=pot-pot[n//2,n//2,n//2]  #set it so the potential at the edge goes to zero
    # pot[0,0,0]=(pot[0,1,0]+pot[1,0,0]+pot[0,0,1]+pot[-1,0,0]+pot[0,-1,0]+pot[0,0,-1])/4-0.25 #we know the Laplacian in 2D picks up rho/4 at the zero point
    # set a square place where the potential is same to set the force to 0
    # if sr != 0:
        
    #     pot[:sr,:sr,:sr]=p
    #     pot[-sr:,:sr,:sr] = p
    #     pot[:sr,-sr:,:sr] = p
    #     pot[:sr,:sr,-sr:]= p
    #     pot[:sr,-sr:,-sr:]= p
    #     pot[-sr:,-sr:,:sr]=p
    #     pot[-sr:,:sr,-sr:]=p
    #     pot[-sr:,-sr:,-sr:]=p
    return pot

# # Check if the potential for one particle in 3d is correct
# def wrap_for_image(img):
#     return np.roll(img, img.shape[0]//2, axis=(0,1))
# m =128
# pot = greens(2*m,5)
# pot2d = np.zeros([2*m,2*m])
# x = 0
# for i in range(2*m):
#     for j in range(2*m):
#         pot2d[i,j]=pot[i,j,0]       
# plt.imshow(wrap_for_image(pot2d))
# plt.colorbar()


def density(x,n,m):   
    bc,edges = np.histogramdd(x,bins=(n,n,n),range=((0,n),(0,n),(0,n)),weights=m)
    mask = np.zeros([n,n,n],dtype='bool')
    return bc, mask
def get_forces(pot,dx):
    grad = np.gradient(pot,dx)
    return grad
def get_forces_2(pot,deltax):
    dx,dy,dz = np.gradient(pot,deltax)
    return dx,dy,dz
def rho2pot(rho,kernelft):
    tmp=rho.copy()
    tmp=np.pad(tmp,(0,tmp.shape[0]))

    tmpft=np.fft.rfftn(tmp)
    tmp=np.fft.irfftn(tmpft*kernelft)
    # if len(rho.shape)==2:
    #     tmp=tmp[:rho.shape[0],:rho.shape[1]]
    #     return tmp
    if len(rho.shape)==3:
        tmp=tmp[:rho.shape[0],:rho.shape[1],:rho.shape[2]]
        return tmp
    print("error in rho2pot - unexpected number of dimensions")
    assert(1==0)

# def rho2pot_masked(rho,mask,kernelft,return_mat=False):
#     rhomat=np.zeros(mask.shape)
#     rhomat[mask]=rho
#     potmat=rho2pot(rhomat,kernelft)
#     if return_mat:
#         return potmat
#     else:
#         return potmat[mask]
def take_step(x,v,dt,n,kernelft,m):
    xx=x+0.5*v*dt
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            if xx[i,j]<= 0.5:
                xx[i,j] = n-0.5
               
            if xx[i,j]>= n-0.5:
                xx[i,j] = 0.5
               
            else:
                xx[i,j] = xx[i,j]
    den = density(xx,n,m)[0]
    pot = rho2pot(den,kernelft)
    # ff=np.asarray(get_forces(pot))
    f = np.zeros(xx.shape)
    ff = get_forces_2(pot,1/n)
    ffx = ff[0]
    ffy = ff[1]
    ffz = ff[2]
    for i in range(xx.shape[0]):
        xx_int = np.rint(xx[i])
        # fx = ff[0][int(xx_int[0]),int(xx_int[1]),int(xx_int[2])]/m[i]
        # fy = ff[1][int(xx_int[0]),int(xx_int[1]),int(xx_int[2])]/m[i]
        # fz = ff[2][int(xx_int[0]),int(xx_int[1]),int(xx_int[2])]/m[i]
        fx = ffx[int(xx_int[0]),int(xx_int[1]),int(xx_int[2])]
        fy = ffy[int(xx_int[0]),int(xx_int[1]),int(xx_int[2])]
        fz = ffz[int(xx_int[0]),int(xx_int[1]),int(xx_int[2])]
        f[i] = [fx,fy,fz]
    vv=v+0.5*dt*f
    x=x+dt*vv
    v=v+dt*f
    # print(f)
    # print("den=",den)
    # print("pot=",pot)
    # print("Force",f)
    return  f, x, v, den, pot, ff
def energy(x,v,pot,m):
    potential = np.sum(pot)
    vx= v[:,0]
    vy= v[:,1]
    vz = v[:,2]
    v_abs = vx**2+vy**2+vz**2
    kin = np.sum(0.5*m*v_abs)
    return potential+kin
# Convert x in grid to the real position
def covpos(x,n):
    x_r = x/2*n
    return x_r
# transfer the speed in grid to the real speed
def covvol(v,n):
    v_r = v/np.sqrt(2*n)
    return v_r
# grid size
n =256
# number of particles
N =100000
dt=0.1
oversample = 8
T=0

x = n*np.random.rand(N,3)
v = 0*np.random.rand(N,3)

# x= np.array([[64,64,64],[64,44,64]])
# v=np.array([[0.,0.,0],[40.47,0,0]])
# x= np.array([[16,16,16]])
# v=np.array([[0.,0.,0]])

print(x)
m=np.ones(N)
m[0]=1
kernel=greens(2*n,0)
kernelft=np.fft.rfftn(kernel)
den = density(x,n,m)[0]
pot = rho2pot(den,kernelft)
# # Check the force plot
# den = density(x,n,m)[0]
# pot = rho2pot(den,kernelft)
# nf = n
# dx,dy,dz = get_forces_2(pot,1/n)
# x = np.arange(nf)
# X, Y = np.meshgrid(x, x)
# dx2d=np.zeros([nf,nf])
# dy2d=np.zeros([nf,nf])
# dz2d=np.zeros([nf,nf])
# for i in range(nf):
#     for j in range(nf):
#           dx2d[i,j]=dx[i,j,nf//2]
#           dy2d[i,j]=dy[i,j,nf//2]
#           dz2d[i,j]=dz[i,j,nf//2]

# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# zero=np.zeros(dy.shape)
# ax2.imshow(dx[:,:,nf//2])
# ax1.quiver(Y, X,dx[:,:,nf//2],dy[:,:,nf//2])

# ax1.xaxis.set_ticks([])
# ax1.yaxis.set_ticks([])
# ax1.set_aspect('equal')
# plt.show()
# # r = take_step(x,v,dt,n,kernelft,m)
# Check if the potential for one particle in 3d is correct

# pot2d = np.zeros([nf,nf])
# x = 0
# for i in range(nf):
#     for j in range(nf):
#         pot2d[i,j]=kernel[i,j,nf//2]       
# ax1.imshow(pot2d)


# Start the simulation
fig=plt.figure()#Create 3D axes
ax=fig.add_subplot(projection="3d")
# ax.scatter(x[:,0],x[:,1],x[:,2],color='blue',marker=".",s=0.02)
# ax.set_xlim(0,n)
# ax.set_ylim(0,n)
# ax.set_zlim(0,n)
# for i in range (x.shape[0]):
    
#     ax.scatter(x[i][0],x[i][1],x[i][2],color='blue',marker=".",s=0.02)
#     ax.set_xlim(0,n)
#     ax.set_ylim(0,n)
#     ax.set_zlim(0,n)
# fig.savefig('D:/git_code/PHYS512/project/ps_c/3dinitial.png', dpi=600)
a=0
position = []
f= open(' ','ab')
ener = []
time = []

for t in range(2000):
     T=T+dt
     position.append(x)
     np.save(f,position)
     E = energy(x,v,pot,m)
     ener.append(E)
     time.append(T)
     if t%oversample==0:
         plt.cla()
         ax.scatter(x[:,0],x[:,1],x[:,2],color='blue',marker=".",s=0.02)
         ax.set_xlim(0,n)
         ax.set_ylim(0,n)
         ax.set_zlim(0,n)
         ax.set_title('Time ='+str(T)+"\nEnergy="+str(E))
         fig.savefig('D:/git_code/PHYS512/project/ps_c/'+'bla'+str(t)+'.png', dpi=600)
         
                
         plt.pause(0.001)
     np.savetxt('Energy.txt',ener)
     x_tmp= x
     v_tmp = v
     step = take_step(x_tmp,v_tmp,dt,n,kernelft,m)
     x_new = step[1]
     v_new = step[2]
     pot = step[4]
    
     x= x_new
     v = v_new
     for i in range(x.shape[0]):
         for j in range(x.shape[1]):
             if x[i,j]< 0.5:
                 x[i,j] = n-0.5
               
             if x[i,j]> n-0.5:
                 x[i,j] = 0.5
               
             else:
                 x[i,j] = x[i,j]
     print(t)

     a=a+1
     print(E)