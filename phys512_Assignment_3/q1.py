import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('dish_zenith.txt')
#part a)
# The function is z = A0 (x2 + y2) + A1 x + A2 y + A3
def get_A(x,y):
    A = np.zeros([len(x),4])
    A[:,0]=x**2+y**2
    A[:,1]=x
    A[:,2]=y
    A[:,3]=1
    return A

def fitp(A,z):
    u,s,v=np.linalg.svd(A,0)
    pars=v.T@(np.diag(1/s)@(u.T@z))
    return pars
#part b)
x = data[:,0]
y = data[:,1]
z = data[:,2]

A = get_A(x,y)
pars = fitp(A,z)

#part c)
# focal length
zz = A@pars

#what is the noise on the data points?
rms=np.std(zz-z)
N=rms**2
chisq=np.sum((zz-z)**2)/N**2
print("RMS scatter about model is ",rms)

Ninv=np.eye(len(zz))/N
lhs=A.T@Ninv@A
errs=np.sqrt(np.diag(np.linalg.inv(lhs)))
for i in range(len(pars)):
    print('paramter ',i,' has value ',pars[i],' and error ',errs[i])

f = (1/(4*pars[0]))*1e-3    
print('The focal length is ', f ,'meters')