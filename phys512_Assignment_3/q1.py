import numpy as np
import matplotlib.pyplot as plt
# new parameter is a, b/-2a, c/-2a, a(x_0**2+y_0**2)-d
# The function is z = a(x**2+y**2)+bx+cy+d

def get_A(x,y):
    A = np.zeros([len(x),4])
    # d
    A[:,0]=1
    A[:,1]=y
    A[:,2]=x
    A[:,3]=x**2+y**2
    return A
def lin_fit(A,z):
    u,s,v=np.linalg.svd(A,0)
    fitp=v.T@(np.diag(1/s)@(u.T@z))
    return fitp

#load data file
data=np.loadtxt('dish_zenith.txt')
x = data[:,0]
y = data[:,1]
z = data[:,2]
A = get_A(x,y)
fitp = lin_fit(A,z)
pred = A@fitp
e = np.std(pred-z)
# focal length in meter
f = (1/(4*fitp[3]))*1e-3