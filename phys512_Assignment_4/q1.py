import numpy as np
from matplotlib import pyplot as plt

def shift(y,dx):
    N=y.size
    kvec=np.arange(N)
    yft=np.fft.fft(y)
    J=np.complex(0,1)
    yft_shift=yft*np.exp((-2*np.pi*J*kvec*dx)/N)
    y_shift=np.fft.ifft(yft_shift)
    return y_shift

x=np.arange(-10,10,0.1)
y=np.exp(-0.5*x**2)
dx=100.0
y_shift=shift(y,dx)

plt.plot(x,y,label='original fuction')
plt.plot(x,y_shift,'r',label='shifted function')
plt.legend()
plt.savefig('shift gaussian.png')
plt.show()

