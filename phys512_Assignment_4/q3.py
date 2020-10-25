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

def conv(f,g):
    ft_f=np.fft.fft(f)
    ft_g=np.fft.fft(g)
    ft_g=np.conj(ft_g)
    convolution_before_shift=np.fft.ifft(ft_f*ft_g)
    convolution=np.fft.fftshift(convolution_before_shift)
    return convolution, convolution_before_shift

x=np.arange(-10,10,0.1)
y=np.exp(-0.5*x**2)
dx=10.0

y_shift=shift(y,dx)
convolution,convolution_before_shift=conv(y_shift,y)

plt.plot(x,y,label='original fuction')
plt.plot(x,y_shift,'r',label='shifted function')
plt.plot(x,convolution,label='convolution')
plt.legend()
plt.savefig('convolution with shifted gaussian.png')
plt.show()

plt.plot(x,y,label='original fuction')
plt.plot(x,y_shift,'r',label='shifted function')
plt.plot(x,convolution_before_shift,label='convolution')
plt.legend()
plt.savefig('convolution with shifted gaussian_1.png')
plt.show()

