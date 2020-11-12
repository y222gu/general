import numpy as np
from matplotlib import pyplot as plt

def conv(f,g):
    ft_f=np.fft.fft(f)
    ft_g=np.fft.fft(g)
    ft_g=np.conj(ft_g)
    convolution_before_shift=np.fft.ifft(ft_f*ft_g)
    convolution=np.fft.fftshift(convolution_before_shift)
    return convolution, convolution_before_shift

x=np.arange(-10,10,0.1)
y=np.exp(-0.5*x**2)
convolution,convolution_before_shift=conv(y,y)

plt.plot(x,y,label='gaussian fuction')
plt.plot(x,convolution,label='convolution with itself')
plt.legend()
plt.savefig('convolution with gaussian itself.png')
plt.show()

plt.plot(x,y,label='gaussian fuction')
plt.plot(x,convolution_before_shift,label='convolution with itself')
plt.legend()
plt.savefig('convolution with gaussian itself_1.png')
plt.show()

