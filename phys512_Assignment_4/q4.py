import numpy as np
from matplotlib import pyplot as plt

x=np.arange(1000)
tau=50
nhit=200
x_hit=np.asarray(np.floor(len(x)*np.random.rand(nhit)),dtype='int')
y_hit=np.random.rand(nhit)**2

# NO padding, wrapi-around with danger
f1=0.0*x
for i in range(nhit):
    f1[x_hit[i]]=f1[x_hit[i]]+y_hit[i]
g1=np.exp(-1.0*x/tau)
T1=np.fft.irfft(np.fft.rfft(g1)*np.fft.rfft(f1))
T1=T1[:len(x)]

#Padding, wrap-around without danger
f2=0.0*x
f2=np.pad(f2,[0,len(f2)])
for i in range(nhit):
    f2[x_hit[i]]=f2[x_hit[i]]+y_hit[i]
g2=np.exp(-1.0*x/tau)
g2=np.pad(g2,[0,len(g2)])
T2=np.fft.irfft(np.fft.rfft(g2)*np.fft.rfft(f2))
T2=T2[:len(x)]

plt.plot(x,T1,label='No padding')
plt.plot(x,T2,label='With padding')
plt.legend()
plt.savefig('circulant nature.png')
plt.show()
