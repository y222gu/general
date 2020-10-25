import numpy as np
import matplotlib.pyplot as plt

def mydft(k_sin,y):
    N = len(y)
    x=np.arange(N)
    kvec = np.fft.fftfreq(N,1/N) 
    F = np.zeros(N)
    for i in range(N):
        F_i= np.sum(np.exp(-2*np.pi*1j*(kvec[i]+k_sin)*x/N)/2j-np.exp(-2*np.pi*1j*(kvec[i]-k_sin)*x/N)/2j)
        F[i] = abs(F_i)
    return F

#part c

N=1024
x=np.arange(N)
k =15.4
y=np.sin(2*np.pi*x*k/N)
yft1 = np.fft.fft(y)
yft2= mydft(k,y)


delta = np.zeros(len(yft2))
delta[15]=512 #N/2=512
delta[1011]=512 #N/2=512


fig1, ax1  =plt.subplots()
plt.plot(abs(yft1),'.',label = "numpy fft")
plt.plot(abs(yft2),'.',label = "my dft")
plt.plot(abs(delta),'.',label = "delta function")
plt.legend()
plt.savefig('specctral leakage.png')
plt.show()
print("Error between my written DFT and numpy FFT=",np.std(np.abs(yft2)-np.abs(yft1)))
print("Error between my written DFT and delta function=",np.std(np.abs(yft2)-np.abs(delta)))
# part d

N=1024
x=np.arange(N)
xx=np.linspace(0,1,N)*2*np.pi
win=0.5-0.5*np.cos(xx)

k=15.4
y1=np.sin(2*np.pi*x*k/N)
y2=np.sin(2*np.pi*x*k/N)*win
yft1 = np.fft.fft(y1)
yft2 = np.fft.fft(y2)

plt.plot(abs(yft1),'.',label="unwindowed FFT")
plt.plot(abs(yft2),'.',label="windowed FFT")
plt.plot(abs(delta),'.',label="Delta function")
plt.legend()
plt.savefig('windowed FFT.png')
plt.show()
print("Error between delta and FFT without window",np.std(np.abs(yft1)-np.abs(delta)))
print("Error between delta and FFT with window", np.std(np.abs(yft2)-np.abs(delta)))
