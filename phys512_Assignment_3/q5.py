import numpy as np
import camb
from matplotlib import pyplot as plt

def dofft(y):
    f = np.fft.rfft(y)
    x = np.arange(f.size)
    return x[1:], f[1:]

chain1 = np.loadtxt("chain_noprior.txt")
chivec1 = np.loadtxt("chivec_noprior.txt")
chain2 = np.loadtxt("chain_withprior.txt")
chivec2 = np.loadtxt("chivec_withprior.txt")
n = np.linspace(1,10000,10000)
scale = [1e1,1e-2,1e-1,1e-1,1e-9,1e-1]
fig1,ax1 = plt.subplots(2,1)
chain1_rescale = chain1/scale
chain2_rescale = chain2/scale
mean1 = np.mean(chain1,axis=0)
sigma1 = np.std(chain1,axis=0)
mean2 = np.mean(chain2,axis=0)
sigma2 = np.std(chain2,axis=0)
print("No prior tau, pararmeters = "+str(mean1)+ "Error in parameters= "+str(sigma1))

for i in range(6):
    ax1[0].plot(n,chain1_rescale[:,i],label = "parameter number = "+str(i))
    # ax1[1].loglog(n,chain1_rescale[:,i],label = "parameter number = "+str(i))
    fft_x, fft_y = dofft(chain1_rescale[:,i])
    ax1[1].plot(fft_x,fft_y,label = "parameter number = "+str(i))
plt.legend()
plt.show()
# With prior tau
fig2,ax2 = plt.subplots(2,1)
print("With prior tau, pararmeters = "+str(mean2)+ "Error in parameters= "+str(sigma2))

for i in range(6):
    ax2[0].plot(n,chain2_rescale[:,i],label = "parameter number = "+str(i))
    # ax2[1].loglog(n,chain2_rescale[:,i],label = "parameter number = "+str(i))
    fft_x, fft_y = dofft(chain2_rescale[:,i])
    ax2[1].plot(fft_x,fft_y,label = "parameter number = "+str(i))
plt.legend()
plt.show()    
    
# Important Sampling
# get weight vector
wtvec=np.exp(-0.5*((chain1[:,3]-0.0544)/0.0073)**2)
chain1_scat=chain1.copy()
means=np.zeros(chain1.shape[1])
chain1_errs=np.zeros(chain1.shape[1])
for i in range(chain1.shape[1]):
    # weight the parameters
    means[i]=np.sum(wtvec*chain1[:,i])/np.sum(wtvec)
    #subtract the mean from the warm chain so we can calculate the
    #standard deviation
    chain1_scat[:,i]=chain1_scat[:,i]-means[i]
    chain1_errs[i]=np.sqrt(np.sum(chain1_scat[:,i]**2*wtvec)/np.sum(wtvec))

print("With important sample by tau, pararmeters = "+str(means)+ "Error in parameters= "+str(chain1_errs))