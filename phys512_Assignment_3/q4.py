import numpy as np
import camb
from matplotlib import pyplot as plt
import time

def get_spectrum(pars,y,lmax=2000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[2:len(y)+2,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt

def our_chisq(data,pars):
    #we need a function that calculates chi^2 for us for the MCMC
    #routine to call
    y=data[:,1]
    noise=data[:,2]
    cmb_model=get_spectrum(pars,y)
    chisq=np.sum((y-cmb_model)**2/noise**2)
    return chisq

def num_deriv(get_spectrum,y,pars,par_step):
    #calculate numerical derivatives of 
    #a function for use in e.g. Newton's method or LM
    derivs=np.zeros([len(y),len(pars)])
    for i in range(len(pars)):
        pars2=pars.copy()
        
        pars2[i]=pars2[i]+par_step[i]
        f_right=get_spectrum(pars2,y)
        
        pars2[i]=pars[i]-par_step[i]
        f_left=get_spectrum(pars2,y)

        derivs[:,i]=(f_right-f_left)/(2*par_step[i])
    return derivs

def run_mcmc(pars,data,par_step,our_chisq,nstep=5000):
    npar=len(pars)
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)  
    chi_cur=our_chisq(data,pars)
    
    for i in range(nstep):
        pars_trial=pars+par_step*np.random.randn(npar)
        while pars_trial[3]<=0:
            pars_trial = pars+par_step*np.random.randn(npar)    
        
        chi_trial=our_chisq(data,pars_trial)
        #we now have chi^2 at our current location
        #and chi^2 in our trial location. decide if we take the step
        accept_prob=np.exp(-0.5*(chi_trial-chi_cur))
        if np.random.rand(1)<accept_prob: #accept the step with appropriate probability
            pars=pars_trial
            chi_cur=chi_trial
        chain[i,:]=pars
        chivec[i]=chi_cur
        np.savetxt('chain.txt', chain)
        np.savetxt('chi.txt', chivec)
        print('mcmc',i,'step')
    return chain,chivec

def cov_step(covmat):
    r = np.linalg.cholesky(covmat)
    #step = np.dot(r,np.random.randn(r.shape[0]))
    step = np.squeeze(np.asarray(r@ np.random.randn(r.shape[0])))
    return step


#MCMC
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
y=wmap[:,1]
noise=np.array(wmap[:,2])
Ninv=np.eye(len(y))/noise**2

pars=np.asarray([65,0.02,0.1,0.05,2e-9,0.96])
par_step=pars/100
grad = np.matrix(num_deriv(get_spectrum,y,pars,par_step))
covmat= np.linalg.inv(grad.T* np.diag(1.0/wmap[:,2]**2)*grad)

par_step=cov_step(covmat)

chain,chivec=run_mcmc(pars,wmap,par_step,our_chisq,nstep=10000)
par_sigs=np.std(chain,axis=0)
par_means=np.mean(chain,axis=0)
chisq=our_chisq(wmap,par_means)

print('Fitted parameters are',par_means)
print('Their errors are',par_sigs)
print('The chi square is ',chisq)  

plt.ion()
plt.clf();
plt.plot(wmap[:,0],wmap[:,1],'.')
cmb=get_spectrum(par_means,y)
plt.plot(cmb)
plt.savefig('correlated MCMC.png')