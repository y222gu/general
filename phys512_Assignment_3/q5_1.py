import numpy as np
import matplotlib.pyplot as plt
import camb

def get_spectrum(pars,y,tau=0.0544,lmax=2000):
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=tau
    As=pars[3]
    ns=pars[4]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    pred = cmb[2:len(y)+2,0]
    return pred

def get_spectrum_full(pars,y,lmax=2000):
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
    pred = cmb[2:len(y)+2,0]
    return pred

def our_chisq(data,pars):
    #we need a function that calculates chi^2 for us for the MCMC
    #routine to call
    y=data[:,1]
    noise=data[:,2]
    cmb_model=get_spectrum_full(pars,y)
    chisq=np.sum((y-cmb_model)**2/noise**2)
    return chisq

def cov_step(covmat):
    r = np.linalg.cholesky(covmat)
    step = np.squeeze(np.asarray(r@ np.random.randn(r.shape[0])))
    return step

def num_deriv(get_spectrum,y,pars,par_step):
    derivs=np.zeros([len(y),len(pars)])
    for i in range(len(pars)):
        pars2=pars.copy()
        
        pars2[i]=pars2[i]+par_step[i]
        f_right=get_spectrum(pars2,y)
        
        pars2[i]=pars[i]-par_step[i]
        f_left=get_spectrum(pars2,y)

        derivs[:,i]=(f_right-f_left)/(2*par_step[i])
    return derivs

def run_chain_corr(pars,data,par_step,our_chisq,nstep=5000):

    npar=len(pars)
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)  
    chi_cur=our_chisq(data,pars)
    for i in range(nstep):
        
        pars_trial=pars+0.1*par_step*np.random.randn(npar)
        tau = pars_trial[3]
        
# set tau to prior with 3 sigma
        while tau< 0.0544-3*0.0073 or tau > 0.0544+3*0.0073:
            pars_trial=pars+0.1*par_step*np.random.randn(npar)
            tau = pars_trial[3]
            
        chi_trial=our_chisq(data,pars_trial)
        accept_prob=np.exp(-0.5*(chi_trial-chi_cur))
        
        if np.random.rand(1)<accept_prob: #accept the step with appropriate probability
            pars=pars_trial
            chi_cur=chi_trial
        chain[i,:]=pars
        chivec[i]=chi_cur
        
        np.savetxt("chain_with_prior.txt",chain)
        np.savetxt("chivec_with_prior.txt",chivec)
    return chain,chivec

wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
y=wmap[:,1]
noise=np.array(wmap[:,2])
Ninv=np.eye(len(y))/noise**2

pars=np.asarray([65,0.02,0.1,2e-9,0.96])
par_step=pars/10000
tau = 0.0544

#fixed tau and get the other parameters  
for i in range(10):
    model=get_spectrum(pars,y)
    derivs=num_deriv(get_spectrum,y,pars,par_step)
    resid=y-model
    lhs=derivs.T@Ninv@derivs
    rhs=derivs.T@Ninv@resid
    lhs_inv=np.linalg.inv(lhs)
    step=lhs_inv@rhs
    pars=pars+step
    print(pars)
par_sigs=np.sqrt(np.diag(lhs_inv))
print('When fix tau, the other parameters are ',pars,' with errors ',par_sigs)

#  get the full list of all parameters
# pars_better = pars
pars = np.insert(pars,3,tau)
par_step=pars/1000

#get parameter step size for MCMC
grad = np.matrix(num_deriv(get_spectrum_full,y,pars,par_step))
covmat= np.linalg.inv(grad.T* np.diag(1.0/wmap[:,2]**2)*grad)
par_step=cov_step(covmat)

#run chain
chain,chivec=run_chain_corr(pars,wmap,par_step,our_chisq,10000)

#plot
n = np.linspace(1,10000,10000)
fig1,ax1 = plt.subplots()
paras = np.mean(chain,axis=0)
error = np.std(chain,axis=0)
for i in range(6):
    ax1.plot(n,chain[:,i],label=("chain"+str(i)))

