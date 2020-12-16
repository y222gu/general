import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
import scipy.signal as sig
import scipy.integrate as intg
import scipy.interpolate as intp
import scipy.ndimage as ndimage

plt.ion()

def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl

def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]
 
    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc

def smooth(a,n):
    vec = np.zeros(len(a))
    vec[:n]=1
    vec[-n:]=1
    fft1 = np.fft.rfft(a)
    fft2 = np.fft.rfft(vec)
    return np.fft.irfft((fft1*fft2),len(a))

def noise_model(signal,smooth_factor):
    #I also tried welch's method but it couldn't work
    #split the each signal into 8 segments
    #segment_length=len(signal)/8
    # Using Welch's Method to get a smoothed psd
    # The window chose here is hann
    #freqs, psd = sig.welch(signal,window="hann",nperseg=segment_length)
    
    # Using hann windowing to smooth the signal
    window = sig.get_window('hann',len(signal))
    signal_win = signal*window
    sft = np.fft.rfft(signal_win)
    N = np.abs(sft)**2
    N_smooth=smooth(N,smooth_factor)
    N = np.maximum(N,N_smooth)
    return N

def match_filter(strain, template, dt,smooth_factor):
    #window both the signal and strain with hann window
    window = sig.get_window('hann',len(strain))
    sft = np.fft.rfft(strain*window)
    tft = np.fft.rfft(th*window)
    freq = np.fft.fftfreq(len(window),dt)
    #noise
    noise = noise_model(strain,smooth_factor)
    # Get freq spacing
    df = freq[1]-freq[0]
    # Do the matched filter
    mf_ft = np.conj(tft)*(sft/noise)
    mf = np.fft.irfft(mf_ft)
    int = intg.cumtrapz(np.abs(mf_ft), dx=df, initial=0)
    mid_idx = np.argmin(np.abs(int - max(int)/2))
    return mf, mf_ft, mid_idx, noise

def SNR_scatter(mf):
    SNR = np.max(np.abs(mf))/np.std(mf)
    return SNR

def SNR_analytic(template,noise,mf):
   
#  fft of template
    t_ft = np.fft.rfft(template)
    rhs = mf
    #  calcualte sigma
    lhs_ft= np.conj(t_ft)*(t_ft/noise)
    lhs= np.fft.irfft(lhs_ft)
    N = np.sqrt(np.abs(lhs))
#  get the siganl to noise ratio
    return np.max(np.abs(rhs/N))

#Part A
#data loading
dataFolder = "./data"    
templates = glob.glob(dataFolder + "/*template*")
datasets  = glob.glob(dataFolder + "/*LOSC*.hdf5")
noise_smooth_test=[]
L_data=[]
H_data=[]
h_psd = []
h_noise= []
l_psd = []
l_noise = []

#find the smooth factor for the noise model
test_smooth_factor=[1,5,10,15,20]
strain,dt,utc=read_file(datasets[0])
for smooth_factor in test_smooth_factor:
    noise= noise_model(strain,smooth_factor)
    noise_smooth_test.append(noise)

plt.figure()
for i in range(5):
    plt.loglog(noise_smooth_test[i], label='smooth_factor='+str(test_smooth_factor[i]))
plt.title("Smooth Factor Test")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Noise ASD ")
plt.legend(loc=3,prop={'size':8})
plt.savefig('Smooth Factor Test for Noise model.png')
plt.show()

#get the noise model
#from the last section, the smooth factor is decided to be 5
smooth_factor=5
for fname in datasets:
    # Read Data
    strain,dt,utc=read_file(fname)
    #Nyquist = int(len(strain)/2+1)
    #freq = np.fft.fftfreq(len(strain),dt)
    # Add the psd and noise to the list of corresponding detector
    if "H-H1" in fname:
        noise= noise_model(strain,smooth_factor)
        h_noise.append(noise)
        
    elif "L-L1" in fname:
        noise= noise_model(strain,smooth_factor)
        l_noise.append(noise)
    
#plot the noise model
plt.figure()
for i in range(4):
    plt.loglog(h_noise[i], label=datasets[i])
plt.title("Hanford Noise Model")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Noise ASD ")
plt.legend(loc=3,prop={'size':8})
plt.savefig('Noise model of Hanford with hann windowing.png')
plt.show()

for i in range(4):
    plt.loglog(l_noise[i], label=datasets[i])
plt.title("Livingston Noise Model")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Noise ASD ")
plt.legend(loc=3,prop={'size':8})
plt.savefig('Noise model of Livingston with hann windowing.png')
plt.show()

#Part B
# sort the datasets
for fname in datasets:
    if "H-H1" in fname:
        H_data.append(fname)
    elif "L-L1" in fname:
        L_data.append(fname)

#find the every events in every data set
for tname in templates:
    th, tl = read_template(tname)
    for i in range(4):
        strain,dt,utc = read_file(H_data[i])
        mf_h = match_filter(strain,th,dt,smooth_factor)[0]
        midfreq_h = match_filter(strain,th,dt,smooth_factor)[2]
        noise_h = match_filter(strain,th,dt,smooth_factor)[3]
        
        strain,dt,utc = read_file(L_data[i])
        mf_l = match_filter(strain,th,dt,smooth_factor)[0]
        midfreq_l = match_filter(strain,tl,dt,smooth_factor)[2]
        noise_l = match_filter(strain,tl,dt,smooth_factor)[3]

        plt.plot(mf_h,label=H_data[i])
        plt.plot(mf_l,label=L_data[i])
        plt.title("Template:"+tname)
        plt.xlabel("Time")
        plt.ylabel("Match filter")
        plt.legend()
        
#Part C 
#calculate the SNR
        SNR_scatter_h = SNR_scatter(mf_h)
        SNR_scatter_l= SNR_scatter(mf_l)
        print('The SNR of Hanford from scatter:'+str(SNR_scatter_h))
        print('The SNR of Livingston from scatter:'+str(SNR_scatter_l))
        print('The SNR of combined Hanford and Livingston from scatter:'+str(np.sqrt(SNR_scatter_h**2+SNR_scatter_l**2)))

#Part D 
        SNR_analytic_h = SNR_analytic(th,noise_h,mf_h)  
        SNR_analytic_l = SNR_analytic(tl,noise_l,mf_l)
        print('The analytic SNR of Hanford:'+str(SNR_analytic_h))
        print('The analytic SNR of Livingston with noise model :'+str(SNR_analytic_l))
        print('The analytic SNR of combined Hanford and Livingston with noise model:'+str(np.sqrt(SNR_analytic_h**2+SNR_analytic_l**2)))

#Part E

        print("The Mid frequency of Hanford:"+str(midfreq_h))
        print("The Mid frequency of Liveingston:"+str(midfreq_l))
        
        if SNR_scatter_h>9:
            dataname=H_data[i]
            plt.savefig('Template'+tname[7:-6]+'with data'+dataname[13:-6])
        plt.show()