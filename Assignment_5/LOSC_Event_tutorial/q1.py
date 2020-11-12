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

def noise_model(signal):
    
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
    return N

#data loading
dataFolder = "./data"    
templates = glob.glob(dataFolder + "/*template*")
datasets  = glob.glob(dataFolder + "/*LOSC*.hdf5")
h_psd = []
h_noise= []
l_psd = []
l_noise = []

for fname in datasets:
    # Read Data
    strain,dt,utc=read_file(fname)
    #Nyquist = int(len(strain)/2+1)
    freq = np.fft.fftfreq(len(strain),dt)
    # Using hann window
    #make noise model
    #Set the strain sqaure as noise model 
    # Add the psd and noise to the list of corresponding detector
    if "H-H1" in fname:
        noise = noise_model(strain)
        h_noise.append(noise)
        
    elif "L-L1" in fname:
        noise = noise_model(strain)
        l_noise.append(noise)
    
#plot the noise model
plt.figure()
for i in range(4):
    plt.loglog(freq, h_noise[i], label=datasets[i])
plt.title("Hanford Noise Model")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Noise ASD ")
plt.legend(loc=3,prop={'size':15})
plt.savefig('Noise model of Hanford with hann windowing.png')
plt.show()

for i in range(4):
    plt.loglog(freq, l_noise[i], label=datasets[i])
plt.title("Livingston Noise Model")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Noise ASD ")
plt.legend(loc=3,prop={'size':15})
plt.savefig('Noise model of Livingston with hann windowing.png')
plt.show()

#make matched filter

def match_filter(strain, template, dt):
    #window both the signal and strain with hann window
    window = sig.get_window('hann',len(strain))
    sft = np.fft.rfft(strain*window)
    tft = np.fft.rfft(template*window)
    freq = np.fft.fftfreq(len(window),dt)
    #noise
    noise = noise_model(strain)
    # Get freq spacing
    df = freq[1]-freq[0]
    # Do the matched filter
    mf_ft = np.conj(tft)*(sft/noise)
    mf = np.fft.irfft(mf_ft)
    int = intg.cumtrapz(np.abs(mf_ft), dx=df, initial=0)
    mid_idx = np.argmin(np.abs(int - max(int)/2))
    return mf, mf_ft, mid_idx

for tname in templates:
    th, tl = read_template(tname)
    for fname in datasets:
        strain,dt,utc = read_file(fname)
        if "H-H1" in fname:
            mf_h = match_filter(strain,th,dt)[0]
            plt.plot(mf_h)
            plt.title("Hanford Strain:"+str(fname))
            plt.xlabel("Time")
            plt.ylabel("Match filter")
            plt.legend()
            plt.show()
        
        elif "L-L1" in fname:
            mf_l = match_filter(strain,tl,dt)[0]
            plt.plot(mf_h,label="Template:"+str(tname))
            plt.title("Livingston Strain::"+fname)
            plt.xlabel("Time")
            plt.ylabel("Match filter")
            plt.legend()
            plt.show()

'''
#th,tl=read_template('GW150914_4_template.hdf5')
template_name='GW150914_4_template.hdf5'
th,tl=read_template(template_name)

#windowing data and template
x=np.linspace(-1,1,len(strain))*np.pi
win=0.5+0.5*np.cos(x)
strain_windowed=win*strain
th_windowed=win*th
tl_windowed=win*tl

#FFT
dft=np.fft.rfft(strain_windowed)
th_ft=np.fft.rfft(th_windowed) # template Hanford
tl_ft=np.fft.rfft(tl_windowed) # template Livingston


#matched filter
mf_th_ft=np.conj(th_ft)*(dft/N) # template Hanford
mf_th=np.fft.irfft(mf_th_ft)

mf_tl_ft=np.conj(tl_ft)*(dft/N) # template Livingston
mf_tl=np.fft.irfft(mf_tl_ft)
'''