import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import scipy.signal as sig
import scipy.integrate as intg
''' The following is just cosmectic change'''
plt.rcParams['mathtext.fontset'   ] = 'cm' 
plt.rcParams['font.sans-serif'    ] = 'Arial'
plt.rcParams['figure.figsize'     ] = 10, 8
plt.rcParams['font.size'          ] = 19
plt.rcParams['lines.linewidth'    ] = 2
plt.rcParams['xtick.major.width'  ] = 2
plt.rcParams['ytick.major.width'  ] = 2
plt.rcParams['xtick.major.pad'    ] = 4
plt.rcParams['ytick.major.pad'    ] = 4
plt.rcParams['xtick.major.size'   ] = 10
plt.rcParams['ytick.major.size'   ] = 10
plt.rcParams['axes.linewidth'     ] = 2
plt.rcParams['patch.linewidth'    ] = 0
plt.rcParams['legend.fontsize'    ] = 15
plt.rcParams['xtick.direction'    ] = 'in'
plt.rcParams['ytick.direction'    ] = 'in'
plt.rcParams['ytick.right'        ] = True
plt.rcParams['xtick.top'          ] = True
plt.rcParams['xtick.minor.width'  ] = 1
plt.rcParams['xtick.minor.size'   ] = 4
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.minor.width'  ] = 1
plt.rcParams['ytick.minor.size'   ] = 4
plt.rcParams['axes.labelpad'      ] = 0
plt.rcParams['axes.prop_cycle'    ] = plt.cycler('color',['lightseagreen', 'indigo', 'dodgerblue', 'sandybrown', 'brown', 'coral', 'pink' ])
# Conlvolve function
def cov(f,g):
    fft1 = np.fft.rfft(f)
    fft2 = np.fft.rfft(g)
    return np.fft.irfft((fft1*fft2),len(f))
# Define the function to smooth data 
def smooth(a,n):
    win = sig.get_window('boxcar',n)
    vec = np.zeros(len(a))
    vec[:n]=win
    vec[-n:]=win
    return cov(vec,a)

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

# Here is the function to have noise model
def noise_m(strain,win,n):
#  Apply window
    window = sig.get_window(win,len(strain))
    strain_win = strain*window
# Do fft to strain
    sft  = np.fft.rfft(strain_win)
# Set the strain sqaure as noise model
    N = np.abs(sft)**2
    #  smooth the noise
    N_s = smooth(N,n)
    #  Higher the peak
    N_smax = np.maximum(N,N_s)
    return N, N_smax

def mfilter(strain,tem,win,dt):
    window = sig.get_window(win,len(strain))
    sft = np.fft.rfft(strain*window)
    Aft = np.fft.rfft(tem*window)
    dft = sft
    nm = noise_m(strain,win,6)[1]
    mf_ft = np.conj(Aft)*(dft/nm)
    freq = np.fft.fftfreq(len(window),dt)
    df = freq[1]-freq[0]
    int = intg.cumtrapz(np.abs(mf_ft), dx=df, initial=0)
    mid_idx = np.argmin(np.abs(int - max(int)/2))
    
    return nm, mf_ft, np.fft.irfft(mf_ft), freq[mid_idx]

def SNR_scatter(mf,start,end):
    SNR = np.max(np.abs(mf))/np.std(mf[start:end])
    return SNR

# load template, noise model, match filter, window function
def SNR_nm(temp,nm,mf,win):
    window = sig.get_window(win,len(strain))
#  fft of template
    tft = np.fft.rfft(temp)
    rhs = mf
    #  calcualte sigma
    lhs = np.conj(tft)*(tft/nm)
    lhs_t = np.fft.irfft(lhs)
    noi = np.sqrt(np.abs(lhs_t))
#  get the siganl to noise ratio
    return np.max(np.abs(mf/noi))


# A
# Read the data
fig1, ax1 = plt.subplots()
n = [1,2,6,10,15,20,30]
window ='hann'
H_fname = ['H-H1_LOSC_4_V1-1167559920-32.hdf5','H-H1_LOSC_4_V2-1126259446-32.hdf5','H-H1_LOSC_4_V2-1128678884-32.hdf5','H-H1_LOSC_4_V2-1135136334-32.hdf5']
L_fname = ['L-L1_LOSC_4_V1-1167559920-32.hdf5','L-L1_LOSC_4_V2-1126259446-32.hdf5','L-L1_LOSC_4_V2-1128678884-32.hdf5','L-L1_LOSC_4_V2-1135136334-32.hdf5']
Templates = ['GW150914_4_template.hdf5','GW151226_4_template.hdf5','GW170104_4_template.hdf5','LVT151012_4_template.hdf5']
# for p in n:    
for name in L_fname:
    # name=H_fname[0]
    strain,dt,utc = read_file(name)
    N = noise_m(strain,window,6)[0]
    N_s =noise_m(strain,window,6)[1]
    ax1.loglog(N_s, label=str(name))
# for p in n:    
#     for name in H_fname:
#         strain,dt,utc = read_file(name)
#         N = noise_m(strain,window,p)[0]
#         N_s =noise_m(strain,window,p)[1]
#         ax1.loglog(N_s, label=str(name)+'n='+str(p))
ax1.set_xlabel("frequency")
ax1.set_ylabel("strain")          
plt.legend()
plt.show()


#B make the match filter
fig2, ax2 = plt.subplots()
tname = Templates[3]
temp0, tl0 = read_template(tname)
Hname=H_fname[2]
Lname = L_fname[2]
strainH,dtH,utcH = read_file(Hname)
strainL,dtL,utcL = read_file(Lname)
print("Strain:"+Hname+"and"+Lname)
print("Template:"+tname)
mf_H = mfilter(strainH,temp0,window,dtH)[2]
mf_L = mfilter(strainL,temp0,window,dtL)[2]
ax2.plot(mf_H,label=str(Hname))
ax2.plot(mf_L,label=str(Lname))
ax2.set_xlabel("Time")
ax2.set_ylabel("Match filter")
plt.legend()
plt.show()

# C calculate the SNR
SNRH = SNR_scatter(mf_H,3000,6000)
SNRL = SNR_scatter(mf_L,3000,6000)
print('SNR of Hanford:'+str(SNRH))
print('SNR of Livingston:'+str(SNRL))
print('SNR of combined Hanford and Livingston'+str(np.sqrt(SNRH**2+SNRL**2)))
# D 
nm_H =  mfilter(strainH,temp0,window,dtH)[0]
nm_L =  mfilter(strainL,temp0,window,dtL)[0]
mf_Hfft = mfilter(strainH,temp0,window,dtH)[2]
mf_Lfft = mfilter(strainL,temp0,window,dtL)[2]
SNR_Hnmodel = SNR_nm(temp0,nm_H,mf_Hfft,window)  
SNR_Lnmodel = SNR_nm(temp0,nm_L,mf_Lfft,window)
print('SNR of Hanford with noise model:'+str(SNR_Hnmodel))
print('SNR of Livingston with noise model :'+str(SNR_Lnmodel))
print('SNR of combined Hanford and Livingston with noise model :'+str(np.sqrt(SNR_Hnmodel**2+SNR_Lnmodel**2)))
# E
midfreq_H= mfilter(strainH,temp0,window,dtH)[3]
midfreq_L = nm_L =  mfilter(strainL,temp0,window,dtL)[3]
print("Mid frequency of Hanford:"+str(midfreq_H))
print("Mid frequency of Liveingston:"+str(midfreq_L))