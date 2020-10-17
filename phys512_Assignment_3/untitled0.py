import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm3

lags=np.arange(1,10000)
c=chain[:,2]
fig, ax = plt.subplots()
ax.plot(lags, [pm3.autocorr(c, l) for l in lags])
_ = ax.set(xlabel='lag', ylabel='autocorrelation', ylim=(-.1, 1))
plt.title('Autocorrelation Plot')
plt.show()

lags=np.arange(1,100)

fig, ax = plt.subplots()
ax.plot(lags, [pm3.autocorr(chain, l) for l in lags])
_ = ax.set(xlabel='lag', ylabel='autocorrelation', ylim=(-.1, 1))
plt.title('Autocorrelation Plot')
plt.show()



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot


df = pd.read_csv('transcount.csv')
df = df.groupby('year').aggregate(np.mean)

gpu = pd.read_csv('gpu_transcount.csv')
gpu = gpu.groupby('year').aggregate(np.mean)

df = pd.merge(df, gpu, how='outer', left_index=True, right_index=True)

autocorrelation_plot(np.log(df['trans_count']))
plt.show()