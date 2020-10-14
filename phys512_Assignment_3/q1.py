# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 20:43:38 2020

@author: Yifei Gu
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

#load data file
data = np.loadtxt("wmap_tt_spectrum_9yr_v5.txt")