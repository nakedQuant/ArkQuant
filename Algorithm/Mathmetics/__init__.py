# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np
from scipy import integrate

def zoom(raw):
    scale = (raw - raw.min()) / (raw.max() - raw.min())
    return scale

def standardize(raw):
    standard = (raw - raw.mean()) / raw.std()
    return standard

#弧度转角度
def coef2deg(x):
    rad = np.math.acos(x)
    deg = np.rad2deg(rad)
    return deg

def funcScorer(func,interval):
    area, err = integrate.quad(func, * interval)
    ratio = (area - err) / area
    return area,ratio
