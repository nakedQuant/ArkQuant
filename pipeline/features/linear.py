# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm,numpy as np

def _fit_poly(y,degree):
    #return n_array (dimension ascending = False) p(x) = p[0] * x**deg + ... + p[deg]
    y.dropna(inplace = True)
    res = np.polyfit(range(len(y)), np.array(y), degree)
    return res[0]

def _fit_sklearn(x,y):
    reg = LinearRegression(fit_intercept=False).fit(x, y)
    # reg.intercept_
    coef = reg.coef_
    return coef


def _fit_statsmodel(x,y):
    # statsmodels.regression.linear_model  intercept = model.params[0]，rad = model.params[1]
    X = sm.add_constant(x)
    #const coef
    res = sm.OLS(y,X).fit()
    return res[-1]

def _fit_lstsq(x,y):
    res = np.linalg.lstsq(x,y)
    return res[0][0]