# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, numpy as np
from gateway.driver.client import tsclient

# transform [dict] to DataFrame
frame = pd.read_csv('equity_basics.csv')
frame.set_index('代码', drop=False, inplace=True)
# replace null -- Nan
frame.replace(to_replace='null', value=pd.NA, inplace=True)
frame['发行价格'].fillna('0.00', inplace=True)
frame['发行价格'] = frame['发行价格'].astype(np.float)
print(frame['发行价格'].dtype)
#
d = frame['list_date'].dropna()
print('length', d)

stats = tsclient.to_ts_stats()
print(len(stats), stats)

union = set(stats.index) & set(d.index)
print('common', len(union))
difference = set(stats.index) - union
print('difference set', difference)
#
print('test', stats.loc['T00018',:])
print('test', stats.loc['600018',:])
