# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import redis


r = redis.Redis(host='localhost', port=6379)
r.set('test', 'test_value')
print('get', r['test'])
