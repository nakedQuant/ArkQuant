# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from gateway.database.asset_writer import  AssetWriter
"""
    a. assets --- enroll into mysql
    b. enroll kline
    c. enroll equity adjustments divdend equity structure
"""

if __name__ == '__main__':

    assetDb = AssetWriter()
    assetDb.write()
