#!/usr/bin/env python3
# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from ftplib import FTP

host = '192.168.0.111'
user = 'python'
passwd = 'macpython'

con = FTP().connect(host=host)
FTP.getwelcome()
#
# with FTP(host=host) as ftp:
#     ftp.login(user=user, passwd=passwd)
#     ftp.dir()
