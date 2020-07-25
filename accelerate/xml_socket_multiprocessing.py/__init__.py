#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:56:19 2019

@author: python
"""
import subprocess
from subprocess import Popen,PIPE
from subprocess import TimeoutExpired

with Popen(["ifconfig"], stdout=PIPE) as proc:
    res = proc.stdout.read()
    print(res)


# proc = subprocess.Popen(['ping','www.baidu.com'],stdout = PIPE,encoding= 'utf-8')
# print('------------error', proc.stdout)
# for line in proc.stdout:
#     print(line)


import socket

#返回一个字符串，其中包含Python解释器当前正在执行的机器的主机名
res = socket.gethostname()
print(res)
fq = socket.getfqdn(res)
print(fq)
#将主机名转换为IPv4地址格式。IPv4地址以字符串形式返回，例如 '100.50.200.5'。如果主机名本身是IPv4地址，则返回原样
ip = socket.gethostbyname(res)
print(ip)
#(hostname, aliaslist, ipaddrlist) paddrlist是同一接口上同一接口的IPv4 / v6地址的列表主机
tuple = socket.gethostbyaddr(ip)
print(tuple)


