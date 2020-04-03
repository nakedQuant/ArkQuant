#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
import socket

"""
'AF_INET' 地址是 (主机, 端口)  形式的元组类型，其中 主机 是一个字符串，端口 是整数。

'AF_UNIX' 地址是文件系统上文件名的字符串。

'AF_PIPE' 是这种格式的字符串 r'\.\pipe{PipeName}' 。如果要用 Client() 连接到一个名为 ServerName 的远程命名管道，应该替换为使用 r'\ServerName\pipe{PipeName}' 这种格式。
"""

"""
服务器必须执行序列socket()， bind()，listen()，accept()（可能重复accept()，以服务一个以上的客户端），
而一个客户端只需要在序列socket()，connect()。另请注意，服务器不在sendall()/ recv()侦听的套接字上，而是/ 返回的新套接字 accept()
"""

HOST = '192.168.0.103'                 # Symbolic name meaning all available interfaces
PORT = 50007              # Arbitrary non-privileged port
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(20)
            if not data:
                break
            conn.sendall(data + b' response')