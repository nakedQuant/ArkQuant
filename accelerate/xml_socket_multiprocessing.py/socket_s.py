#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 14:00:14 2019

@author: python
"""
import socket , os

def socket_bind_recv(socket_fn, cmd_handler):
    """
    基于bsd系统的进程间socket通信，接受消息，处理消息
    :param socket_fn: socket文件名称
    :param cmd_handler: cmd处理函数，callable类型
    """
    if not callable(cmd_handler):
        print('socket_bind_recv cmd_handler must callable!')

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_fn)
    server.listen(0)
    while True:
        connection, _ = server.accept()
        socket_cmd = connection.recv(1024).decode()
        # 把接收到的socket传递给外部对应的处理函数
        cmd_handler(socket_cmd)
        connection.close()


def socket_send_msg(socket_fn, msg):
    """
    基于bsd系统的进程间socket通信，发送消息
    :param socket_fn: : socket文件名称
    :param msg: 字符串类型需要传递的数据，不需要encode，内部进行encode
    """
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(socket_fn)
    client.send(msg.encode())
    client.close()


def show_msg(title, msg):
    """
    使用osascript脚步提示弹窗，主要用在长时间且耗时的任务中，提示重要问题信息
    :param title: 弹窗标题
    :param msg: 弹窗信息
    """
    msg_cmd = 'osascript -e \'display notification "%s" with title "%s"\'' % (msg, title)
    os.system(msg_cmd)


def fold_free_size_mb(folder):
    """
    mac os下剩余磁盘空间获取
    :param folder: 目标目录
    :return: 返回float，单位mb
    """
    st = os.statvfs(folder)
    return st.f_bavail * st.f_frsize / 1024 / 1024


"""
'AF_INET' 地址是 (主机, 端口)  形式的元组类型，其中 主机 是一个字符串，端口 是整数。

'AF_UNIX' 地址是文件系统上文件名的字符串。

'AF_PIPE' 是这种格式的字符串 r'\.\pipe{PipeName}' 。如果要用 Client() 连接到一个名为 ServerName 的远程命名管道，应该替换为使用 r'\ServerName\pipe{PipeName}' 这种格式。

socket.AF_UNIX
socket.AF_INET
socket.AF_INET6
These constants represent the address (and protocol) families, used for the first argument to socket(). If the AF_UNIX constant is not defined then this protocol is unsupported. More constants may be available depending on the system.

socket.SOCK_STREAM
socket.SOCK_DGRAM
socket.SOCK_RAW
socket.SOCK_RDM
socket.SOCK_SEQPACKET
These constants represent the socket types, used for the second argument to socket(). More constants may be available depending on the system. (Only SOCK_STREAM and SOCK_DGRAM appear

服务器必须执行序列socket()， bind()，listen()，accept()（可能重复accept()，以服务一个以上的客户端），
而一个客户端只需要在序列socket()，connect()。另请注意，服务器不在sendall()/ recv()侦听的套接字上，而是/ 返回的新套接字 accept()

socket.sendmsg（缓冲区[，ancdata [，标志[，地址] ] ] ）
将普通数据和辅助数据发送到套接字，从一系列缓冲区中收集非辅助数据，并将其串联为一条消息。所述缓冲器参数指定为可迭代的非辅助数据 字节状物体 （例如bytes对象）; 操作系统可能会设置可使用的缓冲区数的限制（sysconf()值SC_IOV_MAX）。所述ancdata参数指定所述辅助数据（控制消息），为迭代的零个或多个元组 ，其中cmsg_level和 cmsg_type分别指定协议级和协议特定的类型整数，且cmsg_data(cmsg_level, cmsg_type, cmsg_data)是保存相关数据的类似字节的对象。请注意，某些系统（特别是没有的系统CMSG_SPACE()）可能支持每个呼叫仅发送一条控制消息。该 标志参数默认为0，有用法相同 send()。如果没有提供addressNone，那么它设置消息的目标地址。返回值是发送的非辅助数据的字节数。

以下函数在支持该机制的系统上， 通过套接字 发送文件描述符fds的列表。另请参阅。AF_UNIXSCM_RIGHTSrecvmsg()

import socket, array

def send_fds(sock, msg, fds):
    return sock.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))])

socket.recv（bufsize [，flags ] ）
从套接字接收数据。返回值是一个字节对象，代表接收到的数据。一次要接收的最大数据量由bufsize指定。有关可选参数标志的含义，请参见Unix手册页recv（2）。它默认为零。

注意 为了与硬件和网络的实际情况达到最佳匹配，bufsize的值 应为2的相对较小的幂，例如4096。
"""

HOST = '192.168.0.103'  # The remote host
PORT = 50007  # The same port as used by the server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b'Hello, world')
    data = s.recv(1024)
print('Received', repr(data))

HOST = 'daring.cwi.nl'  # The remote host
PORT = 50007  # The same port as used by the server
s = None
for res in socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC, socket.SOCK_STREAM):
    af, socktype, proto, canonname, sa = res
    try:
        s = socket.socket(af, socktype, proto)
    except OSError as msg:
        s = None
        continue
    try:
        s.connect(sa)
    except OSError as msg:
        s.close()
        s = None
        continue
    break


# --------------------------------------------------------------


"""
'AF_INET' 地址是 (主机, 端口)  形式的元组类型，其中 主机 是一个字符串，端口 是整数。

'AF_UNIX' 地址是文件系统上文件名的字符串。

'AF_PIPE' 是这种格式的字符串 r'\.\pipe{PipeName}' 。如果要用 Client() 连接到一个名为 ServerName 的远程命名管道，应该替换为使用 r'\ServerName\pipe{PipeName}'
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
