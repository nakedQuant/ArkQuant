#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:56:19 2019

@author: python
"""

#服务端
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
"""
SimpleXMLRPCServer.register_introspection_functions（）
注册XML-RPC内省功能system.listMethods， system.methodHelp和system.methodSignature。

SimpleXMLRPCServer.register_multicall_functions（）
注册XML-RPC多重调用函数system.multicall。

SimpleXMLRPCRequestHandler.rpc_paths
一个属性值，必须是一个元组，列出接收XML-RPC请求的URL的有效路径部分。发布到其他路径的请求将导致404“无此页面” HTTP错误。如果该元组为空，则所有路径都将被视为有效。默认值为。('/', '/RPC2')
"""

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

# Create server

if __name__ == '__main__':

    with SimpleXMLRPCServer(('192.168.0.103', 8000),
                            requestHandler=RequestHandler) as server:
        server.register_introspection_functions()
        server.register_multicall_functions()
        # Register pow() function; this will use the value of
        # pow.__name__ as the name, which is just 'pow'.
        server.register_function(pow)

        # def adder_function(x, y):
        #     return x + y
        # server.register_function(adder_function, 'add')

        @server.register_function(name='add')
        def adder_function(x, y):
            return x + y

        # Register a function under function.__name__.
        @server.register_function
        def mul(x, y):
            return x * y

        # Register an instance; all the methods of the instance are
        # published as XML-RPC methods (in this case, just 'mul').
        class MyFuncs:
            def mul(self, x, y):
                return x * y

        server.register_instance(MyFuncs())

        # Run the server's main loop
        server.serve_forever()


# create client
# import xmlrpc.client
#
# s = xmlrpc.client.ServerProxy('http://localhost:8000')
# print(s.pow(2,3))  # Returns 2**3 = 8
# print(s.add(2,3))  # Returns 5
# print(s.mul(5,2))  # Returns 5*2 = 10
#
# # Print list of available methods
# print(s.system.listMethods())

from xmlrpc.client import ServerProxy, Error,MultiCall

server = ServerProxy("http://localhost:8000")

# try:
#     print(server.currentTime.getCurrentTime())
# except Error as v:
#     print("ERROR", v)

multi = MultiCall(server)
multi.pow(2,9)
multi.add(1,2)
try:
    for response in multi():
        print(response)
except Error as v:
    print("ERROR", v)

"""
DocXMLRPCServer对象
本DocXMLRPCServer类源自SimpleXMLRPCServer 并提供创建自我记录的手段，独立的XML-RPC服务器。HTTP POST请求作为XML-RPC方法调用处理。通过生成pydoc样式的HTML文档来处理HTTP GET请求。这允许服务器提供其自己的基于Web的文档。

DocXMLRPCServer.set_server_title（server_title ）
设置在生成的HTML文档中使用的标题。该标题将在HTML“ title”元素中使用。

DocXMLRPCServer.set_server_name（server_name ）
设置在生成的HTML文档中使用的名称。此名称将出现在“ h1”元素内生成的文档的顶部。

DocXMLRPCServer.set_server_documentation（server_documentation ）
设置在生成的HTML文档中使用的描述。该说明将在文档中的服务器名称下方以一段显示。
"""

###########################################################
#Examples of Client
# simple test program (from the XML-RPC specification)
# from xmlrpc.client import ServerProxy, Error
#
# # server = ServerProxy("http://localhost:8000") # local server
# with ServerProxy("http://betty.userland.com") as proxy:
#
#     print(proxy)
#
#     try:
#         print(proxy.examples.getStateName(41))
#     except Error as v:
#         print("ERROR", v)
#
#
# import http.client
# import xmlrpc.client
#
# #要通过HTTP代理访问XML-RPC服务器，您需要定义一个自定义传输
# class ProxiedTransport(xmlrpc.client.Transport):
#
#     def set_proxy(self, host, port=None, headers=None):
#         self.proxy = host, port
#         self.proxy_headers = headers
#
#     def make_connection(self, host):
#         connection = http.client.HTTPConnection(*self.proxy)
#         connection.set_tunnel(host, headers=self.proxy_headers)
#         self._connection = host, connection
#         return connection
#
# transport = ProxiedTransport()
# transport.set_proxy('proxy-server', 8080)
# server = xmlrpc.client.ServerProxy('http://betty.userland.com', transport=transport)
# print(server.examples.getStateName(41))
