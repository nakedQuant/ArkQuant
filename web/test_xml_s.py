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