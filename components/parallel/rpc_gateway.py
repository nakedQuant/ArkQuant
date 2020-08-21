#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:56:19 2019

@author: python
"""

class RpcGateway:
    """
    VN Trader Gateway for RPC service.
    """

    default_setting = {
        "主动请求地址": "tcp://127.0.0.1:2014",
        "推送订阅地址": "tcp://127.0.0.1:4102"
    }

    def __init__(self, event_engine):
        """Constructor"""
        super().__init__(event_engine, "RPC")

        self.symbol_gateway_map = {}

        # self.client = RpcClient()
        # self.client.callback = self.client_callback

    def connect(self, setting: dict):
        """"""
        req_address = setting["主动请求地址"]
        pub_address = setting["推送订阅地址"]

        self.client.subscribe_topic("")
        self.client.start(req_address, pub_address)

        self.write_log("服务器连接成功，开始初始化查询")

        self.query_all()

    def query_account(self):
        """"""
        pass

    def query_position(self):
        """"""
        pass

    def close(self):
        """"""
        self.client.stop()

    def client_callback(self, topic: str, event: Event):
        """"""
        if event is None:
            print("none event", topic, event)
            return

        data = event.data

        if hasattr(data, "gateway_name"):
            data.gateway_name = self.gateway_name

        self.event_engine.put(event)
