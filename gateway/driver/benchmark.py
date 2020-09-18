# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:27:02 2018

@author: hengxinliu
"""
import pandas as pd, json, numpy as np
from gateway.driver.tools import _parse_url
from gateway.spider.url import BENCHMARK_URL
from gateway.driver import lookup_benchmark

__all__ = [
    'get_benchmark_returns',
    'get_alternative_returns'
]


def get_benchmark_returns(sid):
    """
        date --- 19900101
    """
    symbol = '1.' + sid if sid.startswith('0') else '0.' + sid
    url = BENCHMARK_URL['kline'].format(symbol, '30000101')
    obj = _parse_url(url, bs=False)
    data = json.loads(obj)
    raw = data['data']
    if raw and len(raw['klines']):
        raw = [item.split(',') for item in raw['klines']]
        kline = pd.DataFrame(raw, columns=['trade_dt', 'open', 'close', 'high', 'low',
                                           'turnover', 'volume', 'amount'])
        kline.set_index('trade_dt', inplace=True)
        kline.sort_index(inplace=True)
        close = kline['close'].astype(np.float)
        # returns = kline['close'] / kline['close'].shift(1) - 1
        returns = close / close.shift(1) - 1
        return returns


def get_outer_returns(index_name):
    """
        dt --- 1990-01-01
    """
    index = lookup_benchmark[index_name]
    url = BENCHMARK_URL['periphera_kline'] % (index, '3000-01-01')
    text = _parse_url(url, bs=False, encoding='utf-8')
    raw = json.loads(text)
    kline = pd.DataFrame(raw['data'][index]['day'], columns=[
                                'trade_dt', 'open', 'close',
                                'high', 'low', 'turnover'])
    kline.set_index('trade_dt', inplace=True)
    kline.sort_index(inplace=True)
    returns = kline['close'] / kline['close'].shift(1) - 1
    return returns


# if __name__ == '__main__':
#
#     rets = get_benchmark_returns('000001')
#     print('ret', rets)
#    outer_ret = get_outer_returns()
