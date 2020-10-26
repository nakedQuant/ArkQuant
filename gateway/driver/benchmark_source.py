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


class BenchmarkSource(object):

    def __init__(self,
                 sessions):
        self.sessions = sessions
        self.symbol_mappings = self._initialize_symbols()

    @staticmethod
    def _initialize_symbols():
        raw = _parse_url(BENCHMARK_URL['symbols'], encoding='utf-8')
        data = json.loads(raw.text)
        index_mappings = {item['f14']: item['f12'] for item in data['data']['diff']}
        return index_mappings

    def _compute_session_returns(self, returns):
        daily_returns = returns.reindex(self.sessions).fillna(0)
        return daily_returns

    def _validate_benchmark(self, symbol):
        # 判断是否是中文
        try:
            symbol.encode('ascii')
        except UnicodeEncodeError:
            symbol = self.symbol_mappings[symbol]
        return symbol

    def _calculate_returns(self, sid):
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
            returns = close / close.shift(1) - 1
            daily_returns = self._compute_session_returns(returns)
            return daily_returns

    def _calculate_alternative_returns(self, index_name):
        """
            dt --- 1990-01-01
        """
        try:
            index = lookup_benchmark[index_name]
        except KeyError:
            raise ValueError
        url = BENCHMARK_URL['periphera_kline'] % (index, '3000-01-01')
        text = _parse_url(url, bs=False, encoding='utf-8')
        raw = json.loads(text)
        kline = pd.DataFrame(raw['data'][index]['day'], columns=[
                                    'trade_dt', 'open', 'close',
                                    'high', 'low', 'turnover'])
        kline.set_index('trade_dt', inplace=True)
        kline.sort_index(inplace=True)
        kline = kline.astype('float64')
        returns = kline['close'] / kline['close'].shift(1) - 1
        daily_returns = self._compute_session_returns(returns)
        return daily_returns

    def calculate_returns(self, proxy_name):
        if proxy_name in set(lookup_benchmark):
            returns = self._calculate_alternative_returns(proxy_name)
        else:
            symbol = self._validate_benchmark(proxy_name)
            returns = self._calculate_returns(symbol)
        return returns


__all__ = ['BenchmarkSource']
