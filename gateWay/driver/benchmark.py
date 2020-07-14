# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:27:02 2018

@author: hengxinliu
"""
import json,pandas as pd,datetime
from functools import lru_cache

from gateWay.tools import _parse_url
from gateWay.spider._config import  BENCHMARK_REQUEST_URL , index_lookup



BENCHMARK_REQUEST_URL = {'benchmark':'http://71.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=12'
                                  '&po=1&np=2&fltt=2&invt=2&fid=&fs=b:MK0010&fields=f12,f14',
                         'index_kline':'http://push2his.eastmoney.com/api/qt/stock/kline/get?secid='
                                     '{}&fields1=f1&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf'
                                     '57%2Cf58&klt=101&fqt=0&beg=19900101&end={}',
                         'foreign_index_kline':'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?&param=%s,day,1990-01-01,%s,100000,qfq'}

index_lookup = {'道琼斯':'us.DJI',
                '纳斯达克':'us.IXIC',
                '标普500':'us.INX',
                '香港恒生指数':'hkHSI',
                '香港国企指数':'hkHSCEI',
                '香港红筹指数':'hkHSCCI'}


class BenchmarkSource(object):

    @staticmethod
    def native_benchmark_symbols():
        raw = json.loads(_parse_url(BENCHMARK_REQUEST_URL['benchmark'],encoding='utf-8',bs= False))
        index_set = raw['data']['diff']
        return index_set

    @staticmethod
    def download_periphera_index(name, date):
        """
        获取外围指数
        :param index: 指数名称
        :param exchange: 地域
        :return:
        """
        if isinstance(date,datetime.datetime):
            format_date = datetime.datetime.strftime(date,'%Y-%m-%d')
        else:
            date = str(date)
            format_date = date if ('-') in date  else ('-').\
                          join(date[:4],date[4:6],date[6:])
        url = index_lookup['foreign_index_kline']%(index_lookup[name], format_date)
        raw = _parse_url(url, bs=False, encoding='utf-8')
        raw = json.loads(raw)
        data = raw['data'][index_lookup[name]]['day']
        df = pd.DataFrame(data, columns=['trade_dt', 'open', 'close', 'high', 'low', 'turnvoer'])
        df.index  = df.loc[:,'trade_dt']
        df.sort_index(inplace = True)
        return df

    @staticmethod
    def _attach_prefix(k):
        if k['f12'].startswith('0'):
            prefix = '1.' + k['f12']
        else:
            prefix = '0.' + k['f12']
        return prefix

    @lru_cache(maxsize= 30)
    def request_benchmark_kline(self,k,date):
        if isinstance(date,datetime.datetime):
            format_date = datetime.datetime.strftime(date,'%Y%m%d')
        else:
            format_date = date.replace('-','')
        url = BENCHMARK_REQUEST_URL['index_kline'].format(self._attach_prefix(k),format_date)
        obj = _parse_url(url,bs = False)
        data = json.loads(obj)
        raw = data['data']
        if raw and len(raw['klines']):
            raw = [item.split(',') for item in raw['klines']]
            benchmark = pd.DataFrame(raw,
                                     columns = ['trade_dt','open','close','high','low','turnover','volume','amount'])
            return benchmark

    def get_benchmark_returns(self,date,index_symbol):
        try:
            index = index_lookup[index_symbol]
            kline = self.download_periphera_index(index,date)
        except KeyError:
            kline = self.request_benchmark_kline(index_symbol,date)
        benchmark_returns = kline.loc[:,'close'] / kline.loc[:,'close'].shift(1) -1
        return benchmark_returns
