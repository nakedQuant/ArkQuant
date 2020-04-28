# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:27:02 2018

@author: hengxinliu
"""
import requests,json,pandas as pd ,sqlalchemy as sa
from bs4 import BeautifulSoup
from functools import lru_cache

index_lookup = {'上证指数':'000001' ,
                '深证成指':'399001' ,
                'Ｂ股指数':'000003',
                '深成指R':'399002',
                '成份Ｂ指':'399003',
                '深证综指':'399106',
                '上证180':'000010',
                '基金指数':'000011',
                '深证100R':'399004',
                '国债指数':'000012',
                '企债指数':'000013',
                '上证50':'000016',
                '上证380':'000009',
                '沪深300':'000300',
                '中证500':'000905',
                '中小板指':'399005',
                '新指数':'399100',
                '中证100':'000903',
                '中证800':'000906',
                '深证300':'399007',
                '中小300':'399008',
                '创业板指':'399006',
                '上证100':'000132',
                '上证150':'000133',
                '央视50':'399550',
                '创业大盘':'399293',
                '道琼斯':'us.DJI',
                '纳斯达克':'us.IXIC',
                '标普500':'us.INX',
                '香港恒生指数':'hkHSI',
                '香港国企指数':'hkHSCEI',
                '香港红筹指数':'hkHSCCI'}

def _parse_url(url, encoding='gbk', bs=True):
    Header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36(KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'}
    req = requests.get(url, headers=Header, timeout=1)
    if encoding:
        req.encoding = encoding
    if bs:
        raw = BeautifulSoup(req.text, features='lxml')
    else:
        raw = req.text
    return raw


class Benchmark(object):

    def __init__(self,engine):
        self.engine = engine

    @lru_cache(maxsize= 30)
    def load_index_from_sqlite(self,sdate,edate,index):
        tbl = self.engine.table_names()['index_price']
        rp = sa.select([tbl.c.trade_dt,
                        sa.cast(tbl.c.close,sa.Numeric(12,2)).label('close'),
                      sa.cast(tbl.c.volume,sa.Numeric(15,2)).label('amount')]). \
             where(sa.and_(tbl.c.sid == index,
                        tbl.c.trade_dt.between(sdate, edate))).execute()
        arrays = [[r.trade_dt,r.close,r.high,r.amount] for r in rp.fetchall()]
        df = pd.DataFrame(arrays,columns = ['trade_dt','close','amount'])
        df.index = df.loc[:,'trade_dt']
        df.sort_index(inplace = True)
        return df

    def get_benchmark_returns(self,sdate,edate,index):
        if index in ['道琼斯','纳斯达克','标普500','香港恒生指数','香港国企指数','香港红筹指数']:
            raw_array = self.download_periphera_index(index,sdate,edate)
        else:
            raw_array  = self.load_index_from_sqlite(sdate,edate,index)
        benchmark_returns = raw_array.loc[:,'close'] / raw_array.loc[:,'close'].shift(1)
        return benchmark_returns

    @staticmethod
    def download_periphera_index(cname, sdate, edate, lmt=10000):
        """
        获取外围指数
        :param index: 指数名称
        :param exchange: 地域
        :return:
        """
        tencent = 'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?&param=%s,day,%s,%s,%d,qfq' % (
        index_lookup[cname], sdate, edate)
        raw = _parse_url(tencent, bs=False, encoding='utf-8')
        raw = json.loads(raw)
        data = raw['data'][index_lookup[cname]]['day']
        df = pd.DataFrame(data, columns=['trade_dt', 'open', 'close', 'high', 'low', 'turnvoer'])
        df.index  = df.loc[:,'trade_dt']
        df.sort_index(inplace = True)
        return df