import warnings
from datetime import datetime
from os import listdir
import os.path

import pandas as pd
import pytz


DATE_FORMAT = "%Y%m%d"
zipline_dir = os.path.dirname(zipline.__file__)
SECURITY_LISTS_DIR = os.path.join(zipline_dir, 'resources', 'security_lists')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:39:46 2019

@author: python
"""
import json,re,datetime,time,pandas as pd,numpy as np
from itertools import chain
from sqlalchemy import select
from sqlalchemy.sql import func
from abc import ABC,abstractmethod


class Astock(object):
    """
        1、股票日线
        2、股票的基本信息
        3、股权结构
        4、分红配股信息
        参数 frequency : daily  表示每天都要跑 ;否则 历史数据
    """
    __name__ = 'Astock'

    def __init__(self):
        """每次进行初始化"""
        self._failure = []
        self._init_db()
        self.conn = self.db.db_init()
        self.basics_assets = self._get_stock_assets() if self.frequency else []
        self.mode = self.switch_mode()

    @staticmethod
    def _get_prefix(code):
        if code.startswith('0') or code.startswith('3'):
            prefix = '0.' + code
        elif code.startswith('6'):
            prefix = '1.' + code
        return prefix

    def _download_assets(self):
        # 获取存量股票包括退市
        html = 'http://70.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12'
        raw = json.loads(self._parse_url(html,bs= False))
        q = [item['f12'] for item in raw['data']['diff']]
        # sci_tech = list(filter(lambda x: x.startswith('688'), q))
        # tradtional = set(q) - set(sci_tech)
        return q

    def _get_stock_assets(self):
        basics = getattr(self.db,'description')
        ins = select([basics.c.代码.distinct().label('code')])
        rp = self.conn.engine.execute(ins)
        basics_code = [r.code for r in rp]
        return set(basics_code)

    def _download_ipo(self):
        """获取最近新上市的股票"""
        html = 'http://81.push2.eastmoney.com/api/qt/clist/get?&pn=1&pz=100&po=1&np=1&fltt=2&invt=2&fid=f26&fs=m:0+f:8,m:1+f:8&fields=f12,f26'
        obj = self._parse_url(html,bs = False)
        raw = json.loads(obj)
        raw = [list(i.values()) for i in raw['data']['diff']]
        df = pd.DataFrame(raw, columns=['code', 'timeTomarket'])
        return df

    def _download_bascis(self,code):
        """获取股票基础信息"""
        if not self.frequency or code not in self.basics_assets:
            # 公司基本情况
            html = 'https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/%s.phtml' % code
            obj = self._parse_url(html)
            table = obj.find('table', {'id': 'comInfo1'})
            tag = [item.findAll('td') for item in table.findAll('tr')]
            tag_chain = list(chain(*tag))
            raw = [item.get_text() for item in tag_chain]
            # 去除格式
            raw = [i.replace('：', '') for i in raw]
            raw = [i.strip() for i in raw]
            info = list(zip(raw[::2], raw[1::2]))
            info_dict = {item[0]: item[1] for item in info}
            #用于过滤股权结构分布
            self._timeTomarket = info_dict['上市日期']
            info_dict.update({'代码': code})
            #入库
            self.db.enroll('description',info_dict,self.conn)

    def _run_session(self,code):
        """基于事务 基于一个连接"""
        transaction = self.conn.begin()
        try:
            self._download_bascis(code)
            transaction.commit()
            print('enroll kline,splits_divdend,equity,baseinfo from code : %s'%code)
        except Exception as error:
            transaction.rollback()
            print('---------------astock error :%s'%code,error)
            self._failure.append(code)
        finally:
            transaction.close()

    def run_bulks(self):
        quantity = self._download_assets()
        for q in quantity:
            self._run_session(q)
        res = {self.__name__:self._failure}
        return res


class ETF:
    """
        获取ETF列表 以及 对应的日线
    """
    __name__ = 'ETF'

    def __init__(self):
        self._failure = []
        self._init_db()
        self.conn = self.db.db_init()
        self.mode = self.switch_mode()

    def _download_assets(self):
        """获取ETF列表 page num --- 20 40 80"""
        accumulate = []
        page = 1
        while True:
            html = "http://vip.stock.finance.sina.com.cn/quotes_service/api/jsonp.php/IO.XSRV2.CallbackList['v7BkqPkwcfhoO1XH']/"\
                   "Market_Center.getHQNodeDataSimple?page=%d&num=80&sort=symbol&asc=0&node=etf_hq_fund" % page
            obj = self._parse_url(html)
            text = obj.find('p').get_text()
            mid = re.findall('s[z|h][0-9]{6}', text)
            if len(mid) > 0:
                accumulate.extend(mid)
            else:
                break
            page = page + 1
        return accumulate


class Convertible:
    """
        可转换公司债券的期限最短为1年，最长为6年，自发行结束之日起6个月方可转换为公司股票
        可转债赎回：公司股票在一段时间内连续高于转换价格达到一pp-....≥....................//..
        [p.p[“≥≥≥≥≥≥≥≥≥≥≥≥≥≥≥.p[/定幅"?时，公司可按照事先约定的赎回价格买回发行在外尚未转股的可转换公司债券
        可转债强制赎回：连续三十个交易日中至少有十五个交易日的收盘价格不低于当期转股价格的 130%（含130%）；
        存在向下修订转股价
        可转债回售：最后两个计息年度内，一旦正股价持续约30天低于转股价的70%，上市公司必须以100+e 的价格来赎回可转债
        集思录获取可转债数据列表，请求方式post
        可债转基本情况 --- 强制赎回价格 回售价格  时间
    """
    __name__ = 'Convertible'

    def __init__(self):
        """每次进行初始化"""
        self._failure = []
        self._init_db()
        self.conn = self.db.db_init()
        self._basics_id = self._get_convertible_assets() if self.frequency else []
        self.mode = self.switch_mode()

    def _download_assets(self):
        url = 'https://www.jisilu.cn/data/cbnew/cb_list/?'
        text = self._parse_url(url,bs = False,encoding = None)
        text = json.loads(text)
        return text['rows']

    def _get_convertible_assets(self):
        table = self.db.metadata.tables['convertibleDesc']
        ins = select([table.c.bond_id])
        rp = self.conn.engine.execute(ins)
        bond_id = [r.bond_id for r in rp]
        return set(bond_id)

    def _enroll_desc(self,item):
        if not self.frequency or item['id'] not in self._basics_id:
            self.db.enroll('convertible',item['cell'],self.conn)

    def _run_session(self,item):
        """基于事务 基于一个连接"""
        transaction = self.conn.begin()
        try:
            self._enroll_desc(item)
            transaction.commit()
            print('crawler %s kline from 集思录'%item['id'])
        except Exception as e:
            transaction.rollback()
            print('---------- convertible error : %s'%item['id'],e)
            self._failure.append(item)
        finally:
            transaction.close()

    def run_bulks(self):
        quantity = self._download_assets()
        for q in quantity:
            self._run_session(q)
        res = {self.__name__:self._failure}
        return res


class ExtraOrdinary:
    """
    　　中国股市:
            9：15开始参与交易，11：30-13：00为中午休息时间，下午13：00再开始，15：00交易结束。
    　　中国香港股市:
        　　开市前时段 上午9时30分至上午10时正
        　　早市上午10时正至中午12时30分
        　　延续早市中午12时30分至下午2时30分
        　　午市下午2时30分至下午4时正
    　　美国股市：
        　　夏令时间 21：30至04：00
        　　冬令时间 22：30至05：00
    　　欧洲股市：
    　　    4点到晚上12点半
    """
    __name__ = 'ExtraOrdinary'

    def __init__(self):
        """每次进行初始化"""
        self._failure = []
        self._init_db()
        self.conn = self.db.db_init()
        self.mode = self.switch_mode()

    def _get_prefix(self,code,exchange ='hk'):
        if exchange == 'us':
            code = exchange + '.' + code
        elif exchange == 'hk':
            code = exchange + code
        else:
            raise NotImplementedError
        return code

    def switch_mode(self):
        """决定daily or init"""
        edate = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
        if self.frequency:
            sdate = edate
        else:
            sdate = '1990-01-01'
        return (sdate,edate)

    def _download_assets(self):
        """
            A股与H股同时上市的股票
        """
        df = pd.DataFrame()
        count = 1
        while True:
            html = 'http://19.push2.eastmoney.com/api/qt/clist/get?pn=%d&pz=20&po=1&np=1&invt=2&fid=f3&fs=b:DLMK0101&fields=f12,f191,f193'%count
            raw = self._parse_url(html,bs = False,encoding = None)
            raw = json.loads(raw)
            diff = raw['data']
            if diff and len(diff['diff']):
                diff = [[item['f12'],item['f191'],item['f193']] for item in diff['diff']]
                raw = pd.DataFrame(diff,columns = ['h_code','code','name'])
                df = df.append(raw)
                count = count +1
            else:
                break
        return df

    def _run_session(self,item):
        try:
            self._download_kline(item,*self.mode)
            print('hcode:%s successfully run kline'%item[0])
        except Exception as e:
            print('hcode :%s run kline failure'%item[0],e)
            self._failure.append(item)

    def run_bulks(self):
        quantity = self._download_assets()
        for r in quantity.values.tolist():
            self._run_session(r)
        res = {self.__name__:self._failure}
        return res

class Index:
    """
        获取A股基准数据
    """
    __name__ = 'Index'

    def __init__(self):

        self._failure = []
        self._init_db()
        self.conn = self.db.db_init()
        self.mode = self.switch_mode()

    def _download_assets(self):
        url = 'http://71.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=12&po=1&np=2&fltt=2&invt=2&fid=&fs=b:MK0010&fields=f12,f14'
        raw = json.loads(self._parse_url(url,encoding='utf-8',bs= False))
        index_set = raw['data']['diff']
        return index_set

    def _get_prefix(self,k):
        if k['f12'].startswith('0'):
            prefix = '1.' + k['f12']
        else:
            prefix = '0.' + k['f12']
        return prefix

    def switch_mode(self):
        """决定daily or init"""
        if self.frequency:
            s = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')
        else:
            s = '19900101'
        return s

    def _run_session(self,name):
        try:
            self._download_kline(name)
        except Exception as e:
            print('error%s enroll index :%s kline' % (name, e))
            self._failure.append(name)

    def run_bulks(self):
        quantity = self._download_assets()
        # mode = self.switch_mode()
        for k in quantity.values():
            self._run_session(k)
        res = {self.__name__:self._failure}
        return res
