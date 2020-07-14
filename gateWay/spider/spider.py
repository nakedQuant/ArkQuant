#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:39:46 2019

@author: python
"""
import json,re,pandas as pd ,time ,numpy as np
from itertools import chain
from sqlalchemy import select
from sqlalchemy.sql import func

from gateWay.tools import  _parse_url,parse_content_from_header
from gateWay.driver.db_schema import engine,metadata
from gateWay.driver.db_writer import  DBWriter
from _config import (
                    ASSETS_BUNDLES_URL,
                    ASSETS_BASICS_URL,
                    ASSETS_EVENT_URL,
                    ASSETS_STRUCTURE_DIVDEND
                    )


__all__ = ['BundlesRequest','BasicsRequest','EventRequest']


dBwriter = DBWriter()

BUNDLES_COLUMNS_NAMES = frozenset(['trade_dt', 'open', 'close', 'high', 'low', 'volume', 'turnover'])


class BundlesRequest(object):

    def __init__(self,lmt):
        self.lmt = lmt
        self._default_cols = ['trade_dt', 'open', 'close', 'high', 'low', 'volume', 'amount']

    @property
    def default(self):
        return self._default_cols

    def _request_kline_for_asset(self,sid,tbl,extend = False):
        url = ASSETS_BUNDLES_URL[tbl].format(sid['request_sid'],self.lmt)
        obj = _parse_url(url,bs =False)
        raw = json.loads(obj)
        kline = raw['data']
        cols = self.default + ['pct'] if extend else self.default
        if kline and len(kline['klines']):
            kl_pd = pd.DataFrame([item.split(',') for item in kline['klines']],
                                 columns=cols)
            kl_pd.loc[:,'sid'] = sid['sid']
            if hasattr(sid,'bond_id'):
                kl_pd.loc[:,'bond_id'] = sid['bond_id']
            dBwriter.writer(tbl,kl_pd)

    def request_equity_kline(self,sid):
        sid_id = '1.' + sid if sid.startswith('6') else '0.' + sid
        self._request_kline_for_asset({'request_sid':sid_id,'sid':sid},'equity_price',extend = True)

    def request_fund_kline(self,fund):
        fund_id = '1.'+ fund[2:] if fund.startswith('sh') else '0.' + fund[2:]
        self._request_kline_for_asset({'request_sid':fund_id,'sid':fund},'fund_price')

    def request_bond_kline(self,bond):
        sid = bond['cell']['stock_id']
        bond_id = '0.' + bond['id'] if sid.startswith('sz') else '1.' + bond['id']
        self._request_kline_for_asset({'request_sid':bond_id,'sid':sid,'bond_id':bond['id']},'bond_price')

    def request_dual_kline(self,h_symbol):
        sid = ('.').join(['116',h_symbol])
        self._request_kline_for_asset({'request_sid':sid,'sid':h_symbol},'dual_price')


class AdjustmentsRequest(object):

    alter_tbls = frozenset(['symbol_splits',
                  'symbol_rights',
                  'symbol_equity_structure'])

    def __init__(self):
        """
            将数据库已经存在的标的时间缓存
        """
        self._cache()

    def _cache(self):
        self.sid_deadlines = dict()
        for tbl in self.alter_tbls:
            self._retrieve_from_sqlite(tbl)

    def _retrieve_from_sqlite(self,tbl):
        table = metadata.tables[tbl]
        ins = select([func.max(table.c.declared_date),table.c.sid])
        ins = ins.groupby(table.c.sid)
        rp = engine.execute(ins)
        deadlines = pd.DataFrame(rp.fetchall(),columns = ['declared_date','sid'])
        deadlines.set_index('sid',inplace = True)
        self.sid_deadlines[tbl] = deadlines.iloc[:,0]

    def _parse_symbol_issues(self,content,code):
        """配股"""
        resource  = content['divdend']
        table = resource.find('table', {'id': 'sharebonus_2'})
        body = table.tbody
        raw = []
        [raw.append(item.get_text()) for item in body.findAll('tr')]
        if len(raw) ==1 and raw[0] == '暂时没有数据！':
            print('------------code : %s has not 配股'%code,raw[0])
        else:
            parse_raw = [item.split('\n')[1:-2] for item in raw]
            pairwise = pd.DataFrame(parse_raw, columns=['declared_date', 'rights_bonus', 'rights_price',
                                                        'benchmark_share','pay_date', 'record_date',
                                                        '缴款起始日','缴款终止日','effective_date','募集资金合计'])
            pairwise.loc[:,'sid'] = code
            max_date = self.sid_deadlines['symbol_rights'][code]
            res =  pairwise[pairwise['公告日期'] > max_date] if max_date else pairwise
            dBwriter.writer('symbol_rights', res)

    def _parse_symbol_divdend(self,content,code):
        """获取分红配股数据"""
        resource = content['divdend']
        table = resource.find('table', {'id': 'sharebonus_1'})
        body = table.tbody
        raw = []
        [raw.append(item.get_text()) for item in body.findAll('tr')]
        if len(raw) ==1 and raw[0] == '暂时没有数据！':
            print('------------code : %s has not splits and divdend'%code,raw[0])
        else:
            parse_raw = [item.split('\n')[1:-2] for item in raw]
            split_divdend = pd.DataFrame(parse_raw, columns=['declared_date', 'sid_bonus', 'sid_transfer', 'bonus',
                                                             'progress', 'pay_date', 'record_date', 'effective_date'])
            split_divdend.loc[:,'sid'] = code
            max_date = self.sid_deadlines['symbol_divdends'][code]
            res =  split_divdend[split_divdend['公告日期'] > max_date] if max_date else split_divdend
            dBwriter.writer('symbol_divdends', res)

    def _parse_symbol_equity(self,content,code):
        """获取股票股权结构分布"""
        resource = content['equity']
        tbody = resource.findAll('tbody')
        if len(tbody) == 0:
            print('due to sina error ,it raise cannot set a frame with no defined index and a scalar when tbody is null')
        equity = pd.DataFrame()
        for th in tbody:
            formatted = parse_content_from_header(th)
            equity = equity.append(formatted)
        #调整
        equity.loc[:,'代码'] = code
        equity.index = range(len(equity))
        max_date = self.sid_deadlines['symbol_equity_structure'][code]
        filter_equity= equity[equity['公告日期'] > max_date] if max_date else equity
        dBwriter.writer('symbol_equity_structure',filter_equity)

    def _request_for_sid(self,sid):
        content = dict()
        for category,path in ASSETS_STRUCTURE_DIVDEND.itmes():
            req = path%sid
            content[category] = _parse_url(req)
        return content

    def request_for_structure(self,sid):
        try:
            contents = self._request_for_sid(sid)
        except Exception as e:
            print('%s occur due to high prequency',e)
            #retry
            # time.sleep(np.random.randint(0,1))
            contents = self._request_for_sid(sid)
        self._parse_symbol_divdend(contents,sid)
        self._parse_symbol_issues(contents,sid)
        self._parse_symbol_equity(contents,sid)




class FundamentalRequest(object):

    def __init__(self):
        self._init_cache()

    def _init_cache(self):
        self._retrieve_from_sqlite()

    def _retrieve_from_sqlite(self):
        table = metadata.tables['shareHolder_change']
        ins = select([func.max(table.c.declared_date)])
        rp = engine.execute(ins)
        self.deadline = rp.scalar()

    def request_holder(self):
        """股票增持、减持、变动情况"""
        page = 1
        while True:
            url = ASSETS_EVENT_URL['shareHolder_change']%page
            raw = _parse_url(url, bs=False)
            match = re.search('\[(.*.)\]', raw)
            data = json.loads(match.group())
            data = [item.split(',')[:-1] for item in data]
            holdings = pd.DataFrame(data, columns=['代码', '中文', '现价', '涨幅', '股东', '方式', '变动股本', '占总流通比', '途径', '总持仓',
                                                   '占总股本比', '总流通股', '占流通比', '变动开始日', '变动截止日', '公告日'])

            filter_holdings = holdings[holdings['declared_date'] > self.deadline]
            if len(filter_holdings) == 0:
                break
            dBwriter.writer('shareholder',filter_holdings)
            page = page + 1

    def request_massive(self, sdate, edate):
        """
            获取时间区间内股票大宗交易，时间最好在一个月之内
        """
        newcols =['trade_dt', 'sid', 'cname', 'bid_price', 'bid_volume', 'amount', 'buyer_code',
                 'buyer','seller_code', 'seller', 'type', 'unit', 'pct', 'close', 'YSSLTAG',
                 'discount','cjeltszb','1_pct', '5_pct', '10_pct', '20_pct', 'TEXCH']
        count = 1
        prefix ='js={"data":(x)}&filter=(Stype=%27EQA%27)' + \
                '(TDATE%3E=^{}^%20and%20TDATE%3C=^{}^)'.format(sdate,edate)
        while True:
            url = ASSETS_EVENT_URL['massive']%count + prefix
            raw = _parse_url(url,bs = False,encoding=None)
            raw = json.loads(raw)
            if raw['data'] and len(raw['data']):
                massive = pd.DataFrame(raw['data'])
                massive.columns = newcols
                dBwriter.writer('massive', massive)
                count = count +1
            else:
                break

    def request_release(self, sdate, edate):
        """
            获取A股解禁数据
        """
        count = 1
        prefix = '(ltsj%3E=^{}^%20and%20ltsj%3C=^{}^)'.format(sdate,edate) +\
                  '&js={"data":(x)}'
        while True:
            url = ASSETS_EVENT_URL['release']%count + prefix
            text = _parse_url(url,encoding=None,bs = False)
            text = json.loads(text)
            if text['data'] and len(text['data']):
                info = text['data']
                raw = [[item['gpdm'],item['ltsj'],item['xsglx'],item['zb']] for item in info]
                # df = pd.DataFrame(raw,columns = ['代码','解禁时间','类型','解禁占流通市值比例'])
                release = pd.DataFrame(raw,columns = ['sid','release_date','release_type','cjeltszb'])
                dBwriter.writer('release', release)
                count = count + 1
            else:
                break