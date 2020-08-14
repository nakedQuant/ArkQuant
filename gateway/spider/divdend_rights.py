# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from sqlalchemy import select, func
import pandas as pd, time, numpy as np
from gateway.spider.base import Crawler
from gateway.database.db_writer import db
from gateway.spider.xml import DIVDEND


class AdjustmentsWriter(Crawler):
    """
        a. obtain mysql divdends , splits , rights
        b. request sina
        c. update mysql
    """
    adjustment_tables = frozenset(['equity_splits', 'equity_rights'])

    def __init__(self, delay=1):
        """
        :param delay: int --- sleep for seconds
        """
        self.delay = delay
        self.deadlines = dict()

    def __setattr__(self, key, value):
        raise NotImplementedError()

    def _record_deadlines(self):
        """
            record the declared date of equities in mysql
        """
        for tbl in self.adjustment_tables:
            self._retrieve_from_sqlite(tbl)

    def _retrieve_from_sqlite(self, tbl):
        table = self.metadata.tables[tbl]
        ins = select([func.max(table.c.declared_date), table.c.sid])
        rp = self.engine.execute(ins)
        deadlines = pd.DataFrame(rp.fetchall(), columns=['declared_date', 'sid'])
        deadlines.set_index('sid', inplace=True)
        self.deadlines[tbl] = deadlines.iloc[:, 0]

    def _parse_equity_issues(self, content, symbol):
        """配股"""
        raw = list()
        table = content.find('table', {'id': 'sharebonus_2'})
        [raw.append(item.get_text()) for item in table.tbody.findAll('tr')]
        if len(raw) == 1 and raw[0] == '暂时没有数据！':
            print('------------code : %s has not 配股' % symbol, raw[0])
        else:
            delimeter = [item.split('\n')[1:-2] for item in raw]
            frame = pd.DataFrame(delimeter, columns=['declared_date', 'rights_bonus', 'rights_price',
                                                        'benchmark_share', 'pay_date', 'record_date',
                                                        '缴款起始日', '缴款终止日', 'effective_date', '募集资金合计'])
            frame.loc[:, 'sid'] = symbol
            deadline = self.deadlines['equity_rights'][symbol]
            rights = frame[frame['公告日期'] > deadline] if deadline else frame
            db.writer('equity_rights', rights)

    def _parse_equity_divdend(self, content, sid):
        """获取分红配股数据"""
        raw = list()
        table = content.find('table', {'id': 'sharebonus_1'})
        [raw.append(item.get_text()) for item in table.tbody.findAll('tr')]
        if len(raw) == 1 and raw[0] == '暂时没有数据！':
            print('------------code : %s has not splits and divdend' % sid, raw[0])
        else:
            delimeter = [item.split('\n')[1:-2] for item in raw]
            frame = pd.DataFrame(delimeter, columns=['declared_date', 'sid_bonus', 'sid_transfer', 'bonus',
                                                             'progress', 'pay_date', 'record_date', 'effective_date'])
            frame.loc[:, 'sid'] = sid
            deadline = self.deadlines['equity_divdends'][sid]
            divdends = frame[frame['公告日期'] > deadline] if deadline else frame
            db.writer('equity_divdends', divdends)

    def writer(self, sid):
        try:
            contents = self.tool(DIVDEND % sid)
        except Exception as e:
            print('%s occur due to high prequency', e)
            time.sleep(np.random.randint(0, 1))
            contents = self.tool(DIVDEND % sid)
        #获取数据库的最新时点
        self._record_deadlines()
        #解析网页内容
        self._parse_equity_issues(contents, sid)
        self._parse_equity_divdend(contents, sid)
        # reset dict
        self.deadlines.clear()
