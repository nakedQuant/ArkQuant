# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
from gateway.spider import Crawler
from gateway.spider.url import OWNERSHIP
from gateway.driver.tools import parse_content_from_header
from gateway.database.db_writer import db
from gateway.driver.tools import _parse_url

__all__ = ['OwnershipWriter']

# ownership
OwnershipFields = {'变动日期': 'ex_date',
                   '公告日期': 'declared_date',
                   '总股本': 'general',
                   '流通A股': 'float',
                   ' 高管股': 'manager',
                   '限售A股': 'strict',
                   '流通B股': 'b_float',
                   '限售B股': 'b_strict',
                   '流通H股': 'h_float'}


class OwnershipWriter(Crawler):

    def __init__(self):
        self.deadlines = None
        self.missed = set()

    def _parse_equity_ownership(self, content, symbol):
        """获取股票股权结构分布"""
        frame = pd.DataFrame()
        tbody = content.findAll('tbody')
        if len(tbody) == 0:
            print('cannot set a frame with no defined index and a scalar when tbody is null')
        for th in tbody:
            formatted = parse_content_from_header(th)
            frame = frame.append(formatted)
        # rename columns
        frame.rename(columns=OwnershipFields, inplace=True)
        # 调整
        frame.loc[:, 'sid'] = symbol
        frame.index = range(len(frame))
        deadline_date = self.deadlines.get(symbol, None)
        equity = frame[frame['declared_date'] > deadline_date] if deadline_date else frame
        db.writer('ownership', equity)

    def rerun(self):
        if len(self.missed):
            print('missing', self.missed)
            missed = self.missed.copy()
            # RuntimeError: Set changed size during iteration
            self._writer_internal(missed)
            self.rerun()
        self.missed = set()

    def _writer_internal(self, equities):
        for sid in equities:
            try:
                content = _parse_url(OWNERSHIP % sid)
                self._parse_equity_ownership(content, sid)
                print('successfully spider ownership of code : %s' % sid)
            except Exception as e:
                print('spider code: % s  ownership failure due to % r' % (sid, e))
                self.missed.add(sid)
            else:
                self.missed.discard(sid)

    def writer(self):
        # initialize deadline
        self.deadlines = self._retrieve_deadlines_from_sqlite('ownership')
        equities = self._retrieve_assets_from_sqlite()['equity']
        self._writer_internal(equities)
        self.rerun()


# if __name__ == '__main__':
#
#     w = OwnershipWriter()
#     w.writer()
