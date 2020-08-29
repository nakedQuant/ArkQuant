# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
from sqlalchemy import select, func
from gateway.spider import Crawler
from gateway.spider.url import OWNERSHIP
from gateway.driver.tools import parse_content_from_header
from gateway.database.db_writer import db
from gateway.driver.tools import _parse_url

__all__ = ['OwnershipWriter']

# ownership
COLUMNS = {'变动日期': 'ex_date',
           '公告日期': 'declared_date',
           '总股本': 'general',
           '流通A股': 'float',
           ' 高管股': 'manager',
           '限售A股': 'strict',
           '流通B股': 'b_float',
           '限售B股': 'b_strict',
           '流通H股': 'h_float'}


class OwnershipWriter(Crawler):

    def _retrieve_deadlines_from_sqlite(self):
        table = self.metadata.tables['ownership']
        ins = select([func.max(table.c.declared_date), table.c.sid])
        ins = ins.group_by(table.c.sid)
        rp = self.engine.execute(ins)
        deadlines = pd.DataFrame(rp.fetchall(), columns=['declared_date', 'sid'])
        deadlines.set_index('sid', inplace=True)
        return deadlines

    @staticmethod
    def _parse_equity_ownership(content, symbol, deadline):
        """获取股票股权结构分布"""
        frame = pd.DataFrame()
        tbody = content.findAll('tbody')
        if len(tbody) == 0:
            print('cannot set a frame with no defined index and a scalar when tbody is null')
        for th in tbody:
            formatted = parse_content_from_header(th)
            frame = frame.append(formatted)
        print('frame', frame)
        # rename columns
        frame.rename(columns=COLUMNS, inplace=True)
        # 调整
        frame.loc[:, 'sid'] = symbol
        frame.index = range(len(frame))
        deadline_date = deadline.get(symbol, None)
        equity = frame[frame['公告日期'] > deadline_date] if deadline_date else frame
        db.writer('ownership', equity)

    def writer(self):
        # initialize deadline
        deadline = self._retrieve_deadlines_from_sqlite()
        q = self._retrieve_assets_from_sqlite()
        equities = q['equity'].values()
        # equities = ['300357']
        for asset in equities:
            content = _parse_url(OWNERSHIP % asset)
            self._parse_equity_ownership(content, asset, deadline)


if __name__ == '__main__':

    w = OwnershipWriter()
    w.writer()
