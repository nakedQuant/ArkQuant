# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
from sqlalchemy import select, func
from gateway.spider.base import Crawler
from gateway.spider.xml import OWNERSHIP
from gateway.spider import COLUMNS
from gateway.driver.tools import parse_content_from_header
from gateway.database.db_writer import db


class OwnershipWriter(Crawler):

    def _retrieve_from_sqlite(self):
        table = self.metadata.tables['onwership']
        ins = select([func.max(table.c.declared_date), table.c.sid])
        rp = self.engine.execute(ins)
        deadlines = pd.DataFrame(rp.fetchall(), columns=['declared_date', 'sid'])
        deadlines.set_index('sid', inplace=True)
        return deadlines

    def _retrieve_equities_from_sqlite(self):
        table = self.metadata.tables['asset_router']
        ins = select([table.c.sid])
        ins = ins.where(table.c.asset_type == 'equity')
        rp = self.engine.execute(ins)
        equities = [r[0] for r in rp.fetchall()]
        return equities

    @staticmethod
    def _parse_equity_ownership(content, symbol, deadline):
        """获取股票股权结构分布"""
        frame = pd.DataFrame()
        tbody = content.findAll('tbody')
        if len(tbody) == 0:
            print('due to sina error ,it raise cannot set a frame with no defined index and a scalar when tbody is null')
        for th in tbody:
            formatted = parse_content_from_header(th)
            frame = frame.append(formatted)
        # rename columns
        frame.rename(columns=COLUMNS, inplace=True)
        #调整
        frame.loc[:, 'sid'] = symbol
        frame.index = range(len(frame))
        deadline_date = deadline['equity_structure'][symbol]
        equity = frame[frame['公告日期'] > deadline_date] if deadline_date else frame
        db.writer('equity_structure', equity)

    def writer(self):
        # initialize deadline
        deadline = self._retrieve_from_sqlite()
        # obtain asset
        assets = self._retrieve_equities_from_sqlite()
        for asset in assets:
            content = self.tool(OWNERSHIP % asset)
            self._parse_equity_ownership(content, asset, deadline)
