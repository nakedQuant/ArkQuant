# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
from sqlalchemy import select, func
from gateway.spider import Crawler
from gateway.spider.xml import OWNERSHIP, COLUMNS
from gateway.driver.tools import parse_content_from_header
from gateway.database.db_writer import db
from gateway.driver.tools import _parse_url


class OwnershipWriter(Crawler):

    def _retrieve_from_sqlite(self):
        table = self.metadata.tables['onwership']
        ins = select([func.max(table.c.declared_date), table.c.sid])
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
        q = self._retrieve_assets_from_sqlite()
        for asset in q['equity'].values():
            content = _parse_url(OWNERSHIP % asset)
            self._parse_equity_ownership(content, asset, deadline)
