# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from sqlalchemy import select, func
import json, re, pandas as pd
from gateway.spider import Crawler
from gateway.database.db_writer import db
from gateway.spider.xml import ASSET_FUNDAMENTAL_URL, HolderFields


class HolderWriter(Crawler):

    def _retrieve_from_sqlite(self):
        table = self.metadata.tables['ownership']
        ins = select([func.max(table.c.declared_date), table.c.sid])
        rp = self.engine.execute(ins)
        deadlines = pd.DataFrame(rp.fetchall(), columns=['declared_date', 'sid'])
        deadlines.set_index('sid', inplace=True)
        return deadlines

    def writer(self):
        """股票增持、减持、变动情况"""
        deadline = self._retrieve_from_sqlite()
        page = 1
        while True:
            url = ASSET_FUNDAMENTAL_URL['shareholder'] % page
            raw = self.tool(url, bs=False)
            match = re.search('\[(.*.)\]', raw)
            data = json.loads(match.group())
            data = [item.split(',')[:-1] for item in data]
            frame = pd.DataFrame(data, columns=HolderFields)
            deadline = deadline['shareholder'].max()
            holdings = frame[frame['declared_date'] > deadline] if deadline else frame
            if len(holdings) == 0:
                break
            db.writer('shareholder', holdings)
            page = page + 1
