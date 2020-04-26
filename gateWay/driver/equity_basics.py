
import os,datetime,pandas as pd,json,bcolz
from functools import partial
from decimal import Decimal
from gateWay.spider.ts_engine import TushareClient
from gateWay.spider import spider_engine
from env.common import XML

def load_stock_basics(self, asset):
    """ 股票基础信息"""
    table = self.tables['ashareInfo']
    if asset:
        ins = select([table]).where(table.c.代码 == asset)
    else:
        ins = select([table])
    rp = self._proc(ins)
    basics = rp.fetchall()
    return basics


def load_convertible_basics(self, asset):
    """ 可转债基础信息"""
    table = self.tables['convertibleDesc']
    ins = select([table]).where(table.c.bond_id == asset)
    rp = self._proc(ins)
    basics = rp.fetchall()
    return basics[0]


def load_equity_structure(self, asset):
    """
        股票的总股本、流通股本，公告日期,变动日期结构
        Warning: (1366, "Incorrect DECIMAL value: '0' for column '' at row -1")
        Warning: (1292, "Truncated incorrect DECIMAL value: '--'")
        --- 将 -- 变为0
    """
    table = self.tables['ashareEquity']
    ins = select([table.c.代码, table.c.变动日期, table.c.公告日期, cast(table.c.总股本, Numeric(20, 3)).label('总股本'),
                  cast(table.c.流通A股, Numeric(20, 3)), cast(table.c.限售A股, Numeric(20, 3)),
                  cast(table.c.流通B股, Numeric(20, 3)), cast(table.c.流通H股, Numeric(20, 3))]).where(table.c.代码 == asset)
    rp = self._proc(ins)
    equtiy = rp.fetchall()
    return equtiy
