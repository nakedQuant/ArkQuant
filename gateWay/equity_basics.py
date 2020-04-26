import os,datetime,pandas as pd,json,bcolz
from functools import partial
from decimal import Decimal
from gateWay.spider.ts_engine import TushareClient
from gateWay.spider import spider_engine
from env.common import XML

class BarReader:

    def __init__(self):
        self.loader = Core()
        self.ts = TushareClient()
        self.extra = spider_engine.ExtraOrdinary()

    def _verify_fields(self,f,asset):
        """如果asset为空，fields必须asset"""
        field = f.copy()
        if not isinstance(field,list):
            raise TypeError('fields must be list')
        elif asset is None:
            field.append('code')
        return field

    def load_ashare_basics(self, sid=None):
        basics = self.loader.load_stock_basics(sid)
        return basics

    def load_convertible_basics(self, bond):
        brief = self.loader.load_convertible_basics(bond)
        return brief

    def load_equity_info(self, sid):
        raw = self.loader.load_equity_structure(sid)
        structure = pd.DataFrame(raw,columns = ['code','change_dt','announce_dt','total_assets','float_assets','strict_assets','b_assets','h_assets'])
        return structure
