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

    def load_stock_holdings(self,sdate,edate,asset):
        """股东持仓变动"""
        raw = self.loader.load_ashare_holdings(sdate,edate,asset)
        holding = pd.DataFrame(raw,columns = ['变动截止日','代码','变动股本','占总流通比例','总持仓','占总股本比例','总流通股'])
        return holding

    def load_ashare_mass(self, sdate, edate):
        """
            获取时间区间内股票大宗交易，时间最好在一个月之内
        """
        mass = self.extra.download_mass(sdate, edate)
        return mass

    def load_ashare_release(self, sdate, edate):
        """
            获取A股解禁数据
        """
        release = self.extra.download_release(sdate, edate)
        return release

    def load_stock_pledge(self,code):
        """股票质押率"""
        pledge = self.ts.to_ts_pledge(code)
        return pledge

    def load_ashare_hk_con(self, exchange, flag=1):
        """获取沪港通、深港通股票 , exchange 交易所 ; flag :1 最新的， 0 为历史的已经踢出的"""
        assets = self.ts.to_ts_con(exchange, flag)
        return assets

    def load_market_margin(self,sdate,edate):
        """整个A股市场融资融券余额"""
        margin = self.loader.load_market_margin(sdate,edate)
        market_margin = pd.DataFrame(margin,columns = ['交易日期','融资余额','融券余额','融资融券总额','融资融券差额'])
        return market_margin

    def load_gdp(self):
        gdp = self.extra.download_gross_value()
        gdp['总值'] = gdp['总值'].astype('float64')
        return gdp