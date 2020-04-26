import os,datetime,pandas as pd,json,bcolz
from functools import partial
from decimal import Decimal
from gateWay.spider.ts_engine import TushareClient
from gateWay.spider import spider_engine
from env.common import XML

class BarReader:

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

    def load_gdp(self):
        gdp = self.extra.download_gross_value()
        gdp['总值'] = gdp['总值'].astype('float64')
        return gdp

    def load_market_margin(self,sdate,edate):
        """整个A股市场融资融券余额"""
        table = self.tables['marketMargin']
        rp = select([table.c.交易日期,cast(table.c.融资余额,Integer),cast(table.c.融券余额,Integer),cast(table.c.融资融券总额,Integer),cast(table.c.融资融券差额,Integer)]).where\
            (table.c.交易日期.between(start,end)).execute().fetchall()
        res = rp.fetchall()
        market_margin = pd.DataFrame(res,columns = ['交易日期','融资余额','融券余额','融资融券总额','融资融券差额'])
        return market_margin

    def load_ashare_holdings(self,sdate,edate,asset):
        """股东持仓变动"""
        table = self.tables['ashareHolding']
        if asset:
            ins = select(
                [table.c.变动截止日,table.c.代码, cast(table.c.变动股本, Numeric(10,2)), cast(table.c.占总流通比例, Numeric(10,5)), cast(table.c.总持仓, Numeric(10,2)),cast(table.c.占总股本比例,Numeric(10,5)),\
                 cast(table.c.总流通股, Numeric(10,2))]).where \
                (and_(table.c.代码 == asset,table.c.变动截止日.between(sdate,edate)))
        else:
            ins = select(
                [table.c.变动截止日,table.c.代码, cast(table.c.变动股本, Numeric(10,2)), cast(table.c.占总流通比例, Numeric(10,5)), cast(table.c.总持仓, Numeric(10,2)),cast(table.c.占总股本比例,Numeric(10,5)),\
                 cast(table.c.总流通股, Numeric(10,2))]).where \
                (table.c.变动截止日.between(sdate,edate))
        rp = self._proc(ins)
        margin = rp.fetchall()
        holding = pd.DataFrame(raw,columns = ['变动截止日','代码','变动股本','占总流通比例','总持仓','占总股本比例','总流通股'])
        return margin