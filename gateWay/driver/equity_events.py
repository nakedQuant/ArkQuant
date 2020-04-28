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




    def download_gross_value(self):
        page = 1
        gdp = pd.DataFrame()
        while True:
            html = 'http://data.eastmoney.com/cjsj/grossdomesticproduct.aspx?p=%d'%page
            obj = self._parse_url(html)
            raw = obj.findAll('div', {'class': 'Content'})
            text = [t.get_text() for t in raw[1].findAll('td')]
            text = [item.strip() for item in text]
            data = zip(text[::9], text[1::9])
            data = pd.DataFrame(data, columns=['季度', '总值'])
            gdp = gdp.append(data)
            if len(gdp) != len(gdp.drop_duplicates(ignore_index=True)):
                gdp.drop_duplicates(inplace = True,ignore_index= True)
                return gdp
            page = page +1

    def download_mass(self,sdate,edate):
        """
            获取每天股票大宗交易
        """
        df = pd.DataFrame()
        count = 1
        while True:
            html = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=DZJYXQ&' \
                   'token=70f12f2f4f091e459a279469fe49eca5&cmd=&st=SECUCODE&sr=1&p=%d&ps=50&'%count +\
                   'js={"data":(x)}&filter=(Stype=%27EQA%27)'+'(TDATE%3E=^{}^%20and%20TDATE%3C=^{}^)'.format(sdate,edate)
            raw = self._parse_url(html,bs = False,encoding=None)
            raw = json.loads(raw)
            if raw['data'] and len(raw['data']):
                mass = pd.DataFrame(raw['data'])
                df = df.append(mass)
                count = count +1
            else:
                break
        df.index = range(len(df))
        return df

    def download_release(self,sdate,edate):
        """
            获取每天A股的解禁
        """
        release = pd.DataFrame()
        count = 1
        while True:
            html = 'http://dcfm.eastmoney.com/EM_MutiSvcExpandInterface/api/js/get?type=XSJJ_NJ_PC' \
                   '&token=70f12f2f4f091e459a279469fe49eca5&st=kjjsl&sr=-1&p=%d&ps=10&filter=(mkt=)'%count + \
                   '(ltsj%3E=^{}^%20and%20ltsj%3C=^{}^)'.format(sdate,edate) + '&js={"data":(x)}'
            text = self._parse_url(html,encoding=None,bs = False)
            text = json.loads(text)
            if text['data'] and len(text['data']):
                info = text['data']
                raw = [[item['gpdm'],item['ltsj'],item['xsglx'],item['zb']] for item in info]
                df = pd.DataFrame(raw,columns = ['代码','解禁时间','类型','解禁占流通市值比例'])
                release = release.append(df)
                count = count + 1
            else:
                break
        release.index = range(len(release))
        return release