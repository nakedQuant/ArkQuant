# -*- coding : utf-8 -*-

from sqlalchemy import MetaData,select,func
import json,re,pandas as pd
from gateWay.driver.reconstruct import (_parse_url, parse_content_from_header)
from gateWay.driver.db_writer import  DBWriter

from gateWay.driver._config import ASSET_FUNDAMENTAL_URL

dBwriter = DBWriter()


class FundamentalWriter(object):

    def __init__(self,engine):
        self.engine = engine

    @property
    def metadata(self):
        return MetaData(bind = self.engine)

    def _init_cache(self):
        self._retrieve_from_sqlite()

    def _retrieve_from_sqlite(self):
        self.deadline_cache = {}
        for tbl in ['shareholder','equity_structure']:
            table = self.metadata.tables[tbl]
            ins = select([func.max(table.c.declared_date),table.c.sid])
            rp = self.engine.execute(ins)
            deadlines = pd.DataFrame(rp.fetchall(), columns=['declared_date', 'sid'])
            deadlines.set_index('sid', inplace=True)
            self.deadline_cache[tbl] = deadlines

    def _retrieve_equities_from_sqlite(self):
        table = self.metadata.tables['asset_router']
        ins = select([table.c.sid])
        ins = ins.where(table.c.asset_type == 'equity')
        rp = self.engine.execute(ins)
        equities = [ r[0] for r in rp.fetchall()]
        return equities

    def _parse_symbol_equity(self,content,code):
        """获取股票股权结构分布"""
        resource = content['equity']
        tbody = resource.findAll('tbody')
        if len(tbody) == 0:
            print('due to sina error ,it raise cannot set a frame with no defined index and a scalar when tbody is null')
        equity = pd.DataFrame()
        for th in tbody:
            formatted = parse_content_from_header(th)
            equity = equity.append(formatted)
        #调整
        equity.loc[:,'sid'] = code
        equity.index = range(len(equity))
        max_date = self.deadline_cache['equity_structure'][code]
        _equity= equity[equity['公告日期'] > max_date] if max_date else equity
        # 需要rename cols
        dBwriter.writer('equity_structure',_equity)

    def request_structure(self):
        assets = self._retrieve_equities_from_sqlite()
        for asset in assets:
            _url = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vCI_StockStructure/stockid/%s.phtml'%asset
            content = _parse_url(_url)
            self._parse_symbol_equity(content,asset)

    def request_holder(self):
        """股票增持、减持、变动情况"""
        page = 1
        while True:
            url = ASSET_FUNDAMENTAL_URL['shareholder']%page
            raw = _parse_url(url, bs=False)
            match = re.search('\[(.*.)\]', raw)
            data = json.loads(match.group())
            data = [item.split(',')[:-1] for item in data]
            holdings = pd.DataFrame(data, columns=['代码', '中文', '现价', '涨幅', '股东', '方式', '变动股本', '占总流通比', '途径', '总持仓',
                                                   '占总股本比', '总流通股', '占流通比', '变动开始日', '变动截止日', '公告日'])

            filter_holdings = holdings[holdings['declared_date'] > self.deadline_cache['shareholder'].max()]
            if len(filter_holdings) == 0:
                break
            dBwriter.writer('shareholder',filter_holdings)
            page = page + 1

    def request_massive(self, sdate, edate):
        """
            获取时间区间内股票大宗交易，时间最好在一个月之内
        """
        newcols =['trade_dt', 'sid', 'cname', 'bid_price', 'bid_volume', 'amount', 'buyer_code',
                 'buyer','seller_code', 'seller', 'type', 'unit', 'pct', 'close', 'YSSLTAG',
                 'discount','cjeltszb','1_pct', '5_pct', '10_pct', '20_pct', 'TEXCH']
        count = 1
        prefix ='js={"data":(x)}&filter=(Stype=%27EQA%27)' + \
                '(TDATE%3E=^{}^%20and%20TDATE%3C=^{}^)'.format(sdate,edate)
        while True:
            url = ASSET_FUNDAMENTAL_URL['massive']%count + prefix
            raw = _parse_url(url,bs = False,encoding=None)
            raw = json.loads(raw)
            if raw['data'] and len(raw['data']):
                massive = pd.DataFrame(raw['data'])
                massive.columns = newcols
                dBwriter.writer('massive', massive)
                count = count +1
            else:
                break

    def request_release(self, sdate, edate):
        """
            获取A股解禁数据
        """
        count = 1
        prefix = '(ltsj%3E=^{}^%20and%20ltsj%3C=^{}^)'.format(sdate,edate) +\
                  '&js={"data":(x)}'
        while True:
            url = ASSET_FUNDAMENTAL_URL['release']%count + prefix
            text = _parse_url(url,encoding=None,bs = False)
            text = json.loads(text)
            if text['data'] and len(text['data']):
                info = text['data']
                raw = [[item['gpdm'],item['ltsj'],item['xsglx'],item['zb']] for item in info]
                release = pd.DataFrame(raw,columns = ['sid','release_date','release_type','cjeltszb'])
                dBwriter.writer('release', release)
                count = count + 1
            else:
                break