
from sqlalchemy import select,func,MetaData
import pandas as pd,time,numpy as np

from gateWay.driver.db_writer import  DBWriter
from gateWay.driver.tools import  _parse_url
from gateWay.driver._config import  EQUITY_DIVDEND

dBwriter = DBWriter()


class AdjustmentsWriter(object):

    adjustment_tbls = frozenset(['equity_splits','equity_rights'])

    def __init__(self,engine):
        self.engine = engine
        self._metadata = MetaData(bind = engine)

    @property
    def metadata(self):
        return self._metadata

    def _update_cache(self):
        self.sid_deadlines = dict()
        for tbl in self.adjustment_tbls:
            self._retrieve_from_sqlite(tbl)

    def _retrieve_from_sqlite(self,tbl):
        table = self.metadata.tables[tbl]
        ins = select([func.max(table.c.declared_date),table.c.sid])
        # ins = ins.groupby(table.c.sid)
        rp = self.engine.execute(ins)
        deadlines = pd.DataFrame(rp.fetchall(),columns = ['declared_date','sid'])
        deadlines.set_index('sid',inplace = True)
        self._deadlines_for_sid[tbl] = deadlines.iloc[:,0]

    def _parse_equity_issues(self,content,code):
        """配股"""
        table = content.find('table', {'id': 'sharebonus_2'})
        body = table.tbody
        raw = []
        [raw.append(item.get_text()) for item in body.findAll('tr')]
        if len(raw) ==1 and raw[0] == '暂时没有数据！':
            print('------------code : %s has not 配股'%code,raw[0])
        else:
            parse_raw = [item.split('\n')[1:-2] for item in raw]
            pairwise = pd.DataFrame(parse_raw, columns=['declared_date', 'rights_bonus', 'rights_price',
                                                        'benchmark_share','pay_date', 'record_date',
                                                        '缴款起始日','缴款终止日','effective_date','募集资金合计'])
            pairwise.loc[:,'sid'] = code
            max_date = self._deadlines_for_sid['equity_rights'][code]
            res =  pairwise[pairwise['公告日期'] > max_date] if max_date else pairwise
            dBwriter.writer('equity_rights', res)

    def _parse_equity_divdend(self,content,code):
        """获取分红配股数据"""
        table = content.find('table', {'id': 'sharebonus_1'})
        body = table.tbody
        raw = []
        [raw.append(item.get_text()) for item in body.findAll('tr')]
        if len(raw) ==1 and raw[0] == '暂时没有数据！':
            print('------------code : %s has not splits and divdend'%code,raw[0])
        else:
            parse_raw = [item.split('\n')[1:-2] for item in raw]
            split_divdend = pd.DataFrame(parse_raw, columns=['declared_date', 'sid_bonus', 'sid_transfer', 'bonus',
                                                             'progress', 'pay_date', 'record_date', 'effective_date'])
            split_divdend.loc[:,'sid'] = code
            max_date = self._deadlines_for_sid['equity_divdends'][code]
            res =  split_divdend[split_divdend['公告日期'] > max_date] if max_date else split_divdend
            dBwriter.writer('equity_divdends', res)

    def _request_for_sid(self,sid):
        req_url = EQUITY_DIVDEND%sid
        content = _parse_url(req_url)
        return content

    def writer(self,sid):
        try:
            contents = self._request_for_sid(sid)
        except Exception as e:
            print('%s occur due to high prequency',e)
            time.sleep(np.random.randint(0,1))
            contents = self._request_for_sid(sid)
        #获取数据库的最新时点
        self._update_cache()
        #解析网页内容
        self._parse_symbol_divdend(contents,sid)
        self._parse_symbol_issues(contents,sid)
