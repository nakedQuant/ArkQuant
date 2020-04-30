#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:39:46 2019

@author: python
"""
import json,re,datetime,time,pandas as pd,numpy as np
from itertools import chain
from sqlalchemy import select
from sqlalchemy.sql import func
from abc import ABC,abstractmethod

__all__ = ['Astock','Index','ETF','Convertible']

class Spider(object):
    """
        1. kline
        2. market_margin
        3. ashare_holding
        4. ts --- delist
    """
    def download_assets(self):
        pass

    def download_kline(self):
        pass



class Ancestor(ABC):

    frequency = None
    nowdays = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')

    @classmethod
    def _init_db(cls,init = False):

        pass

    def switch_mode(self):
        """决定daily or init"""
        if self.frequency:
            lmt = 1
        else:
            lmt = 10000
        return lmt

    @abstractmethod
    def _get_prefix(self):

        raise NotImplementedError

    @abstractmethod
    def _download_assets(self):

        raise NotImplementedError

    @abstractmethod
    def _download_kline(self):

        raise NotImplementedError

    @abstractmethod
    def _run_session(self):

        raise NotImplementedError

    @abstractmethod
    def run_bulks(self):

        raise NotImplementedError


class Astock(Ancestor):
    """
        1、股票日线
        2、股票的基本信息
        3、股权结构
        4、分红配股信息
        参数 frequency : daily  表示每天都要跑 ;否则 历史数据
    """
    __name__ = 'Astock'

    def __init__(self):
        """每次进行初始化"""
        self._failure = []
        self._init_db()
        self.conn = self.db.db_init()
        self.basics_assets = self._get_stock_assets() if self.frequency else []
        self.mode = self.switch_mode()

    @staticmethod
    def _get_prefix(code):
        if code.startswith('0') or code.startswith('3'):
            prefix = '0.' + code
        elif code.startswith('6'):
            prefix = '1.' + code
        return prefix

    def _download_assets(self):
        # 获取存量股票包括退市
        html = 'http://70.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12'
        raw = json.loads(self._parse_url(html,bs= False))
        q = [item['f12'] for item in raw['data']['diff']]
        # sci_tech = list(filter(lambda x: x.startswith('688'), q))
        # tradtional = set(q) - set(sci_tech)
        return q

    def _get_announce_date(self,code,tbl_name):
        table = getattr(self.db,tbl_name)
        ins = select([func.max(table.c.公告日期).label('max_date')]).where(table.c.代码 == code)
        rp = self.conn.engine.execute(ins)
        res = rp.first()
        return res.max_date

    def _get_stock_assets(self):
        basics = getattr(self.db,'description')
        ins = select([basics.c.代码.distinct().label('code')])
        rp = self.conn.engine.execute(ins)
        basics_code = [r.code for r in rp]
        return set(basics_code)

    def _download_ipo(self):
        """获取最近新上市的股票"""
        html = 'http://81.push2.eastmoney.com/api/qt/clist/get?&pn=1&pz=100&po=1&np=1&fltt=2&invt=2&fid=f26&fs=m:0+f:8,m:1+f:8&fields=f12,f26'
        obj = self._parse_url(html,bs = False)
        raw = json.loads(obj)
        raw = [list(i.values()) for i in raw['data']['diff']]
        df = pd.DataFrame(raw, columns=['code', 'timeTomarket'])
        return df

    def _download_kline(self,code):
        """获取行情数据,默认为日频数据"""
        daily = 'http://64.push2his.eastmoney.com/api/qt/stock/kline/get?&secid={}&fields1=f1'\
               '&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&end=30000101&lmt={}'.format(self._get_prefix(code),self.mode)
        obj = self._parse_url(daily,bs =False)
        raw = json.loads(obj)
        kline = raw['data']
        if kline and len(kline['klines']):
            transform = [item.split(',') for item in kline['klines']]
            kl_pd = pd.DataFrame(transform,
                                 columns=['trade_dt', 'open', 'close', 'high', 'low', 'volume', 'turnover',
                                          'amplitude'])
            if not self.frequency or (self.frequency and self.nowdays in kl_pd['trade_dt'].values):
                kl_pd['code'] = raw['data']['code']
                self.db.enroll('kline',kl_pd,self.conn)
        else:
            print('stock code :%s have no kline due to 未上市'%code)

    def _download_bascis(self,code):
        """获取股票基础信息"""
        if not self.frequency or code not in self.basics_assets:
            # 公司基本情况
            html = 'https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/%s.phtml' % code
            obj = self._parse_url(html)
            table = obj.find('table', {'id': 'comInfo1'})
            tag = [item.findAll('td') for item in table.findAll('tr')]
            tag_chain = list(chain(*tag))
            raw = [item.get_text() for item in tag_chain]
            # 去除格式
            raw = [i.replace('：', '') for i in raw]
            raw = [i.strip() for i in raw]
            info = list(zip(raw[::2], raw[1::2]))
            info_dict = {item[0]: item[1] for item in info}
            #用于过滤股权结构分布
            self._timeTomarket = info_dict['上市日期']
            info_dict.update({'代码': code})
            #入库
            self.db.enroll('description',info_dict,self.conn)

    def _download_equity_issues(self,code):
        html = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/%s.phtml'%code
        obj = self._parse_url(html)
        self._download_splits_divdend(obj,code)
        self._download_issues(obj,code)

    def _download_issues(self,obj,code):
        """配股"""
        table = obj.find('table', {'id': 'sharebonus_2'})
        body = table.tbody
        raw = []
        [raw.append(item.get_text()) for item in body.findAll('tr')]
        if len(raw) ==1 and raw[0] == '暂时没有数据！':
            print('------------code : %s has not 配股'%code,raw[0])
        else:
            parse_raw = [item.split('\n')[1:-2] for item in raw]
            pairwise = pd.DataFrame(parse_raw, columns=['公告日期', '配股方案', '配股价格','基准股本','除权日', '股权登记日','缴款起始日','缴款终止日','配股上市日','募集资金合计'])
            pairwise.loc[:,'代码'] = code
            if self.frequency:
                max_date = self._get_announce_date(code,'pairwise')
                res =  pairwise[pairwise['公告日期'] > max_date] if max_date else pairwise
                self.db.enroll('pairwise',res,self.conn)
            else:
                self.db.enroll('pairwise',pairwise,self.conn)

    def _download_splits_divdend(self,obj,code):
        """获取分红配股数据"""
        table = obj.find('table', {'id': 'sharebonus_1'})
        body = table.tbody
        raw = []
        [raw.append(item.get_text()) for item in body.findAll('tr')]
        if len(raw) ==1 and raw[0] == '暂时没有数据！':
            print('------------code : %s has not splits and divdend'%code,raw[0])
        else:
            parse_raw = [item.split('\n')[1:-2] for item in raw]
            split_divdend = pd.DataFrame(parse_raw, columns=['公告日期', '送股', '转增', '派息', '进度', '除权除息日', '股权登记日', '红股上市日'])
            split_divdend.loc[:,'代码'] = code
            if self.frequency:
                max_date = self._get_announce_date(code,'splits_divdend')
                res =  split_divdend[split_divdend['公告日期'] > max_date] if max_date else split_divdend
                self.db.enroll('splits_divdend',res,self.conn)
            else:
                self.db.enroll('splits_divdend',split_divdend,self.conn)

    def _download_equity(self,code):
        """获取股票股权结构分布"""
        def tranform(th):
            cols = [t.get_text() for t in th.findAll('td', {'width': re.compile('[0-9]+')})]
            raw = [t.get_text() for t in th.findAll('td')]
            # xa0为空格
            raw = [''.join(item.split()) for item in raw]
            # 去除格式
            raw = [re.sub('·', '', item) for item in raw]
            # 调整字段
            raw = [re.sub('\(历史记录\)', '', item) for item in raw]
            raw = [item.replace('万股', '') for item in raw]
            # 结构处理
            num = int(len(raw) / len(cols))
            text = {}
            for count in range(len(cols)):
                idx = count * num
                mid = raw[idx:idx + num]
                text.update({mid[0]: mid[1:]})
                df = pd.DataFrame.from_dict(text)
            return df
        #解析
        html = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vCI_StockStructure/stockid/%s.phtml' % code
        obj = self._parse_url(html)
        tbody = obj.findAll('tbody')
        if len(tbody) == 0:
            print('due to sina error ,it raise cannot set a frame with no defined index and a scalar when tbody is null')
        equity = pd.DataFrame()
        for th in tbody:
            formatted = tranform(th)
            equity = equity.append(formatted)
        #调整
        equity.loc[:,'代码'] = code
        equity.index = range(len(equity))
        if self.frequency:
            max_date = self._get_announce_date(code,'equity')
            filter_equity= equity[equity['公告日期'] > max_date] if max_date else equity
            self.db.enroll('equity',filter_equity,self.conn)
        else:
            self.db.enroll('equity',equity,self.conn)

    def _run_session(self,code):
        """基于事务 基于一个连接"""
        transaction = self.conn.begin()
        try:
            self._download_kline(code)
            self._download_bascis(code)
            self._download_equity_issues(code)
            time.sleep(np.random.randint(0,2))
            self._download_equity(code)
            transaction.commit()
            print('enroll kline,splits_divdend,equity,baseinfo from code : %s'%code)
        except Exception as error:
            transaction.rollback()
            print('---------------astock error :%s'%code,error)
            self._failure.append(code)
        finally:
            transaction.close()

    def run_bulks(self):
        quantity = self._download_assets()
        for q in quantity:
            self._run_session(q)
        res = {self.__name__:self._failure}
        return res


class Index(Ancestor):
    """
        获取A股基准数据
    """
    __name__ = 'Index'

    def __init__(self):

        self._failure = []
        self._init_db()
        self.conn = self.db.db_init()
        self.mode = self.switch_mode()

    def _download_assets(self):
        url = 'http://71.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=12&po=1&np=2&fltt=2&invt=2&fid=&fs=b:MK0010&fields=f12,f14'
        raw = json.loads(self._parse_url(url,encoding='utf-8',bs= False))
        index_set = raw['data']['diff']
        return index_set

    def _get_prefix(self,k):
        if k['f12'].startswith('0'):
            prefix = '1.' + k['f12']
        else:
            prefix = '0.' + k['f12']
        return prefix

    def switch_mode(self):
        """决定daily or init"""
        if self.frequency:
            s = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')
        else:
            s = '19900101'
        return s

    def _download_kline(self,k):
        html = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={}&fields1=f1'\
               '&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={}&end=30000101'.format(self._get_prefix(k),self.mode)
        obj = self._parse_url(html,bs = False)
        data = json.loads(obj)
        raw = data['data']
        if raw and len(raw['klines']):
            raw = [item.split(',') for item in raw['klines']]
            benchmark = pd.DataFrame(raw,columns = ['trade_dt','open','close','high','low','volume','turnover','amplitude'])
            if not self.frequency or (self.frequency and self.nowdays in benchmark['trade_dt'].values):
                benchmark.loc[:,'code'] = k['f12']
                benchmark.loc[:,'name'] = k['f14']
                #入库
                self.db.enroll('index',benchmark,self.conn)
                print('enroll index :%s kline' % k)
        else:
            print('index :%s have not kline'%k)

    def _run_session(self,name):
        try:
            self._download_kline(name)
        except Exception as e:
            print('error%s enroll index :%s kline' % (name, e))
            self._failure.append(name)

    def run_bulks(self):
        quantity = self._download_assets()
        # mode = self.switch_mode()
        for k in quantity.values():
            self._run_session(k)
        res = {self.__name__:self._failure}
        return res


class ETF(Ancestor):
    """
        获取ETF列表 以及 对应的日线
    """
    __name__ = 'ETF'

    def __init__(self):
        self._failure = []
        self._init_db()
        self.conn = self.db.db_init()
        self.mode = self.switch_mode()

    def _download_assets(self):
        """获取ETF列表 page num --- 20 40 80"""
        accumulate = []
        page = 1
        while True:
            html = "http://vip.stock.finance.sina.com.cn/quotes_service/api/jsonp.php/IO.XSRV2.CallbackList['v7BkqPkwcfhoO1XH']/"\
                   "Market_Center.getHQNodeDataSimple?page=%d&num=80&sort=symbol&asc=0&node=etf_hq_fund" % page
            obj = self._parse_url(html)
            text = obj.find('p').get_text()
            mid = re.findall('s[z|h][0-9]{6}', text)
            if len(mid) > 0:
                accumulate.extend(mid)
            else:
                break
            page = page + 1
        return accumulate

    def _get_prefix(self,code):
        if code.startswith('sz'):
            prefix = '0.' + code[-6:]
        elif code.startswith('sh'):
            prefix = '1.' + code[-6:]
        return prefix

    def _download_kline(self,code):
        # 获取EFT数据
        html = 'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={}&fields1=f1'\
                   '&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&lmt={}&klt=101&fqt=1&end=30000101'.format(self._get_prefix(code),self.mode)
        obj = self._parse_url(html,bs = False)
        raw = json.loads(obj)
        etf = raw['data']
        if etf and len(etf['klines']):
            kl_pd = pd.DataFrame([item.split(',') for item in etf['klines']],
                                  columns=['trade_dt', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            if not self.frequency or (self.frequency and self.nowdays in kl_pd['trade_dt'].values):
                kl_pd['code'] = code[-6:]
                #入库
                self.db.enroll('etf',kl_pd,self.conn)
                print('enroll etf:%s kline' % code)
        else:
            print('etf :%s have not kline'%code)

    def _run_session(self,code):
        try:
            self._download_kline(code)
        except Exception as e:
            print('etf:%s kline failure' % code,e)
            self._failure.append(code)

    def run_bulks(self):
        quantity = self._download_assets()
        for q in quantity:
            self._run_session(q)
        res = {self.__name__:self._failure}
        return res


class Convertible(Ancestor):
    """
        可转换公司债券的期限最短为1年，最长为6年，自发行结束之日起6个月方可转换为公司股票
        可转债赎回：公司股票在一段时间内连续高于转换价格达到一pp-....≥....................//..
        [p.p[“≥≥≥≥≥≥≥≥≥≥≥≥≥≥≥.p[/定幅"?时，公司可按照事先约定的赎回价格买回发行在外尚未转股的可转换公司债券
        可转债强制赎回：连续三十个交易日中至少有十五个交易日的收盘价格不低于当期转股价格的 130%（含130%）；
        存在向下修订转股价
        可转债回售：最后两个计息年度内，一旦正股价持续约30天低于转股价的70%，上市公司必须以100+e 的价格来赎回可转债
        集思录获取可转债数据列表，请求方式post
        可债转基本情况 --- 强制赎回价格 回售价格  时间
    """
    __name__ = 'Convertible'

    def __init__(self):
        """每次进行初始化"""
        self._failure = []
        self._init_db()
        self.conn = self.db.db_init()
        self._basics_id = self._get_convertible_assets() if self.frequency else []
        self.mode = self.switch_mode()

    def _get_prefix(self,item):
        stock = item['cell']['stock_id']
        if stock.startswith('sz'):
            prefix = '0.' + item['id']
        elif stock.startswith('sh'):
            prefix = '1.' + item['id']
        return prefix

    def _download_assets(self):
        url = 'https://www.jisilu.cn/data/cbnew/cb_list/?'
        text = self._parse_url(url,bs = False,encoding = None)
        text = json.loads(text)
        return text['rows']

    def _get_convertible_assets(self):
        table = self.db.metadata.tables['convertibleDesc']
        ins = select([table.c.bond_id])
        rp = self.conn.engine.execute(ins)
        bond_id = [r.bond_id for r in rp]
        return set(bond_id)

    def _download_kline(self,item):
        html = 'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={}&fields1=f1' \
               '%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&lmt={}&klt=101&fqt=1&end=30000101'.format(self._get_prefix(item),self.mode)
        obj = self._parse_url(html,bs = False)
        raw = json.loads(obj)
        kline = raw['data']
        if kline and len(kline['klines']) :
            kl_pd = [item.split(',') for item in kline['klines']]
            df = pd.DataFrame(kl_pd, columns=['trade_dt', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            if not self.frequency or (self.frequency and self.nowdays in df['trade_dt'].values):
                df.loc[:, 'code'] = item['id']
                #入库
                self.db.enroll('bond_price',df,self.conn)
        else:
            print('--------------convertible :%s doesnot have kline'%item['id'])

    def _enroll_desc(self,item):
        if not self.frequency or item['id'] not in self._basics_id:
            self.db.enroll('convertible',item['cell'],self.conn)

    def _run_session(self,item):
        """基于事务 基于一个连接"""
        transaction = self.conn.begin()
        try:
            self._download_kline(item)
            self._enroll_desc(item)
            transaction.commit()
            print('crawler %s kline from 集思录'%item['id'])
        except Exception as e:
            transaction.rollback()
            print('---------- convertible error : %s'%item['id'],e)
            self._failure.append(item)
        finally:
            transaction.close()

    def run_bulks(self):
        quantity = self._download_assets()
        for q in quantity:
            self._run_session(q)
        res = {self.__name__:self._failure}
        return res


class ExtraOrdinary(Ancestor):
    """
    　　中国股市:
            9：15开始参与交易，11：30-13：00为中午休息时间，下午13：00再开始，15：00交易结束。
    　　中国香港股市:
        　　开市前时段 上午9时30分至上午10时正
        　　早市上午10时正至中午12时30分
        　　延续早市中午12时30分至下午2时30分
        　　午市下午2时30分至下午4时正
    　　美国股市：
        　　夏令时间 21：30至04：00
        　　冬令时间 22：30至05：00
    　　欧洲股市：
    　　    4点到晚上12点半
    """
    __name__ = 'ExtraOrdinary'

    def __init__(self):
        """每次进行初始化"""
        self._failure = []
        self._init_db()
        self.conn = self.db.db_init()
        self.mode = self.switch_mode()

    def _get_prefix(self,code,exchange ='hk'):
        if exchange == 'us':
            code = exchange + '.' + code
        elif exchange == 'hk':
            code = exchange + code
        else:
            raise NotImplementedError
        return code

    def switch_mode(self):
        """决定daily or init"""
        edate = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d')
        if self.frequency:
            sdate = edate
        else:
            sdate = '1990-01-01'
        return (sdate,edate)

    def _download_assets(self):
        """
            A股与H股同时上市的股票
        """
        df = pd.DataFrame()
        count = 1
        while True:
            html = 'http://19.push2.eastmoney.com/api/qt/clist/get?pn=%d&pz=20&po=1&np=1&invt=2&fid=f3&fs=b:DLMK0101&fields=f12,f191,f193'%count
            raw = self._parse_url(html,bs = False,encoding = None)
            raw = json.loads(raw)
            diff = raw['data']
            if diff and len(diff['diff']):
                diff = [[item['f12'],item['f191'],item['f193']] for item in diff['diff']]
                raw = pd.DataFrame(diff,columns = ['h_code','code','name'])
                df = df.append(raw)
                count = count +1
            else:
                break
        return df

    def _download_kline(self,asset,begin,end,mode = 'qfq'):
        """
            获取港股Kline , 针对于同时在A股上市的 , AH
        """
        tencent = 'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=%s,day,%s,%s,100000,%s'%(self._get_prefix(asset[0]),begin,end,mode)
        raw = self._parse_url(tencent,bs = False,encoding= None)
        raw = json.loads(raw)
        data = raw['data']
        if data and len(data):
            daily = [item[:6] for item in data[self._get_prefix(asset[0])]['day']]
            df = pd.DataFrame(daily,columns=['trade_dt','open','close','high','low','volume'])
            if not self.frequency or (self.frequency and self.nowdays in df['trade_dt'].values):
                df.loc[:,'h_code'] = asset[0]
                df.loc[:,'code'] = asset[1]
                df.loc[:,'category'] = mode
                self.db.enroll('hkline',df,self.conn)

    def _run_session(self,item):
        try:
            self._download_kline(item,*self.mode)
            print('hcode:%s successfully run kline'%item[0])
        except Exception as e:
            print('hcode :%s run kline failure'%item[0],e)
            self._failure.append(item)

    def run_bulks(self):
        quantity = self._download_assets()
        for r in quantity.values.tolist():
            self._run_session(r)
        res = {self.__name__:self._failure}
        return res

    def _get_max_margin_date(self):
        table = getattr(self.db,'margin')
        ins = select([func.max(table.c.交易日期)])
        rp = self.conn.engine.execute(ins)
        res = rp.first()
        return res[0]

    def download_market_margin(self):
        """
            获取沪深两市每日融资融券明细
            rzye	float	融资余额(元)
            rqye	float	融券余额(元)
            rzmre	float	融资买入额(元)
            rqyl	float	融券余量（手）
            rzche	float	融资偿还额(元)
            rqchl	float	融券偿还量(手)
            rqmcl	float	融券卖出量(股,份,手)
            rzrqye	float	融资融券余额(元)
        """
        page = 1
        margin = pd.DataFrame()
        if self.frequency:
            url = 'http://api.dataide.eastmoney.com/data/get_rzrq_lshj?orderby=dim_date&order=desc&pageindex=%d&pagesize=50'%page
            raw = self._parse_url(url,bs = False)
            raw = json.loads(raw)
            raw = [[item['dim_date'],item['rzye'],item['rqye'],item['rzrqye'],item['rzrqyecz'],item['new'],item['zdf']]for item in raw['data']]
            data = pd.DataFrame(raw,columns = ['交易日期','融资余额','融券余额','融资融券总额','融资融券差额','hs300','涨跌幅'])
            data['交易日期'] = [datetime.datetime.fromtimestamp(dt/1000) for dt in data['交易日期']]
            data['交易日期'] = [datetime.datetime.strftime(t,'%Y-%m-%d') for t in data['交易日期']]
            margin = data[data['交易日期'] > self._get_max_margin_date()]
        else:
            while True:
                url = 'http://api.dataide.eastmoney.com/data/get_rzrq_lshj?orderby=dim_date&order=desc&pageindex=%d&pagesize=50'%page
                raw = self._parse_url(url,bs = False)
                raw = json.loads(raw)
                raw = [[item['dim_date'],item['rzye'],item['rqye'],item['rzrqye'],item['rzrqyecz'],item['new'],item['zdf']]for item in raw['data']]
                data = pd.DataFrame(raw,columns = ['交易日期','融资余额','融券余额','融资融券总额','融资融券差额','hs300','涨跌幅'])
                if not len(data):
                    break
                margin = margin.append(data)
                page = page+1
                time.sleep(np.random.randint(0,3))
            #将时间戳转为具体日期
            margin['交易日期'] = [datetime.datetime.fromtimestamp(dt/1000) for dt in margin['交易日期']]
            margin['交易日期'] = [datetime.datetime.strftime(t,'%Y-%m-%d') for t in margin['交易日期']]
            margin.index = range(len(margin))
        self.db.enroll('margin',margin,self.conn)

    def _get_holding_max_deadline(self):
        table = getattr(self.db,'holding')
        ins = select([func.max(table.c.变动截止日)])
        rp = self.conn.engine.execute(ins)
        res = rp.first()
        return res[0]

    def download_ashare_holding(self):
        """股票增持、减持、变动情况"""
        page = 1
        if self.frequency:
            while True:
                url = "http://data.eastmoney.com/DataCenter_V3/gdzjc.ashx?pagesize=50&page=%d&param=&sortRule=-1&sortType=BDJZ"%page
                raw = self._parse_url(url, bs=False)
                match = re.search('\[(.*.)\]', raw)
                data = json.loads(match.group())
                data = [item.split(',')[:-1] for item in data]
                holdings = pd.DataFrame(data, columns=['代码', '中文', '现价', '涨幅', '股东', '方式', '变动股本', '占总流通比例', '途径', '总持仓', '占总股本比例','总流通股', '占流通比例', '变动开始日', '变动截止日', '公告日'])
                holdings = holdings[holdings['变动截止日'] > self._get_holding_max_deadline()]
                if len(holdings) == 0:
                    break
                # 入库
                self.db.enroll('holding', holdings, self.conn)
                page = page + 1
                time.sleep(np.random.randint(0,3))
        else:
            while True:
                url = "http://data.eastmoney.com/DataCenter_V3/gdzjc.ashx?pagesize=50&page=%d&param=&sortRule=-1&sortType=BDJZ"%page
                raw = self._parse_url(url, bs=False)
                match = re.search('\[(.*.)\]', raw)
                if match:
                    data = json.loads(match.group())
                    data = [item.split(',')[:-1] for item in data]
                    holdings = pd.DataFrame(data, columns=['代码', '中文', '现价', '涨幅', '股东', '方式', '变动股本', '占总流通比例', '途径', '总持仓', '占总股本比例','总流通股', '占流通比例', '变动开始日', '变动截止日', '公告日'])
                    if not len(holdings):
                        break
                    # 入库
                    self.db.enroll('holding', holdings, self.conn)
                    print('page : %d' % page, holdings.head())
                    page = page + 1
                    time.sleep(np.random.randint(5,10))
                else:
                    print('match is null')
                    break

from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib,sys,time


class Orient:
    """
        Orient 模块用于初始化数据库并基于Spider批量运行入库操作
        数据库里面ashareEquity里面1900-01-01，5条错误记录 ，
        splitsDivdend --- 进度为实施:
        000001 --- 除权除息日1991-04-03  化为 1991-05-02 ，登记日 1991-03-12 调整为 1991-04-30
        000001 --- 删除1900-01-01 ，增加 修订 (1992-03-04 5 0 2  1992-03-23 1992-03-20)
        000007 --- 1900-01-01 修订 1992-10-22
    """
    nowdays = time.strftime('%Y-%m-%d',time.localtime())

    @classmethod
    def set_mode(cls,init = False):
        if init:
            """初始化将spider 里面 table drop 重新创建"""
            Ancestor._init_db(init)
        else:
            Ancestor.frequency = 'daily'
        cls.frequency = Ancestor.frequency

    def __init__(self):
        self.module_names = ['Index', 'Astock', 'ETF', 'Convertible','ExtraOrdinary']

    @staticmethod
    def import_cls(name):
        """获取Crawler里面的类"""
        sys.path.append('/Users/python/PycharmProjects/git/Alert/GateWay/Spider')
        module = importlib.import_module('Crawler', 'GateWay.Spider')
        cls = getattr(module, name)
        return cls

    def _parallel(self):
        restart = dict()
        def run(attr):
            module = self.import_cls(attr)
            cls = module()
            r = cls.run_bulks()
            restart.update(r)

        with ThreadPoolExecutor(max_workers = 3) as executor:
            to_do = []
            # 线程池
            for name in self.module_names:
                future = executor.submit(run, name)
                to_do.append(future)
            # 线程处理
            for f in as_completed(to_do):
                f.result()
        return restart

    def rerun(self,name,objects):
        """基于跑失败的参数重新执行爬虫"""
        module = self.import_cls(name)
        print('---------load module:',module)
        cls = module()
        for asset in objects:
            cls._run_session(asset)
        record = cls._failure
        if len(record):
            time.sleep(3)
            self.rerun(name,record)
        else:
            print('module :%s run successfully'%name)

    def update_extra(self):
        """更新非基础数据的模块 --- 市值、市场融资融券、股票增减持、交易日、股票状态"""
        m = MarketValue(self.frequency,self.nowdays)
        m.parallel()
        print('update market value via MarketValue')
        """更新市场融资融券和股东增减持"""
        module = self.import_cls('ExtraOrdinary')
        module().download_market_margin()
        print('update market_margin daily successfully')
        ts =  TushareClient()
        ts.update_via_ts()
        print('update trade_dt and status successfully via TushareClient')
        module().download_ashare_holding()
        print('update ashare_holding daily successfully')

    def initialize(self):
        """
            执行爬虫任务
        """
        targets = self._parallel()
        print(targets)
        if len(targets):
            for _name,obj in targets.items():
                self.rerun(_name,obj)
        # self.update_extra()