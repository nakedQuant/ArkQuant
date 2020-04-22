# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import tushare as ts
from gateWay.driver import DataLayer

class TushareClient:
    """
        query calendar , industry , supendInfo from tushare
        token = '7325ca7b347c682eabdd9e9335f16526d01f6dff2de6ed80792cde25'
    """

    token = '7325ca7b347c682eabdd9e9335f16526d01f6dff2de6ed80792cde25'

    def __init__(self,_token = '7325ca7b347c682eabdd9e9335f16526d01f6dff2de6ed80792cde25'):

        ts.set_token(_token)
        self.pro = ts.pro_api()

    def _get_prefix(self,code):
        if code.startswith('6'):
            code = ('.').join([code,'SH'])
        else:
            code = ('.').join([code,'SZ'])
        return code

    def to_ts_status(self,status):
        """status : D -- delist ; P --- suspend """
        abnormal = self.pro.stock_basic(exchange='', list_status=status, fields='symbol,name,delist_date')
        abnormal.columns = ['code', 'name', 'delist_date']
        abnormal.loc[:, 'status'] = status
        abnormal.fillna('',inplace=True)
        return abnormal

    def to_ts_calendar(self,s,e,):
        """获取交易日, SSE , SZSE"""
        calendar = self.pro.trade_cal(start_date = s,end_date = e,exchange = 'SSE',is_open = '1')
        calendar.columns = ['exchange','trade_dt','is_open']
        return calendar

    def to_ts_con(self,exchange,new):
        """获取沪港通和深港通股票数据 , is_new = 1 表示沪港通的标的， is_new = 0 表示已经被踢出的沪港通的股票,exchange : SH SZ"""
        const = self.pro.hs_const(hs_type = exchange,is_new = new)
        return const

    def to_ts_index_component(self,index,sdate,edate):
        s = sdate.replace('-','')
        e = edate.replace('-','')
        """基准成分股以及对应权重"""
        # df = self.pro.index_weight(index_code='399300.SZ', start_date='20180901', end_date='20180930')
        df = self.pro.index_weight(index_code=index, start_date = s, end_date = e)
        return df

    def to_ts_pledge(self,code):
        """股票质押率"""
        code = self._get_prefix(code)
        df = self.pro.pledge_stat(ts_code=code)
        return df

    def to_ts_suspend(self,date,code=''):
        """缺点 --- 不定期更新"""
        if code:
            asset = self._get_prefix(code)
        else:
            asset = code
        d = date.replace('-','')
        suspend = self.pro.suspend(ts_code=asset, suspend_date=d, resume_date='', ann_date = '')
        return suspend

    def to_ts_coef(self,code):
        code = self._get_prefix(code)
        coef = self.pro.adj_factor(ts_code = code)
        return coef

    def update_via_ts(self):
        """
            基于tushare模块对股票退市或者暂停上市的状态更新
            暴力更新 获取 清空 入库
        """
        db = DataLayer()
        try:
            #退市、暂停上市
            delist = self.to_ts_status('D')
            suspend = self.to_ts_status('P')
            status = delist.append(suspend)
            status.index = range(len(status))
            #交易日
            calendar = self.to_ts_calendar('19900101','30000101')
        except Exception as e:
            print('tusahre module error:',e)
        else:
            conn = db.db_init()
            transaction = conn.begin()
            db.empty_table('status',conn)
            db.empty_table('calendar',conn)
            db.enroll('status',status,conn)
            db.enroll('calendar',calendar,conn)
            transaction.commit()
            transaction.close()
            print('successfully run tushare client module')


if __name__ == '__main__':

    ts = TushareClient()
    # ts.to_ts_pledge('002570.SZ')
    coef = ts.to_ts_coef('000001')
    # print(coef[coef['trade_date'] < '19910601'])
    ts.update_via_ts()