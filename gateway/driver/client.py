#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:39:46 2019

@author: python
"""
import tushare as tu, pandas as pd

__all__ = ['tsclient']


class TsClient:

    def __init__(self, _token='7325ca7b347c682eabdd9e9335f16526d01f6dff2de6ed80792cde25'):

        tu.set_token(_token)
        self.pro = tu.pro_api()

    @staticmethod
    def _adapt_symbol_prefix(code):
        code_prefix = '.'.join([code, 'SH']) if code.startswith('6') else '.'.join([code, 'SZ'])
        return code_prefix

    def to_ts_calendar(self, s, e):
        """获取交易日, SSE , SZSE"""
        calendar = self.pro.trade_cal(start_date=s,
                                      end_date=e,
                                      exchange='SSE',
                                      is_open='1')
        calendar.columns = ['exchange', 'trade_dt', 'is_open']
        sessions = calendar['trade_dt'].apply(lambda x: pd.Timestamp(str(x)).strftime('%Y-%m-%d'))
        return sessions

    def to_ts_con(self, exchange, new):
        """获取沪港通和深港通股票数据 , is_new = 1 表示沪港通的标的， is_new = 0 表示已经被踢出的沪港通的股票,exchange : SH SZ"""
        const = self.pro.hs_const(hs_type=exchange,
                                  is_new=new)
        return const

    def to_ts_index_component(self, index, sdate, edate):
        """
            基准成分股以及对应权重
            df = self.pro.index_weight(index_code='399300.SZ', start_date='20180901', end_date='20180930')
        """
        df = self.pro.index_weight(index_code=index, start_date=sdate, end_date=edate)
        return df

    def to_ts_pledge(self, code):
        """股票质押率"""
        code = self._adapt_symbol_prefix(code)
        df = self.pro.pledge_stat(ts_code=code)
        return df

    def to_ts_adjfactor(self, code):
        code = self._adapt_symbol_prefix(code)
        coef = self.pro.adj_factor(ts_code=code)
        return coef

    def _get_ts_status(self, status):
        """status : D -- delist ; P --- suspend """
        abnormal = self.pro.stock_basic(exchange='', list_status=status, fields='symbol,name, delist_date')
        abnormal.loc[:, 'status'] = status
        return abnormal

    def to_ts_status(self):
        """
            基于tushare模块对股票退市或者暂停上市的状态更新
            暴力更新 获取 清空 入库
        """
        withdraw = self._get_ts_status('D')
        # print('withdraw', withdraw)
        suspend = self._get_ts_status('P')
        # print('suspend', suspend)
        status = withdraw.append(suspend)
        status.rename(columns={'symbol': 'sid', 'delist_date': 'last_traded'}, inplace=True)
        # 入库序列号存在重复
        status.index = range(len(status))
        return status


tsclient = TsClient()


# if __name__ == '__main__':
#
#     stats = tsclient.to_ts_status()
#     stats.dropna(axis=0, how='any', inplace=True)

