# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
from gateway.spider import Crawler
from gateway.database.db_writer import db
from gateway.spider.url import DIVDEND
from gateway.driver.tools import _parse_url

__all__ = ['AdjustmentsWriter']


class AdjustmentsWriter(Crawler):
    """
        a. obtain mysql divdends , splits , rights
        b. request sina
        c. update mysql
    """
    adjustment_tables = frozenset(['equity_splits', 'equity_rights'])

    def __init__(self):
        self.deadlines = dict()
        self.missed = set()

    def _record_deadlines(self):
        """
            record the declared date of equities in mysql
        """
        self.deadlines.clear()
        for tbl in self.adjustment_tables:
            self.deadlines[tbl] = self._retrieve_deadlines_from_sqlite(tbl, date_type='ex_date')

    def _parse_equity_rights(self, content, symbol):
        """配股"""
        text = list()
        table = content.find('table', {'id': 'sharebonus_2'})
        [text.append(item.get_text()) for item in table.tbody.findAll('tr')]
        if len(text) == 1 and text[0] == '暂时没有数据！':
            print('------------code : %s has not 配股' % symbol, text[0])
        else:
            sep_text = [item.split('\n')[1:-2] for item in text]
            frame = pd.DataFrame(sep_text, columns=['declared_date', 'rights_bonus', 'rights_price',
                                                    'benchmark_share', 'pay_date', 'ex_date',
                                                    '缴款起始日', '缴款终止日', 'effective_date', '募集资金合计'])
            frame.loc[:, 'sid'] = symbol
            # deadline = self.deadlines['equity_rights'].get(symbol, None)
            # rights = frame[frame['declared_date'] > deadline] if deadline else frame
            ex_deadline = self.deadlines['equity_rights'].get(symbol, None)
            rights = frame[frame['ex_date'] > ex_deadline] if ex_deadline else frame
            db.writer('equity_rights', rights)

    def _parse_equity_divdend(self, content, sid):
        """获取分红配股数据"""
        text = list()
        table = content.find('table', {'id': 'sharebonus_1'})
        [text.append(item.get_text()) for item in table.tbody.findAll('tr')]
        if len(text) == 1 and text[0] == '暂时没有数据！':
            print('------------code : %s has not splits and divdend' % sid, text[0])
        else:
            sep_text = [item.split('\n')[1:-2] for item in text]
            frame = pd.DataFrame(sep_text, columns=['declared_date', 'sid_bonus', 'sid_transfer', 'bonus',
                                                    'progress', 'pay_date', 'ex_date', 'effective_date'])
            frame.loc[:, 'sid'] = sid
            # deadline = self.deadlines['equity_splits'].get(sid, None)
            # divdends = frame[frame['declared_date'] > deadline] if deadline else frame
            ex_deadline = self.deadlines['equity_splits'].get(sid, None)
            divdends = frame[frame['ex_date'] > ex_deadline] if ex_deadline else frame
            db.writer('equity_splits', divdends)

    def _parser_writer(self, sid):
        contents = _parse_url(DIVDEND % sid)
        # 解析网页内容
        self._parse_equity_rights(contents, sid)
        self._parse_equity_divdend(contents, sid)

    def rerun(self):
        if len(self.missed):
            print('missing', self.missed)
            missed = self.missed.copy()
            # RuntimeError: Set changed size during iteration
            self._writer_internal(missed)
            self.rerun()
        # reset
        self.missed = set()
        # self.deadlines.clear()

    def _writer_internal(self, equities):
        for sid in equities:
            try:
                self._parser_writer(sid)
            except Exception as e:
                print('spider divdends and splits of code:%s failure due to %r' % (sid, e))
                self.missed.add(sid)
            else:
                self.missed.discard(sid)

    def writer(self):
        # 获取数据库的最新时点
        self._record_deadlines()
        # 获取所有股票
        equities = self._retrieve_assets_from_sqlite()['equity']
        self._writer_internal(equities)
        self.rerun()


# if __name__ == '__main__':
#
#     w = AdjustmentsWriter()
#     w.writer()
