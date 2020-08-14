# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
# massive
MassiveFields = frozenset(['trade_dt', 'sid', 'cname', 'bid_price', 'bid_volume',
                           'amount', 'buyer_code', 'buyer', 'seller_code', 'seller',
                           'type', 'unit', 'pct', 'close', 'YSSLTAG', 'discount', 'cjeltszb',
                           '1_pct', '5_pct', '10_pct', '20_pct', 'TEXCH'])
# holder
HolderFields = frozenset(['代码', '中文', '现价', '涨幅', '股东', '方式', '变动股本', '占总流通比', '途径',
                          '总持仓', '占总股本比', '总流通股', '占流通比', '变动开始日', '变动截止日', 'declared_date'])

# ownership
COLUMNS = {'变动日期': 'ex_date', '公告日期': 'declared_date', '总股本': 'general', '流通A股': 'float',
           '限售A股': 'strict', '流通B股': 'b_float', '限售B股': 'b_strict', '流通H股': 'h_float'}


__all__ = [
    'MassiveFields',
    'HolderFields',
    'COLUMNS'
]