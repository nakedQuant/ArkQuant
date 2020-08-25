# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
__all__ = [
    'ASSETS_BUNDLES_URL',
    'ASSET_FUNDAMENTAL_URL',
    'DIVDEND',
    'OWNERSHIP',
    'BENCHMARK_URL',
    'ASSERT_URL_MAPPING',
    'ASSET_SUPPLEMENT_URL',
    'MassiveFields',
    'HolderFields',
    'COLUMNS'
]


# kline
equity_bundles = 'http://64.push2his.eastmoney.com/api/qt/stock/kline/get?&secid={}&fields1=f1&' \
                'fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&end=30000101&lmt={}'

bond_bundles = 'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5' \
               '&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&lmt={}&klt=101&fqt=1&end=30000101'

fund_bundles = 'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={}&fields1=f1&' \
               'fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&lmt={}&klt=101&fqt=1&end=30000101',

dual_bundles = 'http://94.push2his.eastmoney.com/api/qt/stock/kline/get?secid=116.08231&fields1=f1' \
               '&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&klt=101&fqt=1&end=20500101&lmt=2'

ASSETS_BUNDLES_URL = {
    'equity_bundles': equity_bundles,
    'bond_bundles': bond_bundles,
    'fund_bundles': fund_bundles,
    'dual_bundles': dual_bundles
}

# fundamental
holder = 'http://data.eastmoney.com/DataCenter_V3/gdzjc.ashx?pagesize=50&page=%d' \
              '&param=&sortRule=-1&sortType=BDJZ'

massive = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=DZJYXQ&' \
          'token=70f12f2f4f091e459a279469fe49eca5&cmd=&st=SECUCODE&sr=1&p=%d&ps=50&'

release = 'http://dcfm.eastmoney.com/EM_MutiSvcExpandInterface/api/js/get?type=XSJJ_NJ_PC' \
          '&token=70f12f2f4f091e459a279469fe49eca5&st=kjjsl&sr=-1&p=%d&ps=10&filter=(mkt=)'

gross_url = 'http://data.eastmoney.com/cjsj/grossdomesticproduct.aspx?p=%d'

margin_url = 'http://api.dataide.eastmoney.com/data/get_rzrq_lshj?' \
             'orderby=dim_date&order=desc&pageindex=%d&pagesize=50'

ASSET_FUNDAMENTAL_URL = {
    'shareholder': holder,
    'massive': massive,
    'release': release,
    'gross': gross_url,
    'margin': margin_url
}

# benchmark
benchmark_kline = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={}&fields1=f1&' \
                  'fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg=19900101&end={}'
periphera_kline = 'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?&param=%s,day,1990-01-01,%s,100000,qfq'


BENCHMARK_URL = {
    'kline': benchmark_kline,
    'periphera_kline': periphera_kline
}

# ownership
OWNERSHIP = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vCI_StockStructure/stockid/%s.phtml'

# splits and divdend
DIVDEND = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/%s.phtml'


# assets --- equity bond etf
equity_url = 'http://70.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&' \
             'fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12'

dual_url = 'http://19.push2.eastmoney.com/api/qt/clist/get?pn=%d&pz=20&po=1&np=1&invt=2&fid=f3&' \
           'fs=b:DLMK0101&fields=f12,f191,f193'

convertible_url = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=KZZ_LB2.0' \
           '&token=70f12f2f4f091e459a279469fe49eca5&cmd=&sr=-1&p=%d&ps=50&js={"pages":(tp),"data":(x)} '

fund_url = 'http://fund.eastmoney.com/cnjy_jzzzl.html'

benchmark_url = 'http://71.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=12&po=1&np=2&fltt=2&invt=2&' \
            'fid=&fs=b:MK0010&fields=f12,f14'


ASSERT_URL_MAPPING = {
                    'equity': equity_url,
                    'convertible': convertible_url,
                    'fund': fund_url,
                    'dual': dual_url,
                    'benchmark': benchmark_url,
                    }

# assets bonds
equity_supplement_url = 'https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/%s.phtml'

bond_supplement_url = 'https://www.jisilu.cn/data/cbnew/cb_list/?'

ASSET_SUPPLEMENT_URL = {
                    'equity_supplement': equity_supplement_url,
                    'convertible_supplement': bond_supplement_url
                        }


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