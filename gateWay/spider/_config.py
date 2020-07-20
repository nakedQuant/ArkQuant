# -*- coding : utf-8 -*-

#kline
equity_bundles = 'http://64.push2his.eastmoney.com/api/qt/stock/kline/get?&secid={}&fields1=f1&' \
                'fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&end=30000101&lmt={}'

bond_bundles = 'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5' \
               '&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&lmt={}&klt=101&fqt=1&end=30000101'

fund_bundles = 'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={}&fields1=f1&' \
               'fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&lmt={}&klt=101&fqt=1&end=30000101',

dual_bundles = 'http://94.push2his.eastmoney.com/api/qt/stock/kline/get?secid=116.08231&fields1=f1' \
               '&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&klt=101&fqt=1&end=20500101&lmt=2'

ASSETS_BUNDLES_URL = {'equity_bundles':equity_bundles,'bond_bundles':bond_bundles,
                      'fund_bundles':fund_bundles,'dual_bundles':dual_bundles}


# splits and divdend
divdend = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/%s.phtml'

equity =  'http://vip.stock.finance.sina.com.cn/corp/go.php/vCI_StockStructure/stockid/%s.phtml'

ASSETS_STRUCTURE_DIVDEND = {'divdend':divdend,'equity':equity}


#fundamental
shareholder = 'http://data.eastmoney.com/DataCenter_V3/gdzjc.ashx?pagesize=50&page=%d&param=&sortRule=-1&sortType=BDJZ'

massive = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=DZJYXQ&' \
          'token=70f12f2f4f091e459a279469fe49eca5&cmd=&st=SECUCODE&sr=1&p=%d&ps=50&'

release = 'http://dcfm.eastmoney.com/EM_MutiSvcExpandInterface/api/js/get?type=XSJJ_NJ_PC' \
          '&token=70f12f2f4f091e459a279469fe49eca5&st=kjjsl&sr=-1&p=%d&ps=10&filter=(mkt=)'

ASSETS_EVENT_URL = {'shareholder':shareholder,'massive':massive,'release':release}


__all__ = [ASSETS_BUNDLES_URL,ASSETS_STRUCTURE_DIVDEND,ASSETS_EVENT_URL]