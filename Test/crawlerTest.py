#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:39:46 2019

@author: python
"""
from urllib.request import urlopen
from bs4 import BeautifulSoup
import json,re,requests
from itertools import chain
import pandas as pd
import requests

Headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'}

#from Alert.GateWay.Driver import db

#获取存量股票包括退市
# html = 'http://70.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12'
# obj = urlopen(html).read()
# raw = json.loads(obj)
# code = [item['f12'] for item in raw['data']['diff']]
# print('--------length%d'%len(code),code)
# sci_tech = list(filter(lambda x : x.startswith('688'),code))
# print('-----lenth:%d'%len(sci_tech),sci_tech)
# tradional = set(code) - set(sci_tech)
# print('-------normal stock length:%d'%len(tradional))
# # #新股 与上市时间
# html = 'http://81.push2.eastmoney.com/api/qt/clist/get?&pn=1&pz=100&po=1&np=1&fltt=2&invt=2&fid=f26&fs=m:0+f:8,m:1+f:8&fields=f12,f26'
# obj = urlopen(html).read()
# raw = json.loads(obj)
# new = [list(i.values()) for i in raw['data']['diff']]
# new_df = pd.DataFrame(new,columns=['code','timeTomarket'])
# new_df.sort_values(by = 'timeTomarket',inplace = True,ascending=False)
# print(new_df.head())
# res = new_df['code'][new_df['timeTomarket'] > 20200204].values
# print(res)


#获取当日日内数据
# html_minutes = urlopen('http://push2.eastmoney.com/api/qt/stock/trends2/get?fields1=f1&fields2=f51,f52,f53,f54,f55,f56,f57,f58&iscr=0&secid=0.002570')
# obj = html_minutes.read()
# obj = str(obj,'utf-8')
# d = json.loads(obj)
# minutes = d['data']['trends']
# print(minutes)

#
# #获取历史日内数据
# html_minutes_his = urlopen('http://push2his.eastmoney.com/api/qt/stock/trends2/get?fields1=f1&fields2=f51,f52,f53,f54,f55,f56,f57,f58&ndays=2&iscr=3&secid=0.002570')
# obj = html_minutes_his.read()
# obj = str(obj,'utf-8')
# print(obj)
# d = json.loads(obj)
# print(d)
# minutes_his = d['data']['trends']
# print(minutes_his)

# def query_new(self):
#     # #新股 与上市时间
#     html = 'http://81.push2.eastmoney.com/api/qt/clist/get?&pn=1&pz=100&po=1&np=1&fltt=2&invt=2&fid=f26&fs=m:0+f:8,m:1+f:8&fields=f12,f26'
#     obj = urlopen(html).read()
#     print(obj)
#     raw = json.loads(obj)
#     new = [list(i.values()) for i in raw['data']['diff']]
#     print(new)
#     new_df = pd.DataFrame(new, columns=['code', 'timeTomarket'])
#     print(new_df.head())
#     print(len(new_df))


code = '000002'
#
# #获取区间日频数据
#html_daily = urlopen('http://64.push2his.eastmoney.com/api/qt/stock/kline/get?&secid=1.688266&fields1=f1&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&end=20200210&lmt=100')
#obj = html_daily.read()
# html_daily = 'http://64.push2his.eastmoney.com/api/qt/stock/kline/get?&secid=0.{}&fields1=f1&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&end=20200203&lmt=1'.format(code)
# req = requests.get(html_daily,headers = Headers)
# req.encoding = 'gbk'
# d = json.loads(req.text)
# obj = req.content
# obj = str(obj,'utf-8')
# d = json.loads(obj)
# kline = d['data']['klines']
# transform = [item.split(',') for item in kline]
# kl_pd = pd.DataFrame(transform,columns = ['trade_dt','open','close','high','low','volume','turnover','amplitude'])
# print('kline',kl_pd.head())
# # kl_pd['code'] = d['data']['code']
# print(kl_pd[kl_pd['trade_dt'] == '1991-05-08'])
# test = kl_pd.iloc[0,:].to_dict()
# #
# cls = getattr(db,'price')
# print('----------class',cls)
# print(cls.columns)
# #删除
# from sqlalchemy import delete
# #u = delete(db.price).where(db.price.c.trade_dt == test['trade_dt'])
# #u = delete(cls).where(db.price.c.trade_dt == test['trade_dt'])
# u = delete(cls).where(cls.c['trade_dt'] == test['trade_dt'])
# db.connection.execute(u)
# #入库
# #ins = db.price.insert()
# ins = cls.insert()
# db.connection.execute(ins,test)
#
# #更新
# from sqlalchemy import update
# #u = update(db.price).where(db.price.c.trade_dt == test['trade_dt'])
# u = update(cls).where(cls.c['trade_dt'] == test['trade_dt'])
# u=u.values(volume = 0)
# db.connection.execute(u)
# #

# 分红配股
# html = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/%s.phtml'%code
# splits = urlopen(html)
# splits = requests.get(html,headers = Headers)
# splits.encoding='gbk'
# obj = BeautifulSoup(splits.text,features='lxml')
# #分红
# # table = obj.find('table',{'id':'sharebonus_1'})
# #配股
# table = obj.find('table',{'id':'sharebonus_2'})
#
# print(table)
# # header = table.thead
# # print('-------------header',header.findAll('tr'))
# # # cols=[]
# # # [cols.append(item.get_text()) for item in header.findAll('tr')]
# # # print('------------------col',cols)
# raw = table.tbody
# print(raw)
# splits_raw = []
# [splits_raw.append(item.get_text()) for item in raw.findAll('tr')]
# print('---------',splits_raw)
# if splits_raw[0] == '暂时没有数据！':
#     pass
# else:
#     splits_parse = [item.split('\n')[1:-2] for item in splits_raw]
#     # split_divdend = pd.DataFrame(splits_parse,columns = ['公告日期','送股','转增','派息','进度','除权除息日','股权登记日','红股上市日'])
#     split_divdend = pd.DataFrame(splits_parse)
#     # split_divdend['代码'] = code
#     # split_divdend.sort_values(by='公告日期',inplace = True)
#     print(split_divdend.tail())
# test = split_divdend.iloc[0,:]
# # #删除
# #
# # #入库
# # ins = db.splits_divdend.insert()
# # test = split_divdend.iloc[0,:].to_dict()
# # db.connection.execute(ins,test)
# # #







# #
#公司基本情况
# html = 'https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/%s.phtml'%code
# #obj = BeautifulSoup(urlopen(html),features='lxml')
# req = requests.get(html,headers = Headers)
# req.encoding = 'gbk'
# obj = BeautifulSoup(req.text,features='lxml')
# table = obj.find('table',{'id':'comInfo1'})
# tag = [item.findAll('td') for item in table.findAll('tr')]
# tag_chain = list(chain(*tag))
# raw = [item.get_text() for item in tag_chain]
# #去除格式
# raw = [i.replace('：','') for i in raw]
# raw = [i.strip() for i in raw]
# info = list(zip(raw[::2],raw[1::2]))
# info_dict = {item[0]:item[1] for item in info}
# print(info_dict)



# info_split = [parse(item) for item in info if len(item) >1]
# info_chain = list(chain(*info_split))
# info_dict = {item[0].strip():item[1].strip() for item in info_chain}
# info_dict.update({'代码':code})
# print(info_dict)
# #删除记录
# from sqlalchemy import delete
# u = delete(db.description).where(db.description.c.上市日期 == info_dict['上市日期'])
# db.connection.execute(u)
# #insert
# ins = db.description.insert()
# db.connection.execute(ins,info_dict)
#
#
# # #流通股本
# html = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vCI_StockStructure/stockid/%s.phtml'%code
# obj = urlopen(html).read()
# obj = requests.get(html,headers = Headers)
# obj.encoding = 'gbk'
# res = BeautifulSoup(obj.text,features = 'lxml')
# print('res----------',res)
#
# def tranform(th):
#     cols = [t.get_text() for t in th.findAll('td',{'width':re.compile('[0-9]+')})]
#     print('cols',cols)
#     raw = [t.get_text() for t in th.findAll('td')]
#     #xa0为空格
#     raw = [''.join(item.split()) for item in raw]
#     #去除格式
#     raw = [re.sub('·','',item) for item in raw]
#     #调整字段
#     raw = [re.sub('\(历史记录\)','',item) for item in raw]
#     raw = [item.replace('万股','') for item in raw]
#     print(raw)
#     #结构处理
#     num = int(len(raw)/len(cols))
#     print('------num',num)
#     text = {}
#     for count in range(len(cols)):
#         idx = count*num
#         mid = raw[idx:idx+num]
#         text.update({mid[0]:mid[1:]})
#         print(text)
#         df = pd.DataFrame.from_dict(text)
#         print(df)
#     return df
#
# tbody= res.findAll('tbody')
# print(len(tbody))
#
# join_df = pd.DataFrame()
# for th in tbody:
#     formatted = tranform(th)
#     join_df = join_df.append(formatted)
#
# join_df.index = range(len(join_df))
# #排序
# join_df.sort_values(by='变动日期',inplace = True)
# print(join_df)
# #删除数据
# from sqlalchemy import delete
# u = delete(db.structure).where(db.structure.c.变动日期 == res['变动日期'])
# db.connection.execute(u)
# #入库
# ins = db.structure.insert()
# db.connection.execute(ins,res)
#
#
# # '1.000001 上证'
# # '1.000300 沪深300'
# # '1.000016 上证50'
# # '1.000132 上证100'
# # '1.000010 上证180'
# # '1.000133 上证150'
# # '1.000009 上证380'
# # '0.399001 深圳'
# # '0.399006 创业板'
# # '0.399005 中小板'
#获取指数信息
# url = 'http://71.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=12&po=1&np=2&fltt=2&invt=2&fid=&fs=b:MK0010&fields=f12,f14'
# req = requests.get(url, headers= Headers, timeout=5)
# req.encoding = 'utf-8'
# #obj = str(obj,encoding = 'utf-8')
# data = json.loads(req.text)
# index_set = data['data']['diff']
# print(index_set)
# print(len(index_set))



# html = 'http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.000001&fields1=f1&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg=20200203&end=20200203'
# obj = urlopen(html).read()
# obj = str(obj,encoding = 'utf-8')
# data = json.loads(obj)
# benchmark = data['data']['code']
# raw = data['data']['klines']
# trans = [item.split(',') for item in raw]
# price = pd.DataFrame(trans,columns = ['trade_dt','open','close','high','low','volume','turnover','amplitude'])
# price['code'] = benchmark
# print('index',price.head())
# test = price.iloc[0,:].to_dict()
# #入库
# ins = db.benchmark_price.insert()
# db.connection.execute(ins,test)
# #删除
# u = delete(db.benchmark_price).where(db.benchmark_price.c.trade_dt == test['trade_dt'])
# db.connection.execute(u)
#
#
# #获取ETF列表 page num --- 20 40 80
# res = []
# page = 1
# while True:
#     html = "http://vip.stock.finance.sina.com.cn/quotes_service/api/jsonp.php/IO.XSRV2.CallbackList['v7BkqPkwcfhoO1XH']/Market_Center.getHQNodeDataSimple?page=%d&num=80&sort=symbol&asc=0&node=etf_hq_fund"%page
#     #obj = BeautifulSoup(urlopen(html).read(),features = 'lxml')
#     req = requests.get(html,headers = Headers)
#     req.encoding = 'gbk'
#     obj = BeautifulSoup(req.text,features = 'lxml')
#     text = obj.find('p').get_text()
#     mid = re.findall('s[z|h][0-9]{6}',text)
#     if len(mid) >0 :
#         res.extend(mid)
#     else:
#         break
#     page = page +1
# print('length of etf:%d'%len(res),res)
#
# #获取EFT数据
# html_etf = 'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.515900&fields1=f1&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&lmt=1&klt=101&fqt=1&end=20200203'
# obj = urlopen(html_etf).read()
# raw = json.loads(obj)
# etf = raw['data']['klines']
# etf_df = pd.DataFrame([item.split(',') for item in etf],columns = ['trade_dt','open','close','high','low','volume','turnover'])
# print('etf',etf_df.head())
# etf_df['code'] = '515900'
# test_etf = etf_df.iloc[0,:].to_dict()
# #入库
# ins = db.etf_price.insert()
# db.connection.execute(ins,test_etf)
#
# # 可转换公司债券的期限最短为1年，最长为6年，自发行结束之日起6个月方可转换为公司股票
# # 可转债赎回：公司股票在一段时间内连续高于转换价格达到一pp-....≥....................//..
# # [p.p[“≥≥≥≥≥≥≥≥≥≥≥≥≥≥≥.p[/定幅"?时，公司可按照事先约定的赎回价格买回发行在外尚未转股的可转换公司债券
# # 可转债强制赎回：连续三十个交易日中至少有十五个交易日的收盘价格不低于当期转股价格的 130%（含130%）；
# # 存在向下修订转股价
# # 可转债回售：最后两个计息年度内，一旦正股价持续约30天低于转股价的70%，上市公司必须以100+e 的价格来赎回可转债
# # 集思录获取可转债数据列表，请求方式post
# # 可债转基本情况 --- 强制赎回价格 回售价格  时间
# url = 'https://www.jisilu.cn/data/cbnew/cb_list/?'
# params = {'___jsl=LST___t':1579323463915}
# r = requests.post(url,data = params)
# r = requests.get(url,headers = Headers)
# #r.encoding = 'gbk'
# #print(r.text)
# text = json.loads(r.text)
# # #text = json.loads(r.content)
# info = text['rows']
# print('info-----',info[0])
# print('-----info',info[:2])
# convertible = [item['id'] for item in info]
# print(convertible)
# #入库
# ins = db.convertible.insert()
# db.connection.execute(ins,info[0]['cell'])
#
#
#获取可转债数据 --- 数据结构： open close high low volume vp
# html = 'https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.113562&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&lmt=1&klt=101&fqt=1&end=20200203'
# obj = urlopen(html).read()
# req = requests.get(html,headers = Headers)
# req.encoding = 'gbk'
# obj = req.text
# print('--------------------',obj)
# dict_data= json.loads(obj)
# print(dict_data)
# kline = dict_data['data']['klines']
# split = [item.split(',') for item in kline]
# df = pd.DataFrame(split,columns = ['trade_dt','open','close','high','low','volume','turnover'])
# print('convertible',df.head())

# df.loc[:,'code'] = code
# #入库
# ins = db.bond_price.insert()
# db.connection.execute(ins,df.iloc[0,:].to_dict())

# # #分时数据
# # http://web.ifzq.gtimg.cn/appstock/app/minute/query?_var=min_data_sh600050&code=sh600050&r=0.001288643980735138
# # tencent = 'http://web.ifzq.gtimg.cn/appstock/app/minute/query?&code=sh600050'
# # obj = urlopen(tencent)
# # res = obj.read()
# # res = str(res,'utf-8')
# # print(res)
#
# # #五日分时数据
# # http://web.ifzq.gtimg.cn/appstock/app/day/query?_var=fdays_data_sh600050&code=sh600050&r=0.009685318738638315
# # tencent = 'http://web.ifzq.gtimg.cn/appstock/app/day/query?&code=sh600050'
# # obj = urlopen(tencent)
# # res = obj.read()
# # res = str(res,'utf-8')
# # print(res)
# # res_dict = json.loads(res)
# # print(res_dict['data']['sh600050'].keys())
#
# # #每日数据
# http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_dayqfq2019&param=sh600050,day,2019-01-01,2020-12-31,640,qfq&r=0.6938006980324525
# #月度数据
# http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?_var=kline_monthqfq&param=sh600050,month,,,320,qfq&r=0.07390350028346937
#tencent = 'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?&param=sh600050,day,,,700,qfq'
# tencent = 'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?&param=us.DJI,day,,,700,qfq'
#tencent = 'http://web.ifzq.gtimg.cn/appstock/app/dayus/query?_var=fdays_data_usDJI&code=us.DJI&r=0.1674241891960231'
#http://web.ifzq.gtimg.cn/appstock/app/day/query?_var=fdays_data_hkHSI&code=hkHSI&r=0.5609046262183592

"""
code=us.DJI 道琼斯
code=us.IXIC 纳斯达克
code=us.INX  标普500

code=hkHSI 香港恒生指数
code=hkHSCEI 香港国企指数
code=hkHSCCI 香港红筹指数
"""
# html = 'http://dcfm.eastmoney.com/EM_MutiSvcExpandInterface/api/js/get?type=ZD_QL_LB&token=70f12f2f4f091e459a279469fe49eca5&cmd=&st=amtshareratio&sr=-1&p=61&ps=50&js=var%20meClDXgi={pages:(tp),data:(x),font:(font)}&filter=(tdate=%272020-02-07%27)&rt=52708317'
# obj = urlopen(html)
# res = obj.read()
# res = str(res,'utf-8')
# print(res)

# res_dict = json.loads(res)
# print(res_dict)
# data = res_dict['data']['us.DJI']
# print(data.keys())
# #print(len(data['qfqmonth']))
# print(data['day'])
# #trade_dt open preclose high low
# #print(data['qt'])

#获取a/h比率
#ah股列表
# html = 'http://web.ifzq.gtimg.cn/appstock/invest/investment/aph?app=web&_callback=__.app.mstats.listTPL.HGTAH.__CALLBACK_0_&p=1&l=20&order=hayj&way='

#http://web.ifzq.gtimg.cn/appstock/app/day/query?_var=fdays_data_hk02359&code=hk02359&r=0.9632662621250254

# ths = 'http://d.10jqka.com.cn/v6/line/hs_601816/01/all.js'
# Headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'}
# import requests
# obj = requests.get(ths,headers = Headers)
# # obj = urlopen(ths)
# print(obj)

# def download_ashare_holding(self):
#     """股票增持、减持、变动情况 没跑"""
#     page = 1209
#     while True:
#         url = "http://data.eastmoney.com/DataCenter_V3/gdzjc.ashx?pagesize=50&page=%d&param=&sortRule=-1&sortType=BDJZ" % page
#         raw = self._parse_url(url, bs=False)
#         match = re.search('\[(.*.)\]', raw)
#         if match:
#             data = json.loads(match.group())
#             data = [item.split(',')[:-1] for item in data]
#             holdings = pd.DataFrame(data, columns=['代码', '中文', '现价', '涨幅', '股东', '方式', '变动股本', '占总流通比例', '途径', '总持仓',
#                                                    '占总股本比例', '总流通股', '占流通比例', '变动开始日', '变动截止日', '公告日'])
#             # 入库
#             self.db.enroll('holding', holdings, self.conn)
#             print('page : %d' % page, holdings.head())
#             page = page + 1
#             time.sleep(np.random.randint(5, 8))
#         else:
#             print('match is null')
#             break

# def _update_market_value(self):
#     """计算流通市值、限售A股市值，流通B股,流通H股，总市值"""
#     bar = BarReader()
#     db = DataLayer()
#     conn = db.db_init()
#     basics = bar.load_ashare_basics()
#     quantity = [item[0] for item in basics]
#     tables = DataLayer().metadata.tables
#     table = tables['ashareValue']
#     from sqlalchemy import select
#     ins = select([table.c.code.distinct()])
#     rp = db.engine.execute(ins)
#     raw = rp.fetchall()
#     q_enroll = [r[0] for r in raw]
#     print('length', len(q_enroll))
#     assets = set(quantity) - set(q_enroll)
#     print('assets', len(assets))
#     not_kline_asset = []
#     sdate = '1990-01-01'
#     for asset in assets:
#         print('asset', asset)
#         # 获取变动流通股本、总流通股本
#         raw = bar.load_equity_info(asset)
#         print('raw', raw)
#         # 将日期转为交易日
#         raw.loc[:, 'trade_dt'] = [t if bar.is_market_caledar(t) else bar.load_calendar_offset(t, 1) for t in
#                                   raw.loc[:, 'change_dt'].values]
#         """由于存在一个变动时点出现多条记录，保留最大total_assets的记录,先按照最大股本降序，保留第一个记录"""
#         raw.sort_values(by='total_assets', ascending=False, inplace=True)
#         raw.drop_duplicates(subset='trade_dt', keep='first', inplace=True)
#         print('raw---', raw)
#         raw.index = raw['trade_dt']
#         print(sdate, self.nowdays)
#         close = bar.load_stock_kline(sdate, self.nowdays, ['close'], asset)
#         print('close', close)
#         if len(close) == 0:
#             print('code:%s has not kline' % asset)
#             not_kline_asset.append(asset)
#         else:
#             close.loc[:, 'total'] = raw.loc[:, 'total_assets']
#             close.loc[:, 'float'] = raw.loc[:, 'float_assets']
#             close.loc[:, 'strict'] = raw.loc[:, 'strict_assets']
#             close.loc[:, 'b_assets'] = raw.loc[:, 'b_assets']
#             close.loc[:, 'h_assets'] = raw.loc[:, 'h_assets']
#             close.fillna(method='ffill', inplace=True)
#             close.fillna(method='bfill', inplace=True)
#             print('close----', close)
#             # 计算不同类型市值
#             mkt = close.loc[:, 'total'] * close.loc[:, 'close']
#             cap = close.loc[:, 'float'] * close.loc[:, 'close']
#             strict = close.loc[:, 'strict'] * close.loc[:, 'close']
#             b = close.loc[:, 'b_assets'] * close.loc[:, 'close']
#             h = close.loc[:, 'h_assets'] * close.loc[:, 'close']
#             print(mkt, cap, strict, b, h)
#             # 调整格式并入库
#             data = pd.DataFrame([mkt, cap, strict, b, h]).T
#             data.columns = ['mkt', 'cap', 'strict', 'foreign', 'hk']
#             data.loc[:, 'trade_dt'] = data.index
#             data.loc[:, 'code'] = asset
#             print('data', data)
#             db.enroll('mkt_value', data, conn)
#     conn.close()
# html = 'http://data.eastmoney.com/cjsj/grossdomesticproduct.aspx?p=4'
# gdp = requests.get(html,headers = Headers)
# gdp.encoding='gbk'
# obj = BeautifulSoup(gdp.text,features='lxml')
# # print('obj',obj)
# raw  = obj.findAll('div',{'class':'Content'})
# # print(raw)
# # print(raw[1].findAll('td'))
# #
# text = [t.get_text() for t in raw[1].findAll('td')]
# text = [item.strip() for item in text]
# print('text',text)
# raw = text[::9]
# print('slice',text)
# data = zip(text[::9],text[1::9])
# #
# data = pd.DataFrame(data,columns = ['季度','总值'])
# print('data',data)