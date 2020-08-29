
import requests, json, pandas as pd
from collections import defaultdict
from toolz import partition_all
from bs4 import BeautifulSoup
from functools import partial
from itertools import chain


def _parse_url(url, encoding='gbk', bs=True):
    Header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36(KHTML, like Gecko)'
                      ' Chrome/79.0.3945.130 Safari/537.36'}
    req = requests.get(url, headers=Header, timeout=10)
    req.encoding = encoding
    if bs:
        raw = BeautifulSoup(req.text, features='lxml')
    else:
        raw = req.text
    return raw


# url = 'https://www.kuaidaili.com/free/inha/2/'

# res = _parse_url(url, proxy, encoding='utf-8')
# # print(res)
# # ip = res.find('td', {'data_title': 'IP'})
# # ip = res.find_all(data_title='IP')
# # print('ip', ip)
# table = res.find('table')
# # print(table)
# # item = [item.findAll('td') for item in table.findAll('tr')]
# ip_item = [item.find('td', {'data-title': 'IP'}) for item in table.findAll('tr')]
# ip = [ele.get_text() for ele in ip_item if ele]
# print(ip)
# print(len(ip))
# #
# port_item = [item.find('td', {'data-title': 'PORT'}) for item in table.findAll('tr')]
# port = [ele.get_text() for ele in port_item if ele]
# print(port)
# print(len(port))
# # category
# category_item = [item.find('td', {'data-title': '类型'}) for item in table.findAll('tr')]
# category = [ele.get_text() for ele in category_item if ele]
# # construct proxy
# proxies = []
# for item in zip(category, ip, port):
#     proxy = item[0].lower() + '://' + item[1] + ':' + item[-1]
#     proxies.append(proxy)
# print(proxies)

# def func(a, b=3, c=4, d=None):
#     result = a+b+c+d if d else a+b+c
#     return result
#
#
# p = partial(func, c=10)
#
# test = p(5, b=5, d=2)
# print(test)

# proxy = {'http': 'http://0825fq1t1d659:0825fq1t1d659@218.87.56.38:65000'}
#
# url = 'http://70.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12'
#
# raw = _parse_url(url, proxy)
# print(raw)
#
# text = _parse_url('https://www.jisilu.cn/data/cbnew/cb_list/?', encoding=None, bs=False)
# text = json.loads(text)
# # print('text', text)
# data = text['rows']
# print(data)
# # 327
# print('size', len(data))
# cell = [basic['cell'] for basic in text['rows']]
# print('cell size', len(cell))
# print('cell', cell)
#
# 获取上市的可转债的标的
# page = 1
# bonds = []
# convertible_url = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=KZZ_LB2.0' \
#            '&token=70f12f2f4f091e459a279469fe49eca5&cmd=&sr=-1&p=%d&ps=50&js={"pages":(tp),"data":(x)} '
#
# while page <= 3:
#     bond_url = convertible_url % page
#     text = _parse_url(bond_url, encoding='utf-8', bs=False)
#     text = json.loads(text)
#     data = text['data']
#     if data:
#         bonds = chain(bonds, data)
#         page = page + 1
#     else:
#         break
# bonds = list(bonds)
# print('bond size', len(bonds))
# print(bonds)
# mappings = []
# # 过滤未上市的可转债 bond_id : bond_basics
# bond_mappings = {bond['BONDCODE']: bond for bond in bonds if bond['LISTDATE'] != '-'}
# print('bond_mappings', bond_mappings['113599'])


#
# url = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=KZZ_LB2.0' \
#       '&token=70f12f2f4f091e459a279469fe49eca5&cmd=&st=STARTDATE&sr=-1&p=1&ps=50&js={"pages":(tp),"data":(x)}'
# text = _parse_url(url, encoding='utf-8', bs=False)
# print(text)
# text = json.loads(text)
# data = text['data']
# print(len(data))


# fund_url = 'http://fund.eastmoney.com/cnjy_jzzzl.html'
# obj = _parse_url(fund_url)
# raw = [data.find_all('td') for data in obj.find_all(id='tableDiv')]
# text = [t.get_text() for t in raw[0]]
# frame = pd.DataFrame(partition_all(14, text[18:]), columns=text[2:16]).iloc[:, :-2]
# frame['基金简称'] = frame['基金简称'].apply(lambda x: x[:-5])
# frame.loc[:, 'exchange'] = frame['基金代码'].apply(lambda x: 'sh' if x.startswith('5') else 'sz')
#
# # frame['基金简称'] = frame['基金简称'].apply(lambda x: x[:-5])
# # frame.drop_duplicates(inplace=True)
# # frame = frame.apply(lambda x: x['基金简称'][:-5], axis=1)
# _rename_fund_cols = {
#     '基金代码': 'sid',
#     '基金简称': 'asset_name',
#     '类型': 'asset_type'
# }
#
# frame.rename(columns=_rename_fund_cols, inplace=True)
# print('rename frame', frame)
#
# _rename_router_cols = frozenset(['sid',
#                                  'asset_name',
#                                  'asset_type',
#                                  'first_traded',
#                                  'last_traded',
#                                  # 'status',
#                                  'exchange'])
#
# renamed_frame = frame.reindex(columns=_rename_router_cols, fill_value='')
# print('frame', renamed_frame)
# print('test', renamed_frame.iloc[0, :])

#
# update_funds = set(frame['基金代码'].values) - set(existing_assets.get('fund', []))
# print('update_funds', update_funds)
# fund_frame = frame[frame['基金代码'].isin(update_funds)] if update_funds else frame
# print('fund_frame', fund_frame)

massive = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=DZJYXQ&' \
          'token=70f12f2f4f091e459a279469fe49eca5&cmd=&st=SECUCODE&sr=1&p={page}&ps=50&' \
          'js={{"data":(x)}}&filter=(Stype=%27EQA%27)(TDATE%3E=^{start}^%20and%20TDATE%3C=^{end}^)'

# sr=-1 --- 表示倒序 ； sr=1 --- 顺序
release = 'http://dcfm.eastmoney.com/EM_MutiSvcExpandInterface/api/js/get?type=XSJJ_NJ_PC' \
          '&token=70f12f2f4f091e459a279469fe49eca5&st=kjjsl&sr=-1&p={page}&ps=10&filter=(mkt=)' \
          '(ltsj%3E=^{start}^%20and%20ltsj%3C=^{end}^)&js={{"data":(x)}}'


# test = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=DZJYXQ' \
#        '&token=70f12f2f4f091e459a279469fe49eca5&cmd=&st=SECUCODE&sr=1&p={page}&ps=50&' \
#        'js={{"data":(x)}}&filter=(Stype=%27EQA%27)'
# test.format(page=100)

massive.format(page=1000, start='2020-08-20', end='2020-08-29')
release.format(page=1000, start='2020-08-20', end='2020-08-29')

#
# add exchange
# basics_frame.loc[:, 'exchange'] = basics_frame['BONDCODE'].apply(lambda x:
#                                                                  'sh' if x.startswith('11') else 'sz')
# convertible_basics '-' --- nan
# basics_frame.replace(to_replace='-', value=pd.NA, inplace=True)
# basics_frame['asset_type'] = 'convertible'
# -------------------------------------------------------------------------------------------------------
# frame.fillna('', inplace=True)
# # append asset_type
# frame.loc[:, 'sid'] = frame.index
# frame.loc[:, 'asset_type'] = 'equity'

# -------------------------------------------------------------------------------------------------------
# frame['基金简称'] = frame['基金简称'].apply(lambda x: x[:-5])
# frame.loc[:, 'exchange'] = frame['基金代码'].apply(lambda x: '上海证券交易所' if x.startswith('5') else '深圳证券交易所')