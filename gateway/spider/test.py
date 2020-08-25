
import requests
from bs4 import BeautifulSoup
from functools import partial


def _parse_url(url, proxy, encoding='gbk', bs=True):
    Header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36(KHTML, like Gecko)'
                      ' Chrome/79.0.3945.130 Safari/537.36'}
    req = requests.get(url, headers=Header, proxies=proxy, timeout=10)
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

proxy = {'http': 'http://0825fq1t1d659:0825fq1t1d659@218.87.56.38:65000'}

url = 'http://70.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12'

raw = _parse_url(url, proxy)
print(raw)



