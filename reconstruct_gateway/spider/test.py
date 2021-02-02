
# 无头浏览器
from collections import defaultdict
from urllib.request import urlopen
from selenium.webdriver import Chrome
from bs4 import BeautifulSoup
import requests
# 存在反爬虫
ths = 'http://d.10jqka.com.cn/v6/line/hs_002570/01/all.js'
#ths = 'http://stockpage.10jqka.com.cn/HQ_v4.html'
#obj = urlopen(ths)
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36'}
obj = requests.get(url=ths,headers = headers)
# drive = Chrome()
# obj = drive.get(ths)
# input = drive.find_element_by_link_text('002570')
# search = drive.find_element_by_id('su')
# input.send_keys('002570.SZ')
# search.click()
#ths = 'https://link.jianshu.com/?t=http://stockpage.10jqka.com.cn/600196/finance/#view'
#obj = urlopen(ths)
