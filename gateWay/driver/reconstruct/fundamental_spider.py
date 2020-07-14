
class FundamentalRequest(object):

    def __init__(self):
        self._init_cache()

    def _init_cache(self):
        self._retrieve_from_sqlite()

    def _retrieve_from_sqlite(self):
        table = metadata.tables['shareHolder_change']
        ins = select([func.max(table.c.declared_date)])
        rp = engine.execute(ins)
        self.deadline = rp.scalar()

    def request_holder(self):
        """股票增持、减持、变动情况"""
        page = 1
        while True:
            url = ASSETS_EVENT_URL['shareHolder_change']%page
            raw = _parse_url(url, bs=False)
            match = re.search('\[(.*.)\]', raw)
            data = json.loads(match.group())
            data = [item.split(',')[:-1] for item in data]
            holdings = pd.DataFrame(data, columns=['代码', '中文', '现价', '涨幅', '股东', '方式', '变动股本', '占总流通比', '途径', '总持仓',
                                                   '占总股本比', '总流通股', '占流通比', '变动开始日', '变动截止日', '公告日'])

            filter_holdings = holdings[holdings['declared_date'] > self.deadline]
            if len(filter_holdings) == 0:
                break
            dBwriter.writer('shareholder',filter_holdings)
            page = page + 1

    def request_massive(self, sdate, edate):
        """
            获取时间区间内股票大宗交易，时间最好在一个月之内
        """
        newcols =['trade_dt', 'sid', 'cname', 'bid_price', 'bid_volume', 'amount', 'buyer_code',
                 'buyer','seller_code', 'seller', 'type', 'unit', 'pct', 'close', 'YSSLTAG',
                 'discount','cjeltszb','1_pct', '5_pct', '10_pct', '20_pct', 'TEXCH']
        count = 1
        prefix ='js={"data":(x)}&filter=(Stype=%27EQA%27)' + \
                '(TDATE%3E=^{}^%20and%20TDATE%3C=^{}^)'.format(sdate,edate)
        while True:
            url = ASSETS_EVENT_URL['massive']%count + prefix
            raw = _parse_url(url,bs = False,encoding=None)
            raw = json.loads(raw)
            if raw['data'] and len(raw['data']):
                massive = pd.DataFrame(raw['data'])
                massive.columns = newcols
                dBwriter.writer('massive', massive)
                count = count +1
            else:
                break

    def request_release(self, sdate, edate):
        """
            获取A股解禁数据
        """
        count = 1
        prefix = '(ltsj%3E=^{}^%20and%20ltsj%3C=^{}^)'.format(sdate,edate) +\
                  '&js={"data":(x)}'
        while True:
            url = ASSETS_EVENT_URL['release']%count + prefix
            text = _parse_url(url,encoding=None,bs = False)
            text = json.loads(text)
            if text['data'] and len(text['data']):
                info = text['data']
                raw = [[item['gpdm'],item['ltsj'],item['xsglx'],item['zb']] for item in info]
                # df = pd.DataFrame(raw,columns = ['代码','解禁时间','类型','解禁占流通市值比例'])
                release = pd.DataFrame(raw,columns = ['sid','release_date','release_type','cjeltszb'])
                dBwriter.writer('release', release)
                count = count + 1
            else:
                break