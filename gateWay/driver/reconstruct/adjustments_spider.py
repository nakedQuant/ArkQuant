class AdjustmentsRequest(object):

    alter_tbls = frozenset(['symbol_splits',
                  'symbol_rights',
                  'symbol_equity_structure'])

    def __init__(self):
        """
            将数据库已经存在的标的时间缓存
        """
        self._cache()

    def _cache(self):
        self.sid_deadlines = dict()
        for tbl in self.alter_tbls:
            self._retrieve_from_sqlite(tbl)

    def _retrieve_from_sqlite(self,tbl):
        table = metadata.tables[tbl]
        ins = select([func.max(table.c.declared_date),table.c.sid])
        ins = ins.groupby(table.c.sid)
        rp = engine.execute(ins)
        deadlines = pd.DataFrame(rp.fetchall(),columns = ['declared_date','sid'])
        deadlines.set_index('sid',inplace = True)
        self.sid_deadlines[tbl] = deadlines.iloc[:,0]

    def _parse_symbol_issues(self,content,code):
        """配股"""
        resource  = content['divdend']
        table = resource.find('table', {'id': 'sharebonus_2'})
        body = table.tbody
        raw = []
        [raw.append(item.get_text()) for item in body.findAll('tr')]
        if len(raw) ==1 and raw[0] == '暂时没有数据！':
            print('------------code : %s has not 配股'%code,raw[0])
        else:
            parse_raw = [item.split('\n')[1:-2] for item in raw]
            pairwise = pd.DataFrame(parse_raw, columns=['declared_date', 'rights_bonus', 'rights_price',
                                                        'benchmark_share','pay_date', 'record_date',
                                                        '缴款起始日','缴款终止日','effective_date','募集资金合计'])
            pairwise.loc[:,'sid'] = code
            max_date = self.sid_deadlines['symbol_rights'][code]
            res =  pairwise[pairwise['公告日期'] > max_date] if max_date else pairwise
            dBwriter.writer('symbol_rights', res)

    def _parse_symbol_divdend(self,content,code):
        """获取分红配股数据"""
        resource = content['divdend']
        table = resource.find('table', {'id': 'sharebonus_1'})
        body = table.tbody
        raw = []
        [raw.append(item.get_text()) for item in body.findAll('tr')]
        if len(raw) ==1 and raw[0] == '暂时没有数据！':
            print('------------code : %s has not splits and divdend'%code,raw[0])
        else:
            parse_raw = [item.split('\n')[1:-2] for item in raw]
            split_divdend = pd.DataFrame(parse_raw, columns=['declared_date', 'sid_bonus', 'sid_transfer', 'bonus',
                                                             'progress', 'pay_date', 'record_date', 'effective_date'])
            split_divdend.loc[:,'sid'] = code
            max_date = self.sid_deadlines['symbol_divdends'][code]
            res =  split_divdend[split_divdend['公告日期'] > max_date] if max_date else split_divdend
            dBwriter.writer('symbol_divdends', res)

    def _parse_symbol_equity(self,content,code):
        """获取股票股权结构分布"""
        resource = content['equity']
        tbody = resource.findAll('tbody')
        if len(tbody) == 0:
            print('due to sina error ,it raise cannot set a frame with no defined index and a scalar when tbody is null')
        equity = pd.DataFrame()
        for th in tbody:
            formatted = parse_content_from_header(th)
            equity = equity.append(formatted)
        #调整
        equity.loc[:,'代码'] = code
        equity.index = range(len(equity))
        max_date = self.sid_deadlines['symbol_equity_structure'][code]
        filter_equity= equity[equity['公告日期'] > max_date] if max_date else equity
        dBwriter.writer('symbol_equity_structure',filter_equity)

    def _request_for_sid(self,sid):
        content = dict()
        for category,path in ASSETS_STRUCTURE_DIVDEND.itmes():
            req = path%sid
            content[category] = _parse_url(req)
        return content

    def request_for_structure(self,sid):
        try:
            contents = self._request_for_sid(sid)
        except Exception as e:
            print('%s occur due to high prequency',e)
            #retry
            # time.sleep(np.random.randint(0,1))
            contents = self._request_for_sid(sid)
        self._parse_symbol_divdend(contents,sid)
        self._parse_symbol_issues(contents,sid)
        self._parse_symbol_equity(contents,sid)
