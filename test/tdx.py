
# ticker_file_day = '/Users/python/Library/Mobile Documents/com~apple~CloudDocs/行情数据/tdx_data/szlday/sz000001.day'
# with open(ticker_file_day,'rb') as f:
#     buf = f.read()
#     size = int(len(buf) / 32)
#     data = []
#     for num in range(size):
#         idx = 32 * num
#         struct_line = struct.unpack('IIIIIfII', buf[idx:idx + 32])
#         data.append(struct_line)
#     df = pd.DataFrame(data,columns = ['trade_dt','open','high','low','close','amount','volume','appendix'])
#
# print(df)
# test = df[(df['trade_dt'] < 19910601)&(df['trade_dt'] > 19910520)]
# print('test',test)

# b = bcolz.ctable.fromdataframe(df)
# b.flush()
# print('-----------b',b)

def to_timestamp(df):
    df['year'] = df['dates'] // 2048 + 2004
    df['month'] = (df['dates'] % 2048) // 100
    df['day'] = (df['dates'] % 2048) % 100
    df['hour'] = df['sub_dates'] // 60
    df['minutes'] = df['sub_dates'] % 60
    # df['year'] = df['year'].astype('int')
    # df['month'] = df['month'].astype('int')
    # df['day'] = df['day'].astype('int')
    # df['hour'] = df['hour'].astype('int')
    # df['minutes'] = df['minutes'].astype('int')
    df['ticker'] = df.apply(lambda x : pd.Timestamp(
                            datetime.datetime(int(x['year']),int(x['month']),int(x['day']),
                                              int(x['hour']),int(x['minutes']))),
                            axis = 1)
    # df['timestamp'] = df['ticker']
    # df['ticker'] = df.apply(lambda x : pd.Timestamp(
    #                         datetime.datetime(x['year'],x['month'],x['day'],x['hour'],x['minutes'])),
    #                         axis = 1)
    df['timestamp'] = df['ticker'].apply(lambda x : x.timestamp())
    cols = ['timestamp','open','high','low','close','amount','volume','appendix']
    return df.loc[:,cols]

ticker_file_min = '/Users/python/Library/Mobile Documents/com~apple~CloudDocs/行情数据/tdx_data/sz5fz/sz300554.5'
with open(ticker_file_min,'rb') as f:
    buf = f.read()
    size = int(len(buf) / 32)
    data = []
    for num in range(size):
        idx = 32 * num
        struct_line = struct.unpack('HhIIIIfii', buf[idx:idx + 32])
        data.append(struct_line)
    df = pd.DataFrame(data,columns = ['dates','sub_dates','open','high','low','close','amount','volume','appendix'])
    ticker_df = to_timestamp(df)

print('ticker_df',ticker_df.head())
# -----------------
b = bcolz.ctable.fromdataframe(ticker_df)
b.flush()
print('-----------b',b)

#bcolz 操作
initial_array = np.empty(0, np.uint32)
bcolz_path = '/Users/python/Library/Mobile Documents/com~apple~CloudDocs/bcolz/test.bcolz'
init = bcolz.ctable(
    rootdir=bcolz_path,
    columns = [
        initial_array,
        initial_array,
        initial_array,
        initial_array,
        initial_array,
        initial_array,
        initial_array,
        initial_array
    ],
    names = [
        'timestamp',
        'open',
        'high',
        'low',
        'close',
        'amount',
        'volume',
        'appendix'
    ],
    mode = 'w'
)

init.append(b)
# b.append(init)
print('init apppend',init)

# s = pd.Timestamp('2020-01-01')
# e = pd.Timestamp('2020-05-01')
# # filter = init.fetchwhere('100000 <= volume')
# filter = init.fetchwhere("({0} <= ticker) & (ticker < {1})".format(s.timestamp(),e.timestamp()))
# print('filter',filter)
#eval
day_seconds =  24 * 60 * 60
ticker = 9 * 60 * 60 + 30 * 60
print('init',init)
print(init.names)
print(len(init))
# test = init.eval('(ticker - 34200) / 86400')
test = init.fetchwhere("(timestamp - 34200) % 86400 == 0")
# test = init.eval('2 * ticker')
print('test',test)
#
# ticker_file_day = '/Users/python/Library/Mobile Documents/com~apple~CloudDocs/行情数据/tdx_data/szlday/sz000002.day'
# with open(ticker_file_day,'rb') as f:
#     buf = f.read()
#     size = int(len(buf) / 32)
#     data = []
#     for num in range(size):
#         idx = 32 * num
#         struct_line = struct.unpack('IIIIIfII', buf[idx:idx + 32])
#         data.append(struct_line)
#     df_ = pd.DataFrame(data,columns = ['trade_dt','open','high','low','close','amount','volume','appendix'])
#
# b_ = bcolz.ctable.fromdataframe(df_)
# print('-----------b_',b_)
# # b_to = b_.todataframe(orient='index')
# # print('b_to-----',b_to)
#
# b.append(b_)
# print('---------',len(b))
# b.flush()
# print(b.names)
# filter = b.fetchwhere('(19910130 <= trade_dt) & (trade_dt < 19910203)')
# print('filter',filter)
# print(b.rootdir)
# init.append(b_)
# print('init append b_',init)
#
# print(len(init))
# print(init.rootdir)
#
# init.attrs['metadata'] = ['test']
#
# print('init attr',init)
# print(len(init))
# print(init.cparams)
# print(init.attrs.attrs)
# init.attrs['metadata'] = 'update'
# init.resize(10)
# print(init.attrs.attrs)
# print(init.names)
# print('init',init)
#
# to_dataframe = pd.DataFrame(init)
# print('to_dataframe',to_dataframe)

namespace = dict()
with open('/Users/python/Library/Mobile Documents/com~apple~CloudDocs/simulation/test/test_driver.py','r') as f:
    exec(f.read(), namespace)

print(namespace.keys())
test = namespace['UnionEngine']
print(test)
# ins = test()
# print(ins)
# print(namespace['__builtins__'])
# print(namespace['signature'])

# exec eval compile将字符串转化为可执行代码 , exec compile source into code or AST object ,if filename is None ,'<string>' is used
# code = compile(self.algoscript, algo_filename, 'exec')
# exec_(code, self.namespace)
#
# def noop(*args, **kwargs):
#     pass
#
#
# if algo_filename is None:
#     algo_filename = '<string>'
#     # exec eval compile将字符串转化为可执行代码 , exec compile source into code or AST object ,
#     # if filename is None ,'<string>' is used
#     code = compile(self.algoscript, algo_filename, 'exec')
#     exec(code, self.namespace)
#
#     # dict get参数可以为方法或者默认参数
#     self._initialize = self.namespace.get('initialize', noop)
#     self._handle_data = self.namespace.get('handle_data', noop)
#     self._before_trading_start = self.namespace.get(
#         'before_trading_start',
#     )
#     # Optional analyze function, gets called after run
#     self._analyze = self.namespace.get('analyze')
# else:
#     self._initialize = initialize or (lambda self: None)
#     self._handle_data = handle_data
#     self._before_trading_start = before_trading_start
#     self._analyze = analyze
