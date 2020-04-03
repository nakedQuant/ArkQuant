# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:27:02 2018

@author: hengxinliu
"""
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine , MetaData,Table,Column,Integer,Numeric,String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import func

# from Alert.GateWay.Driver import DataLayer
#
# db = DataLayer()

metadata = MetaData()

engine = create_engine('mysql+pymysql://root:macpython@localhost:3306/spider',
                       pool_size=100, max_overflow=200, pool_timeout=-1)

metadata.create_all(engine)

#基于sqlalchemy orm 构建映射
# Base = automap_base()

# metadata = MetaData()
# engine = create_engine('mysql+pymysql://root:macpython@localhost/spider',echo=True)
# Base = declarative_base()
# cls = getattr(db,'kline')
# print(cls)

# Base.prepare(engine,reflect = True)
#
# print('---------base',Base.classes.keys())

# price = Table('asharePrice',db.metadata,autoload =True,autoload_with = db.engine)
metadata.reflect(bind = engine)
print('-------core reflect',metadata.tables.keys())

from sqlalchemy import select

"""
    kline 返回特定时间区间日K线
"""
#进行数据校对
kline = metadata.tables['asharePrice']
# ins = select([func.count(kline.c.code.distinct()).label('code')])
# rp = engine.execute(ins)
# res = rp.first()
# print(res)
# kline_code = res.code
# print('kline distinct code:',kline_code)
ins = select([kline.c.code.distinct().label('code')])
rp = engine.execute(ins)
kline_code = []
for r in rp:
    kline_code.append(r.code)
print('kline distinct code:',kline_code)
print('kline distinct code quantity',len(kline_code))
#
basics = metadata.tables['ashareInfo']
# ins = select([func.count(kline.c.code.distinct()).label('code')])
# rp = engine.execute(ins)
# res = rp.first()
# print(res)
# kline_code = res.code
# print('kline distinct code:',kline_code)
ins = select([basics.c.代码.distinct().label('code')])
rp = engine.execute(ins)
basics_code = []
for r in rp:
    basics_code.append(r.code)
print('basics distinct code:',basics_code)
print('basics distinct code length',len(basics_code))
#
flag = set(kline_code).issubset(set(basics_code))
print('kline_code suset of basics_code',flag)
missing = set(basics_code) - set(kline_code)
print('missing code bewteen basics and kline',missing)
print('missing code length from basics',len(missing))
missing_ = set(kline_code) - set(basics_code)
print('missing code bewteen kline and basics',missing_)
print('missing code length from kline',len(missing_))





# s= select([kline]).limit(5)
# print(str(s))
# #resultproxy可迭代对象
# rp= db.engine.execute(s)
# for r in rp:
#     print(r.trade_dt)
# data = rp.fetchall()
# #first 返回一条记录并关闭连接，fetchone 光标打开，更多调用
# s = select([kline.c.trade_dt,kline.c.close])
# #order 升序 , desc 降序
# #s = s.order_by(kline.c.close)
# from sqlalchemy import desc
# s = s.order_by(desc(kline.c.close))
# s = s.limit(10)
# rp = db.engine.execute(s)
# print(rp.keys())
# print(rp.fetchall())
#
# #内置函数
from sqlalchemy.sql import func
# s = select([func.count(kline.c.code).label('count_code')])
# rp = db.engine.execute(s)
# print(rp.keys())
# res = rp.first()
# print(res)
# print(res.count_code)
#distinct
# s = select([func.count(func.distinct(kline.c.code).label('unique_code')).label('quantity')])
#s = select([func.count(kline.c.unique_code)])
# print(str(s))
# rp = db.engine.execute(s)
# res = rp.first()
# print(res.quantity)
#别名alias , manager = employee.alias()
#分组 groupy_by ,不同表联合 或者同一张不同字段名
# test = select([kline.c.code])
# test = test.group_by(kline.c.code)
# rp = db.engine.execute(test)
#print(rp.fetchall())
#过滤操作
#s = select([kline]).where(kline.c.code == '000719')
#clauseElement
"""
    between,concat ,distinct,in_,is_,endswith,startswith
"""
#s = select([kline]).where(kline.c.code.like('0000%'))
#s = select([kline]).where(kline.c.close.between('1.0','2.0'))
# s = select([func.count(kline.c.code.distinct())])#.where(kline.c.code.distinct())
# #s = s.limit(10)
# print(str(s))
# rp = db.engine.execute(s)
# print(rp.fetchall())
#运算符 --- 对数据数据操作
# s = select([kline.c.trade_dt + 'test'])
# s = s.limit(10)
# rp = db.engine.execute(s)
# print(rp.fetchall())
#计算,cast 类型转换函数
from sqlalchemy import cast ,Numeric
# s = select([cookie.name,
#             cast((cookie.c.quantity * cookie.c.price),Numeric(12,2)).label('cose')])
#连接词
# from sqlalchemy import and_,or_,not_
# s = select([kline]).where(and_(kline.c.trade_dt > '2020-01-01',kline.c.trade_dt < '2020-02-01'))
# s =s.limit(10)
# rp = db.engine.execute(s)
# print(rp.fetchall())
#更新数据
# from sqlalchemy import update,delete
# u = update(kline).where(kline.c.trade_dt =='2022-03-02')
# u = u.values({'volume':20})
# rp = db.engine.execute(s)
#连接多张表 select_from -- (join) ,outerjoin(返回所有
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text


# from sqlalchemy import create_engine , MetaData,Table,Column,Integer,Numeric,String
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy import select
# from sqlalchemy.sql import func


# from Alert.GateWay.Driver import DataLayer
#
# db = DataLayer()

#基于sqlalchemy orm 构建映射

# from sqlalchemy.ext.automap import automap_base
# Base = automap_base()
# Base.prepare(db.engine,reflect = True)
# print('---------base',Base.classes.keys())
#
#
# db.metadata.reflect(bind = db.engine)
# print('-------core reflect',db.metadata.tables.keys())
# table = db.metadata.tables['convertibleDesc']
# ins = select([table.c.bond_id])
# rp = db.engine.execute(ins)
# res = [r.bond_id for r in rp]
# print(res)


#price = Table('asharePrice',db.metadata,autoload =True,autoload_with = db.engine)

# kline = db.metadata.tables['asharePrice']
# s= select([kline]).limit(5)
# print(str(s))
# #resultproxy可迭代对象
# rp= db.engine.execute(s)
# for r in rp:
#     print(r.trade_dt)
# data = rp.fetchall()
# #first 返回一条记录并关闭连接，fetchone 光标打开，更多调用
# s = select([kline.c.trade_dt,kline.c.close])
# #order 升序 , desc 降序
# #s = s.order_by(kline.c.close)
# from sqlalchemy import desc
# s = s.order_by(desc(kline.c.close))
# s = s.limit(10)
# rp = db.engine.execute(s)
# print(rp.keys())
# print(rp.fetchall())
#
# #内置函数
# s = select([func.count(kline.c.code).label('count_code')])
# rp = db.engine.execute(s)
# print(rp.keys())
# res = rp.first()
# print(res)
# print(res.count_code)
#distinct
# s = select([func.count(func.distinct(kline.c.code).label('unique_code')).label('quantity')])
#s = select([func.count(kline.c.unique_code)])
# print(str(s))
# rp = db.engine.execute(s)
# res = rp.first()
# print(res.quantity)
#别名alias , manager = employee.alias()
#分组 groupy_by ,不同表联合 或者同一张不同字段名
# test = select([kline.c.code])
# test = test.group_by(kline.c.code)
# rp = db.engine.execute(test)
#print(rp.fetchall())
#过滤操作
#s = select([kline]).where(kline.c.code == '000719')
#clauseElement
"""
    between,concat ,distinct,in_,is_,endswith,startswith
"""
#s = select([kline]).where(kline.c.code.like('0000%'))
#s = select([kline]).where(kline.c.close.between('1.0','2.0'))
# s = select([func.count(kline.c.code.distinct())])#.where(kline.c.code.distinct())
# #s = s.limit(10)
# print(str(s))
# rp = db.engine.execute(s)
# print(rp.fetchall())
# #运算符 --- 对数据数据操作
# s = select([kline.c.trade_dt + 'test'])
# s = s.limit(10)
# rp = db.engine.execute(s)
# print(rp.fetchall())
#计算,cast 类型转换函数
# s = select([cookie.name,
#             cast((cookie.c.quantity * cookie.c.price),Numeric(12,2)).label('cose')])
#连接词
# from sqlalchemy import and_,or_,not_
# s = select([kline]).where(and_(kline.c.trade_dt > '2020-01-01',kline.c.trade_dt < '2020-02-01'))
# s =s.limit(10)
# rp = db.engine.execute(s)
# print(rp.fetchall())
#更新数据
# from sqlalchemy import update,delete
# u = update(kline).where(kline.c.trade_dt =='2022-03-02')
# u = u.values({'volume':20})
# rp = db.engine.execute(s)
#连接多张表 select_from -- (join) ,outerjoin(返回所有
from sqlalchemy import text

# def enroll(self,tablename,data):
#     cls = getattr(self,tablename)
#     ins = cls.insert()
#     if len(data):
#         if isinstance(data,pd.DataFrame):
#             _to_dict = data.T.to_dict()
#             formatted = list(_to_dict.values())
#         elif isinstance(data,pd.Series):
#             formatted = data.to_dict()
#         elif isinstance(data,dict):
#             formatted = data
#         else:
#             raise ValueError
#         conn = self.db_init()
#         conn.execute(ins,formatted)
#         conn.close()

# def establish_session(self):
#     conn = self.db_init()
#     transaction = conn.begin()
#     return transaction

# def empty_table(self,tbl_name):
#     conn = self.db_init()
#     cls = getattr(self,tbl_name)
#     ins = delete(cls)
#     conn.execute(ins)
#     conn.close()
# def _update_market_value(self):
#     """计算流通市值、限售A股市值，流通B股,流通H股，总市值"""
#     bar = BarReader()
#     db = DataLayer()
#     conn = db.db_init()
#     basics = bar.load_ashare_basics()
#     assets = [item[0] for item in basics]
#     not_kline_asset = []
#     sdate = '1990-01-01'
#     for asset in assets:
#         print('asset', asset)
#         # 获取变动流通股本、总流通股本
#         raw = bar.load_equity_info(asset)
#         # 将日期转为交易日
#         raw.loc[:, 'trade_dt'] = [t if bar.is_market_caledar(t) else bar.load_calendar_offset(t, 1) for t in
#                                   raw.loc[:, 'change_dt'].values]
#         """由于存在一个变动时点出现多条记录，保留最大total_assets的记录,先按照最大股本降序，保留第一个记录"""
#         raw.sort_values(by='total_assets', ascending=False, inplace=True)
#         raw.drop_duplicates(subset='trade_dt', keep='first', inplace=True)
#         raw.index = raw['trade_dt']
#         close = bar.load_stock_kline(sdate, self.nowdays, ['close'], asset)
#         if len(close) == 0:
#             print('code:%s has not kline' % asset)
#             not_kline_asset.append(asset)
#         else:
#             # 数据对齐
#             close.loc[:, 'total'] = raw.loc[:, 'total_assets']
#             close.loc[:, 'float'] = raw.loc[:, 'float_assets']
#             close.loc[:, 'strict'] = raw.loc[:, 'strict_assets']
#             close.loc[:, 'b_assets'] = raw.loc[:, 'b_assets']
#             close.loc[:, 'h_assets'] = raw.loc[:, 'h_assets']
#             close.fillna(method='ffill', inplace=True)
#             close.fillna(method='bfill', inplace=True)
#             # 计算不同类型市值
#             mkt = close.loc[:, 'total'] * close.loc[:, 'close']
#             cap = close.loc[:, 'float'] * close.loc[:, 'close']
#             strict = close.loc[:, 'strict'] * close.loc[:, 'close']
#             b = close.loc[:, 'b_assets'] * close.loc[:, 'close']
#             h = close.loc[:, 'h_assets'] * close.loc[:, 'close']
#             # 调整格式并入库
#             data = pd.DataFrame([mkt, cap, strict, b, h]).T
#             data.columns = ['mkt', 'cap', 'strict', 'foreign', 'hk']
#             data.loc[:, 'trade_dt'] = data.index
#             data.loc[:, 'code'] = asset
#             db.enroll('mkt_value', data, conn)
#     print('not kline asset', not_kline_asset)
#     conn.close()