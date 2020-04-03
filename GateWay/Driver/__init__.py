# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:27:02 2018

@author: hengxinliu
"""
from sqlalchemy import MetaData, create_engine,Table,Column,Integer,Numeric,String,Index
from sqlalchemy import delete
import pandas as pd

class DataLayer:
    """
        通用类型：
        sqlalchemy   python             sql
        BigInteger   int                BIGINT
        Boolean      bool               BOOLEAN
        Date         datetime.date      DATE
        DateTime     datetime.datetime  DATETIME
        Enum         str                ENUM
        Float        float              Float
        Integer      int                INTEGER
        Interval     datetime.timedelta INTERVAL
        LargeBinary  byte               BLOB
        Numeric      decimal.Decimal    NUMERIC
        Unicode      unicode            UNICODE
        Text         str                CLOB
        Time         datetime.time      DATETIME
        #autoincrement
        #元数据 Table对象目录，包含与引擎和连接的相关的信息,MetaData.table 对象目录
        #事务
        transaction = connection.begin()
        transaction.commit()
        transaction.rollback()
        #反射单个表
        price = Table('convertible',metadata,autoload = True,autoload_with = engine)
        print(price.columns.keys())
        #反射数据库
        metadata.reflect(bind = engine)
        print(metadata.tables.keys())
    """
    metadata = MetaData()

    engine = create_engine('mysql+pymysql://root:macpython@localhost:3306/spider',
                           pool_size=50, max_overflow=100, pool_timeout=-1)

    kline = Table('asharePrice',metadata,
                    Column('trade_dt',String(10),primary_key= True),
                    Column('open',String(10)),
                    Column('close',String(10)),
                    Column('high',String(10)),
                    Column('low',String(10)),
                    Column('volume',String(20)),
                    Column('turnover',String(20)),
                    Column('amplitude',String(20)),
                    Column('code',String(10),primary_key = True)
                    )

    description = Table('ashareInfo',metadata,
                    Column('代码',String(10),primary_key= True),
                    Column('上市日期',String(20)),
                    Column('发行价格',String(10)),
                    Column('证券简称更名历史',String(200)),
                    Column('主承销商',String(200)),
                    Column('注册地址',String(200))
                    )

    splits_divdend = Table('splitsDivdend',metadata,
                    Column('代码', String(20),primary_key= True),
                    Column('公告日期',String(20),primary_key= True),
                    Column('送股', String(20)),
                    Column('转增', String(20)),
                    Column('派息', String(20)),
                    Column('进度', String(20)),
                    Column('除权除息日', String(20)),
                    Column('股权登记日', String(20)),
                    Column('红股上市日',String(20))
                           )

    pairwise = Table('ashareIssue', metadata,
                     Column('代码', String(20), primary_key=True),
                     Column('公告日期', String(20),primary_key=True),
                     Column('配股方案', Numeric(5,4)),
                     Column('配股价格', Numeric(10,2)),
                     Column('除权日', String(20)),
                     Column('股权登记日', String(20)),
                     Column('配股上市日', String(20)),
                     )

    equity = Table('ashareEquity',metadata,
                      Column('代码', String(10)),
                      Column('变动日期',String(10)),
                      Column('公告日期',String(10)),
                      Column('总股本', String(20)),
                      Column('流通A股', String(20)),
                      Column('高管股', String(20)),
                      Column('限售A股', String(20)),
                      Column('流通B股', String(20)),
                      Column('限售B股', String(20)),
                      Column('流通H股', String(20)),
                      Column('国家股', String(20)),
                      Column('国有法人股', String(20)),
                      Column('境内发起人股', String(20)),
                      Column('募集法人股', String(20)),
                      Column('一般法人股', String(20)),
                      Column('战略投资者持股', String(20)),
                      Column('基金持股', String(20)),
                      Column('转配股', String(20)),
                      Column('内部职工股', String(20)),
                      Column('优先股', String(20)),
                      )

    mkt_value = Table('ashareValue',metadata,
                      Column('trade_dt', String(10),primary_key=True),
                      Column('code',String(10),primary_key= True),
                      Column('mkt',String(20)),
                      Column('cap', String(20)),
                      Column('strict', String(20)),
                      Column('hk', String(20)),
                      Column('foreign', String(20)),
                      )

    holding = Table('ashareHolding', metadata,
                    Column('代码', String(10)),
                    Column('股东', String(200)),
                    Column('方式', String(10)),
                    Column('变动股本', String(20)),
                    Column('占总流通比例', String(10)),
                    Column('途径', String(20)),
                    Column('总持仓', String(20)),
                    Column('占总股本比例', String(10)),
                    Column('总流通股', String(20)),
                    Column('变动开始日', String(10)),
                    Column('变动截止日', String(10)),
                    Column('公告日', String(10)),
                    )

    index = Table('ashareIndex',metadata,
                    Column('trade_dt',String(10),primary_key= True),
                    Column('open',String(10)),
                    Column('close',String(10)),
                    Column('high',String(10)),
                    Column('low',String(10)),
                    Column('volume',String(20)),
                    Column('turnover',String(20)),
                    Column('amplitude',String(10)),
                    Column('code',String(10),primary_key= True),
                    Column('name',String(50)),
                    Index('index','trade_dt','code',unique = True)
                    )

    etf = Table('fundPrice',metadata,
                    Column('trade_dt',String(10),primary_key=True),
                    Column('open',String(10)),
                    Column('close',String(10)),
                    Column('high',String(10)),
                    Column('low',String(10)),
                    Column('volume',String(20)),
                    Column('turnover',String(20)),
                    Column('code', String(10),primary_key = True),
                )

    convertible = Table('convertibleDesc',metadata,
                        Column('bond_id',String(10),nullable = True,primary_key= True),
                        Column('stock_id', String(10),primary_key= True),
                        Column('put_price', String(10)),
                        Column('convert_price', String(10)),
                        Column('convert_dt', String(10)),
                        Column('maturity_dt', String(10)),
                        Column('force_redeem_price', String(10)),
                        Column('put_convert_price', String(10)),
                        Column('guarantor', String(100)),
                        )

    bond_price = Table('convertiblePrice',metadata,
                    Column('trade_dt',String(10),primary_key = True),
                    Column('open',String(10)),
                    Column('close',String(10)),
                    Column('high',String(10)),
                    Column('low', String(10)),
                    Column('volume', String(20)),
                    Column('turnover', String(20)),
                    Column('code',String(10),primary_key= True),
                    )

    hkline = Table('hkPrice',metadata,
                    Column('trade_dt',String(10),primary_key= True),
                    Column('open',String(10)),
                    Column('close',String(10)),
                    Column('high',String(10)),
                    Column('low',String(10)),
                    Column('volume',String(20)),
                    Column('category',String(20)),
                    Column('h_code', String(10), primary_key=True),
                    Column('code',String(10),primary_key = True),
                    )

    margin = Table('marketMargin', metadata,
                   Column('交易日期', String(10), primary_key=True),
                   Column('融资余额', String(20)),
                   Column('融券余额', String(20)),
                   Column('融资融券总额', String(20)),
                   Column('融资融券差额', String(20)),
                   )

    status = Table('ashareStatus',metadata,
                    Column('code',String(10),primary_key = True),
                    Column('name', String(20)),
                    Column('delist_date',String(10)),
                    Column('status',String(10)),
                    )

    calendar = Table('ashareCalendar',metadata,
                    Column('trade_dt',String(10)),
                    Column('id', Integer,autoincrement= True,primary_key= True),
                     )

    metadata.create_all(bind = engine)

    @classmethod
    def initialize(cls):
        cls.metadata.drop_all(bind=cls.engine)
        cls.metadata.create_all(bind = cls.engine)

    @classmethod
    def drop_table(cls,table_name):
        tbl = getattr(cls,table_name)
        cls.metadata.drop_all(bind = cls.engine,tables = [tbl])
        delattr(cls,table_name)

    def __new__(cls,*args,**kwargs):
        if not hasattr(cls,'_instance'):
             DataLayer._instance = object.__new__(cls)
        return DataLayer._instance

    def db_init(self):
        """针对于每一个线程设立连接，多线程不能共用连接"""
        connection = self.engine.connect()
        return connection

    def enroll(self,tablename,data,conn):
        cls = getattr(self,tablename)
        ins = cls.insert()
        if len(data):
            if isinstance(data,pd.DataFrame):
                _to_dict = data.T.to_dict()
                formatted = list(_to_dict.values())
            elif isinstance(data,pd.Series):
                formatted = data.to_dict()
            elif isinstance(data,dict):
                formatted = data
            else:
                raise ValueError
            if not conn:
                conn = self.db_init()
            conn.execute(ins,formatted)

    def empty_table(self,tbl_name,conn):
        if not conn:
            conn = self.db_init()
        cls = getattr(self,tbl_name)
        ins = delete(cls)
        conn.execute(ins)


# if __name__ =='__main__':

    #DataLayer.initialize()
    #DataLayer.drop_table('aHk')
    # db= DataLayer()
#     db.empty_table('kline')
