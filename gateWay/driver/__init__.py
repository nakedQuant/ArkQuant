"""
    1. adjustments 与 price 分离
    2. 构建price ajustments 接口
    3. 不同数据源接口 : 数据库 h5 csv json
    4. 读取数据的方式 : bcolz blaze
    5. 构建管道的数据接口

    资产类别: 股票 ETF 基准 ， 其中可转债 | H股作为股票属性

    碰到问题是否需要将spider数据爬取与数据入库分离
    spider作为数据下载端入口，现在问题数据入口端

    data_portal --- asset.roller_finder.py
                --- bar.py
                --- dispatcher_bar_reader.py
                --- history_loader.py
                --- resample.py

    roller_finder.py :
    from zipline.asset import (
    Asset,
    AssetConvertible,
    Equity,
    Future,
    PricingDataAssociable,

    AssertFinder

    bar.py --- data_frequency
                  --- load_raw_arrays (columns, start_date, end_date, asset)
                  --- first_trading_day
                  --- last_avaiable_dt
                  --- trading_calendar
                  --- get_value (sid, dt, field)
                  --- get_last_trade_dt (asset, dt)

)

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

    isolation_level
    READ COMMITTED

    READ UNCOMMITTED

    REPEATABLE READ

    SERIALIZABLE

    AUTOCOMMIT

    To set isolation level using create_engine():

    对于使用 os.fork 系统调用，例如python multiprocessing 模块，通常要求 Engine 用于每个子进程,
    因为 Engine 维护对最终引用DBAPI连接的连接池的引用-这些连接通常不可跨进程边界移植 --- 多线程原理
"""
