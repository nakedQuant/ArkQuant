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