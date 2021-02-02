# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import sqlalchemy as sa

# Define a version number for the database generated by these writers
# Increment this version number any time a change is made to the schema of the
# asset database
# NOTE: When upgrading this remember to add a downgrade in:
# .asset_db_migrations
ASSET_DB_VERSION = 8
SQLITE_MAX_VARIABLE_NUMBER = 999
PoolSize = 1000
OVerFlow = 20

# A frozenset of the names of all tables in the asset db
# NOTE: When modifying this schema, update the ASSET_DB_VERSION value
config = {
    "user": 'root',
    "password": 'macpython',
    "host": 'localhost',
    "port": '3306',
    'database': 'ark'
}

# engine_path = 'mysql+pymysql://root:macpython@localhost:3306/test01'
engine_path = 'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'.format(user=config['user'],
                                                                                  password=config['password'],
                                                                                  host=config['host'],
                                                                                  port=config['port'],
                                                                                  database=config['database'],)
# READ COMMITTED
# READ UNCOMMITTED
# REPEATABLE READ
# SERIALIZABLE
# AUTOCOMMIT
# """
# engine = create_engine('mysql+pymysql://root:macpython@localhost:3306/c_test',
#                         pool_size=50, max_overflow=100, pool_timeout=-1，
#                         isolation_level="READ UNCOMMITTED")
# engine.execution_options(isolation_level="READ COMMITTED")
# print(engine.get_execution_options())
# #代理
# from sqlalchemy import inspect
# insp = inspect(engine)
# print(insp.get_table_names())
# print(insp.get_columns('asharePrice'))
# print(insp.get_schema_names())
# # get_pk_constraint get_primary_keys get_foreign_keys get_indexes
# sa.CheckConstraint('id <= 1')
# ins = ins.order_by(table.c.trade_dt)

engine = sa.create_engine(engine_path, pool_size=PoolSize, max_overflow=OVerFlow)

metadata = sa.MetaData(bind=engine)