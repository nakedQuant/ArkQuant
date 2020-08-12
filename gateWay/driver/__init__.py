# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import sqlalchemy as sa

ASSET_DB_VERSION = 7
# A frozenset of the names of all tables in the asset db
# NOTE: When modifying this schema, update the ASSET_DB_VERSION value
engine_path = 'mysql+pymysql://root:macpython@localhost:3306/spider'

engine = sa.create_engine(engine_path)

metadata = sa.MetaData(bind=engine)
