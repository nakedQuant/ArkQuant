# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from collections import defaultdict, namedtuple
import json, pandas as pd
from itertools import chain
from toolz import partition_all

from gateWay.assets._config import ASSERT_URL_MAPPING, ASSET_SUPPLEMENT_URL
from gateWay.driver.tools import _parse_url

AssetData = namedtuple(
    'AssetData', (
        'equities',
        'convertibles',
        'funds',
    ),
)


class AssetSpider(object):

    def __init__(self):
        self._assets_cache = defaultdict(list)

    @staticmethod
    def _fetch_equities_from_dfcf():
        # 获取存量股票包括退市
        raw = json.loads(_parse_url(ASSERT_URL_MAPPING['equity'], bs=False))
        equities = [item['f12'] for item in raw['data']['diff']]
        return equities

    @staticmethod
    def _fetch_duals_from_dfcf():
        dual_mappings = {}
        page = 1
        while True:
            url =  ASSERT_URL_MAPPING['dual'] % page
            raw = _parse_url(url, bs=False, encoding=None)
            raw = json.loads(raw)
            diff = raw['data']
            if diff and len(diff['diff']):
                # f12 -- hk ; 191 -- code
                diff = {item['f12']: item['f191'] for item in diff['diff']}
                dual_mappings.update(diff)
                page = page + 1
            else:
                break
        return dual_mappings

    @staticmethod
    def _fetch_convertibles_from_dfcf():
        # 剔除未上市的
        page = 1
        bonds = []
        while True:
            bond_url =  ASSERT_URL_MAPPING['convertible']%page
            text = _parse_url(bond_url, encoding='utf-8', bs=False)
            text = json.loads(text)
            data = text['data']
            if data:
                bonds.append(data)
                page = page + 1
            else:
                break
        # 过滤未上市的可转债
        bond_mappings = {bond['BONDCODE']: bond for bond in bonds if bond['LISTDATE'] != '-'}
        return bond_mappings

    @staticmethod
    def _fetch_funds_from_dfcf():
        # 基金主要分为 固定收益 分级杠杆（A/B） (ETF场内|QDII-ETF)
        obj = _parse_url( ASSERT_URL_MAPPING['equity']['fund'])
        # print(obj.prettify())
        raw = [data.find_all('td') for data in obj.find_all(id='tableDiv')]
        text = [t.get_text() for t in raw[0]]
        df = pd.DataFrame(partition_all(14, text[18:]), columns=text[2:16])
        df = df.apply(lambda x: x['基金简称'][:-5])
        return df

    def _update_cache_for_asset(self):
        self._request_cache = {}
        equities = self._fetch_equities_from_dfcf()
        self._request_cache['equity'] = set(equities) - set(self._assets_cache['equity'])
        self._assets_cache['equity'] = equities

        convertibles = self._fetch_convertibles_from_dfcf()
        self._request_cache['convertible'] = set(convertibles) - set(self._assets_cache['convertible'])
        self._assets_cache['convertible'] = convertibles

        funds = self._fetch_funds_from_dfcf()
        self._request_cache['fund'] = set(funds['基金代码'].values) - set(self._assets_cache['fund'])
        self._assets_cache['fund'] = funds

        duals = self._fetch_duals_from_dfcf()
        self._request_cache['dual'] = set(duals) - set(self._assets_cache['dual'])
        self._assets_cache['dual'] = duals

    def _request_supplemnet_for_equity(self):
        # 获取dual
        dual_equity = self._request_cache['equity']
        equities = self._request_cache['equity']
        equity_basics = []
        # 公司基本情况
        for code in equities:
            url = ASSET_SUPPLEMENT_URL['equity_supplement'] % code
            obj = _parse_url(url)
            table = obj.find('table', {'id': 'comInfo1'})
            tag = [item.findAll('td') for item in table.findAll('tr')]
            tag_chain = list(chain(*tag))
            raw = [item.get_text() for item in tag_chain]
            # 去除格式
            raw = [i.replace('：', '') for i in raw]
            raw = [i.strip() for i in raw]
            baiscs = list(zip(raw[::2], raw[1::2]))
            basics_mapping = {item[0]: item[1] for item in baiscs}
            basics_mapping.update({'代码': code})
            try:
                dual = dual_equity[code]
                basics_mapping.update({'港股' :dual})
            except KeyError:
                pass
            equity_basics.append(basics_mapping)
        #转化为DataFrame
        equity_dataframe = pd.DataFrame(equity_basics)
        return equity_dataframe

    def _request_supplement_for_convertible(self):
        # 剔除未上市的
        bond_mappings = self._request_cache['convertible']
        # bond basics 已上市的basics
        text = _parse_url(ASSET_SUPPLEMENT_URL['convertible_supplement'],
                          bs=False,
                          encoding=None)
        text = json.loads(text)
        basics = [basic['cell'].update(bond_mappings[basic['id']]) for basic in text['rows']]
        basics_dataframe = pd.DataFrame(basics)
        return basics_dataframe

    def _to_asset_data(self):
        self._update_cache_for_asset()
        equity_mappings = self._request_supplemnet_for_equity()
        convertible_mappings = self._request_supplement_for_convertible()
        fund_mappings = self._fetch_funds_from_dfcf()
        return AssetData(
            equities=equity_mappings,
            convertibles=convertible_mappings,
            funds=fund_mappings,
        )

    def load_data(self):
        """
            Returns a standard set of pandas.DataFrames:
            equities, futures, exchanges, root_symbols
        """
        asset_data = self._to_asset_data()
        return asset_data
