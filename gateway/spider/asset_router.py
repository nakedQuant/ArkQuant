# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from collections import namedtuple, defaultdict
import json, pandas as pd, time
from itertools import chain
from toolz import partition_all, keyfilter
from gateway.spider.url import ASSERT_URL_MAPPING, ASSET_SUPPLEMENT_URL
from gateway.spider import Crawler
from gateway.database.asset_writer import AssetWriter
from gateway.driver.tools import _parse_url


AssetData = namedtuple(
    'AssetData', (
        'equities',
        'convertibles',
        'funds',
    ),
)


class AssetSpider(Crawler):
    """
        a.获取全部的资产标的 --- equity convertible etf
        b.筛选出需要更新的标的（与缓存进行比较）
        # 字段非string一定要单独进行格式处理
    """

    @staticmethod
    def _request_equities():
        # 获取存量股票包括退市
        raw = json.loads(_parse_url(ASSERT_URL_MAPPING['equity'], bs=False))
        equities = [item['f12'] for item in raw['data']['diff']]
        return equities

    @staticmethod
    def _request_convertibles():
        # 获取上市的可转债的标的
        page = 1
        bonds = []
        while True:
            bond_url = ASSERT_URL_MAPPING['convertible'] % page
            text = _parse_url(bond_url, encoding='utf-8', bs=False)
            text = json.loads(text)
            data = text['data']
            if data:
                bonds = chain(bonds, data)
                page = page + 1
            else:
                break
        bonds = list(bonds)
        # 过滤未上市的可转债 bond_id : bond_basics
        bond_mappings = {bond['BONDCODE']: bond for bond in bonds if bond['LISTDATE'] != '-'}
        return bond_mappings

    @staticmethod
    def _request_funds():
        # 获取存量的ETF 基金主要分为 固定收益 分级杠杆（A/B） ( ETF场内| QDII-ETF )
        obj = _parse_url(ASSERT_URL_MAPPING['fund'])
        raw = [data.find_all('td') for data in obj.find_all(id='tableDiv')]
        text = [t.get_text() for t in raw[0]]
        # 由于format原因，最后两列为空
        frame = pd.DataFrame(partition_all(14, text[18:]), columns=text[2:16]).iloc[:, :-2]
        return frame

    @staticmethod
    def _request_duals():
        # 获取存量AH两地上市的标的
        dual_mappings = {}
        page = 1
        while True:
            url = ASSERT_URL_MAPPING['dual'] % page
            raw = _parse_url(url, bs=False, encoding=None)
            raw = json.loads(raw)
            diff = raw['data']
            if diff and len(diff['diff']):
                # f12 -- hk ; 191 -- code
                diff = {item['f191']: item['f12'] for item in diff['diff']}
                dual_mappings.update(diff)
                page = page + 1
            else:
                break
        return dual_mappings

    def to_be_updated(self):
        left = dict()
        existing_assets = self._retrieve_assets_from_sqlite()
        # update equity -- list
        equities = self._request_equities()
        left['equity'] = set(equities) - set(existing_assets.get('equity', []))
        print('update equities', len(left['equity']))
        # update convertible --- dict contain basics
        convertibles = self._request_convertibles()
        # print('full convertible', len(convertibles.keys()))
        update_convertibles = set(convertibles) - set(existing_assets.get('convertible', []))
        left['convertible'] = keyfilter(lambda x: x in update_convertibles, convertibles)
        print('update convertibles', len(left['convertible']))
        # update funds -- frame  contain basics
        fund = self._request_funds()
        # print('full fund', len(fund))
        update_funds = set(fund['基金代码'].values) - set(existing_assets.get('fund', []))
        fund_frame = fund[fund['基金代码'].isin(update_funds)] if update_funds else pd.DataFrame()
        left['fund'] = fund_frame
        print('update funds', len(left['fund']))
        # update duals
        duals = self._request_duals()
        # print('dual', duals)
        left['dual'] = duals
        return left

    def _writer_internal(self, *args):
        raise NotImplementedError()


class AssetRouterWriter(Crawler):
    """
        asset status 由于吸收合并代码可能会消失但是主体继续上市存在 e.g. T00018
        对于暂停上市 delist_date 为None(作为一种长期停盘的情况来考虑，由于我们不能存在后视误差不清楚是否能重新上市）
        基于算法发出信号操作暂时上市的标的, 为了避免前视误差，过滤筛选距离已经退市标的而不是暂停上市  e.g. 5个交易日的股票；
        对于暂停上市的股票可能还存在重新上市的可能性也可能存在退市的可能性 --- None ,状态不断的更新
    """
    def __init__(self, engine_path=None):
        self._writer = AssetWriter(engine_path)
        self.missing = defaultdict(set)
        self.spider = AssetSpider()

    @staticmethod
    def _request_equity_basics(code):
        url = ASSET_SUPPLEMENT_URL['equity_supplement'] % code
        obj = _parse_url(url)
        table = obj.find('table', {'id': 'comInfo1'})
        tag = [item.findAll('td') for item in table.findAll('tr')]
        tag_chain = list(chain(*tag))
        raw = [item.get_text() for item in tag_chain]
        # remove format
        raw = [i.replace('：', '') for i in raw]
        raw = [i.strip() for i in raw]
        brief = list(zip(raw[::2], raw[1::2]))
        mapping = {item[0]: item[1] for item in brief}
        mapping.update({'代码': code})
        return mapping

    def _request_equities_basics(self, update_mapping):
        equities = update_mapping['equity']
        t = time.time()
        if len(equities):
            # 获取dual
            dual_equity = update_mapping['dual']
            basics = []
            # 公司基本情况
            for code in equities:
                try:
                    mapping = self._request_equity_basics(code)
                except Exception as e:
                    print('code:%s due to %s' % (code, e))
                    self.missing['equity'].add(code)
                else:
                    self.missing['equity'].discard(code)
                    print('scrapy code % s from sina successfully' % code)
                    dual = dual_equity.get(code, None)
                    mapping.update({'港股': dual})
                    basics.append(mapping)
            frame = pd.DataFrame(basics)
            # frame.to_csv('equity_basics.csv')
        else:
            frame = pd.DataFrame()
        print('spider equity basics elapsed time : %f' % (time.time() - t))
        return frame

    @staticmethod
    def _request_convertible_basics(update_mapping):
        bond_mappings = update_mapping['convertible']
        if bond_mappings:
            # bond basics 已上市的basics
            text = _parse_url(ASSET_SUPPLEMENT_URL['convertible_supplement'], encoding=None, bs=False)
            text = json.loads(text)
            # 取交集
            common_bond = set(bond_mappings) & set([basic['id'] for basic in text['rows']])
            print('common', len(common_bond))
            # combine two dict object --- single --- 保持数据的完整性
            [basic['cell'].update(bond_mappings[basic['id']]) for basic in text['rows'] if basic['id'] in common_bond]
            basics = [basic['cell'] for basic in text['rows'] if basic['id'] in common_bond]
            basics_frame = pd.DataFrame(basics)
        else:
            basics_frame = pd.DataFrame()
        return basics_frame

    def _load_data(self, to_be_updated):
        """
            inner method
            accumulate equities , convertibles, etfs
            :return: AssetData
        """
        convertible_frames = self._request_convertible_basics(to_be_updated)
        print('asset data convertible', convertible_frames.head())
        fund_frames = to_be_updated['fund']
        print('asset data fund', fund_frames.head())
        equity_frames = self._request_equities_basics(to_be_updated)
        print('asset data equity', equity_frames.head())
        return AssetData(
            equities=equity_frames,
            convertibles=convertible_frames,
            funds=fund_frames,
        )

    def load_data(self):
        to_be_updated = self.spider.to_be_updated()
        data = self._load_data(to_be_updated)
        return data

    def rerun(self):
        # print('missing equity', self.missing)
        if len(self.missing['equity']):
            missed = self.missing.copy()
            # RuntimeError: Set changed size during iteration
            equity_frames = self._request_equities_basics(missed)
            AssetData.convertibles = AssetData.funds = pd.DataFrame()
            AssetData.equities = equity_frames
            self._writer.write(AssetData)
            self.rerun()
        print('rerun module finish')
        # self.missing['equity'] = set()

    def _writer_internal(self):
        asset_data = self.load_data()
        self._writer.write(asset_data)
        self.rerun()

    def writer(self):
        self._writer_internal()
        # in case of pressure on spider network
        time.sleep(5)


__all__ = ['AssetSpider', 'AssetRouterWriter']


if __name__ == '__main__':

    t = time.time()
    router = AssetRouterWriter()
    router.writer()
    print('elapsed time : %f' % (time.time() - t))
