# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, sqlalchemy as sa, json
from itertools import chain
from toolz import keyfilter, valfilter, groupby
from functools import partial
from collections import defaultdict
from gateway.database import engine, metadata
from gateway.database.db_schema import asset_db_table_names
from gateway.asset.assets import Equity, Convertible, Fund
from gateway.spider.url import ASSERT_URL_MAPPING
from gateway.driver.client import tsclient
from gateway.driver.tools import _parse_url
from gateway.driver.bar_reader import AssetSessionReader


SectorPrefix = {
                'CYB': '3',
                'KCB': '688',
                'ZXB': '0',
                'ZB': '6'
                }

AssetTypeMappings = {
                'equity': Equity,
                'convertible': Convertible,
                'fund': Fund
}


class AssetFinder(object):
    """
    An AssetFinder is an interface to a database of Asset metadata written by
    an ``AssetDBWriter``.

    This class provides methods for looking up asset by unique integer id or
    by symbol.  For historical reasons, we refer to these unique ids as 'sids'.

    returns assets list not sid string
    """
    def __init__(self):
        for table_name in asset_db_table_names:
            setattr(self, table_name, metadata.tables[table_name])
        self.reader = AssetSessionReader()
        self._asset_type_cache = defaultdict(set)

    # daily synchronize ---- change every day
    def synchronize_assets(self):
        ins = sa.select([self.asset_router.c.sid, self.asset_router.c.asset_type])
        rp = engine.execute(ins)
        asset_frame = pd.DataFrame(rp.fetchall(), columns=['sid', 'asset_type'])
        # update
        asset_frame.set_index('sid', inplace=True)
        assets_updated = set(asset_frame.index) - set(chain(*self._asset_type_cache.values()))
        if assets_updated:
            proxy_frame = asset_frame[asset_frame.index.isin(assets_updated)].groupby('asset_type').groups
            for asset_type, sids in proxy_frame.items():
                for sid in sids:
                    obj = AssetTypeMappings.get(asset_type, Fund)(sid)
                    self._asset_type_cache[obj.asset_type].add(obj)

    def retrieve_asset(self, sids):
        """
        Retrieve asset types for a list of sids.

        Parameters
        ----------
        sids : list[int]

        Returns
        -------
        types : dict[sid -> str or None]
            Asset types for the provided sids.
        """
        dct = groupby(lambda x: x.sid, chain(*(self._asset_type_cache.values())))
        print('dct', dct)
        found = set()
        if len(sids):
            for sid in sids:
                try:
                    asset = dct[sid][0]
                    found.add(asset)
                except KeyError:
                    raise NotImplementedError('missing code : %s' % sid)
        return found

    def retrieve_type_assets(self, category):
        if category == 'fund':
            fund_assets = keyfilter(lambda x: x not in set(['equity', 'convertible']), self._asset_type_cache)
            category_assets = set(chain(*fund_assets.values()))
        else:
            category_assets = self._asset_type_cache[category]
        return category_assets

    def retrieve_all(self):
        """
        Retrieve all asset in `sids`.

        Parameters
        ----------
        sids : iterable of int
            Assets to retrieve.
        default_none : bool
            If True, return None for failed lookups.
            If False, raise `SidsNotFound`.

        Returns
        -------
        asset : list[Asset or None]
            A list of the same length as `sids` containing Assets (or Nones)
            corresponding to the requested sids.

        Raises
        ------
        SidsNotFound
            When a requested sid is not found and default_none=False.
        """
        all_assets = chain(*(self._asset_type_cache.values()))
        return set(all_assets)

    def group_by_type(self, sids):
        """
        Group a list of sids by asset type.

        Parameters
        ----------
        sids : list[int]

        Returns
        -------
        types : dict[str or None -> list[int]]
            A dict mapping unique asset types to lists of sids drawn from sids.
            If we fail to look up an asset, we assign it a key of None.
        """
        found = self.retrieve_asset(sids)
        assets = groupby(lambda x: x.asset_type, found)
        return assets

    def equity_ownership_maps_by_exchange(self, brief):
        """
            exchange --- 深圳 | 上海
            brief --- sh sz
        """
        equities = self.retrieve_type_assets('equity')
        exchange_equities = [symbol for symbol in equities if symbol.exchange == brief]
        return exchange_equities

    def fuzzy_equity_ownership_by_sector(self, sector_code):
        """
            sector_code --- 主板 中小板 创业板 科创板 （首字母缩写）
        """
        prefix = SectorPrefix[sector_code.upper()]
        sector_equities = valfilter(lambda x: x.startswith(prefix), self._asset_type_cache)
        assets = set(chain(*sector_equities.values()))
        return assets

    def fuzzy_equity_ownership_by_district(self, district_code):
        """
            基于区域邮编找到对应的股票集
        """
        equities = self.retrieve_type_assets('equity')
        district_equities = groupby(lambda x: x.district, equities)
        return district_equities[district_code]

    def fuzzy_symbol_ownership_by_broker(self, broke_id):
        """
            基于主承商找到对应的股票代码
        """
        equities = self.retrieve_type_assets('equity')
        broker_equity_mappings = groupby(lambda x: x.broker, equities)
        return broker_equity_mappings[broke_id]

    def fuzzy_dual_equities(self):
        """ 获取A股与H股同时上市的股票"""
        ins = sa.select([self.equity_basics.c.sid, self.equity_basics.c.dual_sid])
        ins = ins.where(self.equity_basics.c.dual_sid != '')
        rp = engine.execute(ins)
        duals = pd.DataFrame(rp.fetchall(), columns=['sid', 'dual'])
        print('duals', duals)
        dual_equities = self.retrieve_asset(duals['sid'].values)
        return dual_equities

    def fuzzy_bond_map_by_guarantor(self, guarantor):
        ins = sa.select([self.convertible_basics.c.sid, self.convertible_basics.c.guarantor])
        rp = engine.execute(ins)
        bond_basics = pd.DataFrame(rp.fetchall(), columns=['sid', 'guarantor'])
        bond_sids = bond_basics['sid'][bond_basics['guarantor'] == guarantor]
        bond_assets = self.retrieve_asset(bond_sids.values())
        return bond_assets

    def lookup_bond_ownership_by_equity(self, sid):
        """
            基于A股代码找到对应的可转债
        """
        ins = sa.select([self.convertible_basics.c.sid, self.convertible_basics.c.swap_code])
        rp = engine.execute(ins)
        bond_basics = pd.DataFrame(rp.fetchall(), columns=['sid', 'swap_code'])
        bond_sids = bond_basics['sid'][bond_basics['swap_code'] == sid]
        bond_assets = self.retrieve_asset(bond_sids.values)
        return bond_assets

    @staticmethod
    def _is_alive(asset, session_labels):
        """
        Whether or not `asset` was active at the time corresponding to
        `reference_date_value`.

        Parameters
        ----------
        asset : Asset
            The asset object to check.

        Returns
        -------
        was_active : bool
            Whether or not the `asset` existed at the specified time.
        """
        start, end = session_labels
        mask = True
        if asset.first_traded:
            mask &= asset.first_traded <= start
        if asset.last_traded:
            mask &= (asset.last_traded >= end)
        return mask

    def lifetimes(self, sessions, category):
        """
        Compute a DataFrame representing asset lifetimes for the specified date
        range.

        Parameters
        ----------
        sessions : tuple or list
            The dates for which to compute lifetimes.
        category : str
            equity or fund or convertible

        Returns
        -------
        assets list
        # 剔除处于退市整理期的股票，一般是30个交易日  --- 收到退市决定后申请复合，最后一步进入退市整理期30个交易日
        """
        assets = self.retrieve_type_assets(category)
        _active = partial(self._is_alive, session_labels=sessions)
        active_assets = [asset for asset in assets if _active(asset)]
        return active_assets

    def can_be_traded(self, session_label):
        """
        Parameters
        ----------

        session_label : Timestamp
            The asset object to check.
        Returns
        -------
        was_active : bool
            Whether or not the `asset` is tradeable at the specified time.

        between first_traded and last_traded ; is tradeable on session label
        """
        alive_assets = [asset for asset in self.retrieve_all()
                        if asset.is_active(session_label)]
        print('alive_assets', alive_assets)
        datas = self.reader.load_raw_arrays([session_label, session_label], alive_assets, ['close'])
        trade_assets = [asset for asset in alive_assets if asset.sid in datas.keys() and datas[asset.sid]['close'][0]]
        return trade_assets

    @classmethod
    def suspend(cls, dt):
        """
            获取dt停盘信息  e.g:2020-07-13
        """
        supspend_url = 'http://datainterface.eastmoney.com/EM_DataCenter/JS.aspx?type=FD&sty=SRB&st=0&sr=-1&p=1&ps=50&'\
                       'js={"pages":(pc),"data":[(x)]}&mkt=1&fd=%s' % dt
        text = _parse_url(supspend_url, bs=False, encoding=None)
        text = json.loads(text)
        text = [t.split(',') for t in text['data']]
        # list(partition(9, text['data']))
        frame = pd.DataFrame(text, columns=['sid', 'name', 'open_ticker', 'close_ticker',
                                            'suspend', 'reason', 'market', 'date', 'market_date'])
        print('frame', frame.iloc[0, :])
        return frame

    @classmethod
    def retrieve_index_symbols(cls):
        raw = json.loads(_parse_url(ASSERT_URL_MAPPING['benchmark'], encoding='utf-8', bs=False))
        symbols = raw['data']['diff']
        frame = pd.DataFrame(symbols.values())
        frame.set_index('f12', inplace=True)
        dct = frame.iloc[:, 0].to_dict()
        return dct

    @staticmethod
    def fuzzy_equities_ownership_by_connection(exchange, flag=1):
        """获取沪港通、深港通股票 , exchange 交易所 --- (SH | SZ) ; flag :1 最新的， 0 为历史的已经踢出的"""
        con_exchange = tsclient.to_ts_con(exchange, flag).iloc[0, :]
        return con_exchange


# if __name__ == '__main__':
#
#     finder = AssetFinder()
#     finder.synchronize_assets()
#     all = finder.retrieve_all()
#     assets = finder.retrieve_asset(['512690', '515110', '603612'])
#     equities = finder.lookup_bond_ownership_by_equity('603612')
#     tradeable = finder.can_be_traded('2020-08-25')
#     print('tradeable', tradeable)
#     fund_assets = finder.retrieve_type_assets('fund')
#     print('fund_assets', fund_assets)
#     dual_assets = finder.fuzzy_dual_equities()
#     print('dual_assets', dual_assets)
#     index_assets = AssetFinder.retrieve_index_symbols()
#     print('index_assets', index_assets)
#     suspend_assets = AssetFinder.suspend('2020-08-25')
#     print('suspend_assets', suspend_assets)
#     live_assets = finder.lifetimes(['2018-09-30', '2018-10-30'], 'equity')
#     print('live assets', live_assets)
