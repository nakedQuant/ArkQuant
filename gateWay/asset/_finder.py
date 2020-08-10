# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, sqlalchemy as sa, json
from itertools import groupby, chain
from toolz import valfilter, keyfilter, valmap
from functools import partial
from .asset_db_schema import asset_db_table_names
from .assets import Equity, Convertible, Fund
from ._config import ASSERT_URL_MAPPING
from gateWay.driver.third_api.client import tsclient
from gateWay.driver.tools import _parse_url
from gateWay.driver.db_schema import engine


Sector_Prefix = {
                'CYB': '3',
                'KCB': '688',
                'ZXB': '0',
                'ZB': '6'
                }

AssetMappings = {
    c.asset_type: c
    for c in [Equity, Convertible, Fund]
}


class AssetFinder(object):
    """
    An AssetFinder is an interface to a database of Asset metadata written by
    an ``AssetDBWriter``.

    This class provides methods for looking up asset by unique integer id or
    by symbol.  For historical reasons, we refer to these unique ids as 'sids'.

    Parameters
    ----------
    engine : str or SQLAlchemy.engine
        An engine with a connection to the asset database to use, or a string
        that can be parsed by SQLAlchemy as a URI.
    """
    def __init__(self):
        metadata = sa.MetaData(bind=engine)
        metadata.reflect(only=asset_db_table_names)
        for table_name in asset_db_table_names:
            setattr(self, table_name, metadata.tables[table_name])
        # setup cache
        self._asset_type_cache = {}

    # daily synchronize
    def synchronize_asset_mappings(self):
        ins = sa.select([self.asset_router.c.sid, self.asset_router.c.asset_type])
        rp = self.engine.execute(ins)
        asset_mappings = pd.DataFrame(rp.fetchall(), columns=['sid', 'asset_type'])
        asset_mappings.set_index('sid', inplace=True)
        # update
        assets_to_be_update = set(asset_mappings.index) - set(self._asset_type_cache)
        if assets_to_be_update:
            proxy_mappings = asset_mappings[assets_to_be_update].groupby('asset_type').groups
            for category, sids in proxy_mappings.items():
                for sid in sids:
                    self._asset_type_cache[sid] = AssetMappings[category](sid, engine)

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
        sids = [sids] if isinstance(sids, str) else sids
        found = set()
        missing = set()
        if sids:
            for sid in sids:
                try:
                    asset = self._asset_type_cache[sid]
                    found.add(asset)
                except KeyError:
                    missing.add(sid)
        return found, missing

    @staticmethod
    def retrieve_benchmarks():
        raw = json.loads(_parse_url(ASSERT_URL_MAPPING['benchmark'], encoding='utf-8', bs=False))
        symbols = raw['data']['diff']
        return symbols

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
        return self._asset_type_cache

    def retrieve_type_assets(self, category):
        assets = valmap(lambda x: x.asset_type == category,
                        self._asset_type_cache)
        return assets

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
        assets = groupby(self.lookup_asset_types(sids), lambda x: x.asset_type)
        return assets

    def equity_ownership_maps_by_exchange(self, brief):
        """
            exchange --- 深圳 | 上海
            brief --- sh sz
        """
        equities = self.retrieve_all('equity').values()
        exchange_equities = valfilter(lambda x: x.exchange == brief, equities)
        return exchange_equities

    def fuzzy_equity_ownership_by_sector(self, sector_code):
        """
            sector_code --- 主板 中小板 创业板 科创板 （首字母缩写）
        """
        prefix = Sector_Prefix[sector_code.upper()]
        equities = self.retrieve_all('equity')
        sector_equities = keyfilter(lambda x: x.startswith(prefix), equities)
        return sector_equities

    def fuzzy_equity_ownership_by_district(self, district_code):
        """
            基于区域邮编找到对应的股票集
        """
        equities = self.retrieve_all('equity').values()
        district_equities = groupby(equities, lambda x: x.district)
        return district_equities[district_code]

    def fuzzy_symbol_ownership_by_broker(self, broke):
        """
            基于主承商找到对应的股票代码
        """
        equities = self.retrieve_all('equity').values()
        broker_equity_mappings = groupby(equities, lambda x: x.broker)
        return broker_equity_mappings[broke]

    def fuzzy_dual_equities(self):
        """ 获取A股与H股同时上市的股票"""
        equities = self.retrieve_all('equity').values()
        dual_equities = [e for e in equities if e.dual != 'null']
        return dual_equities

    def fuzzy_bond_map_by_guarantor(self, guarantor):
        bond_assets = self.retrieve_all('bond')
        bonds = valfilter(lambda x: x.guarantor == guarantor, bond_assets)
        return bonds

    def lookup_bond_ownership_by_equity(self, sid):
        """
            基于A股代码找到对应的可转债
        """
        bonds = self.retrieve_all('bond')
        bond_equities = valfilter(lambda x: x.swap_code == sid, bonds)
        return bond_equities

    def lifetimes(self, sessions, include_start_date=False):
        """
        Compute a DataFrame representing asset lifetimes for the specified date
        range.

        Parameters
        ----------
        sessions : tuple or list
            The dates for which to compute lifetimes.
        include_start_date : bool
            Whether or not to count the asset as alive on its start_date.

        Returns
        -------
        lifetimes : pd.DataFrame
            A frame of dtype bool with `dates` as index and an Int64Index of
            asset as columns.  The value at `lifetimes.loc[date, asset]` will
            be True iff `asset` existed on `date`.  If `include_start_date` is
            False, then lifetimes.loc[date, asset] will be false when date ==
            asset.start_date.

        See Also
        --------
        numpy.putmask
        zipline.pipeline.engine.SimplePipelineEngine._compute_root_mask
        # 剔除处于退市整理期的股票，一般是30个交易日  --- 收到退市决定后申请复合，最后一步进入退市整理期30个交易日
        """
        _active = partial(self._is_alive,
                          sessions=sessions,
                          include=include_start_date)
        active_assets = [asset for sid, asset in self._asset_cache.items()
                         if _active(asset)]
        return active_assets

    @staticmethod
    def _is_alive(asset, session_labels, include):
        """
        Whether or not `asset` was active at the time corresponding to
        `reference_date_value`.

        Parameters
        ----------
        reference_date_value : int
            Date, represented as nanoseconds since EPOCH, for which we want to know
            if `asset` was alive.  This is generally the result of accessing the
            `value` attribute of a pandas Timestamp.
        asset : Asset
            The asset object to check.

        Returns
        -------
        was_active : bool
            Whether or not the `asset` existed at the specified time.
        """
        start, end = session_labels
        mask = (asset.first_traded <= start if include
                else asset.first_traded < start)
        if asset.last_traded:
            mask &= (asset.last_traded > end)
        return mask

    def was_active(self, session_label):
        """
        Parameters
        ----------

        session_label : Timestamp
            The asset object to check.
        Returns
        -------
        was_active : bool
            Whether or not the `asset` is tradeable at the specified time.
        """
        alive_assets = [asset for asset in chain(* self._asset_type_cache.values())
                        if asset.is_active(session_label)]
        return alive_assets

    @staticmethod
    def fuzzy_equities_ownership_by_connection(exchange, flag=1):
        """获取沪港通、深港通股票 , exchange 交易所 --- (SH | SZ) ; flag :1 最新的， 0 为历史的已经踢出的"""
        con_exchange = tsclient.to_ts_con(exchange, flag).iloc[0, :]
        return con_exchange
