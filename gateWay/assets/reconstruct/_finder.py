# Copyright 2016 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json, re, pandas as pd,datetime,sqlalchemy as sa
from functools import partial,lru_cache
from weakref import WeakValueDictionary
from itertools import groupby
from toolz import groupby , keyfilter


class AssetFinder(object):
    """
    An AssetFinder is an interface to a database of Asset metadata written by
    an ``AssetDBWriter``.

    This class provides methods for looking up assets by unique integer id or
    by symbol.  For historical reasons, we refer to these unique ids as 'sids'.

    Parameters
    ----------
    engine : str or SQLAlchemy.engine
        An engine with a connection to the asset database to use, or a string
        that can be parsed by SQLAlchemy as a URI.
    """
    # 按照不同field group by category
    # retrieve sid
    #
    Asset_Types = frozenset(['equity','bond','fund'])

    def __init__(self):
        self._asset_cache = {}
        self._asset_type_cache = {}

    @property
    def exchange_info(self):
        raise NotImplementedError()

    def _cache_assets_type(self,sids,asset_type):
        #cache_type
        _missing = set(sids) -  set(self._asset_type_cache)
        for sid in _missing:
            self._asset_type_cache[sid] = asset_type
        #cache_asset
        missing = set(sids) - set(self._asset_cache)
        for sid in missing:
            asset = asset_type(sid,self.exchange_info)
            self._asset_cache[sid] = asset

    def _retrieve_equity_assets(self):
        """
        Retrieve Equity objects for a list of sids.

        Parameters
        ----------
        sids : iterable[int]

        Returns
        -------
        equities : dict[int -> Equity]

        """


    def _retrieve_bond_assets(self):
        """可转债  --- item['cell']['stock_id'] item['id'] """


    def _retrieve_fund_assets(self):
        """
            retrieve fund etfs
        """

    def retrieve_asset(self,sids):
        """
        Internal function for loading assets from a table.

        This should be the only method of `AssetFinder` that writes Assets into
        self._asset_cache.

        Parameters
        ---------
        sids : iterable of int
            Asset ids to look up.
        asset_tbl : sqlalchemy.Table
            Table from which to query assets.
        asset_type : type
            Type of asset to be constructed.

        Returns
        -------
        assets : dict[int -> Asset]
            Dict mapping requested sids to the retrieved assets.
        """
        assets = []
        missing = []
        for sid in sids:
            try:
                asset = self._asset_cache[sid]
            except KeyError:
                # Look up cache misses by type.
                print('maybe asset:%s has not come to market'%sid)
                missing.append(sid)
            assets.append(asset)
        return assets,missing

    def _supplement_for_cache(self):
        from multiprocessing import Pool
        pool = Pool()
        for asset_type in self.Asset_Types:
            method = '_retrieve_%s_assets'%asset_type
            pool.apply_async(getattr(self,method)())

    def implement_for_cache(self):
        """
            外部实现
        """
        self._supplement_for_cache()

    def retrieve_all(self,_type):
        """
        Retrieve all assets in `sids`.

        Parameters
        ----------
        sids : iterable of int
            Assets to retrieve.
        default_none : bool
            If True, return None for failed lookups.
            If False, raise `SidsNotFound`.

        Returns
        -------
        assets : list[Asset or None]
            A list of the same length as `sids` containing Assets (or Nones)
            corresponding to the requested sids.

        Raises
        ------
        SidsNotFound
            When a requested sid is not found and default_none=False.
        """
        mapping_groups = groupby(lambda x : x.asset_type == _type,self._asset_cache.values())
        return mapping_groups

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
        found = keyfilter(lambda x : x in sids,self._cache_assets_type)
        return found

    def equity_ownership_maps_by_exchange(self,exchange):
        equities = self.retrieve_all('equity')
        lowercase = exchange.lower()

        def select(sid):
            pattern = '~(0|3|6)'
            match = re.match(pattern, sid)
            assert match,('unkown sid :'%sid)
            if lowercase == 'sz' and match.group in ['0','3']:
                return True
            elif lowercase == 'sh' and match.group == '6':
                return True
            return False

        exchange_equities = keyfilter(lambda x : select(x),equities)
        return exchange_equities

    def fuzzy_equity_ownership_by_sector(self,sector_code):
        secotr_prefix = {'CYB':'3','KCB':'688','ZXB':'0','6':'ZB'}
        equities = self.retrieve_all('equity')
        sector_equities = keyfilter(lambda x : x.startswith(secotr_prefix[sector_code]),equities)
        return sector_equities

    def fuzzy_equity_ownership_by_district(self,area):
        """
            基于区域地址找到对应的股票代码
        """
        groups = {}
        for sid,asset in self._asset_cache.items():
            province = self._fuzzify_district(asset.district)
            groups.setdefault(province,[]).append(sid)
        return groups[area]

    def _fuzzify_district(self,district):
        """
            找出省份或者市
        """
        raise NotImplementedError()

    def fuzzy_symbol_ownership_by_broker(self, broker):
        """
            基于主承商找到对应的股票代码
        """
        broker_mappings = {}
        for sid ,asset in self._asset_cache.items():
            broker = asset.brker.split('证券')[0]
            broker_mappings.setdefault(sid,[]).append(broker)
        return broker

    def fuzzy_bond_map_by_guarantor(self,guarantor):
        bond_assets = self.retrieve_all('bond')
        bonds = keyfilter(lambda x : x.guarantor == guarantor,bond_assets)
        return bonds

    def lookup_bond_ownership_by_equity(self,sid):
        """
            基于A股代码找到对应的可转债数据
        """


    def _retrieve_dual(self):
        """ 获取A股与H股同时上市的股票"""


    def lookup_hk_by_equity(self,sid = None):
        """
            基于A股代码找到对应的H股代码
        """


    def fuzzy_equities_ownership_by_connection(self, exchange, flag=1):
        """获取沪港通、深港通股票 , exchange 交易所 ; flag :1 最新的， 0 为历史的已经踢出的"""
        assets = self.ts.to_ts_con(exchange, flag)
        return assets

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
            assets as columns.  The value at `lifetimes.loc[date, asset]` will
            be True iff `asset` existed on `date`.  If `include_start_date` is
            False, then lifetimes.loc[date, asset] will be false when date ==
            asset.start_date.

        See Also
        --------
        numpy.putmask
        zipline.pipeline.engine.SimplePipelineEngine._compute_root_mask
        # 剔除处于退市整理期的股票，一般是30个交易日  --- 收到退市决定后申请复合，最后一步进入退市整理期30个交易日
        """
        _active = partial(self.was_active,
                          sessions=sessions,
                          include=include_start_date)
        active_assets = [asset for sid, asset in self._asset_cache.items()
                         if _active(asset)]
        return active_assets

    @staticmethod
    def was_active(asset, sessions, include):
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
        sdate, edate = sessions
        mask = (asset.first_traded >= sdate if include
                else asset.first_traded > sdate)
        if asset.last_traded:
            mask &= (asset.last_traded > edate)
        return mask

    def can_be_traded(self,dt,sids = None):
        """
            1.剔除停盘的股票
            2.在生存期内的股票
            默认为股票
        """
        if sids:
            assets = [asset  for asset in self.retrieve_asset(sids)
                      if asset._is_active(dt)]
        else:
            assets = [asset  for asset in self.retrieve_all('equity')
                      if asset._is_active(dt)]
        return assets

    supspend_asset_url = 'http://datainterface.eastmoney.com/EM_DataCenter/JS.aspx?type=FD&sty=SRB&st=0&sr=-1&p=1&ps=500&' \
                         'js={"pages":(pc),"data":[(x)]}&mkt=1&fd=%s'