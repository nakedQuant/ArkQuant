# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd, numpy as np, sqlalchemy as sa, json
from sqlalchemy import MetaData, select
from gateWay.driver import engine
from gateWay.driver.bar_reader import AssetSessionReader
from gateWay.driver.tools import _parse_url
from _calendar.trading_calendar import calendar


class Asset(object):
    """
    Base class for entities that can be owned by a trading algorithm.

    Attributes
    ----------
    sid : str
        Persistent unique identifier assigned to the asset.
    engine : str
        sqlalchemy engine
    """
    __slots__ = ['sid']

    reader = AssetSessionReader()

    def __init__(self, sid):
        self.sid = sid
        self._retrieve_asset_mappings()
        self._supplementary_for_asset()

    def _retrieve_asset_mappings(self):
        table = self.metadata.tables['asset_router']
        ins = select([table.c.asset_type, table.c.asset_name, table.c.first_traded,
                      table.c.last_traded, table.c.country_code])
        ins = ins.where(table.c.sid == self.sid)
        rp = self.engine.execute(ins)
        assets = pd.DataFrame(rp.fetchall(), columns=['asset_type', 'asset_name', 'first_traded',
                                                      'last_traded', 'country_code', 'status'])
        for k, v in assets.iloc[0, :].items():
            self.__setattr__(k, v)

    def _supplementary_for_asset(self):
        raise NotImplementedError()

    @property
    def trading_calendar(self):
        return calendar

    @property
    def price_multiplier(self):
        return 1.0

    @property
    def engine(self):
        return engine

    @property
    def metadata(self):
        return MetaData(bind=self.engine)

    @property
    def tick_size(self):
        return 100

    @property
    def increment(self):
        return self.tick_size

    @property
    # 日内交易日
    def is_interday(self):
        return False

    def __setattr__(self, key, value):
        raise NotImplementedError()

    def restricted(self, dt):
        raise NotImplementedError()

    def bid_mechanism(self):
        raise NotImplementedError()

    def _is_active(self, session_label):
        """
        Returns whether the asset is alive at the given dt and not suspend on the given dt

        Parameters
        ----------
        session_label: pd.Timestamp
            The desired session label to check. (midnight UTC)

        Returns
        -------
        boolean: whether the asset is alive at the given dt.
        """
        if self.last_traded != 'null':
            active = self.first_traded <= session_label <= self.last_traded
        else:
            active = self.first_trade <= session_label
        return active

    def __repr__(self):
        return '%s(%d)' % (type(self).__name__, self.sid)

    def __reduce__(self):
        """
        Function used by pickle to determine how to serialize/deserialize this
        class.  Should return a tuple whose first element is self.__class__,
        and whose second element is a tuple of all the attributes that should
        be serialized/deserialized during pickling.
        """
        return (self.__class__, (self.sid,
                                 self.asset_name,
                                 self.asset_type,
                                 self.exchange,
                                 self.first_traded,
                                 self.last_traded,
                                 self.status,
                                 self.tick_size,
                                 self.price_multiplier
                                 ))

    def to_dict(self):
        """Convert to a python dict containing all attributes of the asset.

        This is often useful for debugging.

        Returns
        -------
        as_dict : dict
        """
        return {
            'sid': self.sid,
            'asset_name': self.asset_name,
            'first_traded': self.first_traded,
            'last_traded': self.last_traded,
            'status': self.status,
            'exchange': self.exchange,
            'tick_size': self.tick_size,
            'multiplier': self.price_multiplier
        }


class Equity(Asset):
    """
    Asset subclass representing partial ownership of a company, trust, or
    partnership.
    """
    def __init__(self, sid):
        super(Equity, self).__init__(sid)
        self._retrieve_asset_mappings()
        self._supplementary_for_asset()

    def _supplementary_for_asset(self):
        tbl = self.metadata.tables['equity_basics']
        ins = sa.select([tbl.c.dual,
                         tbl.c.broker,
                         tbl.c.district,
                         tbl.c.initial_price]).where(tbl.c.sid == self.sid)
        rp = self.engine.execute(ins)
        raw = pd.DataFrame(rp.fetchall(), columns=['dual',
                                                   'broker',
                                                   'district',
                                                   'initial_price'])
        for k, v in raw.iloc[0, :].to_dict().items():
            self.__setattr__(k, v)

    @property
    def tick_size(self):
        _tick_size = 200 if self.sid.startswith('688') else 100
        return _tick_size

    @property
    def increment(self):
        per = 1 if self.sid.startswith('688') else self.tick_size
        return per

    def __setattr__(self, key, value):
        raise NotImplementedError()

    def restricted(self, dt):

        """
            科创板股票上市后的前5个交易日不设涨跌幅限制，从第六个交易日开始设置20%涨跌幅限制
        """
        end_dt = self.trading_calendar.dt_window_size(dt, self._restricted_window)

        if self.first_traded == dt:
            _limit = np.inf if self.sid.startwith('688') else 0.44
        elif self.first_traded <= end_dt:
            _limit = np.inf if self.sid.startwith('688') else 0.1
        else:
            _limit = 0.2 if self.sid.startwith('688') else 0.1
        return _limit

    @property
    def bid_mechanism(self):
        """在临时停牌阶段，投资者可以继续申报也可以撤销申报，并且申报价格不受2%的报价限制。
            复牌时，对已经接受的申报实行集合竞价撮合交易，申报价格最小变动单位为0.01"""
        bid_mechanism = 0.02 if self.sid.startwith('688') else None
        return bid_mechanism

    def is_active(self, session_label):
        # between first_traded and last_traded ; is tradeable on session label
        active = self._is_active(session_label)
        data = self.reader.load_raw_arrays([session_label, session_label], self.sid, ['close'])
        active &= (True if data else False)
        return active

    @staticmethod
    def suspend(dt):
        """
            获取时间dt --- 2020-07-13停盘信息
        """
        supspend_url = 'http://datainterface.eastmoney.com/EM_DataCenter/JS.aspx?type=FD&sty=SRB&st=0&sr=-1&p=1&ps=50&'\
                       'js={"pages":(pc),"data":[(x)]}&mkt=1&fd=%s' % dt
        text = _parse_url(supspend_url, bs=False, encoding=None)
        text = json.loads(text)
        return text['data']

    def is_specialized(self, dt):
        """
            equity is special treatment on dt
        :param dt: str e.g. %Y-%m-%d
        :return: bool
        """
        raise NotImplementedError


class Convertible(Asset):
    """
       我国《上市公司证券发行管理办法》规定，可转换公司债券的期限最短为1年，最长为6年，自发行结束之日起6个月方可转换为公司股票
       回售条款 --- 最后两年
       1.强制赎回 --- 股票在任何连续三十个交易日中至少十五个交易日的收盘价格不低于当期转股价格的125%(含 125%)
       2.回售 --- 公司股票在最后两个计息年度任何连续三十个交易日的收盘价格低于当期转股价格的70%时
       3. first_traded --- 可转摘转股日期
       限制条件:
       1.可转换公司债券流通面bai值少于3000万元时，交易所立即公告并在三个交易日后停止交易
       2.可转换公司债券转换期结束前的10个交易日停止交易
       3.中国证监会和交易所认为必须停止交易
    """
    def __init__(self, bond_id):
        super(Convertible, self).__init__(bond_id)
        self._retrieve_asset_mappings()
        self._supplementary_for_asset()

    def _supplementary_for_asset(self):
        tbl = self.metadata.tables['convertible_basics']
        ins = sa.select([tbl.c.swap_code,
                         tbl.c.put_price,
                         tbl.c.redeem_price,
                         tbl.c.convert_price,
                         tbl.c.convert_dt,
                         tbl.c.put_convert_price,
                         tbl.c.guarantor]).\
            where(tbl.c.sid == self.sid)
        rp = self.engine.execute(ins)
        df = pd.DataFrame(rp.fetchall(), columns=['swap_code',
                                                  'put_price',
                                                  'put_price',
                                                  'redeem_price',
                                                  'convert_price',
                                                  'convert_dt',
                                                  'put_convert_price',
                                                  'guarantor'])
        for k, v in df.iloc[0, :].to_dict():
            self.__setattr__(k, v)

    @property
    def is_interday(self):
        return True

    @property
    def bid_mechanism(self):
        return None

    def __setattr__(self, key, value):
        raise NotImplementedError()

    def restricted(self, dt):
        return None

    def is_active(self, dt):
        active = self._is_active(dt)
        return active


class Fund(Asset):
    """
    ETF --- exchange trade fund
    目前不是所有的ETF都是t+0的，只有跨境ETF、债券ETF、黄金ETF、货币ETF实行的是t+0，境内A股ETF暂不支持t+0
    10%
    """
    def __init__(self, fund_id):
        super(Fund, self).__init__(fund_id)
        self._retrieve_asset_mappings()

    def _supplementary_for_asset(self):
        raise NotImplementedError()

    @property
    def bid_mechanism(self):
        return None

    def __setattr__(self, key, value):
        raise NotImplementedError()

    def restricted(self, dt):
        return 0.1

    def is_active(self, session_label):
        active = self._is_active(session_label)
        return active


__all__ = [Equity, Convertible, Fund]
