# -*- coding:utf-8 -*-

from abc import ABC,abstractmethod
from functools import reduce
import operator,pandas as pd,numpy as np , sqlalchemy as sa,json
from sqlalchemy import MetaData,select

from driver.reconstruct.bar_reader import AssetSessionReader
from gateWay.driver.tools import  _parse_url


class Restrictions(ABC):
    """
    Abstract restricted list interface, representing a set of assets that an
    algorithm is restricted from trading.
    """

    @abstractmethod
    def is_restricted(self, assets, dt):
        """
        Is the asset restricted (RestrictionStates.FROZEN) on the given dt?

        Parameters
        ----------
        asset : Asset of iterable of Assets
            The asset(s) for which we are querying a restriction
        dt : pd.Timestamp
            The timestamp of the restriction query

        Returns
        -------
        is_restricted : bool or pd.Series[bool] indexed by asset
            Is the asset or assets restricted on this dt?

        """
        raise NotImplementedError('is_restricted')

    def __or__(self, other_restriction):
        """Base implementation for combining two restrictions.
        """
        # If the right side is a _UnionRestrictions, defers to the
        # _UnionRestrictions implementation of `|`, which intelligently
        # flattens restricted lists
        # 调用 _UnionRestrictions 的__or__
        if isinstance(other_restriction, _UnionRestrictions):
            return other_restriction | self
        return _UnionRestrictions([self, other_restriction])


class _UnionRestrictions(Restrictions):
    """
    A union of a number of sub restrictions.

    Parameters
    ----------
    sub_restrictions : iterable of Restrictions (but not _UnionRestrictions)
        The Restrictions to be added together

    Notes
    -----
    - Consumers should not construct instances of this class directly, but
      instead use the `|` operator to combine restrictions
    """

    def __new__(cls, sub_restrictions):
        # Filter out NoRestrictions and deal with resulting cases involving
        # one or zero sub_restrictions
        sub_restrictions = [
            r for r in sub_restrictions if not isinstance(r, NoRestrictions)
        ]
        if len(sub_restrictions) == 0:
            return NoRestrictions()
        elif len(sub_restrictions) == 1:
            return sub_restrictions[0]

        new_instance = super(_UnionRestrictions, cls).__new__(cls)
        new_instance.sub_restrictions = sub_restrictions
        return new_instance

    def __or__(self, other_restriction):
        """
        Overrides the base implementation for combining two restrictions, of
        which the left side is a _UnionRestrictions.
        """
        # Flatten the underlying sub restrictions of _UnionRestrictions
        if isinstance(other_restriction, _UnionRestrictions):
            new_sub_restrictions = \
                self.sub_restrictions + other_restriction.sub_restrictions
        else:
            new_sub_restrictions = self.sub_restrictions + [other_restriction]

        return _UnionRestrictions(new_sub_restrictions)

    def is_restricted(self, assets, dt):
        if isinstance(assets, Asset):
            return any(
                r.is_restricted(assets, dt) for r in self.sub_restrictions
            )

        return reduce(
            operator.or_,
            (r.is_restricted(assets, dt) for r in self.sub_restrictions)
        )


class NoRestrictions(Restrictions):
    """
    A no-op restrictions that contains no restrictions.
    """
    def is_restricted(self, assets, dt):
        if isinstance(assets, Asset):
            return False
        return pd.Series(index=pd.Index(assets), data=False)


class StaticRestrictions(Restrictions):
    """
    Static restrictions stored in memory that are constant regardless of dt
    for each asset.

    Parameters
    ----------
    restricted_list : iterable of assets
        The assets to be restricted
    """

    def __init__(self, restricted_list):
        self._restricted_set = frozenset(restricted_list)

    def is_restricted(self, assets, dt):
        """
        An asset is restricted for all dts if it is in the static list.
        """
        if isinstance(assets, Asset):
            return assets in self._restricted_set
        return pd.Series(
            index=pd.Index(assets),
            # list 内置的__contains__ 方法
            data=vectorized_is_element(assets, self._restricted_set)
        )


class SecurityListRestrictions(Restrictions):
    """
    Restrictions based on a security list.

    Parameters
    ----------
    restrictions : zipline.utils.security_list.SecurityList
        The restrictions defined by a SecurityList
    """

    def __init__(self, security_list_by_dt):
        self.current_securities = security_list_by_dt.current_securities

    def is_restricted(self, assets, dt):
        securities_in_list = self.current_securities(dt)
        if isinstance(assets, Asset):
            return assets in securities_in_list
        return pd.Series(
            index=pd.Index(assets),
            data=vectorized_is_element(assets, securities_in_list)
        )


class RestrictedBid(Restrictions):
    """
    biding Restrictions based on a security
    """
    def __init__(self):
        self._name = 'bid'

    def is_restricted(self, asset):
        """在临时停牌阶段，投资者可以继续申报也可以撤销申报，并且申报价格不受2%的报价限制。
            复牌时，对已经接受的申报实行集合竞价撮合交易，申报价格最小变动单位为0.01"""
        sid = asset.sid
        bid_limit = 0.02 if sid.startwith('688') else None
        return bid_limit


class RestrictedPrice(Restrictions):
    """
    Pct Restrictions.
    """
    def __init__(self,
                 trading_calendar,
                 length = 5):
        self._calendar = trading_calendar
        self._restricted_window = length
        self._name = 'price'

    def is_restricted(self, asset, dt):
        """
            科创板股票上市后的前5个交易日不设涨跌幅限制，从第六个交易日开始设置20%涨跌幅限制
        """
        sid = asset.sid
        end_dt = self._calendar._roll_forward(dt,self._restricted_window)
        first_traded = asset.first_traded
        if first_traded == dt :
            _limit = np.inf if sid.startwith('688') else 0.44
        elif first_traded <= end_dt:
            _limit = np.inf if self.sid.startwith('688') else 0.1
        else:
            _limit = 0.2 if self.sid.startwith('688') else 0.1
        return _limit


class TemporaryRestriction(object):
    """
        前5个交易日,科创板科创板还设置了临时停牌制度，当盘中股价较开盘价上涨或下跌幅度首次达到30%、60%时，都分别进行一次临时停牌
        单次盘中临时停牌的持续时间为10分钟。每个交易日单涨跌方向只能触发两次临时停牌，最多可以触发四次共计40分钟临时停牌。
        如果跨越14:57则复盘
    """
    def is_restricted(self,asset,dt):
        raise NotImplementedError()


class AfterRestriction(object):
    """
        科创板盘后固定价格交易 15:00 --- 15:30
        若收盘价高于买入申报指令，则申报无效；若收盘价低于卖出申报指令同样无效
        原则 --- 以收盘价为成交价，按照时间优先的原则进行逐笔连续撮合
    """
    def is_restricted(self,asset,dt):
        raise NotImplementedError()


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
    def __init__(self,
                 sid,
                 engine):
        self.sid = sid
        self.engine = engine
        self._retrieve_asset_mappings()
        self._supplementary_for_asset()

    def _retrieve_asset_mappings(self):
        table = self.metadata.tables['asset_router']
        ins = select([table.c.asset_type,table.c.asset_name,table.c.first_traded,
                      table.c.last_traded,table.c.country_code])
        ins = ins.where(table.c.sid == self.sid)
        rp = self.engine.execute(ins)
        assets = pd.DataFrame(rp.fetchall(),columns = ['asset_type','asset_name','first_traded',
                                                       'last_traded','country_code'])
        for k,v in assets.iloc[0,:].items():
            self.__setattr__(k,v)

    @property
    def metadata(self):
        return MetaData(bind = self.engine)

    @property
    def tick_size(self):
        return 100

    @property
    def increment(self):
        return self.tick_size

    def _supplementary_for_asset(self):
        raise NotImplementedError()

    @property
    def intraday(self):
        return False

    def restricted_for_price(self,dt):
        raise NotImplementedError()

    def restricted_for_bid(self):
        raise NotImplementedError()

    def _is_alive(self, session_label):
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
        if self.last_traded:
            active = self.first_traded <= session_label <= self.last_traded
        else:
            active = self.first_trade <= session_label
        return active

    def __repr__(self):
        if self.symbol:
            return '%s(%d [%s])' % (type(self).__name__, self.sid, self.symbol)
        else:
            return '%s(%d)' % (type(self).__name__, self.sid)

    def __reduce__(self):
        """
        Function used by pickle to determine how to serialize/deserialize this
        class.  Should return a tuple whose first element is self.__class__,
        and whose second element is a tuple of all the attributes that should
        be serialized/deserialized during pickling.
        """
        return (self.__class__, (self.sid,
                                 self.exchange_info,
                                 self.symbol,
                                 self.asset_name,
                                 self.start_date,
                                 self.end_date,
                                 self.first_traded,
                                 self.auto_close_date,
                                 self.tick_size,
                                 self.price_multiplier))

    def to_dict(self):
        """Convert to a python dict containing all attributes of the asset.

        This is often useful for debugging.

        Returns
        -------
        as_dict : dict
        """
        return {
            'sid': self.sid,
            'symbol': self.symbol,
            'asset_name': self.asset_name,
            'first_traded': self.first_traded,
            'last_traded': self.last_traded,
            'exchange': self.exchange,
            'tick_size': self.tick_size,
            'multiplier': self.price_multiplier,
        }


class Equity(Asset):
    """
    Asset subclass representing partial ownership of a company, trust, or
    partnership.
    """
    _name = 'equity'

    def __init__(self,
                 sid,
                 engine):
        super(Equity,self).__init__(sid,engine)
        self._reader = AssetSessionReader()
        self._proxy_restrictions = { r._name : r
                                     for r in [RestrictedBid,RestrictedPrice]}
        self._retrieve_asset_mappings()
        self._supplementary_for_asset()

    @property
    def tick_size(self):
        _tick_size = 200 if self.sid.startswith('688') else 100
        return _tick_size

    @property
    def increment(self):
        incre = 1 if self.sid.startswith('688') else self.tick_size
        return incre

    def _supplementary_for_asset(self):
        tbl = self.metadata.tables['equity_supplementary']
        ins = sa.select([tbl.c.dual,
                         tbl.c.sector_canonical,
                         tbl.c.broker,
                         tbl.c.district,
                         tbl.c.initial_price]).where(tbl.c.sid == self.sid)
        rp = self.engine.execute(ins)
        raw = pd.DataFrame(rp.fetchall(),columns = ['dual',
                                                    'sector_canonical',
                                                    'broker',
                                                    'district',
                                                    'initial_price'])
        for k ,v in raw.iloc[0,:].to_dict().items():
            self.setattr(k,v)

    def restricted_for_price(self,dt):
        mechanism = self._proxy_restrictions['price'].is_restricted(self,dt)
        return mechanism

    def restricted_for_bid(self):
        _mechanism = self._proxy_restrictions['bid'].is_restricted(self)
        return _mechanism

    def is_alive(self,session_label):
        active = self._is_alive(session_label)
        #是否停盘
        data = self._reader.load_raw_arrays(session_label,0,'close',self.sid)
        active &= (True if data else False)
        return active

    def suspend(self,dt):
        """
            获取时间dt --- 2020-07-13停盘信息
        """
        supspend_url = 'http://datainterface.eastmoney.com/EM_DataCenter/JS.aspx?type=FD&sty=SRB&st=0&sr=-1&p=1&ps=50&' \
                       'js={"pages":(pc),"data":[(x)]}&mkt=1&fd=%s'%dt
        text = _parse_url(supspend_url, bs=False, encoding=None)
        text = json.loads(text)
        return text['data']


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
    _name = 'convertible'

    def __init__(self,
                 bond_id,
                 engine):
        super(Convertible,self)._init__(bond_id,engine)
        self._retrieve_asset_mappings()
        self._supplementary_for_asset()

    def _supplementary_for_asset(self):
        tbl = self.metadata.tables['convertible_supplementary']
        ins = sa.select([tbl.c.swap_code,
                         tbl.c.put_price,
                         tbl.c.redeem_price,
                         tbl.c.convert_price,
                         tbl.c.convert_dt,
                         tbl.c.put_convert_price,
                         tbl.c.guarantor]).\
            where(tbl.c.sid == self.sid)
        rp = self.engine.execute(ins)
        df = pd.DataFrame(rp.fetchall(),columns = [ 'swap_code',
                                                       'put_price',
                                                       'put_price',
                                                       'redeem_price',
                                                       'convert_price',
                                                       'convert_dt',
                                                       'put_convert_price',
                                                       'guarantor'])
        for k,v in df.iloc[0,:].to_dict():
            setattr(k,v)

    @property
    def intraday(self):
        return True

    def restricted_for_price(self,dt):
        return None

    def restricted_for_bid(self):
        return None

    def is_alive(self,dt):
        active = self._is_alive(dt)
        return active


class Fund(Asset):
    """
    ETF --- exchange trade fund
    目前不是所有的ETF都是t+0的，只有跨境ETF、债券ETF、黄金ETF、货币ETF实行的是t+0，境内A股ETF暂不支持t+0
    10%
    """
    _name = 'fund'

    def __init__(self,
                 fund_id,
                 engine):
        super(Fund,self).__init__(fund_id,engine)
        self._retrieve_asset_mappings()
        self._supplementary_for_asset()

    def _supplementary_for_asset(self):
        pass

    def restricted_for_price(self,dt):
        return 0.1

    def restricted_for_price(self):
        return None

    def is_alive(self,session_label):
        active = self._is_active(session_label)
        return active


__all__ = [Equity,Convertible,Fund]