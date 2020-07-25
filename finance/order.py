# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from enum import Enum
from abc import ABC , abstractmethod
import uuid
import math

import collections
from copy import copy
import datetime
import itertools


class StyleType(Enum):
    """
        Market Price (市价单）
    """
    LMT = 'lmt'
    BOC = 'boc'
    BOP = 'bop'
    ITC = 'itc'
    B5TC = 'b5tc'
    B5TL = 'b5tl'
    FOK =  'fok'
    FAK =  'fak'


class Order(ABC):

    def make_id(self):
        return  uuid.uuid4().hex()

    @property
    def open_amount(self):
        return self.amount - self.filled

    @property
    def sid(self):
        # For backwards compatibility because we pass this object to
        # custom slippage models.
        return self.asset.sid

    @property
    def status(self):
        self._status = OrderStatus.OPEN

    @status.setter
    def status(self,status):
        self._status = status

    def to_dict(self):
        dct = {name : getattr(self.name)
               for name in self.__slots__}
        return dct

    def __repr__(self):
        """
        String representation for this object.
        """
        return "Order(%s)" % self.to_dict().__repr__()

    def __getstate__(self):
        """ pickle -- __getstate__ , __setstate__"""
        return self.__dict__()

    @abstractmethod
    def check_trigger(self,price,dt):
        """
        Given an order and a trade event, return a tuple of
        (stop_reached, limit_reached).
        For market orders, will return (False, False).
        For stop orders, limit_reached will always be False.
        For limit orders, stop_reached will always be False.
        For stop limit orders a Boolean is returned to flag
        that the stop has been reached.

        Orders that have been triggered already (price targets reached),
        the order's current values are returned.
        """
        raise NotImplementedError


class TickerOrder(Order):
    # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    """
        Parameters
        ----------
        asset : AssetEvent
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        dt : str, optional
            The date created order.

        市价单 --- 针对与卖出 --- 被动算法 ，基于时刻去卖出，这样避免被检测到 --- 将大订单拆分多个小订单然后基于时点去按照市价卖出

    """
    __slot__ = ['asset','_created_dt','capital']

    def __init__(self,asset,ticker,capital):
        self.asset = asset
        self._created_dt = ticker
        self.order_capital = capital
        self.direction = math.copysign(1,capital)
        self.filled = 0.0
        self.broker_order_id = self.make_id()
        self.order_type = StyleType.BOC

    def check_trigger(self,dts):
        if dts >= self._created_dt:
            return True
        return False


class RealtimeOrder(Order):
    # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    """
        Parameters
        ----------
        asset : AssetEvent
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        dt : str, optional
            The date created order.

        市价单 --- 针对与卖出 --- 被动算法 ，基于时刻去卖出，这样避免被检测到 --- 将大订单拆分多个小订单然后基于时点去按照市价卖出
        实时订单
    """
    __slot__ = ['asset', 'capital']

    def __init__(self, asset, capital):
        self.asset = asset
        self.order_capital = capital
        self.direction = math.copysign(1, capital)
        self.filled = 0.0
        self.broker_order_id = self.make_id()
        self.order_type = StyleType.BOC

    def check_trigger(self, dts):
        return True


class PriceOrder(Order):
    # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    """
        Parameters
        ----------
        asset : AssetEvent
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        dt : str, optional
            The date created order.

        限价单 --- 执行买入算法， 如果确定标的可以买入，偏向于涨的概率，主动买入而不是被动买入

        买1 价格超过卖1，买方以卖1价成交
    """
    __slot__ = ['asset','amount','lmt']

    def __init__(self,asset,amount,price):
        self.asset = asset
        self.amount = amount
        self.lmt_price = price
        self._created_dt = dt
        self.direction = math.copysign(1,self.amount)
        self.filled = 0.0
        self.broker_order_id = self.make_id()
        self.order_type = StyleType.LMT

    def check_trigger(self,bid):
        if bid <= self.lmt_price:
            return True
        return False


# order  multiplier - 100


class OrderExecutionBit(object):
    '''
    Intended to hold information about order execution. A "bit" does not
    determine if the order has been fully/partially executed, it just holds
    information.

    Member Attributes:

      - dt: datetime (float) execution time
      - size: how much was executed
      - price: execution price
      - closed: how much of the execution closed an existing postion
      - opened: how much of the execution opened a new position
      - openedvalue: market value of the "opened" part
      - closedvalue: market value of the "closed" part
      - closedcomm: commission for the "closed" part
      - openedcomm: commission for the "opened" part

      - value: market value for the entire bit size
      - comm: commission for the entire bit execution
      - pnl: pnl generated by this bit (if something was closed)

      - psize: current open position size
      - pprice: current open position price

    '''

    def __init__(self,
                 dt=None, size=0, price=0.0,
                 closed=0, closedvalue=0.0, closedcomm=0.0,
                 opened=0, openedvalue=0.0, openedcomm=0.0,
                 pnl=0.0,
                 psize=0, pprice=0.0):

        self.dt = dt
        self.size = size
        self.price = price

        self.closed = closed
        self.opened = opened
        self.closedvalue = closedvalue
        self.openedvalue = openedvalue
        self.closedcomm = closedcomm
        self.openedcomm = openedcomm

        self.value = closedvalue + openedvalue
        self.comm = closedcomm + openedcomm
        self.pnl = pnl

        self.psize = psize
        self.pprice = pprice


class OrderData(object):
    '''
    Holds actual order data for Creation and Execution.

    In the case of Creation the request made and in the case of Execution the
    actual outcome.

    Member Attributes:

      - exbits : iterable of OrderExecutionBits for this OrderData

      - dt: datetime (float) creation/execution time
      - size: requested/executed size
      - price: execution price
        Note: if no price is given and no pricelimite is given, the closing
        price at the time or order creation will be used as reference
      - pricelimit: holds pricelimit for StopLimit (which has trigger first)
      - trailamount: absolute price distance in trailing stops
      - trailpercent: percentage price distance in trailing stops

      - value: market value for the entire bit size
      - comm: commission for the entire bit execution
      - pnl: pnl generated by this bit (if something was closed)
      - margin: margin incurred by the Order (if any)

      - psize: current open position size
      - pprice: current open position price

    '''
    # According to the docs, collections.deque is thread-safe with appends at
    # both ends, there will be no pop (nowhere) and therefore to know which the
    # new exbits are two indices are needed. At time of cloning (__copy__) the
    # indices can be updated to match the previous end, and the new end
    # (len(exbits)
    # Example: start 0, 0 -> islice(exbits, 0, 0) -> []
    # One added -> copy -> updated 0, 1 -> islice(exbits, 0, 1) -> [1 elem]
    # Other added -> copy -> updated 1, 2 -> islice(exbits, 1, 2) -> [1 elem]
    # "add" and "__copy__" happen always in the same thread (with all current
    # implementations) and therefore no append will happen during a copy and
    # the len of the exbits can be queried with no concerns about another
    # thread making an append and with no need for a lock

    def __init__(self, dt=None, size=0, price=0.0, pricelimit=0.0, remsize=0,
                 pclose=0.0, trailamount=0.0, trailpercent=0.0):

        self.pclose = pclose
        self.exbits = collections.deque()  # for historical purposes
        self.p1, self.p2 = 0, 0  # indices to pending notifications

        self.dt = dt
        self.size = size
        self.remsize = remsize
        self.price = price
        self.pricelimit = pricelimit
        self.trailamount = trailamount
        self.trailpercent = trailpercent

        if not pricelimit:
            # if no pricelimit is given, use the given price
            self.pricelimit = self.price

        if pricelimit and not price:
            # price must always be set if pricelimit is set ...
            self.price = pricelimit

        self.plimit = pricelimit

        self.value = 0.0
        self.comm = 0.0
        self.margin = None
        self.pnl = 0.0

        self.psize = 0
        self.pprice = 0

    def _getplimit(self):
        return self._plimit

    def _setplimit(self, val):
        self._plimit = val

    plimit = property(_getplimit, _setplimit)

    def __len__(self):
        return len(self.exbits)

    def __getitem__(self, key):
        return self.exbits[key]

    def add(self, dt, size, price,
            closed=0, closedvalue=0.0, closedcomm=0.0,
            opened=0, openedvalue=0.0, openedcomm=0.0,
            pnl=0.0,
            psize=0, pprice=0.0):

        self.addbit(
            OrderExecutionBit(dt, size, price,
                              closed, closedvalue, closedcomm,
                              opened, openedvalue, openedcomm, pnl,
                              psize, pprice))

    def addbit(self, exbit):
        # Stores an ExecutionBit and recalculates own values from ExBit
        self.exbits.append(exbit)

        self.remsize -= exbit.size

        self.dt = exbit.dt
        oldvalue = self.size * self.price
        newvalue = exbit.size * exbit.price
        self.size += exbit.size
        self.price = (oldvalue + newvalue) / self.size
        self.value += exbit.value
        self.comm += exbit.comm
        self.pnl += exbit.pnl
        self.psize = exbit.psize
        self.pprice = exbit.pprice

    def getpending(self):
        return list(self.iterpending())

    def iterpending(self):
        return itertools.islice(self.exbits, self.p1, self.p2)

    def markpending(self):
        # rebuild the indices to mark which exbits are pending in clone
        self.p1, self.p2 = self.p2, len(self.exbits)

    def clone(self):
        obj = copy(self)
        obj.markpending()
        return obj


class OrderBase(with_metaclass(MetaParams, object)):
    params = (
        ('owner', None), ('data', None),
        ('size', None), ('price', None), ('pricelimit', None),
        ('exectype', None), ('valid', None), ('tradeid', 0), ('oco', None),
        ('trailamount', None), ('trailpercent', None),
        ('parent', None), ('transmit', True),
        ('simulated', False),
        # To support historical order evaluation
        ('histnotify', False),
    )

    DAY = datetime.timedelta()  # constant for DAY order identification

    # Time Restrictions for orders
    T_Close, T_Day, T_Date, T_None = range(4)

    # Volume Restrictions for orders
    V_None = range(1)

    (Market, Close, Limit, Stop, StopLimit, StopTrail, StopTrailLimit,
     Historical) = range(8)
    ExecTypes = ['Market', 'Close', 'Limit', 'Stop', 'StopLimit', 'StopTrail',
                 'StopTrailLimit', 'Historical']

    OrdTypes = ['Buy', 'Sell']
    Buy, Sell = range(2)

    Created, Submitted, Accepted, Partial, Completed, \
        Canceled, Expired, Margin, Rejected = range(9)

    Cancelled = Canceled  # alias

    Status = [
        'Created', 'Submitted', 'Accepted', 'Partial', 'Completed',
        'Canceled', 'Expired', 'Margin', 'Rejected',
    ]

    refbasis = itertools.count(1)  # for a unique identifier per order

    def _getplimit(self):
        return self._plimit

    def _setplimit(self, val):
        self._plimit = val

    plimit = property(_getplimit, _setplimit)

    def __getattr__(self, name):
        # Return attr from params if not found in order
        return getattr(self.params, name)

    def __setattribute__(self, name, value):
        if hasattr(self.params, name):
            setattr(self.params, name, value)
        else:
            super(Order, self).__setattribute__(name, value)

    def __str__(self):
        tojoin = list()
        tojoin.append('Ref: {}'.format(self.ref))
        tojoin.append('OrdType: {}'.format(self.ordtype))
        tojoin.append('OrdType: {}'.format(self.ordtypename()))
        tojoin.append('Status: {}'.format(self.status))
        tojoin.append('Status: {}'.format(self.getstatusname()))
        tojoin.append('Size: {}'.format(self.size))
        tojoin.append('Price: {}'.format(self.price))
        tojoin.append('Price Limit: {}'.format(self.pricelimit))
        tojoin.append('TrailAmount: {}'.format(self.trailamount))
        tojoin.append('TrailPercent: {}'.format(self.trailpercent))
        tojoin.append('ExecType: {}'.format(self.exectype))
        tojoin.append('ExecType: {}'.format(self.getordername()))
        tojoin.append('CommInfo: {}'.format(self.comminfo))
        tojoin.append('End of Session: {}'.format(self.dteos))
        tojoin.append('Info: {}'.format(self.info))
        tojoin.append('Broker: {}'.format(self.broker))
        tojoin.append('Alive: {}'.format(self.alive()))

        return '\n'.join(tojoin)

    def __init__(self):
        self.ref = next(self.refbasis)
        self.broker = None
        self.info = AutoOrderedDict()
        self.comminfo = None
        self.triggered = False

        self._active = self.parent is None
        self.status = Order.Created

        self.plimit = self.p.pricelimit  # alias via property

        if self.exectype is None:
            self.exectype = Order.Market

        if not self.isbuy():
            self.size = -self.size

        # Set a reference price if price is not set using
        # the close price
        pclose = self.data.close[0] if not self.simulated else self.price
        if not self.price and not self.pricelimit:
            price = pclose
        else:
            price = self.price

        dcreated = self.data.datetime[0] if not self.p.simulated else 0.0
        self.created = OrderData(dt=dcreated,
                                 size=self.size,
                                 price=price,
                                 pricelimit=self.pricelimit,
                                 pclose=pclose,
                                 trailamount=self.trailamount,
                                 trailpercent=self.trailpercent)

        # Adjust price in case a trailing limit is wished
        if self.exectype in [Order.StopTrail, Order.StopTrailLimit]:
            self._limitoffset = self.created.price - self.created.pricelimit
            price = self.created.price
            self.created.price = float('inf' * self.isbuy() or '-inf')
            self.trailadjust(price)
        else:
            self._limitoffset = 0.0

        self.executed = OrderData(remsize=self.size)
        self.position = 0

        if isinstance(self.valid, datetime.date):
            # comparison will later be done against the raw datetime[0] value
            self.valid = self.data.date2num(self.valid)
        elif isinstance(self.valid, datetime.timedelta):
            # offset with regards to now ... get utcnow + offset
            # when reading with date2num ... it will be automatically localized
            if self.valid == self.DAY:
                valid = datetime.datetime.combine(
                    self.data.datetime.date(), datetime.time(23, 59, 59, 9999))
            else:
                valid = self.data.datetime.datetime() + self.valid

            self.valid = self.data.date2num(valid)

        elif self.valid is not None:
            if not self.valid:  # avoid comparing None and 0
                valid = datetime.datetime.combine(
                    self.data.datetime.date(), datetime.time(23, 59, 59, 9999))
            else:  # assume float
                valid = self.data.datetime[0] + self.valid

        if not self.p.simulated:
            # provisional end-of-session
            # get next session end
            dtime = self.data.datetime.datetime(0)
            session = self.data.p.sessionend
            dteos = dtime.replace(hour=session.hour, minute=session.minute,
                                  second=session.second,
                                  microsecond=session.microsecond)

            if dteos < dtime:
                # eos before current time ... no ... must be at least next day
                dteos += datetime.timedelta(days=1)

            self.dteos = self.data.date2num(dteos)
        else:
            self.dteos = 0.0

    def clone(self):
        # status, triggered and executed are the only moving parts in order
        # status and triggered are covered by copy
        # executed has to be replaced with an intelligent clone of itself
        obj = copy(self)
        obj.executed = self.executed.clone()
        return obj  # status could change in next to completed

    def getstatusname(self, status=None):
        '''Returns the name for a given status or the one of the order'''
        return self.Status[self.status if status is None else status]

    def getordername(self, exectype=None):
        '''Returns the name for a given exectype or the one of the order'''
        return self.ExecTypes[self.exectype if exectype is None else exectype]

    @classmethod
    def ExecType(cls, exectype):
        return getattr(cls, exectype)

    def ordtypename(self, ordtype=None):
        '''Returns the name for a given ordtype or the one of the order'''
        return self.OrdTypes[self.ordtype if ordtype is None else ordtype]

    def active(self):
        return self._active

    def activate(self):
        self._active = True

    def alive(self):
        '''Returns True if the order is in a status in which it can still be
        executed
        '''
        return self.status in [Order.Created, Order.Submitted,
                               Order.Partial, Order.Accepted]

    def addcomminfo(self, comminfo):
        '''Stores a CommInfo scheme associated with the asset'''
        self.comminfo = comminfo

    def addinfo(self, **kwargs):
        '''Add the keys, values of kwargs to the internal info dictionary to
        hold custom information in the order
        '''
        for key, val in iteritems(kwargs):
            self.info[key] = val

    def __eq__(self, other):
        return other is not None and self.ref == other.ref

    def __ne__(self, other):
        return self.ref != other.ref

    def isbuy(self):
        '''Returns True if the order is a Buy order'''
        return self.ordtype == self.Buy

    def issell(self):
        '''Returns True if the order is a Sell order'''
        return self.ordtype == self.Sell

    def setposition(self, position):
        '''Receives the current position for the asset and stotres it'''
        self.position = position

    def submit(self, broker=None):
        '''Marks an order as submitted and stores the broker to which it was
        submitted'''
        self.status = Order.Submitted
        self.broker = broker
        self.plen = len(self.data)

    def accept(self, broker=None):
        '''Marks an order as accepted'''
        self.status = Order.Accepted
        self.broker = broker

    def brokerstatus(self):
        '''Tries to retrieve the status from the broker in which the order is.

        Defaults to last known status if no broker is associated'''
        if self.broker:
            return self.broker.orderstatus(self)

        return self.status

    def reject(self, broker=None):
        '''Marks an order as rejected'''
        if self.status == Order.Rejected:
            return False

        self.status = Order.Rejected
        self.executed.dt = self.data.datetime[0]
        self.broker = broker
        return True

    def cancel(self):
        '''Marks an order as cancelled'''
        self.status = Order.Canceled
        self.executed.dt = self.data.datetime[0]

    def margin(self):
        '''Marks an order as having met a margin call'''
        self.status = Order.Margin
        self.executed.dt = self.data.datetime[0]

    def completed(self):
        '''Marks an order as completely filled'''
        self.status = self.Completed

    def partial(self):
        '''Marks an order as partially filled'''
        self.status = self.Partial

    def execute(self, dt, size, price,
                closed, closedvalue, closedcomm,
                opened, openedvalue, openedcomm,
                margin, pnl,
                psize, pprice):

        '''Receives data execution input and stores it'''
        if not size:
            return

        self.executed.add(dt, size, price,
                          closed, closedvalue, closedcomm,
                          opened, openedvalue, openedcomm,
                          pnl, psize, pprice)

        self.executed.margin = margin

    def expire(self):
        '''Marks an order as expired. Returns True if it worked'''
        self.status = self.Expired
        return True

    def trailadjust(self, price):
        pass  # generic interface


class Order(OrderBase):
    '''
    Class which holds creation/execution data and type of oder.

    The order may have the following status:

      - Submitted: sent to the broker and awaiting confirmation
      - Accepted: accepted by the broker
      - Partial: partially executed
      - Completed: fully exexcuted
      - Canceled/Cancelled: canceled by the user
      - Expired: expired
      - Margin: not enough cash to execute the order.
      - Rejected: Rejected by the broker

        This can happen during order submission (and therefore the order will
        not reach the Accepted status) or before execution with each new bar
        price because cash has been drawn by other sources (future-like
        instruments may have reduced the cash or orders orders may have been
        executed)

    Member Attributes:

      - ref: unique order identifier
      - created: OrderData holding creation data
      - executed: OrderData holding execution data

      - info: custom information passed over method :func:`addinfo`. It is kept
        in the form of an OrderedDict which has been subclassed, so that keys
        can also be specified using '.' notation

    User Methods:

      - isbuy(): returns bool indicating if the order buys
      - issell(): returns bool indicating if the order sells
      - alive(): returns bool if order is in status Partial or Accepted
    '''

    def execute(self, dt, size, price,
                closed, closedvalue, closedcomm,
                opened, openedvalue, openedcomm,
                margin, pnl,
                psize, pprice):

        super(Order, self).execute(dt, size, price,
                                   closed, closedvalue, closedcomm,
                                   opened, openedvalue, openedcomm,
                                   margin, pnl, psize, pprice)

        if self.executed.remsize:
            self.status = Order.Partial
        else:
            self.status = Order.Completed

        # self.comminfo = None

    def expire(self):
        if self.exectype == Order.Market:
            return False  # will be executed yes or yes

        if self.valid and self.data.datetime[0] > self.valid:
            self.status = Order.Expired
            self.executed.dt = self.data.datetime[0]
            return True

        return False

    def trailadjust(self, price):
        if self.trailamount:
            pamount = self.trailamount
        elif self.trailpercent:
            pamount = price * self.trailpercent
        else:
            pamount = 0.0

        # Stop sell is below (-), stop buy is above, move only if needed
        if self.isbuy():
            price += pamount
            if price < self.created.price:
                self.created.price = price
                if self.exectype == Order.StopTrailLimit:
                    self.created.pricelimit = price - self._limitoffset
        else:
            price -= pamount
            if price > self.created.price:
                self.created.price = price
                if self.exectype == Order.StopTrailLimit:
                    # limitoffset is negative when pricelimit was greater
                    # the - allows increasing the price limit if stop increases
                    self.created.pricelimit = price - self._limitoffset


class BuyOrder(Order):
    ordtype = Order.Buy


class StopBuyOrder(BuyOrder):
    pass


class StopLimitBuyOrder(BuyOrder):
    pass


class SellOrder(Order):
    ordtype = Order.Sell


class StopSellOrder(SellOrder):
    pass


class StopLimitSellOrder(SellOrder):
    pass