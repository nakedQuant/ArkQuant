# -*- coding : utf-8 -*-
from collections import defaultdict,OrderedDict
import numpy as np , pandas as pd
from functools import partial

from finance.position import Position
from gateWay.driver.reader import BarReader

bar_reader = BarReader()


class PositionStats(object):

    gross_exposure = None
    gross_value = None
    long_exposure = None
    net_exposure = None
    net_value = None
    short_exposure = None
    longs_count = None
    shorts_count = None
    position_exposure_array = np.array()
    position_exposure_series = pd.series()

    def __new__(cls):
        self = cls()
        es = pd.Series(np.array([],dtype = 'float64'),index = np.array([],dtpe = 'int64'))
        self._underlying_value_array = es.values
        self._underlying_index_array = es.index.values
        return self


def calculate_position_tracker_stats(positions,stats):
    """
        stats ---- PositionStats
        return portfolio value and num of position
    """
    count = 0
    net_exposure = 0

    for outer_position in positions.values():
        position = outer_position.inner_position
        #daily更新价格
        exposure = position.amount * position.last_sale_price
        count += 1
        net_exposure += exposure

    stats.net_exposure = net_exposure
    stats.count = count


class PositionTracker(object):
    """
        持仓变动
        the current state of position held
    """
    def __init__(self):

        self.positions = OrderedDict()
        #根据时间记录关闭的交易
        self.record_closed_position = defaultdict(list)
        #cache the stats until
        self._dirty_stats = True
        self._stats = PositionStats.new()

    def _update_position(self,transaction):
        asset = transaction.asset
        try:
            position = self.positions[asset]
        except KeyError:
            position = self.positions[asset] = Position(asset)
        finally:
            cash_flow = position.update(transaction)
        if position._closed:
            self.record_closed_position[transaction.dt].append(position)
            del self.positions[asset]
        return cash_flow

    def execute_transaction(self,transactions):
        """执行完交易cash变动"""
        aggregate_cash_flow = 0.0
        for txn in transactions:
            aggregate_cash_flow += self._update_position(txn)
        self._dirty_stats = False
        return aggregate_cash_flow

    def handle_splits(self,dt):
        total_left_cash = 0
        for asset,position in self.positions.items():
            left_cash = position.handle_split(dt)
            total_left_cash += left_cash
        return total_left_cash

    def sync_last_date(self,dt):
        for outer_position in self.positions.values():
            inner_position = outer_position.inner_position
            inner_position.last_sync_date = dt

    def sync_last_price(self):
        """update last_sale_price of position"""
        assets =[position.inner_position.asset for position in self.positions]
        dts = [position.inner_position.last_sync_date for position in self.positions]
        if len(set(dts)) >1 :
            raise ValueError('sync all the position date')
        dt = dts[0]
        get_price = partial(bar_reader.load_asset_kline,
                            date = dt,
                            window = 0,
                            fields = 'close',
                            )
        last_sync_prices = get_price(assets = assets)
        for asset,outer_position in self.positions.items():
            inner_position = outer_position.inner_position
            asset = inner_position.asset
            inner_position.last_sync_price = last_sync_prices[asset]

    @property
    def stats(self):
        """基于sync_last_sale_price  --- 计算每天的暴露度也就是porfolio"""
        calculate_position_tracker_stats(self.positions,self._stats)
        return self._stats

    def get_positions(self):
        # protocol
        positions = {}
        for asset, position in self.positions.items():
            # Adds the new position if we didn't have one before, or overwrite
            # one we have currently
            positions[asset] = position.protocol_position

        return positions

    def maybe_create_close_position_transaction(self,txn):
        """强制平仓机制 --- 设立仓位 ，改变脚本运行逻辑"""

        raise NotImplementedError('all action determined by alogrithm')