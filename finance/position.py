#  -*- coding : utf-8 -*-
import numpy as np
from gateWay.driver.adjustment_reader import SQLiteAdjustmentReader
from .commission import Commission


class InnerPosition:
    """The real values of a position.

    This exists to be owned by both a
    :class:`zipline.finance.position.Position` and a
    :class:`zipline.protocol.Position` at the same time without a cycle.
    """
    def __init__(self,
                 asset,
                 amount=0,
                 cost_basis=0.0,
                 last_sale_price=0.0,
                 last_sale_date=None):
        self.asset = asset
        self.amount = amount
        self.cost_basis = cost_basis  # per share
        self.last_sync_price = last_sale_price
        self.last_sync_date = last_sale_date

    def __repr__(self):
        return (
            '%s(asset=%r, amount=%r, cost_basis=%r,'
            ' last_sale_price=%r, last_sale_date=%r)' % (
                type(self).__name__,
                self.asset,
                self.amount,
                self.cost_basis,
                self.last_sync_price,
                self.last_sync_date,
            )
        )

# 仓位 protocol
class Position(object):
    """
        a position held by algorithm
    """
    __slots__ = ('_underlying_position')

    def __init__(self,underlying_position):
        object.__setattr__(self,'_underlying_position', underlying_position)

    def __getattr__(self, attr):
        return getattr(self._underlying_position,attr)

    def __setattr__(self, key, value):
        raise AttributeError('cannot mutate position objects')

    @property
    def sid(self):
        return self.asset.sid

    def __repr__(self):
        return 'position(%r)'%{
            k:getattr(self,k)
            for k in (
                'asset',
                'amount',
                'cost_basis',
                'last_sale_price',
                'laost_sale_date'
            )
        }


class Position(object):

    __slots__ = ['inner_position','protocol_position']

    def __init__(self,
                 asset,
                 amount = 0,
                 cost_basis = 0.0,
                 last_sync_price = 0.0,
                 last_sync_date = None,
                 multiplier = 3):

        inner = InnerPosition(
                asset = asset,
                amount = amount,
                cost_basis = cost_basis,
                last_sync_price = last_sync_price,
                last_sync_date = last_sync_date,
        )
        object.__setattr__(self,'inner_position',inner)
        object.__setattr__(self,'protocol_position',Position(inner))
        self.commission = Commission(multiplier)
        self._closed = False

    @property
    def adjust_reader(self):
        reader = SQLiteAdjustmentReader()
        return reader

    def __getattr__(self, item):
        return getattr(self.inner_position,item)

    def __setattr__(self, key, value):
        setattr(self.inner_position,key,value)

    def _calculate_earning_ratio_from_sqlite(self,dt):
        """
            股权登记日，股权除息日（为股权登记日下一个交易日）
            但是红股的到账时间不一致（制度是固定的）
            根据上海证券交易规则，对投资者享受的红股和股息实行自动划拨到账。股权（息）登记日为R日，除权（息）基准日为R+1日，
            投资者的红股在R+1日自动到账，并可进行交易，股息在R+2日自动到帐，
            其中对于分红的时间存在差异

            根据深圳证券交易所交易规则，投资者的红股在R+3日自动到账，并可进行交易，股息在R+5日自动到账，

            持股超过1年：税负5%;持股1个月至1年：税负10%;持股1个月以内：税负20%新政实施后，上市公司会先按照5%的最低税率代缴红利税
        """
        divdend = self.adjust_reader.load_specific_divdends_for_sid(
                                                                    self.sid,
                                                                    dt
                                                                    )
        amount_ratio = (divdend['sid_bonus'] +divdend['sid_transfer']) / 10
        cash_ratio = divdend['bonus'] / 10

        return amount_ratio,cash_ratio

    def handle_split(self,date):
        """
            update the postion by the split ratio and return the fractional share that will be converted into cash (除权）
            零股转为现金 ,重新计算成本,
            散股 -- 转为现金
        """
        earning_amount_ratio,earning_cash_ratio = self._calculate_earning_ratio_from_sqlite(date)
        adjust_share_count = self.amount( 1 + earning_amount_ratio)
        adjust_cost_basics = round(self.cost_basis / earning_amount_ratio,2)
        scatter_cash = (adjust_share_count - np.floor(adjust_share_count)) *  adjust_cost_basics
        left_cash = self.amount * earning_cash_ratio + scatter_cash
        self.cost_basis = adjust_share_count
        self.amount = np.floor(adjust_share_count)
        return left_cash

    def update(self,txn):
        """
            原始 --- 300股 价格100
            交易正 --- 100股 价格120 成本 （300 * 100 + 100 *120 ） / （300+100）
            交易负 --- 100股 价格90  成本
            交易负 --- 300股 价格120 成本 300 * 120 * fee
        """
        if self.asset == txn.asset:
            raise Exception('transaction asset must same with position asset')
        txn_cost = self.commission.calculate(txn)
        total_amount = txn.amount + self.amount
        if total_amount < 0 :
            raise Exception('put action is not allowed in china')
        else:
            total_cost = txn.amount * txn.price + self.amount * self.cost_basis
            try:
                self.cost_basis = total_cost / total_amount
                self.amount = total_amount
                self._adjust_cost_basis_for_commission(txn_cost)
                txn_cash = txn.amount * txn.price
            except ZeroDivisionError :
                """ 仓位结清 , 当持仓为0此时cost_basis为交易成本需要剔除 , _closed为True"""
                self.cost_basis = txn.price - self.cost_basis - txn / txn.amount
                self._closed = True
                txn_cash = txn.amount * txn.price + txn_cost
        return txn_cash

    def _adjust_cost_basis_for_commission(self,txn_cost):
        prev_cost = self.amount * self.cost_basis
        new_cost = prev_cost + txn_cost
        self.cost_basis = new_cost / self.amount

    def __repr__(self):
        template = "asset :{asset} , amount:{amount},cost_basis:{cost_basis}"
        return template.format(
            asset = self.asset,
            amount = self.amount,
            cost_basis = self.cost_basis
        )

    def to_dict(self):
        """
            create a dict representing the state of this position
        :return:
        """
        return {
            'sid':self.asset.sid,
            'amount':self.amount,
            'origin':self.asset_type.source,
            'cost_basis':self.cost_basis,
            'last_sale_price':self.last_sale_price
        }