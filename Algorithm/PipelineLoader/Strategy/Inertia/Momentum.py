# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from GateWay import Event,GateReq,Quandle
from Algorithm.Feature.Technic import DPower

quandle = Quandle()

class Momentum:
    """
        动量策略 --- 基于价格、成交量 -- DPower 度量动量
        参数 ： 最高价格、最低价、收盘价、成交量，时间窗口
        逻辑：
            1、计算固定区间内的所有股票的累计动量
            2、动能最高点与价格最高点的时间偏差，一般来讲动能高点先出现
            3、计算2中时间偏差的收益
        注意点：属于趋势范畴 -- 惯性特质 ，不适用于反弹
        select assets which has events occurred (investor interveiw | asset reconstruction |  share transfer |
        equity private placement) --- 事件（比如举牌、增持、减持、股权转让、重组）
        逻辑：
        1、基于实战的检验，重要上市公告的影响产生具有一个长期的效应，短期的收到投资者热捧，由于短视效应以及获利回吐的影响，但是由于热点性质以及🎧第三方机构的
           盈利需求的存在，股价继续上行，但是前期获利以及观望的投资者由于锚定效应，不敢买入，但是随着价格上升，内心的盈利期望超过恐惧买入，但是机构获利盘退出，
           周而复始这个过程，（螺旋式上升，波浪理论）
        2、监控上述股票回调之后，一旦出现突破前期高点，买入
        3、监控事件的长期影响（心理上投资偏差）
    """
    _n_fields = ['close','high','volume']

    def __init__(self,window):
        self.window = window

    def _load_raw(self,dt,asset):
        event =Event(dt,asset)
        req = GateReq(event,self._n_fields,self.window)
        raw = quandle.query_ashare_kline(req)
        return raw

    def _measure_shift(self,momentum,close):
        loc1 = momentum.idxmax()
        loc2 = (close / close.iloc[0]).idxmax()
        shift = loc2 - loc1
        shift_ret = close[loc2] / close[loc1] -1
        return shift,shift_ret

    def calculate_power(self,dt,asset):
        raw = self._load_raw(dt,asset)
        momentum = DPower.calc_feature(raw)
        delta,delta_ret = self._measure_shift(momentum,raw['close'])
        return {asset:{'delta':delta,'ret':delta_ret}}

    def _analyse(self):
        """
            分析全A股票进行统计分析，火种取栗 --- 在动量到达高点，介入等到动量下降一定到比例的阈值，卖出 --- 由于时间差
        """
        pass

    def run(self,dt):
        assets = quandle.query_basics()
        output_dict = {}
        for asset in assets:
            output_dict.update(self.calculate_power(dt,asset))
        self._analyse(output_dict)


if __name__ == '__main__':

    momenum = Momentum(10)
