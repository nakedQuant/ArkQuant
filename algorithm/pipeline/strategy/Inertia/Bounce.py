# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from gateWay import Event,GateReq,Quandle

quandle = Quandle()

class Rebound:
    """
        参数： backPeriod --- 回测度量周期
              withdraw --- 回调比例
              retPeriod --- 预期时间段的收益率
              注意 backPeriod >= retPeriod ,关键点下一个迭代从哪一个时间点开始，时间交叉的比例（如果比例太大，产生的结果过度产生偏差，但是比例
              如果太小，影响样本同时分析不具有连续性
        逻辑：
            1、筛选出backPeriod的区间收益率最高的top1%(剔除上市不满半年，退市日期不足一个月--- 如果一个股票即将退市，将有公告，避免的存活偏差）
            2、计算top1%股票集的回撤幅度
            3、根据回撤幅度进行label，并计算未来的一段时间的收益率
            4、将时间推导retPeriod之后的，重复
            5、将分类结果收集，统计出最佳的回撤幅度
            6、修改backPeriod、returnPeriod进行迭代
        分析：
            1、回撤比例达到什么范围，股价在给定时间内反弹的可能性最大，由于时间越长，变数越大导致历史分析参考的依据性急剧下降
            2、回补缺口，如果短期内缺口回补了，说明向上的概率增大，惯性越大
    """

    def __init__(self,backPeriod,retPeriod):
        self.sdate = '2000-01-01'
        self.edate = '2020-02-10'
        self._back_period = backPeriod
        self._ret_period = retPeriod

    def _filter_assets(self,dt):
        """
            剔除退市时期在dt之前的股票、以及上市不满半年的股票
        """
        pass

    def load_bar(self,dt):
        event = Event(dt)
        req = GateReq(event, ['close'],self._back_period)
        bar = quandle.query_ashare_kline(req)
        raw = bar.pivot(columns='code', values='close')
        raw.sort_index(inplace=True)
        raw.fillna(method='bfill', inplace=True)
        return bar

    def bound(self,dt):
        raw = self.load_bar(dt)
        #计算区间收益
        period_ret = raw.iloc[-1,:] / raw.iloc[0,:] -1
        #定义区间的回撤收益率
        withdraw = (raw.max()  - raw.iloc[-1,:]) / raw.max() - 1
        self._analyse(period_ret,withdraw)

    def _analyse(self):
        """
            分析区间收益 、 回测比例、 持有收益、回补缺口的统计关系
        """
        pass

    def fulfill(self):
        calendar = quandle.query_calendar_session(self.sdate,self.edate)
        for trade_dt in calendar[::self._ret_period]:
            self.bound(trade_dt)


if __name__ == '__main__':

    rebound = Rebound(10,5)