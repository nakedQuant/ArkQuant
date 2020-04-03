# -*- coding : utf-8 -*-

import pandas as pd
from functools import partial

from GateWay import Event,GateReq,Quandle
from Algorithm.Feature import BaseFeature
from Algorithm.Mathmetics.distance_tool import  SignDistance

class Discretizer(BaseFeature):
    """
        encoding algorithm to classify the dataset and calconf ,dense method is not avaivable for classify the stock ;
        ordinal method ---  encoded as an integer value suits for seek the pattern of stock price change
        原理：股票波动剔除首日以及科创板 ，波动范围：-10%至10% ，将其划分为N档，计算符号相关性
        e.g : 2,3,4,6,7,9
              4,6,8,3,10,4
        序列之差 --- 1，1，2，1，2
                    2，2，-5，7，-6
        转化为sign : 1,1,1,1,1
                    1,1,-1,1,-1
        计算相关性 : 序列相乘之和除以序列长度
    """
    _fields = ['close','alla']
    _thre = 0.7
    _bins = 8

    @staticmethod
    def _cal_conf(x,y = None):
        res = SignDistance.calc_feature(x,y)
        return res

    def genSim(self,data):
        conf = data[data > self._thres].sum()/len(data)
        return conf

    def _filter(self,out):
        out_filter = {k:v for k,v in out.items if v >= self._thres}
        out_sorted = sorted(out_filter.items(),key = lambda x :x[1])
        pd_out = pd.DataFrame.from_dict(out_sorted)
        return pd_out

    # calculate one stock pct spread
    def _calc_feature(self,raw,window):
        Category = pd.cut(raw,bins = self._bins ,labels = range(1,self._bins +1))
        out_dict = {}
        for id in range(len(Category) - window):
            slice_ = Category[id:id + window]
            wrapper = partial(self._cal_conf,y = slice_)
            res_windowed = Category.rolling(window = window).apply(wrapper)
            simi = self.genSim(res_windowed)
            out_dict.update({str(slice_):simi})
        output = self._filter(out_dict)
        return output

    @classmethod
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._fields)
        raw = feed.copy()
        quantity = pd.DataFrame()
        instance = cls()
        for code in raw['code']:
            res = instance._calc_feature(raw['close'][code],window)
            quantity = quantity.append(res,ignore_index= True)
        p_value = (pd.isnull(quantity)).sum() / raw['code']
        result = quantity.columns[p_value.idxmax()]
        return {result:p_value.max()}


if __name__ == '__main__':

    date = '2019-06-01'
    edate = '2020-02-20'
    asset = '000001'
    window = 100

    fields = ['close']
    event = Event(date)
    req = GateReq(event,fields,window)
    quandle = Quandle()
    feed = quandle.query_ashare_kline(req)
    feed.index = pd.DatetimeIndex(feed.index)

    discretizer = Discretizer.calc_feature(feed,5)
    print('discretizer',discretizer)