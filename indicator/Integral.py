# -*- coding : utf-8 -*-
import numpy as np,pandas as pd

from gateWay import Event,GateReq,Quandle
from test.algorithm import BaseFeature
from test.algorithm import zoom

class CurveScorer(BaseFeature):
    """
        intergrate under curve
        input : array or series
        output : float --- area under curve
        logic : if the triangle area exceed curve area means positive
    """

    @classmethod
    def calc_feature(cls,feed):
        raw = feed.copy()
        raw_normalize = zoom(raw)
        print('normalize',raw_normalize)
        area = np.trapz(np.array(raw_normalize),np.array(range(1,len(raw_normalize) + 1)))
        print('area',area)
        ratio = area / len(raw_normalize)
        return ratio


if __name__ == '__main__':

    date = '2019-06-01'
    edate = '2020-02-20'
    asset = '000001'
    window = 20

    fields = ['close']
    event = Event(date,asset)
    req = GateReq(event,fields,window)
    quandle = Quandle()

    feed = quandle.query_ashare_kline(req)
    feed.index = pd.DatetimeIndex(feed.index)

    curve = CurveScorer.calc_feature(feed)
    print('curve',curve)
