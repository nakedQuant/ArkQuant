# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import struct, pandas as pd
from utils.dt_utilty import normalize_date

BcolzMinuteFields = ['ticker', 'open', 'high', 'low', 'close', 'amount', 'volume']


def retrieve_data_from_tdx(path):
    """解析通达信数据"""
    with open(path, 'rb') as f:
        buf = f.read()
        size = int(len(buf) / 32)
        data = []
        for num in range(size):
            idx = 32 * num
            line = struct.unpack('HhIIIIfii', buf[idx:idx + 32])
            data.append(line)
        frame = pd.DataFrame(data, columns=['dates', 'sub_dates', 'open',
                                            'high', 'low', 'close', 'amount',
                                            'volume', 'appendix'])
        frame = normalize_date(frame)
        print('frame', frame)
        ticker_frame = frame.loc[:, BcolzMinuteFields]
        return ticker_frame


if __name__ == '__main__':

    path = r'E:\tdx\200404\vipdoc\sh\minline\sh000001.01'
    frame = retrieve_data_from_tdx(path)