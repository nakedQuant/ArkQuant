# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
import requests
import re
import os
from bs4 import BeautifulSoup
from collections import defaultdict

ONE_HOUR = pd.Timedelta(hours=1)


def _parse_url(url, encoding='gbk', bs=True):
    Header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36(KHTML, like Gecko)'
                      ' Chrome/79.0.3945.130 Safari/537.36'}
    req = requests.get(url, headers=Header, timeout=1)
    req.encoding = encoding
    if bs:
        raw = BeautifulSoup(req.text, features='lxml')
    else:
        raw = req.text
    return raw


def unpack_df_to_component_dict(stack):
    """Returns the set of known tables in the adjustments file in DataFrame
    form.

    Parameters
    ----------
    stack : pd.DataFrame , stack

    Returns
    -------
    dfs : dict{str->DataFrame}
        Dictionary which maps sid name to the corresponding DataFrame
        version of the table, where all date columns have been coerced back
        from int to datetime.
    """
    unpack = defaultdict(pd.DataFrame)
    for index, raw in stack.iterrows():
        unpack[index] = unpack[index].append(raw, ignore_index=True)
    return unpack

# 解析头文件
def parse_content_from_header(header):
    cols = [t.get_text() for t in header.findAll('td', {'width': re.compile('[0-9]+')})]
    raw = [t.get_text() for t in header.findAll('td')]
    # xa0为空格
    raw = [''.join(item.split()) for item in raw]
    # 去除格式
    raw = [re.sub('·', '', item) for item in raw]
    # 调整字段
    raw = [re.sub('\(历史记录\)', '', item) for item in raw]
    raw = [item.replace('万股', '') for item in raw]
    # 结构处理
    num = int(len(raw) / len(cols))
    text = {}
    for count in range(len(cols)):
        idx = count * num
        mid = raw[idx:idx + num]
        text.update({mid[0]: mid[1:]})
    contents = pd.DataFrame.from_dict(text)
    return contents


def last_modified_time(path):
    """
    Get the last modified time of path as a Timestamp.
    """
    return pd.Timestamp(os.path.getmtime(path), unit='s', tz='UTC')


def load_prices_from_csv(filepath, identifier_col, tz='UTC'):
    data = pd.read_csv(filepath, index_col=identifier_col)
    data.index = pd.DatetimeIndex(data.index, tz=tz)
    data.sort_index(inplace=True)
    return data


def load_prices_from_csv_folder(folder, identifier_col, tz='UTC'):
    data = None
    for file in os.listdir(folder):
        if '.csv' not in file:
            continue
        raw = load_prices_from_csv(os.path.join(folder, file),
                                   identifier_col, tz)
        if data is None:
            data = raw
        else:
            data = pd.concat([data, raw], axis=1)
    return data


def has_data_for_dates(series_or_df, first_date, last_date):
    """
    Does `series_or_df` have data on or before first_date and on or after
    last_date?
    """
    dts = series_or_df.index
    if not isinstance(dts, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex, but got %s." % type(dts))
    first, last = dts[[0, -1]]
    return (first <= first_date) and (last >= last_date)


def transfer_to_timestamp(dt):
    if not isinstance(dt, pd.Timestamp):
        try:
            stamp = pd.Timestamp(dt)
        except Exception as e:
            raise TypeError('cannot tranform %r to timestamp due to %s' % (dt, e))
    else:
        stamp = dt
    timestamps = stamp.timestamp()
    return timestamps
