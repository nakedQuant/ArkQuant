import os
import pandas as pd ,requests ,re
from bs4 import BeautifulSoup
from collections import defaultdict

# Mapping from index symbol to appropriate bond data

ONE_HOUR = pd.Timedelta(hours=1)


def _parse_url(url, encoding='gbk', bs=True):
    Header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36(KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'}
    req = requests.get(url, headers=Header, timeout=1)
    # if encoding:
    req.encoding = encoding
    if bs:
        raw = BeautifulSoup(req.text, features='lxml')
    else:
        raw = req.text
    return raw

def unpack_df_to_component_dict(stacked_df):
    """Returns the set of known tables in the adjustments file in DataFrame
    form.

    Parameters
    ----------
    convert_dates : bool, optional
        By default, dates are returned in seconds since EPOCH. If
        convert_dates is True, all ints in date columns will be converted
        to datetimes.

    Returns
    -------
    dfs : dict{str->DataFrame}
        Dictionary which maps sid name to the corresponding DataFrame
        version of the table, where all date columns have been coerced back
        from int to datetime.
    """
    unpack = defaultdict(pd.DataFrame)
    for index, raw in stacked_df.iterrows():
        unpack[index] = unpack[index].append(raw)
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
