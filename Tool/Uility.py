# -*- coding: utf-8  -*-

import os,argparse,numpy as np,pandas as pd,pytz
from io import StringIO

def parse_date_str_series(format_str, tz, date_str_series):
    tz_str = str(tz)
    if tz_str == pytz.utc.zone:

        parsed = pd.to_datetime(
            date_str_series.values,
            format=format_str,
            utc=True,
            errors='coerce',
        )
    else:
        parsed = pd.to_datetime(
            date_str_series.values,
            format=format_str,
            errors='coerce',
        ).tz_localize(tz_str).tz_convert('UTC')
    return parsed


def scale():
    """
        zoom self.x.max() / self.tl.max() y_zoom = zoom_factor * self.tl --- scale
        d = float(w.nnz) / (w.shape[0] * w.shape[1])
    """
    pass

def resample():
    """
        kl_rp = pd_resample(kl, '{}D'.format(resample), how='mean')
    """
    pass


def load_to_file():
    output = StringIO()
    output.write('First line.\n')
    contents = output.getvalue()
    output.close()
    fd = StringIO()
    fd.tell()
    fd.seek(0)
    fd.close()


def nan_proc(x):
    np.nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None)


def signature():
    """
        sig = signature(foo)
        str(sig)
        #'(a, *, b:int, **kwargs)'
        str(sig.parameters['b'])
        #'b:int'
        sig.parameters['b'].annotation
        #<class 'int'>
    """
    pass


def display():
    """
        pandas DataFrame表格最大显示行数
        pd.options.display.max_rows = 20
        pandas DataFrame表格最大显示列数
        pd.options.display.max_columns = 20
        pandas精度浮点数显示4位
        pd.options.display.precision = 4
        numpy精度浮点数显示4位，不使用科学计数法
        np.set_printoptions(precision=4, suppress=True)
    """
    pass


def encryt(obj,method = 'md5',hex = False):

    """
        method : md5(), sha1(), sha224(), sha256(), sha384(), and sha512()
        algorithms may be available depending upon the OpenSSL library that Python uses on your platform.
        e.g. : hashlib.sha224("Nobody inspects the spammish repetition").hexdigest()
    """
    import hashlib
    m = hashlib.md5()
    m.update(obj)
    if hex :
        res = m.hexdigest()
    else:
        res = m.digest()
    return res

def extract(_p_dir,file='RomDataBu/df_kl.h5.zip'):
    """
    解压数据
    """
    data_example_zip = os.path.join(_p_dir, file)
    try:
        from zipfile import ZipFile

        zip_h5 = ZipFile(data_example_zip, 'r')
        unzip_dir = os.path.join(_p_dir, 'RomDataBu/')
        for h5 in zip_h5.namelist():
            zip_h5.extract(h5, unzip_dir)
        zip_h5.close()
    except Exception as e:
        print('example env failed! e={}'.format(e))


def commandParse():
    default={'color':'red','user':'guest'}
    #创建参数实例
    parser=argparse.ArgumentParser()
    #添加
    parser.add_argument('-u','--user')
    parser.add_argument('-c','--color')
    #解析参数
    namespace=parser.parse_args()
    command_line_args={k:v for k,v in vars(namespace).items() if v}


