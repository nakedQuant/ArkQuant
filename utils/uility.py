# -*- coding: utf-8  -*-

import os,argparse,numpy as np,pandas as pd,pytz

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


