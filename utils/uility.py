# -*- coding: utf-8  -*-

import os,inspect

def getargspec(f):
    full_argspec = inspect.getfullargspec(f)
    return inspect.ArgSpec(
        args = full_argspec.args,
        varargs = full_argspec.varargs,
        keywords = full_argspec.varkw,
        defaults = full_argspec.defaults)

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


def encryt(obj,hex = False):

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


def verify_indices_all_unique(obj):

    axis_names = [
        ('index',),                            # Series
        ('index', 'columns'),                  # DataFrame
        ('items', 'major_axis', 'minor_axis')  # Panel
    ][obj.ndim - 1]  # ndim = 1 should go to entry 0,

    for axis_name, index in zip(axis_names, obj.axes):
        if index.is_unique:
            continue

        raise ValueError(
            "Duplicate entries in {type}.{axis}: {dupes}.".format(
                type=type(obj).__name__,
                axis=axis_name,
                dupes=sorted(index[index.duplicated()]),
            )
        )
    return obj


def validate_keys(dict_, expected, funcname):
    """Validate that a dictionary has an expected set of keys.
    """
    expected = set(expected)
    received = set(dict_)

    missing = expected - received
    if missing:
        raise ValueError(
            "Missing keys in {}:\n"
            "Expected Keys: {}\n"
            "Received Keys: {}".format(
                funcname,
                sorted(expected),
                sorted(received),
            )
        )

    unexpected = received - expected
    if unexpected:
        raise ValueError(
            "Unexpected keys in {}:\n"
            "Expected Keys: {}\n"
            "Received Keys: {}".format(
                funcname,
                sorted(expected),
                sorted(received),
            )
        )