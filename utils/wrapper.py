#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 13:56:19 2019

@author: python
"""
import functools, logging, pdb, time
import warnings
from functools import wraps
from contextlib import contextmanager
import sys,weakref


def _deprecated_getitem_method(name, attrs):
    """Create a deprecated ``__getitem__`` method that tells users to use
    getattr instead.

    Parameters
    ----------
    name : str
        The name of the object in the warning message.
    attrs : iterable[str]
        The set of allowed attributes.

    Returns
    -------
    __getitem__ : callable[any, str]
        The ``__getitem__`` method to put in the class dict.
    """
    attrs = frozenset(attrs)
    msg = (
        "'{name}[{attr!r}]' is deprecated, please use"
        " '{name}.{attr}' instead"
    )

    def __getitem__(self, key):
        """``__getitem__`` is deprecated, please use attribute access instead.
        """
        warnings.warn(msg.format(name=name, attr=key), DeprecationWarning, stacklevel=2)
        if key in attrs:
            return getattr(self, key)
        raise KeyError(key)

    return __getitem__


class Deprecated(object):
    """
    支持装饰类或者方法，在使用类或者方法时警告Deprecated信息
    wraps --- update_wrapper(包含了update_wrapper) ,如果单独函数修饰不加@wraps(f),需要update_wrapper（wrapper_function,f)
    类作为修饰器，必须实现__call__
    ---(__name__, __module__ and __doc__）
    """

    def __init__(self, tip_info=''):
        # 用户自定义警告信息tip_info
        self.tip_info = tip_info

    def __call__(self, obj):
        if isinstance(obj, six.class_types):
            # 针对类装饰
            return self._decorate_class(obj)
        else:
            # 针对方法装饰
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        """实现类装饰警告Deprecated信息"""

        msg = "class {} is deprecated".format(cls.__name__)
        if self.tip_info:
            msg += "; {}".format(self.tip_info)
        # 取出原始init
        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)

        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.__doc__ = self._update_doc(init.__doc__)
        # init成为deprecated_original，必须要使用这个属性名字，在其它地方，如AbuParamBase会寻找原始方法找它
        wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """实现方法装饰警告Deprecated信息"""

        msg = "func {} is deprecated".format(fun.__name__)
        if self.tip_info:
            msg += "; {}".format(self.tip_info)

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        # 更新func及文档信息
        wrapped.__name__ = fun.__name__
        wrapped.__dict__ = fun.__dict__
        wrapped.__doc__ = self._update_doc(fun.__doc__)

        return wrapped

    def _update_doc(self, func_doc):
        """更新文档信息，把原来的文档信息进行合并格式化, 即第一行为deprecated_doc(Deprecated: tip_info)，下一行为原始func_doc"""
        deprecated_doc = "Deprecated"
        if self.tip_info:
            """如果有tip format tip"""
            deprecated_doc = "{}: {}".format(deprecated_doc, self.tip_info)
        if func_doc:
            # 把原来的文档信息进行合并格式化, 即第一行为deprecated_doc，下一行为原始func_doc
            func_doc = "{}\n{}".format(deprecated_doc, func_doc)
        return func_doc


def warnings_filter(func):
    """
        作用范围：函数装饰器 (模块函数或者类函数)
        功能：被装饰的函数上的警告不会打印，忽略
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.simplefilter('ignore')
        ret = func(*args, **kwargs)
        if not ABuEnv.g_ignore_all_warnings:
            # 如果env中的设置不是忽略所有才恢复
            warnings.simplefilter('default')
        return ret

    return wrapper


def singleton(cls):
    """
        作用范围：类装饰器
        功能：被装饰后类变成单例类
    """

    instances = {}

    @functools.wraps(cls)
    def get_instance(*args, **kw):
        if cls not in instances:
            # 不存在实例instances才进行构造
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return get_instance


def params_to_pandas(func):
    """
        函数装饰器：不定参数装饰器，定参数转换使用ABuScalerUtil中的装饰器arr_to_pandas(func)
        将被装饰函数中的参数中所有可以迭代的序列转换为pd.DataFrame或者pd.Series
    """

    @functools.wraps(func)
    def wrapper(*arg, **kwargs):
        # 把arg中的可迭代序列转换为pd.DataFrame或者pd.Series
        arg_list = [arr_to_pandas(param) for param in arg]
        # 把kwargs中的可迭代序列转换为pd.DataFrame或者pd.Series
        arg_dict = {param_key: arr_to_pandas(kwargs[param_key]) for param_key in kwargs}
        return func(*arg_list, **arg_dict)

    return wrapper


def params_to_numpy(func):
    """
        函数装饰器：不定参数装饰器，定参数转换使用ABuScalerUtil中的装饰器arr_to_numpy(func)
        将被装饰函数中的参数中所有可以迭代的序列转换为np.array
    """

    @functools.wraps(func)
    def wrapper(*arg, **kwargs):
        # 把arg中的可迭代序列转换为np.array
        arg_list = [arr_to_numpy(param) for param in arg]
        # 把kwargs中的可迭代序列转换为np.array
        arg_dict = {param_key: arr_to_numpy(kwargs[param_key]) for param_key in kwargs}
        return func(*arg_list, **arg_dict)

    return wrapper


def catch_error(return_val=None, log=True):
    """
    作用范围：函数装饰器 (模块函数或者类函数)
    功能：捕获被装饰的函数中所有异常，即忽略函数中所有的问题，用在函数的执行级别低，且不需要后续处理
    :param return_val: 异常后返回的值，
                eg:
                    class A:
                        @ABuDTUtil.catch_error(return_val=100)
                        def a_func(self):
                            raise ValueError('catch_error')
                            return 100
                    in: A().a_func()
                    out: 100
    :param log: 是否打印错误日志
    """
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.exception(e) if log else logging.debug(e)
                return return_val

        return wrapper

    return decorate


def consume_time(func):
    """
    作用范围：函数装饰器 (模块函数或者类函数)
    功能：简单统计被装饰函数运行时间
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('{} cost {}s'.format(func.__name__, round(end_time - start_time, 3)))
        return result

    return wrapper


def empty_wrapper(func):
    """
    作用范围：函数装饰器 (模块函数或者类函数)
    功能：空装饰器，为fix版本问题使用，或者分逻辑功能实现使用
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# noinspection PyUnusedLocal
def empty_wrapper_with_params(*p_args, **p_kwargs):
    """
    作用范围：函数装饰器 (模块函数或者类函数)
    功能：带参数空装饰器，为fix版本问题使用，或者分逻辑功能实现使用
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorate



def except_debug(func):
    """
    作用范围：函数装饰器 (模块函数或者类函数)
    功能：debug，调试使用，装饰在有问题函数上，发生问题打出问题后，再运行一次函数，可以用s跟踪问题了
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            pdb.set_trace()
            print(e)
            # 再来一遍用s跟踪进去
            return func(*args, **kwargs)

    return wrapper


class LazyFunc(object):
    """
    描述器类：作用在类中需要lazy的对象方法上
    优先级：
    __getattribute__ > __getattr__
    #1 调用属性会触发该功能，属性存在则会返回相应的值；
    #2 如果属性不存在则会抛出异常AttributeError，所以可以自定义异常信息
    #3 存在__getattr__，若有异常出现则会传递给__getattr__用来接收，执行操作
    #描述符discription
    __get__ __set__ __delete__

    lazy property  描述符 __get__ __set__（其中一个方法） __delete__ ，将函数或者方法变成实例的属性（__dict__）

    class TestDes:
        def __get__(self, instance, owner):
            print(instance, owner)
            return 'TestDes:__get__'

    class TestMain:
        des = TestDes()

    if __name__ == '__main__':
        t = TestMain()
        print(t.des)
        print(TestMain.des)
    """

    def __init__(self, func):
        """
        外部使用eg：
            class BuyCallMixin(object):
                @LazyFunc
                def buy_type_str(self):
                    return "call"

                @LazyFunc
                def expect_direction(self):
                    return 1.0
        """
        self.func = func
        self.cache = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
        """描述器__get__，使用weakref.WeakKeyDictionary将以实例化的instance加入缓存"""
        if instance is None:
            return self
        try:
            return self.cache[instance]
        except KeyError:
            ret = self.func(instance)
            self.cache[instance] = ret
            return ret

    def __set__(self, instance, value):
        """描述器__set__，raise AttributeError，即禁止外部set值"""
        raise AttributeError("LazyFunc set value!!!")

    def __delete__(self, instance):
        """描述器___delete__从weakref.WeakKeyDictionary cache中删除instance"""
        del self.cache[instance]


class LazyClsFunc(LazyFunc):
    """
        描述器类：
        作用在类中需要lazy的类方法上，实际上只是使用__get__(owner, owner)
        替换原始__get__(self, instance, owner)
    """

    def __get__(self, instance, owner):
        """描述器__get__，使用__get__(owner, owner)替换原始__get__(self, instance, owner)"""
        return super(LazyClsFunc, self).__get__(owner, owner)


def add_doc(func, doc):
    """Lazy add doc"""
    func.__doc__ = doc


def import_module(name):
    """Lazy impor _module"""
    __import__(name)
    return sys.modules[name]


# valid_check装饰器工作：
def valid_check(func):
    """检测度量的输入是否正常，非正常显示info，正常继续执行被装饰方法"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.valid:
            return func(self, *args, **kwargs)
        else:
            logging.info('metrics input is invalid or zero order gen!')

    return wrapper


# 单例模式
def singleton(cls):

    instance = {}

    def _singleton(*args,**kwargs):
        if cls not in instance:
            instance[cls] = cls(*args,**kwargs)
        return instance[cls]

    return _singleton


def _validate_type(_type=(list,tuple)):
    def decorate(func):
        def wrap(*args):
            res = func(*args)
            if not isinstance(res, _type):
                raise TypeError('can not algorithm type:%s' % _type)
            return res
        return wrap
    return decorate


def deprecated(msg=None, stacklevel=2):
    """
    Used to mark a function as deprecated.

    Parameters
    ----------
    msg : str
        The message to display in the deprecation warning.
    stacklevel : int
        How far up the stack the warning needs to go, before
        showing the relevant calling lines.

    Examples
    --------
    @deprecated(msg='function_a is deprecated! Use function_b instead.')
    def function_a(*args, **kwargs):
    """
    def deprecated_dec(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(
                msg or "Function %s is deprecated." % fn.__name__,
                category=DeprecationWarning,
                stacklevel=stacklevel
            )
            return fn(*args, **kwargs)
        return wrapper
    return deprecated_dec


def require_not_initialized(exception):
    """
    Decorator for API methods that should only be called during or before
    TradingAlgorithm.initialize.  `exception` will be raised if the method is
    called after initialize.

    Examples
    --------
    @require_not_initialized(SomeException("Don't do that!"))
    def method(self):
        # Do stuff that should only be allowed during initialize.
    """
    def decorator(method):
        @wraps(method)
        def wrapped_method(self, *args, **kwargs):
            if self.initialized:
                raise exception
            return method(self, *args, **kwargs)
        return wrapped_method
    return decorator


def require_initialized(exception):
    """
    Decorator for API methods that should only be called after
    TradingAlgorithm.initialize.  `exception` will be raised if the method is
    called before initialize has completed.

    Examples
    --------
    @require_initialized(SomeException("Don't do that!"))
    def method(self):
        # Do stuff that should only be allowed after initialize.
    """
    def decorator(method):
        @wraps(method)
        def wrapped_method(self, *args, **kwargs):
            if not self.initialized:
                raise exception
            return method(self, *args, **kwargs)
        return wrapped_method
    return decorator


def disallowed_in_before_trading_start(exception):
    """
    Decorator for API methods that cannot be called from within
    TradingAlgorithm.before_trading_start.  `exception` will be raised if the
    method is called inside `before_trading_start`.

    Examples
    --------
    @disallowed_in_before_trading_start(SomeException("Don't do that!"))
    def method(self):
        # Do stuff that is not allowed inside before_trading_start.
    """
    def decorator(method):
        @wraps(method)
        def wrapped_method(self, *args, **kwargs):
            if self._in_before_trading_start:
                raise exception
            return method(self, *args, **kwargs)
        return wrapped_method
    return decorator


def _make_unsupported_method(name):
    def method(*args, **kwargs):
        raise NotImplementedError(
            "Method %s is not supported on LabelArrays." % name
        )
    method.__name__ = name
    method.__doc__ = "Unsupported LabelArray Method: %s" % name
    return method


@contextmanager
def ignore_pandas_nan_categorical_warning():
    with warnings.catch_warnings():
        # Pandas >= 0.18 doesn't like null-ish values in categories, but
        # avoiding that requires a broader change to how missing values are
        # handled in pipe, so for now just silence the warning.
        warnings.filterwarnings(
            'ignore',
            category=FutureWarning,
        )
        yield


def remove_na(f):
    @wraps(f)
    def wrapper(*args):
        result = f(*args)
        if isinstance(result, (pd.DataFrame, pd.Series)):
            result.dropna(inplace=True)
        return result
    return wrapper


def coerce_numbers_to_my_dtype(f):
    """
    A decorator for methods whose signature is f(self, other) that coerces
    ``other`` to ``self.dtype``.

    This is used to make comparison operations between numbers and `Factor`
    instances work independently of whether the user supplies a float or
    integer literal.

    For example, if I write::

        my_filter = my_factor > 3

    my_factor probably has dtype float64, but 3 is an int, so we want to coerce
    to float64 before doing the comparison.
    """
    @wraps(f)
    def method(self, other):
        if isinstance(other, Number):
            other = coerce_to_dtype(self.dtype, other)
        return f(self, other)
    return method