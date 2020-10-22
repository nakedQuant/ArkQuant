# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from textwrap import dedent
# from types import CodeType
from uuid import uuid4
from six import exec_
from functools import wraps
import inspect
from numpy import dtype
import pandas as pd
from operator import attrgetter
from datetime import tzinfo
from pytz import timezone
from toolz import curry

# @api_method
# @require_not_initialized(AttachPipelineAfterInitialize())
# @expect_types(
#     pipeline=Pipeline,
#     name=string_types,
#     chunks=(int, Iterable, type(None)),
# )

_qualified_name = attrgetter('__qualname__')


@curry
def lossless_float_to_int(funcname, func, argname, arg):
    """
    A preprocessor that coerces integral floats to ints.

    Receipt of non-integral floats raises a TypeError.
    """
    if not isinstance(arg, float):
        return arg

    arg_as_int = int(arg)
    if arg == arg_as_int:
        warnings.warn(
            "{f} expected an int for argument {name!r}, but got float {arg}."
            " Coercing to int.".format(
                f=funcname,
                name=argname,
                arg=arg,
            ),
        )
        return arg_as_int

    raise TypeError(arg)


def getargspec(f):
    full_argspec = inspect.getfullargspec(f)
    return inspect.ArgSpec(
        args=full_argspec.args,
        varargs=full_argspec.varargs,
        keywords=full_argspec.varkw,
        defaults=full_argspec.defaults,
    )


NO_DEFAULT = object()


def preprocess(*_unused,** processors):
    """
        保持函数签名一致性 --- 位置参数 关键字参数 基于字典形式
    """
    if _unused:
        raise TypeError("preprocess doesn't accept positional arguments")

    def _decorator(f):
        # varargs varkw --- str
        args, varargs, varkw, defaults = argspec = getargspec(f)
        if not all(isinstance(arg,str) for arg in args):
            raise TypeError('cannot validate function using tuple unpacking')
        if defaults is None:
            defaults = ()
        non_defaults = (NO_DEFAULT,) * (len(args) - len(defaults))
        # 位置参数在关键字参数之前
        args_defaults = list(zip(args,non_defaults + defaults))
        if varargs:
            args_defaults.append((varargs,NO_DEFAULT))
        if varkw:
            args_defaults.append((varargs,NO_DEFAULT))

        #字典keys可以与集合直接处理
        argset = set(args) | {varargs, varkw} - {None}
        #判断是否子集
        bad_names = processors.keys() - argset
        if bad_names:
            raise TypeError(
                "Got processors for unknown arguments: %s." % bad_names
            )
        return _build_preprocessed_function(f,processors,args_defaults,varargs,varkw)
    return _decorator


def _build_preprocessed_function(func,
                               processors,
                               args_defaults,
                               varargs,
                               varkw):
    """
        rebuild a preprocessed function with the same signature as func
    """
    def modified(name):
        #以a开头确保作为变量
        return 'a' + uuid4().hex + name

    mangled_name = modified(func.__name__)
    #执行脚本的全局变量
    exec_globals = {mangled_name: func, 'wraps': wraps}
    defaults_seen = 0
    default_name_template = 'a' + uuid4().hex + '_%d'
    signature = []
    call_args = []
    assignments = []
    star_map = {
        varargs: '*',
        varkw: '**',
    }

    # 核心部分 --- 对参数进行预处理
    def make_processor_assignment(args,process_name):
        template = "{args} = {processor}({func},'args',{args})"
        return template.format(
            args = args,
            processor = process_name,
            func = mangled_name
        )

    # 为了适配可变参数
    def name_as_arg(arg):
        return star_map.get(arg, '') + arg

    for arg,default in args_defaults:
        if default is NO_DEFAULT:
            signature.append(name_as_arg(arg))
        else:
            default_name = default_name_template % defaults_seen
            exec_globals[default_name] = default
            signature.append('='.join([name_as_arg(arg),default_name]))

        if arg in processors:
            procname = modified('__processor__' + arg)
            exec_globals[procname] = processors[arg]
            assignments.append(make_processor_assignment(arg,procname))

        # 包含位置参数以及关键字参数
        call_args.append(arg)

    # 主要执行语句
    exec_str = dedent(
        """
        @wraps({wrapped_funcname})
        def {func_name}({signature}):
            {assignments}
            return {wrapped_funcname}({call_args})
        """
    ).format(
        wrapped_funcname=mangled_name,
        func_name = func.__name__,
        # assignments --- 主要参数预处理过程
        assignments = '\n'.join(assignments),
        #函数签名 --- 函数参数
        signature = ','.join(signature),
        #call_args --- 全部转化为位置参数
        call_args = ','.join(call_args)
    )
    # 将string compile to pyobj
    compiled = compile(
        exec_str,
        func.__code__.co_filename,
        mode='exec',
    )
    #
    exec_locals = {}
    exec_(compiled, exec_globals, exec_locals)
    new_func = exec_locals[func.__name__]
    return new_func


def call(f):
    """
    Wrap a function in a processor that calls `f` on the argument before
    passing it along.

    Useful for creating simple arguments to the `@preprocess` decorator.

    Parameters
    ----------
    f : function
        Function accepting a single argument and returning a replacement.

    Examples
    --------
    >>> @preprocess(x=call(lambda x: x + 1))
    ... def foo(x):
    ...     return x
    ...
    >>> foo(1)
    2
    """
    @wraps(f)
    def processor(func, argname, arg):
        return f(arg)
    return processor


def _ensure_tuple(func, argname, arg):
    if isinstance(arg, tuple):
        return arg
    try:
        return tuple(arg)
    except TypeError:
        raise TypeError(
            "%s() expected argument '%s' to"
            " be iterable, but got %s instead." % (
                func.__name__, argname, arg,
            )
        )


def ensure_upper_case(func, argname, arg):
    if isinstance(arg, str):
        return arg.upper()
    else:
        raise TypeError(
            "{0}() expected argument '{1}' to"
            " be a string, but got {2} instead.".format(
                func.__name__,
                argname,
                arg,
            ),
        )


def ensure_dtype(func, argname, arg):
    """
    Argument preprocessor that converts the input into a numpy dtype.
    """
    try:
        return dtype(arg)
    except TypeError:
        raise TypeError(
            "{func}() couldn't convert argument "
            "{argname}={arg!r} to a numpy dtype.".format(
                func=_qualified_name(func),
                argname=argname,
                arg=arg,
            ),
        )


def ensure_timezone(func, argname, arg):
    """Argument preprocessor that converts the input into a tzinfo object.

    """
    if isinstance(arg, tzinfo):
        return arg
    if isinstance(arg, str):
        return timezone(arg)

    raise TypeError(
        "{func}() couldn't convert argument "
        "{argname}={arg!r} to a timezone.".format(
            func=_qualified_name(func),
            argname=argname,
            arg=arg,
        ),
    )


def ensure_timestamp(func, argname, arg):
    """Argument preprocessor that converts the input into a pandas Timestamp
    object.
    """
    try:
        return pd.Timestamp(arg)
    except ValueError as e:
        raise TypeError(
            "{func}() couldn't convert argument "
            "{argname}={arg!r} to a pandas Timestamp.\n"
            "Original error was: {t}: {e}".format(
                func=_qualified_name(func),
                argname=argname,
                arg=arg,
                t=_qualified_name(type(e)),
                e=e,
            ),
        )


# if __name__ == '__main__':
#
#     @preprocess(arg=_ensure_tuple)
#     def foo(arg, kwargs=4):
#         return arg, kwargs
#     print(foo([1,2,3]))
#
#     @preprocess(arg=ensure_upper_case)
#     def foo(arg, kwargs=4):
#         return arg, kwargs
#
#     print(foo('abc'))
#
#     @preprocess(arg=ensure_dtype)
#     def foo(arg, kwargs=4):
#         return arg, kwargs
#
#     print(foo('a'))
#
#
#     @preprocess(arg=ensure_timezone)
#     def foo(arg, kwargs=4):
#         return arg, kwargs
#     print(foo('utc'))
#
#
#     @preprocess(arg=ensure_timestamp)
#     def foo(arg, kwargs=4):
#         return arg, kwargs
#     print(foo('2010-01-01'))
