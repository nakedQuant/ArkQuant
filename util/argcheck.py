# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import argparse, re, heapq, inspect
from collections import namedtuple
from hashlib import md5


Argspec = namedtuple('Argspec', ['args', 'starargs', 'kwargs'])


def create_args(args, root):
    """
    Encapsulates a set of custom command line arguments in key=value
    or key.namespace=value form into a chain of Namespace objects,
    where each next level is an attribute of the Namespace object on the
    current level

    Parameters
    ----------
    args : list
        A list of strings representing arguments in key=value form
    root : Namespace
        The top-level element of the argument tree
    """

    extension_args = {}

    for arg in args:
        parse_extension_arg(arg, extension_args)

    for name in sorted(extension_args, key=len):
        path = name.split('.')
        update_namespace(root, path, extension_args[name])


def parse_argspec(callable_):
    """
    Takes a callable and returns a tuple with the list of Argument objects,
    the name of *args, and the name of **kwargs.
    If *args or **kwargs is not present, it will be None.
    This returns a namedtuple called Argspec that has three fields named:
    args, starargs, and kwargs.
    """
    args, varargs, keywords, defaults = inspect.signature(callable_)
    defaults = list(defaults or [])

    if getattr(callable_, '__self__', None) is not None:
        # This is a bound method, drop the self param.
        args = args[1:]

    first_default = len(args) - len(defaults)
    return Argspec(
        [args[n] if n < first_default else defaults[n - first_default]
         for n, arg in enumerate(args)],
        varargs,
        keywords,
    )


def parse_extension_arg(arg, arg_dict):
    """
    Converts argument strings in key=value or key.namespace=value form
    to dictionary entries

    Parameters
    ----------
    arg : str
        The argument string to parse, which must be in key=value or
        key.namespace=value form.
    arg_dict : dict
        The dictionary into which the key/value pair will be added
    """

    match = re.match(r'^(([^\d\W]\w*)(\.[^\d\W]\w*)*)=(.*)$', arg)
    if match is None:
        raise ValueError(
            "invalid extension argument '%s', must be in key=value form" % arg
        )

    name = match.group(1)
    value = match.group(4)
    arg_dict[name] = value


def update_namespace(namespace, path, name):
    """
    A recursive function that takes a root element, list of namespaces,
    and the value being stored, and assigns namespaces to the root object
    via a chain of Namespace objects, connected through attributes

    Parameters
    ----------
    namespace : Namespace
        The object onto which an attribute will be added
    path : list
        A list of strings representing namespaces
    name : str
        The value to be stored at the bottom level
    """

    if len(path) == 1:
        setattr(namespace, path[0], name)
    else:
        if hasattr(namespace, path[0]):
            if isinstance(getattr(namespace, path[0]), str):
                raise ValueError("Conflicting assignments at namespace"
                                 " level '%s'" % path[0])
        else:
            a = Namespace()
            setattr(namespace, path[0], a)

        update_namespace(getattr(namespace, path[0]), path[1:], name)


def commandParse():
    default={'color': 'red', 'user': 'guest'}
    # 创建参数实例
    parser = argparse.ArgumentParser()
    # 添加
    parser.add_argument('-u', '--user')
    parser.add_argument('-c', '--color')
    # 解析参数
    namespace = parser.parse_args()
    command_line_args={k: v for k, v in vars(namespace).items() if v}
    return command_line_args


def _decorate_source(source):
    for message in source:
        yield ((message.dt, message.source_id), message)


def date_sorted_sources(*sources):
    """
    Takes an iterable of sources, generating namestrings and
    piping their output into date_sort.
    """
    # merge multi inputs into single return iterable
    sorted_stream = heapq.merge(*(_decorate_source(s) for s in sources))

    # Strip out key decoration
    for _, message in sorted_stream:
        yield message


def hash_args(*args, **kwargs):
    """Define a unique string for any set of representable args."""
    arg_string = '_'.join([str(arg) for arg in args])
    kwarg_string = '_'.join([str(key) + '=' + str(value)
                             for key, value in kwargs.items()])
    combined = ':'.join([arg_string, kwarg_string])
    hasher = md5()
    hasher.update(bytes(combined))
    return hasher.hexdigest()


