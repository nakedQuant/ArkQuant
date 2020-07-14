#
# Copyright 2014 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import namedtuple
from itertools import chain
from six.moves import map, zip_longest
import argparse ,re

from uility import getargspec

Argspec = namedtuple('Argspec', ['args', 'starargs', 'kwargs'])


def singleton(cls):
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance

def parse_argspec(callable_):
    """
    Takes a callable and returns a tuple with the list of Argument objects,
    the name of *args, and the name of **kwargs.
    If *args or **kwargs is not present, it will be None.
    This returns a namedtuple called Argspec that has three fields named:
    args, starargs, and kwargs.
    """
    args, varargs, keywords, defaults = getargspec(callable_)
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

class Namespace(object):
    """
    A placeholder object representing a namespace level
    """

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
    default={'color':'red','user':'guest'}
    #创建参数实例
    parser=argparse.ArgumentParser()
    #添加
    parser.add_argument('-u','--user')
    parser.add_argument('-c','--color')
    #解析参数
    namespace=parser.parse_args()
    command_line_args={k:v for k,v in vars(namespace).items() if v}

