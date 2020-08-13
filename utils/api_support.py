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

from functools import wraps
import threading
"""
多线程编程中的对同一变量的访问冲突的一种技术，TLS会为每一个线程维护一个和该线程绑定的变量的副本。而不是无止尽的传递局部参数的方式编程
每一个线程都拥有自己的变量副本，并不意味着就一定不会对TLS变量中某些操作枷锁了。
Java平台的java.lang.ThreadLocal和Python 中的threading.local()都是TLS技术的一种实现
TLS使用的缺陷是，如果你的线程都不退出，那么副本数据可能一直不被GC回收，会消耗很多资源，比如线程池中，线程都不退出，使用TLS需要非常小心
"""
context = threading.local()


def get_algo_instance():
    return getattr(context, 'algorithm', None)


def set_algo_instance(algo):
    context.algorithm = algo


class ZiplineAPI(object):
    """
    Context manager for making an algorithm instance available to zipline API
    functions within a scoped block.
    """

    def __init__(self, algo_instance):
        self.algo_instance = algo_instance

    def __enter__(self):
        """
        Set the given algo instance, storing any previously-existing instance.
        """
        self.old_algo_instance = get_algo_instance()
        set_algo_instance(self.algo_instance)

    def __exit__(self, _type, _value, _tb):
        """
        Restore the algo instance stored in __enter__.
        """
        set_algo_instance(self.old_algo_instance)


#基于api_method 将方法注册到api
def api_method(f):
    # Decorator that adds the decorated class method as a callable
    # function (wrapped) to zipline.api
    @wraps(f)
    def wrapped(*args, **kwargs):
        # Get the instance and call the method
        algo_instance = get_algo_instance()
        if algo_instance is None:
            raise RuntimeError(
                'zipline api method %s must be called during a simulation.'
                % f.__name__
            )
        return getattr(algo_instance, f.__name__)(*args, **kwargs)
    # Add functor to zipline.api
    # setattr(zipline.api, f.__name__, wrapped)
    # zipline.api.__all__.append(f.__name__)
    # f.is_api_method = True
    return f
