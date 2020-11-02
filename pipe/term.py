# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import glob, os
from toolz import valmap
from weakref import WeakValueDictionary
from pipe.domain import infer_domain


class NotSpecific(Exception):

    def __repr__(self):
        return 'object not specific'


class Term(object):
    """
        term.specialize(domain)
        执行算法 --- 拓扑结构
        退出算法 --- 裁决模块
        Dependency-Graph representation of Pipeline API terms.
        结构:
            1 节点 --- 算法，基于拓扑结构 --- 实现算法逻辑 表明算法的组合方式
            2 不同的节点已经应该继承相同的接口，不需要区分pipeline还是featureUnion
            3 同一层级的不同节点相互独立，一旦有连接不同层级
            4 同一层级节点计算的标的集合交集得出下一层级的输入，不同节点之间不考虑权重分配因为交集（存在所有节点）,也可以扩展
            5 每一个节点产生有序的股票集合，一层所有的交集按照各自节点形成综合排序
            6 最终节点 --- 返回一个有序有限的集合
        节点:
            1 mask --- asset list
            2 compute ---- algorithm list
            3 outputs --- algorithm list & asset list
        term --- 不可变通过改变不同的dependence重构pipeline --- params (final attr to define term)
        logic of term is indicator( return targeted assets)
    """
    default_type = (tuple,)
    _term_cache = WeakValueDictionary()
    namespace = dict()
    # base_dir = os.path.join(os.path.split(os.getcwd())[0], 'strat')
    base_dir = '/Users/python/Library/Mobile Documents/com~apple~CloudDocs/ArkQuant/strat'
    # base_dir = os.path.join(os.path.split(os.path.abspath('__file__'))[0], 'strat')

    def __new__(cls,
                script,
                params,
                dependencies=NotSpecific
                ):
        # p = cls._pop_params(params)
        p = cls._hash_params(params)
        # 设立身份属性防止重复产生实例
        identity = cls._static_identity(script, p, dependencies)
        try:
            return cls._term_cache[identity]
        except KeyError:
            new_instance = cls._term_cache[identity] = super(Term, cls).__new__(cls)._init(script, p, dependencies)
            # print('new_instance', new_instance.domain.domain_window)
            return new_instance

    @staticmethod
    def _hash_params(kwargs):
        kwargs = valmap(lambda x: tuple(x) if isinstance(x, list) else x, kwargs)
        hash_params = tuple(zip(kwargs.keys(), kwargs.values()))
        return hash_params

    @classmethod
    def _static_identity(cls, ins, domain, dependencies):
        return ins, domain, dependencies

    def _init(self, script, p, dependencies):
        """
            __new__已经初始化后，不需要在__init__里面调用
            Noop constructor to play nicely with our caching __new__.  Subclasses
            should implement _init instead of this method.

            When a class' __new__ returns an instance of that class, Python will
            automatically call __init__ on the object, even if a new object wasn't
            actually constructed.  Because we memoize instances, we often return an
            object that was already initialized from __new__, in which case we
            don't want to call __init__ again.
        """
        params = dict(p)
        # 解析信号文件并获取类对象
        # print('base_dir', self.base_dir)
        file = glob.glob(os.path.join(self.base_dir, '%s.py' % script))[0]
        # print(file)
        with open(file, 'r') as f:
            exec(f.read(), self.namespace)
        logic = self.namespace[script.capitalize()]
        try:
            self.signal = logic(params)
            self._validate()
        except TypeError:
            self._subclass_called_validate = False

        assert self._subclass_called_validate, (
            "Term._validate() was not called.\n"
            "This probably means that logic cannot be initialized."
        )
        del self._subclass_called_validate
        # infer domain
        self.domain = infer_domain(params)
        self.dependencies = [dependencies] if isinstance(dependencies, Term) or dependencies == NotSpecific \
            else dependencies
        return self

    def _validate(self):
        """
        Assert that this term is well-formed.  This should be called exactly
        once, at the end of Term._init().
        """
        # mark that we got here to enforce that subclasses overriding _validate
        self._subclass_called_validate = True

    def postprocess(self, data):
        """
            Called with an result of ``self``, unravelled (i.e. 1-dimensional)
            after any user-defined screens have been applied.
            This is mostly useful for transforming the dtype of an output, e.g., to
            convert a LabelArray into a pandas Categorical.
            The default implementation is to just return data unchanged.
            called with an result of self ,after any user-defined screens have been applied
            this is mostly useful for transforming  the dtype of an output
        """
        if type(data) not in self.default_type:
            try:
                data = self.default_type[0](data)
            except Exception as e:
                raise TypeError('cannot transform the style of data to %r due to error %s' % (self.default_type, e))
        return data

    def _compute(self, meta, mask):
        """
            Subclasses should implement this to perform actual computation.
            This is named ``_compute`` rather than just ``compute`` because
            ``compute`` is reserved for user-supplied functions in
            CustomFilter/CustomFactor/CustomClassifier.
            1. subclass should implement when _verify_asset_finder is True
            2. self.postprocess()
        """
        output = self.signal.long_signal(meta, mask)
        validate_output = self.postprocess(output)
        return validate_output

    def compute(self, metadata, mask):
        """
            1. subclass should implement when _verify_asset_finder is True
            2. self.postprocess()
        """
        output = self._compute(metadata, mask)
        # print('term output', output)
        return output

    def withdraw(self, feed):
        signal = self.signal.short_signal(feed)
        return signal

    def __repr__(self):
        return (
            "{type}({args})"
        ).format(
            type=type(self).__name__,
            # args=', '.join([k for i, k in self.__dict__]),
            args=self.signal.name)

    def recursive_repr(self):
        """A short repr to use when recursively rendering terms with inputs.
        """
        # Default recursive_repr is just the name of the type.
        return type(self).__name__


# if __name__ == '__main__':
#
#     kw = {'window': (5, 10)}
#     cross_term = Term('cross', kw)
#     print('sma_term', cross_term)
#     kw = {'window': 10, 'fast': 12, 'slow': 26, 'period': 9}
#     break_term = Term('break', kw, cross_term)
#     print(break_term.dependencies)

