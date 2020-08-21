# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import glob, numpy as np
from weakref import WeakValueDictionary


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
            1 inputs --- asset list
            2 compute ---- algorithm list
            3 outputs --- algorithm list & asset list

        term --- 不可变通过改变不同的dependence重构pipeline
        term --- universe
    """
    inputs = NotSpecified
    outputs = NotSpecified
    window_length = NotSpecified
    mask = NotSpecified
    domain = NotSpecified
    _term_cache = WeakValueDictionary
    namespace = dict()

    __slots__ = ['domain', 'dependence', 'd_type', 'term_logic', '_subclass_called_validate']

    def __new__(cls,
                domain,
                script,
                params,
                d_type=list,
                dependence=NotSpecific
                ):
        # 解析策略文件并获取对象
        script_path = glob.glob('strategy/%s.py' % script)
        with open(script_path, 'r') as f:
            exec(f.read(), cls.namespace)
        logic_cls = cls.namespace[script]
        # 设立身份属性防止重复产生实例
        identity = cls._static_identity(domain, logic_cls, params, d_type, dependence)
        try:
            return cls._term_cache[identity]
        except KeyError:
            new_instance = cls._term_cache[identity] = \
                super(Term, cls).__new__(cls)._init(domain, logic_cls, params, d_type, dependence)
            return new_instance

    @classmethod
    def _static_identity(cls, domain, script_class, script_params, d_type, dependence):
        return domain, script_class, script_params, d_type, dependence

    # __new__已经初始化后，不需要在__init__里面调用
    def _init(self, domain, script, params, d_type, dependence):
        """
            __new__已经初始化后，不需要在__init__里面调用
            Noop constructor to play nicely with our caching __new__.  Subclasses
            should implement _init instead of this method.

            When a class' __new__ returns an instance of that class, Python will
            automatically call __init__ on the object, even if a new object wasn't
            actually constructed.  Because we memoize instances, we often return an
            object that was already initialized from __new__, in which case we
            don't want to call __init__ again.

            Subclasses that need to initialize new instances should override _init,
            which is guaranteed to be called only once.
            Parameters
            ----------
            domain : zipline.pipe.domain.Domain
                The domain of this term.
            dtype : np.dtype
                Dtype of this term's output.
            params : tuple[(str, hashable)]
                Tuple of key/value pairs of additional parameters.
        """
        self.domain = domain
        self.dependence = dependence
        self.d_type = d_type
        try:
            instance = script(params)
            self.term_logic = instance
            self._validate()
        except TypeError:
            self._subclass_called_validate = False

        assert self._subclass_called_validate, (
            "Term._validate() was not called.\n"
            "This probably means that logic cannot be initialized."
        )
        del self._subclass_called_validate
        return self

    def _validate(self):
        """
        Assert that this term is well-formed.  This should be called exactly
        once, at the end of Term._init().
        """
        # mark that we got here to enforce that subclasses overriding _validate
        self._subclass_called_validate = True

    def __setattr__(self, key, value):
        raise NotImplementedError()

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
        if self.d_type == bool:
            if not isinstance(data, self.d_type):
                raise TypeError('style of data is not %r' % self.d_type)
        try:
            data = self.d_type(data)
        except Exception as e:
            raise TypeError('cannot transform the style of data to %s due to error %s' % (self.d_type, e))
        return data

    def _compute(self, inputs, data):
        """
            Subclasses should implement this to perform actual computation.
            This is named ``_compute`` rather than just ``compute`` because
            ``compute`` is reserved for user-supplied functions in
            CustomFilter/CustomFactor/CustomClassifier.
            1. subclass should implement when _verify_asset_finder is True
            2. self.postprocess()
        """
        output = self.term_logic.compute(inputs, data)
        validate_output = self.postprocess(output)
        return validate_output

    def compute(self, inputs, data):
        """
            1. subclass should implement when _verify_asset_finder is True
            2. self.postprocess()
        """
        output = self._compute(inputs, data)
        return output

    @lazyval
    def windowed(self):
        """
        Whether or not this term represents a trailing window computation.

        If term.windowed is truthy, its compute_from_windows method will be
        called with instances of AdjustedArray as inputs.

        If term.windowed is falsey, its compute_from_baseline will be called
        with instances of np.ndarray as inputs.
        """
        return (
            self.window_length is not NotSpecified
            and self.window_length > 0
        )

    @lazyval
    def dependencies(self):
        """
        The number of extra rows needed for each of our inputs to compute this
        term.
        """
        extra_input_rows = max(0, self.window_length - 1)
        out = {}
        for term in self.inputs:
            out[term] = extra_input_rows
        out[self.mask] = 0
        return out

    def to_workspace_value(self, result, assets):
        """
        Called with a column of the result of a pipeline. This needs to put
        the data into a format that can be used in a workspace to continue
        doing computations.

        Parameters
        ----------
        result : pd.Series
            A multiindexed series with (dates, assets) whose values are the
            results of running this pipeline term over the dates.
        assets : pd.Index
            All of the assets being requested. This allows us to correctly
            shape the workspace value.

        Returns
        -------
        workspace_value : array-like
            An array like value that the engine can consume.
        """
        return result.unstack().fillna(self.missing_value).reindex(
            columns=assets,
            fill_value=self.missing_value,
        ).values

    def _downsampled_type(self, *args, **kwargs):
        """
        The expression type to return from self.downsample().
        """
        raise NotImplementedError(
            "downsampling is not yet implemented "
            "for instances of %s." % type(self).__name__
        )

    @expect_downsample_frequency
    @templated_docstring(frequency=PIPELINE_DOWNSAMPLING_FREQUENCY_DOC)
    def downsample(self, frequency):
        """
        Make a term that computes from ``self`` at lower-than-daily frequency.

        Parameters
        ----------
        {frequency}
        """
        return self._downsampled_type(term=self, frequency=frequency)

    def _aliased_type(self, *args, **kwargs):
        """
        The expression type to return from self.alias().
        """
        raise NotImplementedError(
            "alias is not yet implemented "
            "for instances of %s." % type(self).__name__
        )

    @templated_docstring(name=PIPELINE_ALIAS_NAME_DOC)
    def alias(self, name):
        """
        Make a term from ``self`` that names the expression.

        Parameters
        ----------
        {name}

        Returns
        -------
        aliased : Aliased
            ``self`` with a name.

        Notes
        -----
        This is useful for giving a name to a numerical or boolean expression.
        """
        return self._aliased_type(term=self, name=name)

    def __repr__(self):
        return (
            "{type}([{inputs}], {window_length})"
        ).format(
            type=type(self).__name__,
            inputs=', '.join(i.recursive_repr() for i in self.inputs),
            window_length=self.window_length,
        )

    def recursive_repr(self):
        """A short repr to use when recursively rendering terms with inputs.
        """
        # Default recursive_repr is just the name of the type.
        return type(self).__name__
