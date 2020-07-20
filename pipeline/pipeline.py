# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import uuid,warnings,numpy as np
from collections import OrderedDict
from toolz import keyfilter
from contextlib import contextmanager
from functools import reduce

from gateWay.assets._finder import  AssetFinder

class NotSpecific(Exception):

    def __str__(self):
        return ('object not specific')

    __repr__ = __str__()

@contextmanager
def ignore_pandas_nan_categorical_warning():
    with warnings.catch_warnings():
        # Pandas >= 0.18 doesn't like null-ish values in categories, but
        # avoiding that requires a broader change to how missing values are
        # handled in pipeline, so for now just silence the warning.
        warnings.filterwarnings(
            'ignore',
            category=FutureWarning,
        )
        yield


class Pipeline(object):
    """
        拓扑执行逻辑
    """
    __slots__ = ['_term_store','default_type']

    def __init__(self,terms,default):
        self._terms_store = terms
        self.default_type = default
        self._init_graph()
        self.initialzed = False
        self.asset_finder = AssetFinder()

    @property
    def name(self):
        return uuid.uuid4().hex()

    def __add__(self,term):
        if not isinstance(term, Term):
            raise TypeError(
                "{term} is not a valid pipeline column. Did you mean to "
                "append '.latest'?".format(term=term)
            )
        if term in self.pipeline_graph.nodes:
            raise Exception('term object already exists in pipeline')
        self._terms_store.append(term)
        return self

    def __sub__(self,term):
        try:
            self._terms_store.remove(term)
        except Exception as e:
            raise TypeError(e)
        return self

    def __setattr__(self, key, value):

        raise NotImplementedError

    def _init_graph(self):
        self.pipeline_graph = self._to_simple_graph()

    def attach_default(self,dt,window = 30):
        """
            a. 剔除停盘
            b. 剔除进入退市整理期的30个交易日
            c. 剔除上市不足一个月的 --- 为了防止筛选出来的都是无法买入的次新股由于流通性问题 -- 次新股波动性太大缺少一定逻辑性,而且到时基于pipeline
               算法筛选出得都是无法买入的次新股的概率大大增加
        """
        suspend = self.asset_finder.fuzzy_symbols_ownership_by_suspend(dt)
        delist = self.asset_finder.fuzzy_symbol_ownership_by_delist(dt)
        unsufficient = self.asset_finder.fuzzy_symbol_ownership_by_ipo(dt,window)
        self.default = set(unsufficient) - (set(suspend) | set(delist))
        self.initialzed = True

    def _to_simple_graph(self):
        """
        Compile into a simple TermGraph with no extra row metadata.

        Parameters
        ----------
        default_screen : zipline.pipeline.Term
            Term to use as a screen if self.screen is None.

        Returns
        -------
        graph : zipline.pipeline.graph.TermGraph
            Graph encoding term dependencies.
        """
        graph = TermGraph(self._terms_store).graph
        return graph

    def _inputs_for_term(self, term):
        if term.dependencies != NotSpecific :
            slice_inputs = keyfilter(lambda x: x in term.dependencies,
                                     self.initial_workspace_cache)
            input_of_term = reduce(lambda x, y: set(x) & set(y),
                                   slice_inputs.values())
        else:
            input_of_term = self.default
        return input_of_term

    def _decref_recursive(self,metadata):
        """
        Return a topologically-sorted list of the terms in ``self`` which
        need to be computed.

        Filters out any terms that are already present in ``workspace``, as
        well as any terms with refcounts of 0.

        Parameters
        ----------
        workspace : dict[Term, np.ndarray]
            Initial state of workspace for a pipeline execution. May contain
            pre-computed values provided by ``populate_initial_workspace``.
        refcounts : dict[Term, int]
            Reference counts for terms to be computed. Terms with reference
            counts of 0 do not need to be computed.
        """
        decref_nodes = self.pipeline_graph._decref_dependencies()

        for node in decref_nodes:
            _input = self._inputs_for_term(node)
            output = node._compute(_input,metadata)
            self.initial_workspace_cache[node] = output
            self._decref_recursive(metadata)

    def tag(self,outs):
        """将pipeline.name --- outs"""
        for asset in outs:
            asset.origin = self.name

    def to_execution_plan(self,pipeline_metadata,alternative = 1):
        """
            source: accumulated data from all terms
        """
        assert self.initialzied , ValueError('attach_default first')
        self.initial_workspace_cache = OrderedDict()
        self._decref_recursive(pipeline_metadata)
        outputs = self.initial_workspace_cache.popitem(last=True)
        self.initialzed = False
        #tag
        pipes = self.tag(outputs)
        #保留一定个数的标的
        return {self.name : pipes[:alternative + 1]}
