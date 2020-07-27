# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import uuid
from collections import OrderedDict
from toolz import keyfilter
from functools import reduce
from .term import Term , NotSpecific
from .graph import  TermGraph


class Pipeline(object):
    """
        拓扑执行逻辑
    """
    __slots__ = ['_term_store']

    def __init__(self,
                 terms,
                 ):
        self._terms_store = terms
        self._name = uuid.uuid4().hex()
        self._init_graph()

    @property
    def name(self):
        return self._name

    def set_default(self,default):
        self.default = default

    def _initialize_workspace(self):
        self._workspace = OrderedDict()

    def _init_graph(self):
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
        self._graph = TermGraph(self._terms_store).graph

    def __add__(self,term):
        if not isinstance(term, Term):
            raise TypeError(
                "{term} is not a valid pipeline column. Did you mean to "
                "append '.latest'?".format(term=term)
            )
        if term in self._graph.nodes:
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

    def _load_term(self, term):
        if term.dependencies != NotSpecific :
            # 将节点的依赖 --- 交集 作为下一个input
            slice_inputs = keyfilter(lambda x : x in term.dependencies,
                                     self._workspace)
            input_of_term = reduce(lambda x, y: set(x) & set(y),
                                   slice_inputs.values())
        else:
            input_of_term = self.default
        return input_of_term

    def _decref_recursive(self,metadata):
        """
            internal method for decref_recursive
        """
        decref_nodes = self._graph.decref_dependencies()
        for node in decref_nodes:
            _input = self._load_term(node)
            output = node.compute(_input,metadata)
            self._workspace[node] = output
            self._decref_recursive(metadata)

    def decref_recursive(self,metadata):
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
        self._decref_recursive(metadata)

    def fit_out(self,alternative):
        """将pipeline.name --- outs"""
        outputs = self._workspace.popitem(last=True).values()
        #打上标记 pipeline_name : asset
        outputs = [asset.tag(self.name) for asset in outputs[:alternative]]
        return outputs

    def to_execution_plan(self,metadata,alternative):
        """
            source: accumulated data from all terms
        """
        # assert self.initialized, ValueError('attach_default first')
        #initialize
        self._initialize_workspace()
        # main engine
        self.decref_recursive(metadata)
        pipeline_output = self.fit_out(alternative)
        return pipeline_output