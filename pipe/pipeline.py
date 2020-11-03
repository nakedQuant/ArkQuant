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
from pipe.term import Term, NotSpecific
from pipe.graph import TermGraph
from pipe.ump import UmpPickers


class Pipeline(object):
    """
        拓扑执行逻辑
        a. logic of pipe
        b. withdraw logic of pipe --- instance of ump_picker
        c. pipe --- ump_picker
    """
    __slots__ = ['_terms_store', '_name', '_workspace', '_ump']

    def __init__(self, terms, ump_picker=None):
        self._terms_store = [terms] if isinstance(terms, Term) else terms
        self._workspace = OrderedDict()
        self._ump = UmpPickers(ump_picker) if ump_picker else UmpPickers(terms)
        self._name = str(uuid.uuid4())

    @property
    def name(self):
        return self._name

    @property
    def terms(self):
        return self._terms_store

    @property
    def ump_terms(self):
        return self._ump.pickers

    def _init_graph(self):
        """
        Compile into a simple TermGraph with no extra row metadata.

        Parameters
        ----------

        Returns
        -------
        graph : zipline.pipeline.graph.TermGraph
            Graph encoding term dependencies.
        """
        graph = TermGraph(self._terms_store)
        return graph

    def __add__(self, term):
        if not isinstance(term, Term):
            raise TypeError(
                "{term} is not a valid pipe column. Did you mean to "
                "append '.latest'?".format(term=term)
            )
        assert term not in self._terms_store, 'term object already exists in pipe'
        # if term in self._graph.nodes:
        #     raise Exception('term object already exists in pipe')
        self._terms_store.append(term)
        return self

    def __sub__(self, term):
        try:
            self._terms_store.remove(term)
        except Exception as e:
            raise TypeError(e)
        return self

    def _combine_term_dependence(self, term, default_mask):
        # print('dependence', term.dependencies)
        if term.dependencies != [NotSpecific]:
            # 将节点的依赖筛选出来
            dependence_masks = keyfilter(lambda x: x in term.dependencies,
                                         self._workspace)
            # print('dependence_masks', dependence_masks)
            # 将依赖的交集作为节点的input
            input_mask = reduce(lambda x, y: set(x) & set(y),
                                dependence_masks.values())
        else:
            input_mask = default_mask
        return input_mask

    def _decref_recursive(self, graph, metadata, mask):
        """
            internal method for decref_recursive
            decrease by layer
        """
        # print('decref graph', graph.nodes)
        decref_nodes = graph.decref_dependencies()
        # print('decref nodes', decref_nodes)
        if decref_nodes:
            for node in decref_nodes:
                node_mask = self._combine_term_dependence(node, mask)
                # print('node_mask', node_mask)
                output = node.compute(metadata, list(node_mask))
                self._workspace[node] = output
                # print('_workspace', self._workspace)
            self._decref_recursive(graph, metadata, mask)

    def _decref_dependence(self, graph, metadata, mask):
        """
        Return a topologically-sorted list of the terms in ``self`` which
        need to be computed.

        Filters out any terms that are already present in ``workspace``, as
        well as any terms with refcounts of 0.

        Parameters
        ----------
        metadata : dict[Term, np.ndarray]
            Initial state of workspace for a pipe execution. May contain
            pre-computed values provided by ``populate_initial_workspace``.
        mask : asset list
            Reference counts for terms to be computed. Terms with reference
            counts of 0 do not need to be computed.
        """
        self._decref_recursive(graph, metadata, mask)

    def compute_eager_pipeline(self, final):
        """
        Compute pipeline to get eager asset
        """
        outputs = self._workspace.popitem(last=True)
        # asset tag pipeline_name --- 可能存在相同的持仓但是由不同的pipeline产生
        outputs = [r.source_id(self.name) for r in outputs[1]]
        # print('pipeline outputs', outputs)
        try:
            final_out = final.resolve_final(outputs)
        except IndexError:
            print('list index out of range means no pipeline out')
            final_out = None
        return final_out

    def to_execution_plan(self, metadata, mask, final):
        """
            to execute pipe logic
        """
        term_graph = self._init_graph()
        self._decref_dependence(term_graph, metadata, mask)
        result = self.compute_eager_pipeline(final)
        self._workspace = OrderedDict()
        return result

    def to_withdraw_plan(self, position, metadata):
        """
            to execute ump_picker logic
        """
        out = self._ump.evaluate(position, metadata)
        return out


__all__ = ['Pipeline']


# if __name__ == '__main__':
#
#     from pipe.term import Term
#     kw = {'window': (5, 10), 'fields': ['close']}
#     cross_term = Term('cross', kw)
#     # print('sma_term', cross_term)
#     kw = {'fields': ['close'], 'window': 5, 'final': True}
#     break_term = Term('break', kw, cross_term)
#     p = Pipeline(cross_term)
#     print('p', p.terms)
#     p = p + break_term
#     print('after p', p.terms)
#     p = p - break_term
#     print('sub p', p.terms)
#     print('name', p.name)
#     print('name_', p.name)
#     print('name', p.name)
