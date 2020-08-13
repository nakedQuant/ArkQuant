# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import uuid
from collections import OrderedDict, namedtuple
from toolz import keyfilter
from functools import reduce
from pipe.term import Term, NotSpecific
from pipe.graph import TermGraph

Event = namedtuple('event', 'asset name')
NamedPipe = namedtuple('pipe', 'event priority')


class Pipeline(object):
    """
        拓扑执行逻辑
        a. logic of pipe
        b. withdraw logic of pipe --- instance of ump_picker
        c. pipe --- ump_picker
    """
    __slots__ = ['_terms_store', 'graph', '_workspace', '_ump_picker']

    def __init__(self, terms, ump_picker):
        self._terms_store = terms
        # last item --- finalTerm
        self._workspace = OrderedDict()
        self.graph = self._init_graph()
        self._ump_picker = ump_picker

    @property
    def name(self):
        return uuid.uuid4()

    @property
    def terms(self):
        return self._terms_store

    @property
    def ump_terms(self):
        return self._ump_picker.pickers

    def _initialize_workspace(self):
        self._workspace = OrderedDict

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
        graph = TermGraph(self._terms_store).graph
        return graph

    def __add__(self, term):
        if not isinstance(term, Term):
            raise TypeError(
                "{term} is not a valid pipe column. Did you mean to "
                "append '.latest'?".format(term=term)
            )
        if term in self._graph.nodes:
            raise Exception('term object already exists in pipe')
        self._terms_store.append(term)
        return self

    def __sub__(self, term):
        try:
            self._terms_store.remove(term)
        except Exception as e:
            raise TypeError(e)
        return self

    def __setattr__(self, key, value):
        raise NotImplementedError

    def _load_term(self, term, default):
        if term.dependencies != NotSpecific:
            # 将节点的依赖筛选出来
            inter_inputs = keyfilter(lambda x: x in term.dependencies,
                                     self._workspace)
            # 将依赖的交集作为节点的input
            input_of_term = reduce(lambda x, y: set(x) & set(y),
                                   inter_inputs.values())
        else:
            input_of_term = default
        return input_of_term

    def _decref_recursive(self, metadata, default):
        """
            internal method for decref_recursive
            decrease by layer
        """
        # return in_degree == 0 nodes
        decref_nodes = self.graph.decref_dependencies()
        for node in decref_nodes:
            _input = self._load_term(node, default)
            output = node.compute(_input, metadata)
            self._workspace[node] = output
        self._decref_recursive(metadata, default)

    def _inner_decref_recursive(self, metadata, default):
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
        default : asset list
            Reference counts for terms to be computed. Terms with reference
            counts of 0 do not need to be computed.
        return : PIPE list
        """
        self._initialize_workspace()
        self._decref_recursive(metadata, default)

    def _finalize(self, alternative):
        """将pipeline.name --- outs"""
        final = self._workspace.popitem(last=True).values()
        # transform to named_pipe , priority --- 0 highest
        outputs = [NamedPipe(Event(asset, self.name), priority) for priority, asset
                   in enumerate(final[:alternative])]
        return outputs

    def to_execution_plan(self, default, metadata, alternative):
        """
            to execute pipe logic
        """
        try:
            self._inner_decref_recursive(metadata, default)
        except Exception as e:
            print('error means graph decrease to top occur %s' % e)
        result = self._finalize(alternative)
        return result

    def to_withdraw_plan(self, position, metadata):
        """
            to execute ump_picker logic
        """
        out = self._ump_picker.evaluate(position, metadata)
        return out
