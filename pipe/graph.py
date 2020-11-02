# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import uuid, networkx as nx, numpy as np
import matplotlib.pyplot as plt
from toolz import valfilter
from pipe.term import NotSpecific, Term


class TermGraph(object):
    """
    An abstract representation of Pipeline Term dependencies.

    This class does not keep any additional metadata about any term relations
    other than dependency ordering.  As such it is only useful in contexts
    where you care exclusively about order properties (for example, when
    drawing visualizations of execution order).

    Graph represention of Pipeline Term dependencies that includes metadata
    about extra rows required to perform computations.

    Each node in the graph has an `extra_rows` attribute, indicating how many,
    if any, extra rows we should compute for the node.  Extra rows are most
    often needed when a term is an input to a rolling window computation.
    """
    def __init__(self, terms):
        assert np.all([t for t in terms if isinstance(t, Term)]), \
            'terms must be subclass of term'
        self.graph = nx.DiGraph()
        self._frozen = False
        for term in terms:
            self._add_to_graph(term)
        self._outputs = terms
        self._frozen = True

    def _add_to_graph(self, term):
        """
            先增加节点 --- 增加edge
        """
        if self._frozen:
            raise ValueError(
                "Can't mutate %s after construction." % type(self).__name__
            )
        self.graph.add_node(term)
        for dependency in term.dependencies:
            if dependency == NotSpecific:
                pass
            else:
                self._add_to_graph(dependency)
                self.graph.add_edge(dependency, term)

    @property
    def outputs(self):
        """
        Dict mapping names to designated output terms.
        """
        return self._outputs

    @property
    def screen_name(self):
        """Name of the specially-designated ``screen`` term for the pipe.
        """
        screen = 'screen_' + uuid.uuid4().hex
        return screen

    @property
    def nodes(self):
        return self.graph.nodes

    def __contains__(self, term):
        return term in self.graph

    def __len__(self):
        return len(self.graph)

    def ordered(self):
        return iter(nx.topological_sort(self.graph))

    def decref_dependencies(self):
        """
        Decrement in-edges for ``term`` after computation.

        Return
        ------
        terms which need to decref
        """
        refcounts = dict(self.graph.in_degree())
        # print('refcounts', refcounts)
        nodes = valfilter(lambda x: x == 0, refcounts)
        # print('refcounts == 0', nodes)
        for node in nodes:
            self.graph.remove_node(node)
        return nodes

    def draw(self):
        # plt.subplot(121)
        # nx.draw(self.graph)
        # plt.subplot(122)
        # nx.draw(self.graph, pos=nx.circular_layout(self.graph),
        #         node_color='r', edge_color='b')
        nx.draw_networkx(self.graph, pos=nx.circular_layout(self.graph),
                         node_color='r', edge_color='b')
        plt.show()


__all__ = ['TermGraph']


if __name__ == '__main__':

    kw = {'window': (5, 10)}
    cross_term = Term('cross', kw)
    print('sma_term', cross_term)
    kw = {'window': 10, 'fast': 12, 'slow': 26, 'period': 9}
    break_term = Term('break', kw, cross_term)
    terms = [cross_term, break_term]
    graph = TermGraph(terms)
    print('ordered graph', list(graph.ordered()))
    print('nodes', graph.nodes)
    print('length', len(graph))
    # graph.decref_dependencies()
    graph.draw()
