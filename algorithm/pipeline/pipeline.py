
from collections import OrderedDict
from toolz import valfilter, keyfilter
from functools import reduce

class Pipeline(object):
    """
        结合了executionPlan 与 pipeline
    """
    def __init__(self,terms):

        self.initial_workspace_cache = OrderedDict()
        self.termGraph = self.to_simple_graph(terms)
        self.graph = self.termGraph.graph
        self._terms_store = terms

    def add(self,term,overwrite = False):
        if not isinstance(term, Term):
            raise TypeError(
                "{term} is not a valid pipeline column. Did you mean to "
                "append '.latest'?".format(term=term)
            )

        if term in self.graph.nodes:
            if overwrite:
                self.graph.remove_node(term)
            else:
                raise KeyError('item already exists')

        self.graph.add_node(term)

    def _validate_inputs_for_term(self,term):
        """
            验证inputs的输入是否与dependencies一致
        """
        dependencies = term.dependencies
        if dependencies:
            if set(dependencies).issubset(self.initial_workspace_cache.keys()):
                slice_inputs = keyfilter(lambda x : x in dependencies , self.inputs)
                term_input = reduce(lambda x ,y : set(x) | set(y),slice_inputs.values())
        else:
            term_input = term.default_input
        return term_input

    def decref_dependence(self, node_dict):
        """
        Decrement in-edges for ``term`` after computation.

        Parameters
        ----------
        term : zipline.pipeline.Term
            The term whose parents should be decref'ed.
        refcounts : dict[Term -> int]
            Dictionary of refcounts.

        Return
        ------
        garbage : set[Term]
            Terms whose refcounts hit zero after decrefing.
        """
        for node in node_dict.keys():
            self.graph.remove_node(node)

    def remove(self,term):

        self.decref_dependence({term:0})

    def loading_executable_nodes(self):
        """Contextmanager entered when loading a batch of LoadableTerms.

        Parameters
        ----------
        terms : list[zipline.pipeline.LoadableTerm]
            Terms being loaded.
        """
        in_degree = dict(self.graph.in_degree)
        nodes = valfilter(lambda x : x == 0 ,in_degree)
        return nodes

    def _computing_chunked_terms(self, nodes,source):
        """Contextmanager entered when computing a ComputableTerm.

        Parameters
        ----------
        terms : zipline.pipeline.ComputableTerm
            Terms being computed.
        """
        def run(node):
            inputs = self._validate_inputs_for_term(node)
            node.set_asset_finder(inputs)
            output = node._compute(source)
            self.initial_workspace_cache[node] = output

        from multiprocessing.pool import Pool
        for node in nodes:
            Pool.apply_async(run,node)

    def _decref_recursive(self,source):
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
        nodes = self.loading_executable_nodes()
        self._compute_chunked_terms(nodes,source)
        self.decref_dependence(nodes)
        self._decref_recursive(source)

    def to_execution_plan(self,source):
        """
            source: accumulated data from all terms
        """
        self._decref_recursive(source)
        return self.initial_workspace_cache.popitem(last=True)

    @staticmethod
    def to_simple_graph(terms):
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
        return TermGraph(terms)

    def show_graph(self,format = 'svg'):
        """
        Render this Pipeline as a DAG.

        Parameters
        ----------
        format : {'svg', 'png', 'jpeg'}
            Image format to render with.  Default is 'svg'.
        """
        g = self.to_simple_graph(self._terms_store)
        if format == 'svg':
            return g.svg
        elif format == 'png':
            return g.png
        elif format == 'jpeg':
            return g.jpeg
        else:
            # We should never get here because of the expect_element decorator
            # above.
            raise AssertionError("Unknown graph format %r." % format)