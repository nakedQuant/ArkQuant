
import networkx as nx

class TermGraph(object):
    """
    An abstract representation of Pipeline Term dependencies.

    This class does not keep any additional metadata about any term relations
    other than dependency ordering.  As such it is only useful in contexts
    where you care exclusively about order properties (for example, when
    drawing visualizations of execution order).

    Parameters
    ----------
    terms : dict
        A dict mapping names to final output terms.

    Attributes
    ----------
    outputs

    Methods
    -------
    ordered()
        Return a topologically-sorted iterator over the terms in self.
    execution_order(workspace, refcounts)
        Return a topologically-sorted iterator over the terms in self, skipping
        entries in ``workspace`` and entries with refcounts of zero.

    See Also
    --------
    ExecutionPlan
    """
    def __init__(self, terms):

        self.graph = nx.DiGraph()

        self._frozen = False

        for term in terms:
            self._add_to_graph(term)

        self._frozen = True

    def _add_to_graph(self,term):
        """
            先增加节点 --- 增加edge
        """
        if self._frozen:
            raise ValueError(
                "Can't mutate %s after construction." % type(self).__name__
            )

        self.graph.add_node(term)

        for dependency in term.dependencies:
            self._add_to_graph(dependency)
            self.graph.add_edge(dependency,term)

        self._outputs = terms

    @property
    def screen_name(self):
        """Name of the specially-designated ``screen`` term for the pipeline.
        """

    def execution_order(self, workspace, refcounts):
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
        return list(nx.topological_sort(
            self.graph.subgraph(
                {
                    term for term, refcount in refcounts.items()
                    if refcount > 0 and term not in workspace
                },
            ),
        ))

    def ordered(self):
        return iter(nx.topological_sort(self.graph))

    def initial_refcounts(self, initial_terms):
        """
        Calculate initial refcounts for execution of this graph.

        Parameters
        ----------
        initial_terms : iterable[Term]
            An iterable of terms that were pre-computed before graph execution.

        Each node starts with a refcount equal to its outdegree, and output
        nodes get one extra reference to ensure that they're still in the graph
        at the end of execution.
        """
        refcounts = self.graph.out_degree()
        for t in self.outputs.values():
            refcounts[t] += 1

        for t in initial_terms:
            self._decref_dependencies_recursive(t, refcounts, set())

        return refcounts

    def _decref_dependencies_recursive(self, term, refcounts, garbage):
        """
        Decrement terms recursively.

        Notes
        -----
        This should only be used to build the initial workspace, after that we
        should use:
        :meth:`~zipline.pipeline.graph.TermGraph.decref_dependencies`
        """
        # Edges are tuple of (from, to).
        for parent, _ in self.graph.in_edges([term]):
            refcounts[parent] -= 1
            # No one else depends on this term. Remove it from the
            # workspace to conserve memory.
            if refcounts[parent] == 0:
                garbage.add(parent)
                self._decref_dependencies_recursive(parent, refcounts, garbage)

    def decref_dependencies(self, term, refcounts):
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
        garbage = set()
        # Edges are tuple of (from, to).
        for parent, _ in self.graph.in_edges([term]):
            refcounts[parent] -= 1
            # No one else depends on this term. Remove it from the
            # workspace to conserve memory.
            if refcounts[parent] == 0:
                garbage.add(parent)
        return garbage

    def __contains__(self, term):
        return term in self.graph

    def __len__(self):
        return len(self.graph)

    # @lazyval
    def loadable_terms(self):
        return {term for term in self.graph if isinstance(term, LoadableTerm)}

    # @lazyval
    def jpeg(self):
        return display_graph(self, 'jpeg')

    # @lazyval
    def png(self):
        return display_graph(self, 'png')

    # @lazyval
    def svg(self):
        return display_graph(self, 'svg')

    def _repr_png_(self):
        return self.png.data


class ExecutionPlan(TermGraph):
    """
    Graph represention of Pipeline Term dependencies that includes metadata
    about extra rows required to perform computations.

    Each node in the graph has an `extra_rows` attribute, indicating how many,
    if any, extra rows we should compute for the node.  Extra rows are most
    often needed when a term is an input to a rolling window computation.  For
    example, if we compute a 30 day moving average of price from day X to day
    Y, we need to load price data for the range from day (X - 29) to day Y.

    Parameters
    ----------
    domain : zipline.pipeline.domain.Domain
        The domain of execution for which we need to build a plan.
    terms : dict
        A dict mapping names to final output terms.
    start_date : pd.Timestamp
        The first date for which output is requested for ``terms``.
    end_date : pd.Timestamp
        The last date for which output is requested for ``terms``.

    """
    def __init__(self,
                 domain,
                 terms,
                 start_date,
                 end_date,
                 min_extra_rows=0):
        super(ExecutionPlan, self).__init__(terms)


class Term(object):

    def __init__(self,default = []):
        self._val = default

    @property
    def dependencies(self):
        return self._val

    @dependencies.setter
    def dependencies(self,val):
        self._val = val


"""
Tools for visualizing dependencies between Terms.
"""
from __future__ import unicode_literals

from contextlib import contextmanager
import errno
from functools import partial
from io import BytesIO
from subprocess import Popen, PIPE

from networkx import topological_sort
from six import iteritems

from zipline.pipeline.data import BoundColumn
from zipline.pipeline import Filter, Factor, Classifier, Term
from zipline.pipeline.term import AssetExists


class NoIPython(Exception):
    pass


def delimit(delimiters, content):
    """
    Surround `content` with the first and last characters of `delimiters`.

    >>> delimit('[]', "foo")  # doctest: +SKIP
    '[foo]'
    >>> delimit('""', "foo")  # doctest: +SKIP
    '"foo"'
    """
    if len(delimiters) != 2:
        raise ValueError(
            "`delimiters` must be of length 2. Got %r" % delimiters
        )
    return ''.join([delimiters[0], content, delimiters[1]])


quote = partial(delimit, '""')
bracket = partial(delimit, '[]')


def begin_graph(f, name, **attrs):
    writeln(f, "strict digraph %s {" % name)
    writeln(f, "graph {}".format(format_attrs(attrs)))


def begin_cluster(f, name, **attrs):
    attrs.setdefault("label", quote(name))
    writeln(f, "subgraph cluster_%s {" % name)
    writeln(f, "graph {}".format(format_attrs(attrs)))


def end_graph(f):
    writeln(f, '}')


@contextmanager
def graph(f, name, **attrs):
    begin_graph(f, name, **attrs)
    yield
    end_graph(f)


@contextmanager
def cluster(f, name, **attrs):
    begin_cluster(f, name, **attrs)
    yield
    end_graph(f)


def roots(g):
    "Get nodes from graph G with indegree 0"
    return set(n for n, d in iteritems(g.in_degree()) if d == 0)


def filter_nodes(include_asset_exists, nodes):
    if include_asset_exists:
        return nodes
    return filter(lambda n: n is not AssetExists(), nodes)


def _render(g, out, format_, include_asset_exists=False):
    """
    Draw `g` as a graph to `out`, in format `format`.

    Parameters
    ----------
    g : zipline.pipeline.graph.TermGraph
        Graph to render.
    out : file-like object
    format_ : str {'png', 'svg'}
        Output format.
    include_asset_exists : bool
        Whether to filter out `AssetExists()` nodes.
    """
    graph_attrs = {'rankdir': 'TB', 'splines': 'ortho'}
    cluster_attrs = {'style': 'filled', 'color': 'lightgoldenrod1'}

    in_nodes = g.loadable_terms
    out_nodes = list(g.outputs.values())

    f = BytesIO()
    with graph(f, "G", **graph_attrs):

        # Write outputs cluster.
        with cluster(f, 'Output', labelloc='b', **cluster_attrs):
            for term in filter_nodes(include_asset_exists, out_nodes):
                add_term_node(f, term)

        # Write inputs cluster.
        with cluster(f, 'Input', **cluster_attrs):
            for term in filter_nodes(include_asset_exists, in_nodes):
                add_term_node(f, term)

        # Write intermediate results.
        for term in filter_nodes(include_asset_exists,
                                 topological_sort(g.graph)):
            if term in in_nodes or term in out_nodes:
                continue
            add_term_node(f, term)

        # Write edges
        for source, dest in g.graph.edges():
            if source is AssetExists() and not include_asset_exists:
                continue
            add_edge(f, id(source), id(dest))

    cmd = ['dot', '-T', format_]
    try:
        proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise RuntimeError(
                "Couldn't find `dot` graph layout program. "
                "Make sure Graphviz is installed and `dot` is on your path."
            )
        else:
            raise

    f.seek(0)
    proc_stdout, proc_stderr = proc.communicate(f.read())
    if proc_stderr:
        raise RuntimeError(
            "Error(s) while rendering graph: %s" % proc_stderr.decode('utf-8')
        )

    out.write(proc_stdout)


def display_graph(g, format='svg', include_asset_exists=False):
    """
    Display a TermGraph interactively from within IPython.
    """
    try:
        import IPython.display as display
    except ImportError:
        raise NoIPython("IPython is not installed.  Can't display graph.")

    if format == 'svg':
        display_cls = display.SVG
    elif format in ("jpeg", "png"):
        display_cls = partial(display.Image, format=format, embed=True)

    out = BytesIO()
    _render(g, out, format, include_asset_exists=include_asset_exists)
    return display_cls(data=out.getvalue())


def writeln(f, s):
    f.write((s + '\n').encode('utf-8'))


def fmt(obj):
    if isinstance(obj, Term):
        r = obj.graph_repr()
    else:
        r = obj
    return '"%s"' % r


def add_term_node(f, term):
    declare_node(f, id(term), attrs_for_node(term))


def declare_node(f, name, attributes):
    writeln(f, "{0} {1};".format(name, format_attrs(attributes)))


def add_edge(f, source, dest):
    writeln(f, "{0} -> {1};".format(source, dest))


def attrs_for_node(term, **overrides):
    attrs = {
        'shape': 'box',
        'colorscheme': 'pastel19',
        'style': 'filled',
        'label': fmt(term),
    }
    if isinstance(term, BoundColumn):
        attrs['fillcolor'] = '1'
    if isinstance(term, Factor):
        attrs['fillcolor'] = '2'
    elif isinstance(term, Filter):
        attrs['fillcolor'] = '3'
    elif isinstance(term, Classifier):
        attrs['fillcolor'] = '4'

    attrs.update(**overrides or {})
    return attrs


def format_attrs(attrs):
    """
    Format key, value pairs from attrs into graphviz attrs format

    Examples
    --------
    >>> format_attrs({'key1': 'value1', 'key2': 'value2'})  # doctest: +SKIP
    '[key1=value1, key2=value2]'
    """
    if not attrs:
        return ''
    entries = ['='.join((key, value)) for key, value in iteritems(attrs)]
    return '[' + ', '.join(entries) + ']'



if __name__ == '__main__':

    term_a = Term()
    term_b = Term()
    term_c = Term()
    term_d = Term()
    term_e = Term()

    term_a.dependencies = [term_b]

    term_a.dependencies = [term_c]

    term_b.dependencies = [term_d]

    term_b.dependencies = [term_e]

    terms = [term_a,term_b,term_c,term_d,term_e]

    graph = TermGraph(terms)

    refcounts = graph.graph.out_degree
    print('refcount',refcounts)

    outputs = graph._outputs

    print(outputs)

    # graph.execution_order()

