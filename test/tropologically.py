
import networkx as nx,pandas as pd
from weakref import WeakValueDictionary
from interface import default, implements, Interface

import glob, uuid

class IDomain(Interface):
    """
    A domain defines two things:

    1. A calendar defining the dates to which the pipeline's inputs and outputs
       should be aligned. The calendar is represented concretely by a pandas
       DatetimeIndex.

    2. The set of assets that the pipeline should compute over. Right now, the only
       supported way of representing this set is with a two-character country code
       describing the country of assets over which the pipeline should compute. In
       the future, we expect to expand this functionality to include more general
       concepts.
    """
    def all_sessions(self,sdate,edate):
        """
        Get all trading sessions for the calendar of this domain.

        This determines the row labels of Pipeline outputs for pipelines run on
        this domain.

        Returns
        -------
        sessions : pd.DatetimeIndex
            An array of all session labels for this domain.
        """

    def all_assets(self,category = 'stock'):
        """
            Get all assets
        """

Domain = implements(IDomain)
Domain.__doc__ = """ """
Domain.__name__ = "Domain"
Domain.__qualname__ = "zipline.pipeline.domain.Domain"

class Generic(Domain):
    """
    This module defines the interface and implementations of Pipeline domains.

    A domain represents a set of labels for the arrays computed by a Pipeline.
    Currently, this means that a domain defines two things:

    1. A calendar defining the dates to which the pipeline's inputs and outputs
       should be aligned. The calendar is represented concretely by a pandas
       DatetimeIndex.

    2. The set of assets that the pipeline should compute over. Right now, the only
       supported way of representing this set is with a two-character country code
       describing the country of assets over which the pipeline should compute. In
       the future, we expect to expand this functionality to include more general
       concepts.
    """
    def all_session(self,s,e):
        raise NotImplementedError

    def all_assets(self,category= 'stock'):
        raise NotImplementedError

    @default
    def roll_forward(self, dt,window):
        dt = pd.Timestamp(dt, tz='UTC')
        trading_days = self.all_sessions()
        idx = trading_days.searchsorted(dt)
        return trading_days[idx - window ,idx]

ALLOWED_DTYPES = [list,tuple]

class Term(object):
    """
        执行算法 --- 拓扑结构
        退出算法 --- 裁决模块

        因子 --- 策略 --- 执行算法
        scripts --- 策略

        Dependency-Graph representation of Pipeline API terms.
        结构:
            1 节点 --- 算法，基于拓扑结构 --- 实现算法逻辑 表明算法的组合方式
            2 不同的节点已经应该继承相同的接口，不需要区分pipeline还是featureUnion
            3 同一层级的不同节点相互独立，一旦有连接不同层级
            4 同一层级节点计算的标的集合交集得出下一层级的输入，不同节点之间不考虑权重分配因为交集（存在所有节点）
            5 最终节点 --- 返回一个有序有限的集合
        节点:
            1 inputs --- asset list
            2 compute ---- algorithm list
            3 outputs --- algorithm list & asset list

    """

    _term_cache = WeakValueDictionary

    def __new__(cls,
                script_file,
                params,
                dtype,
                domain = None,
                window_safe = False):
        if domain is None:
            domain = GenericDomain()

        if script_file not in glob.glob('strategy/*.py'):
            raise ValueError

        identity = cls._static_identity(script_file,params,domain,dtype,window_safe)
        try:
            return cls._term_cache[identity]
        except KeyError:
            new_instance = cls._term_cache[identity] = \
                super(Term,cls).__new__(cls)._init(script_file,params,domain,dtype)
            return new_instance

    def _static_identity(self,script_file,params,domain,dtype,window_safe):

        return (script_file,params,domain,dtype,window_safe)

    def _init(self,script_file,params,domain,dtype):
        self.domain = domain
        self.params = params
        self.dtype = dtype

        namespace = dict()
        with open('strategy/%s.py'%script_file,'r') as f:
            exec(f.read(),namespace)

        self._term_core = namespace[script_file]

        return self

    def _validate(self,out):

        if type(out) not in ALLOWED_DTYPES:
            raise TypeError(
                typename=type(self).__name__,
                dtype=self.dtype,
            )

    @property
    def inputs(self,inputs = None):
        """
            验证inputs的输入是否与dependencies保持一直
        """
        if inputs :
            term_string = map(str,self.dependencies)
            if list(inputs.keys()) != term_string:
                raise ValueError('inputs must corride with term dependencies')
            from functools import reduce
            term_input = reduce(lambda x ,y : set(x) | set(y),inputs.values())
        else:
            term_input = self.domain.all_assets()
        return term_input

    @property
    def dependencies(self,terms = []):

        return terms

    def _compute(self,*args):
        """
            subclass should implement
        """
        raise NotImplemented

    @expect(ALLOWED_DTYPES)
    def output(self,date):
        term_out = self._compute(self.inputs,date)
        return term_out


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

        self.graph = nx.DiGraph()

        self._frozen = False

        for term in terms:
            self._add_to_graph(term)

        self._frozen = True

        from collections import defaultdict

        self.inputs = dict()

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


    def execution_order(self):
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
        in_degree = dict(self.graph.in_degree)

        from toolz import valfilter,keyfiler

        layer = valfilter(lambda x : x == 0 ,in_degree)

        test = layer[0]

        inputs = self.inputs[term]

        outputs= layer[0].compute(inputs)

        self.input[test] = outputs

    def compute_inputs(self,term):

        if len(term.dependencies):
            return []


        #iter(nx.topological_sort(self.graph))

        # return list(nx.topological_sort(
        #     self.graph.subgraph(
        #         {
        #             term for term, refcount in refcounts.items()
        #             if refcount > 0 and term not in workspace
        #         },
        #     ),
        # ))

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

    @property
    def screen_name(self):
        """Name of the specially-designated ``screen`` term for the pipeline.
        """
        SCREEN_NAME = 'screen_' + uuid.uuid4().hex

    def iter_item(self,dt,inputs = None):

        in_degree = dict(self.graph.in_degree)

        layer = [key for key,val in in_degree.items() if val ==0 ]

        outputs = term.compute(dt)

        self.iter_item(dt,outputs)

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

    def __contains__(self, term):
        return term in self.graph

    def __len__(self):
        return len(self.graph)


if __name__ == '__main__':

    class Term(object):

        def __init__(self, default=[]):
            self._val = default

        @property
        def dependencies(self):
            return self._val

        @dependencies.setter
        def dependencies(self, val):
            self._val = val


    term_a = Term()
    term_b = Term()
    term_c = Term()
    term_d = Term()
    term_e = Term()

    term_a.dependencies = [term_b,term_c]

    term_b.dependencies = [term_d,term_e]

    terms = [term_a,term_b,term_c,term_d,term_e]

    graph = TermGraph(terms)

    print('graph_indegree',graph.graph.in_degree)

    print('graph_outdegree',graph.graph.out_degree)

    dgraph = nx.DiGraph()

    dgraph.add_edge('e','b')

    dgraph.add_edge('d','b')

    dgraph.add_edge('c','b')

    dgraph.add_edge('c','f')

    dgraph.add_edge('g','f')

    dgraph.add_edge('b','a')

    dgraph.add_edge('f','a')

    dgraph.add_edge('f','h')

    dgraph.add_edge('a','i')

    dgraph.add_edge('h','i')

    print(list(nx.topological_sort(dgraph)))

    print('out_degree',dgraph.out_degree)

    print('in_degree',dgraph.in_degree)

    print('in_edges',dgraph.in_edges)

    print('out_edges',dgraph.out_edges)
