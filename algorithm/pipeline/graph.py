
from weakref import WeakValueDictionary
import glob, uuid,networkx as nx

class Term(object):
    """
        执行算法 --- 拓扑结构
        退出算法 --- 裁决模块
        scripts --- 策略   params :  term_init_params term_fields min_extra_window  term_name(optional)
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
                script,
                term_params,
                dtype,
                window_safe = False):
        format_script = script if script.endswith('.py') else script + '.py'
        if format_script not in glob.glob('strategy/*.py'):
            raise ValueError

        if 'min_extra_window' not in term_params.keys():
            raise ValueError('missing min_extra_window in term params')
        if 'term_fields' not in term_params.keys():
            raise ValueError('missing term_fields means not data input in term')
        if 'name' not in term_params.keys():
            term_params['term_name'] = cls.__name__

        identity = cls._static_identity(format_script,term_params,dtype,window_safe)
        try:
            return cls._term_cache[identity]
        except KeyError:
            new_instance = cls._term_cache[identity] = \
                super(Term,cls).__new__(cls)._init(format_script,term_params,dtype)
            return new_instance

    def _static_identity(self,format_script,params,dtype,window_safe):

        return (format_script,params,dtype,window_safe)

    def _init(self,format_script,params,dtype):
        self.dtype = dtype
        self.default_inputs = Domain().all_assets()

        namespace = dict()
        with open('strategy/%s'%format_script,'r') as f:
            exec(f.read(),namespace)
            obj = namespace[params.pop('name')]
            self._term_core = obj(params.pop('term_init_params'))
        self.domain = params
        self._verify_asset_finder = False

    # @expect_types(data = ndarray)
    def postprocess(self,data):
        """
            called with an result of self ,after any user-defined screens have been applied
            this is mostly useful for transforming  the dtype of an output

            the default implementation is to just return data unchange
        """
        if not isinstance(data,self.dtype):
            try:
                data  = self.dtype(data)
            except:
                raise TypeError('cannot transform the style of data to %s'%self.dtype)
        return data

    @property
    def dependencies(self,terms = None):
        if terms and isinstance(terms,(list,tuple)):
            for item in terms:
                if not isinstance(item,self):
                    raise TypeError('dependencies must be Term')
        return terms

    def set_asset_finder(self,inputs):
        self.assert_finder = inputs
        self._verify_asset_finder = True

    def _compute(self,inputs):
        """
            1. subclass should implement when _verify_asset_finder is True
            2. self.postprocess()
        """
        raise NotImplemented


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

    @property
    def screen_name(self):
        """Name of the specially-designated ``screen`` term for the pipeline.
        """
        SCREEN_NAME = 'screen_' + uuid.uuid4().hex

    def __contains__(self, term):
        return term in self.graph

    def __len__(self):
        return len(self.graph)

    def ordered(self):
        return iter(nx.topological_sort(self.graph))

    @lazyval
    def jpeg(self):
        return display_graph(self, 'jpeg')

    @lazyval
    def png(self):
        return display_graph(self, 'png')

    @lazyval
    def svg(self):
        return display_graph(self, 'svg')

    def _repr_png_(self):
        return self.png.data
