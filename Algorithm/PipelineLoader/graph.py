
from weakref import WeakValueDictionary
from collections import OrderedDict
from abc import ABC, abstractmethod
from functools import partial
from numpy import array, arange
from pandas import DataFrame, MultiIndex
from toolz import groupby,valfilter, keyfilter,partition_all
from functools import reduce
import glob, uuid,pandas as pd,networkx as nx,warnings,copy

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
        if format_script not in glob.glob('Strategy/*.py'):
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
        with open('Strategy/%s'%format_script,'r') as f:
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

    def compute_extra_rows(self,all_dates,start_date,end_date,min_extra_rows):

    def set_asset_finder(self,inputs):
        self.assert_finder = inputs
        self._verify_asset_finder = True

    def _compute(self,inputs):
        """
            subclass should implement when _verify_asset_finder is True
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


class Pipeline(object):
    """
        结合了executionPlan 与 pipeline
    """
    def __init__(self,terms,format = 'svg'):

        g = self.to_simple_graph(terms)
        self.graph = g.graph
        self.pic = self.show_graph(g,format)
        self.initial_workspace_cache = OrderedDict()

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

    def show_graph(self,g,format):
        """
        Render this Pipeline as a DAG.

        Parameters
        ----------
        format : {'svg', 'png', 'jpeg'}
            Image format to render with.  Default is 'svg'.
        """
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

    def _compute_output(self):
        self.to_execution_plan()
        return self.initial_workspace_cache.popitem(last = True)


class PipelineEngine(ABC):
    """
    Computation engines for executing Pipelines.

    This module defines the core computation algorithms for executing Pipelines.

    The primary entrypoint of this file is SimplePipelineEngine.run_pipeline, which
    implements the following algorithm for executing pipelines:

    1、Determine the domain of the pipeline.The domain determines the top-level
        set of dates and field that serves as row and column ---- data needed
        to compute the pipeline

    2. Build a dependency graph of all terms in TernmGraph with information
     about tropological tree of terms.

    3. Combine the domains of all terms to produce a overall data source.
        Each entry nodes(term) calculate outputs based on it.

    4. Iterate over the terms in the order computed . For each term:

       a. Fetch the term's inputs from the workspace and set_assert_finder
          with inputs

       b. Call ``term._compute`` with source . Store the results into
          the workspace.

       c. Decrement terms on the tropological tree and recursive the
          process.

    5. Extract the pipeline's outputs from the workspace and convert them
       into "narrow" format, with output labels dictated by the Pipeline's
       screen. This logic lives in SimplePipelineEngine._to_narrow.
    """
    def resolve_domain(self, terms):
        """Resolve a concrete domain for pipeline from terms.
        """

    def _pipeline_source_cache(self,domain):
        self._pipeline_source = WeakValueDictionary()



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

    def categorical_df_concat(df_list, inplace=False):
        """
        Prepare list of pandas DataFrames to be used as input to pd.concat.
        Ensure any columns of type 'category' have the same categories across each
        dataframe.

        Parameters
        ----------
        df_list : list
            List of dataframes with same columns.
        inplace : bool
            True if input list can be modified. Default is False.

        Returns
        -------
        concatenated : df
            Dataframe of concatenated list.
        """

        if not inplace:
            df_list = copy.deepcopy(df_list)

        # Assert each dataframe has the same columns/dtypes
        df = df_list[0]
        if not all([(df.dtypes.equals(df_i.dtypes)) for df_i in df_list[1:]]):
            raise ValueError("Input DataFrames must have the same columns/dtypes.")

        categorical_columns = df.columns[df.dtypes == 'category']

        for col in categorical_columns:
            new_categories = sorted(
                set().union(
                    *(frame[col].cat.categories for frame in df_list)
                )
            )

            with ignore_pandas_nan_categorical_warning():
                for df in df_list:
                    df[col].cat.set_categories(new_categories, inplace=True)

        return pd.concat(df_list)

    @abstractmethod
    def run_pipeline(self, pipeline, start_date, end_date, hooks=None):
        """
        Compute values for ``pipeline`` from ``start_date`` to ``end_date``.

        Parameters
        ----------
        pipeline : zipline.pipeline.Pipeline
            The pipeline to run.
        start_date : pd.Timestamp
            Start date of the computed matrix.
        end_date : pd.Timestamp
            End date of the computed matrix.
        hooks : list[implements(PipelineHooks)], optional
            Hooks for instrumenting Pipeline execution.

        Returns
        -------
        result : pd.DataFrame
            A frame of computed results.

            The ``result`` columns correspond to the entries of
            `pipeline.columns`, which should be a dictionary mapping strings to
            instances of :class:`zipline.pipeline.Term`.

            For each date between ``start_date`` and ``end_date``, ``result``
            will contain a row for each asset that passed `pipeline.screen`.
            A screen of ``None`` indicates that a row should be returned for
            each asset that existed each day.
        """
        raise NotImplementedError("run_pipeline")


class SimplePipelineEngine(PipelineEngine):
    """
    PipelineEngine class that computes each term independently.

    Parameters
    ----------
    get_loader : callable
        A function that is given a loadable term and returns a PipelineLoader
        to use to retrieve raw data for that term.
    asset_finder : zipline.assets.AssetFinder
        An AssetFinder instance.  We depend on the AssetFinder to determine
        which assets are in the top-level universe at any point in time.
    populate_initial_workspace : callable, optional
        A function which will be used to populate the initial workspace when
        computing a pipeline. See
        :func:`zipline.pipeline.engine.default_populate_initial_workspace`
        for more info.
    default_hooks : list, optional
        List of hooks that should be used to instrument all pipelines executed
        by this engine.

    See Also
    --------
    :func:`zipline.pipeline.engine.default_populate_initial_workspace`
    """
    __slots__ = (
        '_get_loader',
    )

    def __init__(self,
                 get_loader,
                 terms,
                 default_hooks=None):

        self._get_loader = get_loader

        self._populate_initial_workspace = {}

        if default_hooks is None:
            self._default_hooks = []
        else:
            self._default_hooks = list(default_hooks)

    def run_pipeline(self, pipeline, start_date, end_date, hooks=None):
        """
        Compute values for ``pipeline`` from ``start_date`` to ``end_date``.

        Parameters
        ----------
        pipeline : zipline.pipeline.Pipeline
            The pipeline to run.
        start_date : pd.Timestamp
            Start date of the computed matrix.
        end_date : pd.Timestamp
            End date of the computed matrix.
        hooks : list[implements(PipelineHooks)], optional
            Hooks for instrumenting Pipeline execution.

        Returns
        -------
        result : pd.DataFrame
            A frame of computed results.

            The ``result`` columns correspond to the entries of
            `pipeline.columns`, which should be a dictionary mapping strings to
            instances of :class:`zipline.pipeline.Term`.

            For each date between ``start_date`` and ``end_date``, ``result``
            will contain a row for each asset that passed `pipeline.screen`.
            A screen of ``None`` indicates that a row should be returned for
            each asset that existed each day.
        """
        hooks = self._resolve_hooks(hooks)
        with hooks.running_pipeline(pipeline, start_date, end_date):
            return self._run_pipeline_impl(
                pipeline,
                start_date,
                end_date,
                hooks,
            )

    def _run_pipeline_impl(self, pipeline, start_date, end_date, hooks):
        """Shared core for ``run_pipeline`` and ``run_chunked_pipeline``.
        """
        # See notes at the top of this module for a description of the
        # algorithm implemented here.
        if end_date < start_date:
            raise ValueError(
                "start_date must be before or equal to end_date \n"
                "start_date=%s, end_date=%s" % (start_date, end_date)
            )

        domain = self.resolve_domain(pipeline)

        plan = pipeline.to_execution_plan(
            domain, self._root_mask_term, start_date, end_date,
        )
        extra_rows = plan.extra_rows[self._root_mask_term]
        root_mask = self._compute_root_mask(
            domain, start_date, end_date, extra_rows,
        )
        dates, sids, root_mask_values = explode(root_mask)

        workspace = self._populate_initial_workspace(
            {
                self._root_mask_term: root_mask_values,
                self._root_mask_dates_term: as_column(dates.values)
            },
            self._root_mask_term,
            plan,
            dates,
            sids,
        )

        refcounts = plan.initial_refcounts(workspace)
        execution_order = plan.execution_order(workspace, refcounts)

        with hooks.computing_chunk(execution_order,
                                   start_date,
                                   end_date):

            results = self.compute_chunk(
                graph=plan,
                dates=dates,
                sids=sids,
                workspace=workspace,
                refcounts=refcounts,
                execution_order=execution_order,
                hooks=hooks,
            )

        return self._to_narrow(
            plan.outputs,
            results,
            results.pop(plan.screen_name),
            dates[extra_rows:],
            sids,
        )

    def _to_narrow(self, terms, data, mask, dates, assets):
        """
        Convert raw computed pipeline results into a DataFrame for public APIs.

        Parameters
        ----------
        terms : dict[str -> Term]
            Dict mapping column names to terms.
        data : dict[str -> ndarray[ndim=2]]
            Dict mapping column names to computed results for those names.
        mask : ndarray[bool, ndim=2]
            Mask array of values to keep.
        dates : ndarray[datetime64, ndim=1]
            Row index for arrays `data` and `mask`
        assets : ndarray[int64, ndim=2]
            Column index for arrays `data` and `mask`

        Returns
        -------
        results : pd.DataFrame
            The indices of `results` are as follows:

            index : two-tiered MultiIndex of (date, asset).
                Contains an entry for each (date, asset) pair corresponding to
                a `True` value in `mask`.
            columns : Index of str
                One column per entry in `data`.

        If mask[date, asset] is True, then result.loc[(date, asset), colname]
        will contain the value of data[colname][date, asset].
        """
        if not mask.any():
            # Manually handle the empty DataFrame case. This is a workaround
            # to pandas failing to tz_localize an empty dataframe with a
            # MultiIndex. It also saves us the work of applying a known-empty
            # mask to each array.
            #
            # Slicing `dates` here to preserve pandas metadata.
            empty_dates = dates[:0]
            empty_assets = array([], dtype=object)
            return DataFrame(
                data={
                    name: array([], dtype=arr.dtype)
                    for name, arr in iteritems(data)
                },
                index=MultiIndex.from_arrays([empty_dates, empty_assets]),
            )

        final_columns = {}
        for name in data:
            # Each term that computed an output has its postprocess method
            # called on the filtered result.
            #
            # As of Mon May 2 15:38:47 2016, we only use this to convert
            # LabelArrays into categoricals.
            final_columns[name] = terms[name].postprocess(data[name][mask])

        resolved_assets = array(self._finder.retrieve_all(assets))
        index = _pipeline_output_index(dates, resolved_assets, mask)

        return DataFrame(data=final_columns, index=index)

    def _resolve_hooks(self, hooks):
        if hooks is None:
            hooks = []
        return DelegatingHooks(self._default_hooks + hooks)