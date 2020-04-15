'''
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
'''

import uuid

import networkx as nx

# This sentinel value is uniquely-generated at import time so that we can
# guarantee that it never conflicts with a user-provided column name.
#
# (Yes, technically, a user can import this file and pass this as the name of a
# column. If you do that you deserve whatever bizarre failure you cause.)
SCREEN_NAME = 'screen_' + uuid.uuid4().hex

from abc import ABC
from bisect import insort
from collections import Mapping
from weakref import WeakValueDictionary
from numpy import array,dtype as dtype_class , ndarray,searchsorted

import datetime
from textwrap import dedent

from interface import default, implements, Interface
import numpy as np
import pandas as pd
import pytz

from zipline.utils.memoize import lazyval

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

    def all_assets(self,dt,category):
        """
            Get all assets
        """

    @property
    def country_code(self):
        """The country code for this domain.

        Returns
        -------
        code : str
            The two-character country iso3166 country code for this domain.
        """

    @default
    def roll_forward(self, date,window):
        """
        Given a date, align it to the calendar of the pipeline's domain.

        Parameters
        ----------
        dt : pd.Timestamp

        Returns
        -------
        pd.Timestamp
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

    def all_assets(self,dt,category= 'stock'):
        raise NotImplementedError

    @default
    def roll_forward(self, dt,window):
        dt = pd.Timestamp(dt, tz='UTC')
        trading_days = self.all_sessions()
        idx = trading_days.searchsorted(dt)
        return trading_days[idx - window ,idx]


import glob

class Term(object):
    """
        执行算法 --- 拓扑结构
        退出算法 --- 裁决模块

        因子 --- 策略 --- 执行算法
        scripts --- 策略

        numinputs
    """

    _term_cache = WeakValueDictionary
    default = None

    def __new__(cls,
                script_file,
                params,
                dtype,
                domain = None,
                window_safe = False):
        if domain is None:
            domain = GenericDomain()

        if script_file not in glob.glob('Strategy/*.py'):
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
        with open('Strategy/%s.py'%script_file,'r') as f:
            exec(f.read(),namespace)

        self._term_core = namespace[script_file]

        return self

    # def _validate(self,out):
    #
    #     if isinstance(out,(list,tuple)):
    #         return True
    #     return False
    # if self.dtype not in self.ALLOWED_DTYPES:
    #     raise UnsupportedDataType(
    #         typename=type(self).__name__,
    #         dtype=self.dtype,
    #     )

    def _validate(self):
        super(ComputableTerm,self)._validate()

        if self.inputs is NotSpecified:
            raise ValueError('')
        if not isinstance(self.domain,Domain):
            raise TypeError('')

        if self.outputs is NotSpecified:
            pass
        elif not self.outputs:
            raise ValueError('empty')
        else:
            disallowed_names = [
                attr for attr in dir(ComputableTerm)
                if not attr.startswith('_')
            ]

        for output in self.outputs:
            if output.startswith('_') or output in disallowed_names:
                raise ValueError('invalidoutputname')

        if self.window_length is NotSpecified:
            raise ValueError('windowlength error')

        if self.mask is NotSpecified:
            raise AssertionError

        if self.window_length > 1:
            for child in self.inputs:
                if not child.window_safe:
                    raise ValueError('')

    @property
    def inputs(self,input = None):

        init_input = self.domain.all_assets('stock') if input is None else input
        return init_input

    def windowed(self):
        """
            represent a trailing window computation
        :return:
        """
        return (
            self.window_length is not NotSpecified and self.window_length > 0
        )

    @property
    def dependencies(self,terms = []):

        return terms

    def _compute(self):
        """
            subclass should implement
        """
        raise NotImplemented

    def _allocate_output(self, windows, shape):
        """
        Allocate an output array whose rows should be passed to `self.compute`.

        The resulting array must have a shape of ``shape``.

        If we have standard outputs (i.e. self.outputs is NotSpecified), the
        default is an empty ndarray whose dtype is ``self.dtype``.

        If we have an outputs tuple, the default is an empty recarray with
        ``self.outputs`` as field names. Each field will have dtype
        ``self.dtype``.

        This can be overridden to control the kind of array constructed
        (e.g. to produce a LabelArray instead of an ndarray).
        """
        missing_value = self.missing_value
        outputs = self.outputs
        if outputs is not NotSpecified:
            out = recarray(
                shape,
                formats=[self.dtype.str] * len(outputs),
                names=outputs,
            )
            out[:] = missing_value
        else:
            out = full(shape, missing_value, dtype=self.dtype)
        return out

    def _format_inputs(self, windows, column_mask):
        inputs = []
        for input_ in windows:
            window = next(input_)
            if window.shape[1] == 1:
                # Do not mask single-column inputs.
                inputs.append(window)
            else:
                inputs.append(window[:, column_mask])
        return inputs

    def compute_extra_rows(self,
                           all_dates,
                           start_date,
                           end_date,
                           min_extra_rows):
        """
        Ensure that min_extra_rows pushes us back to a computation date.

        Parameters
        ----------
        all_dates : pd.DatetimeIndex
            The trading sessions against which ``self`` will be computed.
        start_date : pd.Timestamp
            The first date for which final output is requested.
        end_date : pd.Timestamp
            The last date for which final output is requested.
        min_extra_rows : int
            The minimum number of extra rows required of ``self``, as
            determined by other terms that depend on ``self``.

        Returns
        -------
        extra_rows : int
            The number of extra rows to compute.  This will be the minimum
            number of rows required to make our computed start_date fall on a
            recomputation date.
        """
        try:
            current_start_pos = all_dates.get_loc(start_date) - min_extra_rows
            if current_start_pos < 0:
                raise NoFurtherDataError.from_lookback_window(
                    initial_message="Insufficient data to compute Pipeline:",
                    first_date=all_dates[0],
                    lookback_start=start_date,
                    lookback_length=min_extra_rows,
                )
        except KeyError:
            before, after = nearest_unequal_elements(all_dates, start_date)
            raise ValueError(
                "Pipeline start_date {start_date} is not in calendar.\n"
                "Latest date before start_date is {before}.\n"
                "Earliest date after start_date is {after}.".format(
                    start_date=start_date,
                    before=before,
                    after=after,
                )
            )

        # Our possible target dates are all the dates on or before the current
        # starting position.
        # TODO: Consider bounding this below by self.window_length
        candidates = all_dates[:current_start_pos + 1]

        # Choose the latest date in the candidates that is the start of a new
        # period at our frequency.
        choices = select_sampling_indices(candidates, self._frequency)

        # If we have choices, the last choice is the first date if the
        # period containing current_start_date.  Choose it.
        new_start_date = candidates[choices[-1]]

        # Add the difference between the new and old start dates to get the
        # number of rows for the new start_date.
        new_start_pos = all_dates.get_loc(new_start_date)
        assert new_start_pos <= current_start_pos, \
            "Computed negative extra rows!"

        return min_extra_rows + (current_start_pos - new_start_pos)





from numpy import (
    array,
    full,
    recarray,
    vstack,
)
from pandas import NaT as pd_NaT

"""
Computation engines for executing Pipelines.

This module defines the core computation algorithms for executing Pipelines.

The primary entrypoint of this file is SimplePipelineEngine.run_pipeline, which
implements the following algorithm for executing pipelines:

1. Determine the domain of the pipeline. The domain determines the
   top-level set of dates and assets that serve as row- and
   column-labels for the computations performed by this
   pipeline. This logic lives in
   zipline.pipeline.domain.infer_domain.

2. Build a dependency graph of all terms in `pipeline`, with
   information about how many extra rows each term needs from its
   inputs. At this point we also **specialize** any generic
   LoadableTerms to the domain determined in (1). This logic lives in
   zipline.pipeline.graph.TermGraph and
   zipline.pipeline.graph.ExecutionPlan.

3. Combine the domain computed in (2) with our AssetFinder to produce
   a "lifetimes matrix". The lifetimes matrix is a DataFrame of
   booleans whose labels are dates x assets. Each entry corresponds
   to a (date, asset) pair and indicates whether the asset in
   question was tradable on the date in question. This logic
   primarily lives in AssetFinder.lifetimes.

4. Call self._populate_initial_workspace, which produces a
   "workspace" dictionary containing cached or otherwise pre-computed
   terms. By default, the initial workspace contains the lifetimes
   matrix and its date labels.

5. Topologically sort the graph constructed in (1) to produce an
   execution order for any terms that were not pre-populated.  This
   logic lives in TermGraph.

6. Iterate over the terms in the order computed in (5). For each term:

   a. Fetch the term's inputs from the workspace, possibly removing
      unneeded leading rows from the input (see ExecutionPlan.offset
      for details on why we might have extra leading rows).

   b. Call ``term._compute`` with the inputs. Store the results into
      the workspace.

   c. Decrement "reference counts" on the term's inputs, and remove
      their results from the workspace if the refcount hits 0. This
      significantly reduces the maximum amount of memory that we
      consume during execution

   This logic lives in SimplePipelineEngine.compute_chunk.

7. Extract the pipeline's outputs from the workspace and convert them
   into "narrow" format, with output labels dictated by the Pipeline's
   screen. This logic lives in SimplePipelineEngine._to_narrow.
"""
from abc import ABCMeta, abstractmethod
from functools import partial

from six import iteritems, with_metaclass, viewkeys
from numpy import array, arange
from pandas import DataFrame, MultiIndex
from toolz import groupby


class PipelineEngine(with_metaclass(ABCMeta)):

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

    @abstractmethod
    def run_chunked_pipeline(self,
                             pipeline,
                             start_date,
                             end_date,
                             chunksize,
                             hooks=None):
        """
        Compute values for ``pipeline`` from ``start_date`` to ``end_date``, in
        date chunks of size ``chunksize``.

        Chunked execution reduces memory consumption, and may reduce
        computation time depending on the contents of your pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to run.
        start_date : pd.Timestamp
            The start date to run the pipeline for.
        end_date : pd.Timestamp
            The end date to run the pipeline for.
        chunksize : int
            The number of days to execute at a time.
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

        See Also
        --------
        :meth:`zipline.pipeline.engine.PipelineEngine.run_pipeline`
        """
        raise NotImplementedError("run_chunked_pipeline")


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
        '_finder',
        '_root_mask_term',
        '_root_mask_dates_term',
        '_populate_initial_workspace',
    )

    @expect_types(
        default_domain=Domain,
        __funcname='SimplePipelineEngine',
    )
    def __init__(self,
                 get_loader,
                 asset_finder,
                 default_domain=GENERIC,
                 populate_initial_workspace=None,
                 default_hooks=None):

        self._get_loader = get_loader
        self._finder = asset_finder

        self._root_mask_term = AssetExists()
        self._root_mask_dates_term = InputDates()

        self._populate_initial_workspace = (
            populate_initial_workspace or default_populate_initial_workspace
        )
        self._default_domain = default_domain

        if default_hooks is None:
            self._default_hooks = []
        else:
            self._default_hooks = list(default_hooks)

    def run_chunked_pipeline(self,
                             pipeline,
                             start_date,
                             end_date,
                             chunksize,
                             hooks=None):
        """
        Compute values for ``pipeline`` from ``start_date`` to ``end_date``, in
        date chunks of size ``chunksize``.

        Chunked execution reduces memory consumption, and may reduce
        computation time depending on the contents of your pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to run.
        start_date : pd.Timestamp
            The start date to run the pipeline for.
        end_date : pd.Timestamp
            The end date to run the pipeline for.
        chunksize : int
            The number of days to execute at a time.
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

        See Also
        --------
        :meth:`zipline.pipeline.engine.PipelineEngine.run_pipeline`
        """
        domain = self.resolve_domain(pipeline)
        ranges = compute_date_range_chunks(
            domain.all_sessions(),
            start_date,
            end_date,
            chunksize,
        )
        hooks = self._resolve_hooks(hooks)

        run_pipeline = partial(self._run_pipeline_impl, pipeline, hooks=hooks)
        with hooks.running_pipeline(pipeline, start_date, end_date):
            chunks = [run_pipeline(s, e) for s, e in ranges]

        if len(chunks) == 1:
            # OPTIMIZATION: Don't make an extra copy in `categorical_df_concat`
            # if we don't have to.
            return chunks[0]

        return categorical_df_concat(chunks, inplace=True)

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

    def _compute_root_mask(self, domain, start_date, end_date, extra_rows):
        """
        Compute a lifetimes matrix from our AssetFinder, then drop columns that
        didn't exist at all during the query dates.

        Parameters
        ----------
        domain : zipline.pipeline.domain.Domain
            Domain for which we're computing a pipeline.
        start_date : pd.Timestamp
            Base start date for the matrix.
        end_date : pd.Timestamp
            End date for the matrix.
        extra_rows : int
            Number of extra rows to compute before `start_date`.
            Extra rows are needed by terms like moving averages that require a
            trailing window of data.

        Returns
        -------
        lifetimes : pd.DataFrame
            Frame of dtype `bool` containing dates from `extra_rows` days
            before `start_date`, continuing through to `end_date`.  The
            returned frame contains as columns all assets in our AssetFinder
            that existed for at least one day between `start_date` and
            `end_date`.
        """
        sessions = domain.all_sessions()

        if start_date not in sessions:
            raise ValueError(
                "Pipeline start date ({}) is not a trading session for "
                "domain {}.".format(start_date, domain)
            )

        elif end_date not in sessions:
            raise ValueError(
                "Pipeline end date {} is not a trading session for "
                "domain {}.".format(end_date, domain)
            )

        start_idx, end_idx = sessions.slice_locs(start_date, end_date)
        if start_idx < extra_rows:
            raise NoFurtherDataError.from_lookback_window(
                initial_message="Insufficient data to compute Pipeline:",
                first_date=sessions[0],
                lookback_start=start_date,
                lookback_length=extra_rows,
            )

        # NOTE: This logic should probably be delegated to the domain once we
        #       start adding more complex domains.
        #
        # Build lifetimes matrix reaching back to `extra_rows` days before
        # `start_date.`
        finder = self._finder
        lifetimes = finder.lifetimes(
            sessions[start_idx - extra_rows:end_idx],
            include_start_date=False,
            country_codes=(domain.country_code,),
        )

        if not lifetimes.columns.unique:
            columns = lifetimes.columns
            duplicated = columns[columns.duplicated()].unique()
            raise AssertionError("Duplicated sids: %d" % duplicated)

        # Filter out columns that didn't exist from the farthest look back
        # window through the end of the requested dates.
        existed = lifetimes.any()
        ret = lifetimes.loc[:, existed]
        num_assets = ret.shape[1]

        if num_assets == 0:
            raise ValueError(
                "Failed to find any assets with country_code {!r} that traded "
                "between {} and {}.\n"
                "This probably means that your asset db is old or that it has "
                "incorrect country/exchange metadata.".format(
                    domain.country_code, start_date, end_date,
                )
            )

        return ret

    @staticmethod
    def _inputs_for_term(term, workspace, graph, domain, refcounts):
        """
        Compute inputs for the given term.

        This is mostly complicated by the fact that for each input we store as
        many rows as will be necessary to serve **any** computation requiring
        that input.
        """
        offsets = graph.offset
        out = []

        # We need to specialize here because we don't change ComputableTerm
        # after resolving domains, so they can still contain generic terms as
        # inputs.
        specialized = [maybe_specialize(t, domain) for t in term.inputs]

        if term.windowed:
            # If term is windowed, then all input data should be instances of
            # AdjustedArray.
            for input_ in specialized:
                adjusted_array = ensure_adjusted_array(
                    workspace[input_], input_.missing_value,
                )
                out.append(
                    adjusted_array.traverse(
                        window_length=term.window_length,
                        offset=offsets[term, input_],
                        # If the refcount for the input is > 1, we will need
                        # to traverse this array again so we must copy.
                        # If the refcount for the input == 0, this is the last
                        # traversal that will happen so we can invalidate
                        # the AdjustedArray and mutate the data in place.
                        copy=refcounts[input_] > 1,
                    )
                )
        else:
            # If term is not windowed, input_data may be an AdjustedArray or
            # np.ndarray. Coerce the former to the latter.
            for input_ in specialized:
                input_data = ensure_ndarray(workspace[input_])
                offset = offsets[term, input_]
                # OPTIMIZATION: Don't make a copy by doing input_data[0:] if
                # offset is zero.
                if offset:
                    input_data = input_data[offset:]
                out.append(input_data)
        return out

    def compute_chunk(self,
                      graph,
                      dates,
                      sids,
                      workspace,
                      refcounts,
                      execution_order,
                      hooks):
        """
        Compute the Pipeline terms in the graph for the requested start and end
        dates.

        This is where we do the actual work of running a pipeline.

        Parameters
        ----------
        graph : zipline.pipeline.graph.ExecutionPlan
            Dependency graph of the terms to be executed.
        dates : pd.DatetimeIndex
            Row labels for our root mask.
        sids : pd.Int64Index
            Column labels for our root mask.
        workspace : dict
            Map from term -> output.
            Must contain at least entry for `self._root_mask_term` whose shape
            is `(len(dates), len(assets))`, but may contain additional
            pre-computed terms for testing or optimization purposes.
        refcounts : dict[Term, int]
            Dictionary mapping terms to number of dependent terms. When a
            term's refcount hits 0, it can be safely discarded from
            ``workspace``. See TermGraph.decref_dependencies for more info.
        execution_order : list[Term]
            Order in which to execute terms.
        hooks : implements(PipelineHooks)
            Hooks to instrument pipeline execution.

        Returns
        -------
        results : dict
            Dictionary mapping requested results to outputs.
        """
        self._validate_compute_chunk_params(graph, dates, sids, workspace)

        get_loader = self._get_loader

        # Copy the supplied initial workspace so we don't mutate it in place.
        workspace = workspace.copy()
        domain = graph.domain

        # Many loaders can fetch data more efficiently if we ask them to
        # retrieve all their inputs at once. For example, a loader backed by a
        # SQL database can fetch multiple columns from the database in a single
        # query.
        #
        # To enable these loaders to fetch their data efficiently, we group
        # together requests for LoadableTerms if they are provided by the same
        # loader and they require the same number of extra rows.
        #
        # The extra rows condition is a simplification: we don't currently have
        # a mechanism for asking a loader to fetch different windows of data
        # for different terms, so we only batch requests together when they're
        # going to produce data for the same set of dates. That may change in
        # the future if we find a loader that can still benefit significantly
        # from batching unequal-length requests.
        def loader_group_key(term):
            loader = get_loader(term)
            extra_rows = graph.extra_rows[term]
            return loader, extra_rows

        # Only produce loader groups for the terms we expect to load.  This
        # ensures that we can run pipelines for graphs where we don't have a
        # loader registered for an atomic term if all the dependencies of that
        # term were supplied in the initial workspace.
        will_be_loaded = graph.loadable_terms - viewkeys(workspace)
        loader_groups = groupby(
            loader_group_key,
            (t for t in execution_order if t in will_be_loaded),
        )

        for term in execution_order:
            # `term` may have been supplied in `initial_workspace`, or we may
            # have loaded `term` as part of a batch with another term coming
            # from the same loader (see note on loader_group_key above). In
            # either case, we already have the term computed, so don't
            # recompute.
            if term in workspace:
                continue

            # Asset labels are always the same, but date labels vary by how
            # many extra rows are needed.
            mask, mask_dates = graph.mask_and_dates_for_term(
                term,
                self._root_mask_term,
                workspace,
                dates,
            )

            if isinstance(term, LoadableTerm):
                loader = get_loader(term)
                to_load = sorted(
                    loader_groups[loader_group_key(term)],
                    key=lambda t: t.dataset
                )
                with hooks.loading_terms(to_load):
                    loaded = loader.load_adjusted_array(
                        domain, to_load, mask_dates, sids, mask,
                    )
                assert set(loaded) == set(to_load), (
                    'loader did not return an AdjustedArray for each column\n'
                    'expected: %r\n'
                    'got:      %r' % (sorted(to_load), sorted(loaded))
                )
                workspace.update(loaded)
            else:
                with hooks.computing_term(term):
                    workspace[term] = term._compute(
                        self._inputs_for_term(
                            term,
                            workspace,
                            graph,
                            domain,
                            refcounts,
                        ),
                        mask_dates,
                        sids,
                        mask,
                    )
                if term.ndim == 2:
                    assert workspace[term].shape == mask.shape
                else:
                    assert workspace[term].shape == (mask.shape[0], 1)

                # Decref dependencies of ``term``, and clear any terms
                # whose refcounts hit 0.
                for garbage in graph.decref_dependencies(term, refcounts):
                    del workspace[garbage]

        # At this point, all the output terms are in the workspace.
        out = {}
        graph_extra_rows = graph.extra_rows
        for name, term in iteritems(graph.outputs):
            # Truncate off extra rows from outputs.
            out[name] = workspace[term][graph_extra_rows[term]:]
        return out

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

    def _validate_compute_chunk_params(self,
                                       graph,
                                       dates,
                                       sids,
                                       initial_workspace):
        """
        Verify that the values passed to compute_chunk are well-formed.
        """
        root = self._root_mask_term
        clsname = type(self).__name__

        # Writing this out explicitly so this errors in testing if we change
        # the name without updating this line.
        compute_chunk_name = self.compute_chunk.__name__
        if root not in initial_workspace:
            raise AssertionError(
                "root_mask values not supplied to {cls}.{method}".format(
                    cls=clsname,
                    method=compute_chunk_name,
                )
            )

        shape = initial_workspace[root].shape
        implied_shape = len(dates), len(sids)
        if shape != implied_shape:
            raise AssertionError(
                "root_mask shape is {shape}, but received dates/assets "
                "imply that shape should be {implied}".format(
                    shape=shape,
                    implied=implied_shape,
                )
            )

        for term in initial_workspace:
            if self._is_special_root_term(term):
                continue

            if term.domain is GENERIC:
                # XXX: We really shouldn't allow **any** generic terms to be
                # populated in the initial workspace. A generic term, by
                # definition, can't correspond to concrete data until it's
                # paired with a domain, and populate_initial_workspace isn't
                # given the domain of execution, so it can't possibly know what
                # data to use when populating a generic term.
                #
                # In our current implementation, however, we don't have a good
                # way to represent specializations of ComputableTerms that take
                # only generic inputs, so there's no good way for the initial
                # workspace to provide data for such terms except by populating
                # the generic ComputableTerm.
                #
                # The right fix for the above is to implement "full
                # specialization", i.e., implementing ``specialize`` uniformly
                # across all terms, not just LoadableTerms. Having full
                # specialization will also remove the need for all of the
                # remaining ``maybe_specialize`` calls floating around in this
                # file.
                #
                # In the meantime, disallowing ComputableTerms in the initial
                # workspace would break almost every test in
                # `test_filter`/`test_factor`/`test_classifier`, and fixing
                # them would require updating all those tests to compute with
                # more specialized terms. Once we have full specialization, we
                # can fix all the tests without a large volume of edits by
                # simply specializing their workspaces, so for now I'm leaving
                # this in place as a somewhat sharp edge.
                if isinstance(term, LoadableTerm):
                    raise ValueError(
                        "Loadable workspace terms must be specialized to a "
                        "domain, but got generic term {}".format(term)
                    )

            elif term.domain != graph.domain:
                raise ValueError(
                    "Initial workspace term {} has domain {}. "
                    "Does not match pipeline domain {}".format(
                        term, term.domain, graph.domain,
                    )
                )

    def resolve_domain(self, pipeline):
        """Resolve a concrete domain for ``pipeline``.
        """
        domain = pipeline.domain(default=self._default_domain)
        if domain is GENERIC:
            raise ValueError(
                "Unable to determine domain for Pipeline.\n"
                "Pass domain=<desired domain> to your Pipeline to set a "
                "domain."
            )
        return domain

    def _is_special_root_term(self, term):
        return (
            term is self._root_mask_term
            or term is self._root_mask_dates_term
        )

    def _resolve_hooks(self, hooks):
        if hooks is None:
            hooks = []
        return DelegatingHooks(self._default_hooks + hooks)


def _pipeline_output_index(dates, assets, mask):
    """
    Create a MultiIndex for a pipeline output.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Row labels for ``mask``.
    assets : pd.Index
        Column labels for ``mask``.
    mask : np.ndarray[bool]
        Mask array indicating date/asset pairs that should be included in
        output index.

    Returns
    -------
    index : pd.MultiIndex
        MultiIndex  containing (date,  asset) pairs  corresponding to  ``True``
        values in ``mask``.
    """
    date_labels = repeat_last_axis(arange(len(dates)), len(assets))[mask]
    asset_labels = repeat_first_axis(arange(len(assets)), len(dates))[mask]
    return MultiIndex(
        levels=[dates, assets],
        labels=[date_labels, asset_labels],
        # TODO: We should probably add names for these.
        names=[None, None],
        verify_integrity=False,
    )

# -*- coding:utf-8 -*-

import importlib,numpy as np
from abc import ABC,abstractmethod
from joblib import Parallel,delayed,Memory
from functools import reduce

from Tool.Wrapper import _validate_type

__all__ = ['Ump','Pipeline', 'FeatureUnion']


class _BaseComposition(ABC):
    """
        将不同的算法通过串行或者并行方式形成算法工厂 ，筛选过滤最终得出目标目标标的组合
        串行：
            1、串行特征工厂借鉴zipline或者scikit_learn Pipeline
            2、串行理论基础：现行的策略大多数基于串行，比如多头排列、空头排列、行业龙头战法、统计指标筛选
            3、缺点：确定存在主观去判断特征顺序，前提对于市场有一套自己的认识以及分析方法
        并行：
            1、并行的理论基础借鉴交集理论
            2、基于结果反向分类strategy
        难点：
            不同算法的权重分配
        input : stategies ,output : list or tuple of filtered assets
    """
    @classmethod
    def _load_from_name(cls,name):
        """Generate names for estimators, if it is instance(already initialized) just return,else return the class  """
        try:
            strat = importlib.__import__(name, 'Algorithm.Strategy')
        except:
            raise ValueError('some of features not implemented')
        return strat

    def _validate_steps(self,steps):
        for item in steps:
            if not hasattr(item, 'fit'):
                raise TypeError('all steps must have calc_feature method')

    @abstractmethod
    def _set_params(self,**params):
        '''
        For this, it enables setting parameters of the various steps using their names and
        params : dict of string -> object
        '''
        raise NotImplemented

    def set_params(self,**kwargs):
        """Set the parameters of estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self._set_params(**kwargs)
        return self

    def register(self, strategy):
        if strategy not in self._n_features:
            self._n_features.append(strategy)

    def unregister(self, feature):
        if feature in self._n_features:
            self._n_features.remove(feature)
        else:
            raise ValueError('特征没有注册')

    @abstractmethod
    def _fit(self,step,res):
        """
        run algorithm  it means already has instance ;else it need to initialize
        """
        raise NotImplemented

    @abstractmethod
    def decision_function(self):
        """Apply transforms, and decision_function of the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the Pipeline.
        Returns
        -------
        output array_like ,stockCode list
        """
        pass


class Pipeline(_BaseComposition):
    """
        pipe of strategy to fit targeted asset
        Parameters
        -----------
        steps :list
            List of strategies
            wgts: List,str or list , default : 'average'
        wgts: List
            List of (name,weight) tuples that allocate the weight of steps means the
            importance, average wgts avoids the unbalance of steps
        memory : joblib.Memory interface used to cache the fitted transformers of
            the Pipeline. By default,no caching is performed. If a string is given,
            it is the path to the caching directory. Enabling caching triggers a clone
            of the transformers before fitting.Caching the transformers is advantageous
            when fitting is time consuming.
    """
    _required_parameters = ['steps']

    def __init__(self,steps,memory = None):
        super()._validate_steps(steps)
        self.steps = steps
        self.cachedir = memory
        self._pipe_params=dict()

    def __len__(self):
        '''
        return the length of Pipeline
        '''
        return len(self.steps)

    def _set_params(self,**params):
        '''
        For this, it enables setting parameters of the various steps using their names and
        params : dict of string -> object
        '''
        for pname ,pval in params.items():
            self._pipeline_params[pname] = pval
        if len(self._pipeline_params) != len(self.steps):
            raise ValueError('all strategies must have params to initialize')

    def register(self, strategy):
        if strategy not in self.steps:
            self.steps.append(strategy)
        else:
            raise ValueError('%s already registered in Pipeline'%strategy)

    def unregister(self, strategy):
        if strategy in self.steps:
            self.steps.remove(strategy)
        else:
            raise ValueError('%s has not registered in Pipeline'%strategy)

    def _fit(self,step) -> list:
        """
        run algorithm ,if param is passthrough , it means already has instance ;else it need to initialize
        """
        strategy = self._load_from_name(step)
        res = strategy(self._self._pipeline_params[step]).run()
        return res

    # Estimator interface
    def _fit_cache(self, step:str,res:list):
        # Setup the memory
        memory = Memory(self.cachedir)
        if hasattr(memory, 'location'):
            if memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                print('no caching is done and the memory object is completely transparent')
        # memory the function
        fit_cached = memory.cache(self._fit)
        # Fit or load from cache the current transfomer.This is necessary when loading the transformer
        out = fit_cached(input,step,res)
        return out

    @_validate_type(_type=(list,tuple))
    def decision_function(self,portfilio:list):
        """
        Based on the steps of algorithm ,we can conclude to predict target assetCode.
        Apply transforms, and predict_proba | predict_log_proba of the final estimator
        If parallel is False(Pipeline),apply all the estimator sequentially by data,then
        predict the target
        """
        for idx,name in enumerate(self.steps):
            portfilio = self._fit_cache(name,portfilio)
        return portfilio


class FeatureUnion(_BaseComposition):
    """
    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters
    ----------
    transformer_list : List of transformer objects to be applied to the data
    n_jobs : int --- Number of jobs to run in parallel,
            -1 means using all processors.`
    allocation: str(default=average) ,dict , callable

    Examples:
        FeatureUnion(_name_estimators(transformers),weight = weight)
    """
    _required_parameters = ["transformer_list"]

    def __init__(self, transformer_list,mode='average'):
        super()._validate_steps(transformer_list)
        self.transformer_list = transformer_list
        self._n_jobs = len(transformer_list)
        self._feature_weight(mode)

    @property
    def _feature_weight(self,mode):
        self.transformer_allocation = dict()
        if not isinstance(mode,(dict , str)):
            raise TypeError('unidentified type')
        elif  isinstance(mode, str) and mode == 'average':
            wgt = 1/len(self.transformer_list)
            self.transformer_allocation = {{name:wgt} for name in self.transformer_list}
        else:
            self.transformer_allocation = mode

    def _set_params(self,**params):
        '''
        For this, it enables setting parameters of the various steps using their names and
        params : dict of string -> object
        '''
        for pname ,pval in params.items():
            self._featureUnion_params[pname] = pval
        if len(self.self._featureUnion_param) != len(self.transformer_list):
            raise ValueError('all  strategies must have params to be initialized')

    def register(self, strategy):
        if strategy not in self.transformer_list:
            self.transformer_list.append(strategy)
        else:
            raise ValueError('%s already registered in featureUnion'%strategy)

    def unregister(self, strategy):
        if strategy in self.transformer_list:
            self.transformer_list.remove(strategy)
        else:
            raise ValueError('%s has not registered in featureUnion'%strategy)

    def _parallel_func(self,porfilio):
        """Runs func in parallel on X and y"""
        return Parallel(n_jobs = self._n_jobs)(delayed(self._fit(name,porfilio)) for name in self.transformer_list)

    def _fit(self,name):
        strategy = self._load_from_name(name)
        outcome = strategy(self._featureUnion_params[name]).run()
        ordered = self._fit_score(name,outcome)
        return (ordered,outcome)

    def _fit_score(self,idx,res):
        """
            Apply score estimator with output
            input : list of ordered assets
            output :
        """
        align = pd.DataFrame(list(range(1, len(res) + 1)), index=res, columns=[idx])
        align_rank = align.rank() * self._feature_weight[idx]
        return align_rank

    def _fit_score_union(self,assets):
        '''
            two action judge if over half of tranformer_list has nothing in common ,that is means union is empty,
            the selection of tranformer_list is not appropriate ,switch to _update_tranformer_list,del tranformer which
            has not intersection with union
        '''
        score_union = pd.DataFrame()
        def union(x,y):
            internal = set(x) | set(y)
            return internal

        r =  self._parallel_func(assets)
        aligning,res = zip(* r)
        intersection = reduce(union,res)
        score_union = [score_union.append(item) for item in aligning]
        return intersection,score_union.sum(axis =1)

    @_validate_type(_type=(list,tuple))
    def decision_function(self):
        assets,scores = self._fit_score_union()
        if not len(targets):
            raise ValueError('union set is empty means strategies need to be modified --- args or change strategy')
        sorted_assets = scores.loc[assets].sort_values(ascending = False)
        return list(sorted_assets.index)


class Ump(_BaseComposition):
    """
        裁决模块 基于有效的特征集，针对特定的asset进行投票抉择
        关于仲裁逻辑：
            普通选股：针对备选池进行选股，迭代初始选股序列，在迭代中再迭代选股因子，选股因子决定是否对
            symbol投出反对票，一旦一个因子投出反对票，即筛出序列
    """

    def __init__(self,poll_workers,thres = 0.8):
        super()._validate_steps(poll_workers)
        self.voters = poll_workers
        self._poll_picker = dict()
        self.threshold = thres

    def _set_params(self,**params):
        for pname ,pval in params.items():
            self._poll_picker[pname] = pval


    def poll_pick(self,res,v):
        """
           vote for feature and quantity the vote action
           simple poll_pick --- calculate rank pct
           return bool
        """
        formatting = pd.Series(range(1,len(res)+1),index = res)
        pct_rank = formatting.rank(pct = True)
        polling = True if pct_rank[v] > self.thres else False
        return polling

    def _fit(self,worker,target):
        '''因子对象针对每一个交易目标的投票结果'''
        picker = super()._load_from_name(worker)
        fit_result = picker(self._poll_picker[worker]).fit()
        poll = self.poll_pick(fit_result,target)
        return poll

    def decision_function(self,asset):
        vote_poll = dict()
        for picker in self.voters:
            vote_poll.update({picker:self._fit(picker,asset)})
        decision = np.sum(list(vote_poll.values))/len(vote_poll)
        return decision

import re
from itertools import chain
from numbers import Number
import numexpr
from numexpr.neconpiler import getExprNames
from numpy import full,inf

#左边
ops_to_methods = {
    '+':'__add__',
    '-':'__sub__',
    '*':'__mul__',
    '/':'__div__',
    '%':'__mod__',
    '**':'__pow__',
    '&':'__and__',
    '|':'__or__',
    '^':'__xor__',
    '<':'__lt__',
    '<=':'__le__',
    '>':'__gt__',
    '>=':'__ge__',
    '==':'__eq__',
    '!=':'__ne__'
}

#右边
ops_to_commuted_methods = {
    '+':'__radd__',
    '-':'__rsub__',
    '*':'__rmul__',
    '/':'__rdiv__',
    '%':'__rmod__',
    '**':'__rpow__',
    '&':'__rand__',
    '|':'__ror__',
    '^':'__rxor__',
    '<':'__gt__',
    '<=':'__ge__',
    '>':'__lt__',
    '>=':'__le__',
    '==':'__eq__',
    '!=':'__ne__'
}

unary_ops_to_methods = {'-':'__neg__','~':'__invert__'}

_Variable_Name_re = re.compile('^(x_)([0-9]*)$')

class NumericalExpression(ComputableTerm):
    """
        term binding to a numexpr expression
    """
    window_length = 0

    def __new__(cls,expr,binds,dtype):

        window_safe = (dtype == bool_type) or all(t.window_safe for t in binds)

        return super(NumericalExpression,cls).__new__(
            cls,
            inputs = binds,
            expr = expr,
            dtype = dtype,
            window_safe = window_safe
        )

    def _init(self,expr,*args,**kwargs):
        self._expr = expr
        return super(NumericalExpression,self)._init(*args,**kwargs)

    def _validate(self):
        variable_names ,_unused = getExprNames(self._expr,{})
        expr_indices = []
        for name in variable_names:
            if name == 'inf':
                pass
            match = _Variable_Name_re.match(name)

            expr_indices.append(int(match.group(2)))

        expr_indices.sort()
        expected_indices = list(range(len(self.inputs)))

        if expr_indices != expected_indices:
            raise ValueError('')

        super(NumericalExpression,self)._validate()

    def _compute(self,arrays,dates,assets,mask):
        """
            compute our stored expression string with numexpr
        """
        out = full(mask.shape,self.missing_value,dtype = self.dtype)
        numexpr.evaluate(self._expr,
                         local_dict = {'x_%d'% idx : array for idx,array in enumerate(arrays)},
                         global_dict = {'inf':inf},
                         out = out)
        return out

    def _rebind_variables(self,new_inputs):
        """
            根据inputs修改
        """
        expr = self._expr

        for idx ,input_ in reversed(list(enumerate(self.inputs))):
            old_varname = 'x_%d'%idx
            temp_new_varname = 'x_temp_%d'%new_inputs.index(input_)
            expr = expr.replace(old_varname,temp_new_varname)
        return expr.replace('_temp_','_')

    def _merge_expression(self,other):
        new_inputs = tuple(set(self.inputs).union(other,inputs))
        new_self_expr = self._rebind_variables(new_inputs)
        new_other_expr = other._rebind_variable(new_inputs)
        return new_self_expr,new_other_expr,new_inputs

    def bulid_binary_op(self,op,other):
        if isinstance(other,NumericalExpression):
            self._expr , other_expr,new_inputs = self._merge_expression(other)
        elif isinstance(other,Term):
            self_expr = self._expr
            new_inputs ,other_idx = _ensure_element(self.inputs,other)
            other_expr = 'x_%d'%other_idx
        elif isinstance(other,Number):
            self_expr = self._expr
            other_expr = str(other)
            new_inputs = self.inputs
        else:
            raise BadBinaryOperate(op,other)
        return self_expr,other_expr,new_inputsa

    @property
    def bindings(self):
        return {
            'x_%d'% i : input_
            for i ,input_ in enumerate(self.inputs)
        }

