
from weakref import WeakValueDictionary
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pandas import Series
from functools import partial
import warnings

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
    _pipeline_cache = WeakValueDictionary()
    _get_loader = None

    @staticmethod
    def _resolve_domain(pipeline):
        """Resolve a concrete domain for pipeline from terms.
           domain keys --- min_extra_window term_fields
        """
        terms = pipeline._terms_store
        window_list = [t.domain['min_extra_window'] for t in terms]
        field_list = [t.domain['term_fields'] for t in terms]
        pipeline_domain = {'min_extra_window': max(window_list),'term_fields':set(field_list)}
        return pipeline_domain

    def _cache_pipeline_metadata(self,pipeline_domain,get_loader,trade_dt):
        """
            fetch data source through pipeline domain
        """
        window = pipeline_domain['min_extra_window']
        _get_loader = get_loader if get_loader else self._get_loader
        for field in pipeline_domain['term_fields']:
            self._pipeline_cache[field] = _get_loader.load_adjust_array(field,window,trade_dt)

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
        A function that is given a loadable term and returns a pipeline
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
        'pipeline',
        '_get_loader',
        'domain'
    )

    def __init__(self,
                 pipeline,
                 Domain,
                 get_loader = None):

        pipeline_domain = self._resolve_domain(pipeline)
        self._get_loader = partial(self._cache_pipeline_metadata,pipeline_domain,get_loader)
        self.pipeline = pipeline
        self.domain = Domain()
        self._populate_initial_workspace = {}

    def run_pipeline(self, start_date, end_date, hooks=None):
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
        sessions = self.domain.all_session(start_date,end_date)
        outputs = Series(index = sessions)
        with hooks.running_pipeline(start_date, end_date):
            for trade_dt in sessions:
                outputs[trade_dt] = self._run_pipeline_impl(
                    trade_dt,
                    hooks,
                )
        return outputs

    @contextmanager
    def _run_pipeline_impl(self, date, hooks):
        """Shared core for ``run_pipeline`` and ``run_chunked_pipeline``.
        """
        # See notes at the top of this module for a description of the
        # algorithm implemented here.
        self._get_loader(date)
        yield self.pipeline.to_execution_plan()
        self._pipeline_cache.clear()