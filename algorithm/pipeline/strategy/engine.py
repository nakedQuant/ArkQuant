# Initialize pipe API data.
def init_engine(self, get_loader):
    """
    Construct and store a PipelineEngine from loader.

    If get_loader is None, constructs an ExplodingPipelineEngine
    """
    if get_loader is not None:
        self.engine = SimplePipelineEngine(
            get_loader,
            self.asset_finder,
            self.default_pipeline_domain(self.trading_calendar),
        )
    else:
        self.engine = ExplodingPipelineEngine()

# Create an already-expired cache so that we compute the first time
# data is requested.
def compute_eager_pipelines(self):
    """
    Compute any pipelines attached with eager=True.
    """
    for name, pipe in self._pipelines.items():
        if pipe.eager:
            self.pipeline_output(name)

##############
# pipe API
##############
def attach_pipeline(self, pipeline, name, chunks=None, eager=True):
    """Register a Pipeline to be computed at the start of each day.

    Parameters
    ----------
    pipeline : Pipeline
        The Pipeline to have computed.
    name : str
        The name of the Pipeline.
    chunks : int or iterator, optional
        The number of days to compute Pipeline results for. Increasing
        this number will make it longer to get the first results but
        may improve the total runtime of the simulation. If an iterator
        is passed, we will run in chunks based on values of the iterator.
        Default is True.
    eager : bool, optional
        Whether or not to compute this Pipeline prior to
        before_trading_start.

    Returns
    -------
    Pipeline : Pipeline
        Returns the Pipeline that was attached unchanged.

    See Also
    --------
    :func:`zipline.api.pipeline_output`
    """
    if chunks is None:
        # Make the first chunk smaller to get more immediate results:
        # (one week, then every half year)
        chunks = chain([5], repeat(126))
    elif isinstance(chunks, int):
        chunks = repeat(chunks)

    if name in self._pipelines:
        raise DuplicatePipelineName(name=name)

    self._pipelines[name] = AttachedPipeline(pipeline, iter(chunks), eager)

    # Return the Pipeline to allow expressions like
    # p = attach_pipeline(pipe(), 'name')
    return pipeline

@api_method
@require_initialized(PipelineOutputDuringInitialize())
def pipeline_output(self, name):
    """
    Get results of the Pipeline attached by with name ``name``.

    Parameters
    ----------
    name : str
        Name of the Pipeline from which to fetch results.

    Returns
    -------
    results : pd.DataFrame
        DataFrame containing the results of the requested Pipeline for
        the current simulation date.

    Raises
    ------
    NoSuchPipeline
        Raised when no Pipeline with the name `name` has been registered.

    See Also
    --------
    :func:`zipline.api.attach_pipeline`
    :meth:`zipline.Pipeline.engine.PipelineEngine.run_pipeline`
    """
    try:
        pipe, chunks, _ = self._pipelines[name]
    except KeyError:
        raise NoSuchPipeline(
            name=name,
            valid=list(self._pipelines.keys()),
        )
    return self._pipeline_output(pipe, chunks, name)

def _pipeline_output(self, pipeline, chunks, name):
    """
    Internal implementation of `pipeline_output`.
    """
    today = normalize_date(self.get_datetime())
    try:
        data = self._pipeline_cache.get(name, today)
    except KeyError:
        # Calculate the next block.
        data, valid_until = self.run_pipeline(
            pipeline, today, next(chunks),
        )
        self._pipeline_cache.set(name, data, valid_until)

    # Now that we have a cached result, try to return the data for today.
    try:
        return data.loc[today]
    except KeyError:
        # This happens if no assets passed the Pipeline screen on a given
        # day.
        return pd.DataFrame(index=[], columns=data.columns)

def run_pipeline(self, pipeline, start_session, chunksize):
    """
    Compute `Pipeline`, providing values for at least `start_date`.

    Produces a DataFrame containing data for days between `start_date` and
    `end_date`, where `end_date` is defined by:

        `end_date = min(start_date + chunksize trading days,
                        simulation_end)`

    Returns
    -------
    (data, valid_until) : tuple (pd.DataFrame, pd.Timestamp)

    See Also
    --------
    PipelineEngine.run_pipeline
    """
    sessions = self.trading_calendar.all_sessions

    # Load data starting from the previous trading day...
    start_date_loc = sessions.get_loc(start_session)

    # ...continuing until either the day before the simulation end, or
    # until chunksize days of data have been loaded.
    sim_end_session = self.sim_params.end_session

    end_loc = min(
        start_date_loc + chunksize,
        sessions.get_loc(sim_end_session)
    )

    end_session = sessions[end_loc]

    return \
        self.engine.run_pipeline(pipeline, start_session, end_session), \
        end_session

@staticmethod
def default_pipeline_domain(calendar):
    """
    Get a default Pipeline domain for algorithms running on ``calendar``.

    This will be used to infer a domain for pipelines that only use generic
    datasets when running in the context of a TradingAlgorithm.
    """
    return _DEFAULT_DOMAINS.get(calendar.name, domain.GENERIC)