def require_not_initialized(exception):
    """
    Decorator for API methods that should only be called during or before
    TradingAlgorithm.initialize.  `exception` will be raised if the method is
    called after initialize.

    Examples
    --------
    @require_not_initialized(SomeException("Don't do that!"))
    def method(self):
        # Do stuff that should only be allowed during initialize.
    """
    def decorator(method):
        @wraps(method)
        def wrapped_method(self, *args, **kwargs):
            if self.initialized:
                raise exception
            return method(self, *args, **kwargs)
        return wrapped_method
    return decorator


def api_method(f):
    # Decorator that adds the decorated class method as a callable
    # function (wrapped) to zipline.api
    @wraps(f)
    def wrapped(*args, **kwargs):
        # Get the instance and call the method
        algo_instance = get_algo_instance()
        if algo_instance is None:
            raise RuntimeError(
                'zipline api method %s must be called during a simulation.'
                % f.__name__
            )
        return getattr(algo_instance, f.__name__)(*args, **kwargs)
    # Add functor to zipline.api
    setattr(zipline.api, f.__name__, wrapped)
    zipline.api.__all__.append(f.__name__)
    f.is_api_method = True
    return f