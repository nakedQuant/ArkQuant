class error_keywords(object):

    def __init__(self, *args, **kwargs):
        self.messages = kwargs

    def __call__(self, func):
        @wraps(func)
        def assert_keywords_and_call(*args, **kwargs):
            for field, message in iteritems(self.messages):
                if field in kwargs:
                    raise TypeError(message)
            return func(*args, **kwargs)
        return assert_keywords_and_call

class BadCallable(TypeError, AssertionError, ZiplineError):
    """
    The given callable is not structured in the expected way.
    """
    _lambda_name = (lambda: None).__name__

    def __init__(self, callable_, args, starargs, kwargs):
        self.callable_ = callable_
        self.args = args
        self.starargs = starargs
        self.kwargsname = kwargs

        self.kwargs = {}

    def format_callable(self):
        if self.callable_.__name__ == self._lambda_name:
            fmt = '%s %s'
            name = 'lambda'
        else:
            fmt = '%s(%s)'
            name = self.callable_.__name__

        return fmt % (
            name,
            ', '.join(
                chain(
                    (str(arg) for arg in self.args),
                    ('*' + sa for sa in (self.starargs,) if sa is not None),
                    ('**' + ka for ka in (self.kwargsname,) if ka is not None),
                )
            )
        )

    @property
    def msg(self):
        return str(self)