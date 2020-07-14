
@object.__new__
class nop_context(object):
    """A nop context manager.
    """
    def __enter__(self):
        pass

    def __exit__(self, *excinfo):
        pass


def _nop(*args, **kwargs):
    pass


class CallbackManager(object):
    """Create a context manager from a pre-execution callback and a
    post-execution callback.

    Parameters
    ----------
    pre : (...) -> any, optional
        A pre-execution callback. This will be passed ``*args`` and
        ``**kwargs``.
    post : (...) -> any, optional
        A post-execution callback. This will be passed ``*args`` and
        ``**kwargs``.
    """
    def __init__(self, pre=None, post=None):
        self.pre = pre if pre is not None else _nop
        self.post = post if post is not None else _nop

    def __call__(self, *args, **kwargs):
        return _ManagedCallbackContext(self.pre, self.post, args, kwargs)

    # special case, if no extra args are passed make this a context manager
    # which forwards no args to pre and post
    def __enter__(self):
        return self.pre()

    def __exit__(self, *excinfo):
        self.post()


class _ManagedCallbackContext(object):
    def __init__(self, pre, post, args, kwargs):
        self._pre = pre
        self._post = post
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        return self._pre(*self._args, **self._kwargs)

    def __exit__(self, *excinfo):
        self._post(*self._args, **self._kwargs)
