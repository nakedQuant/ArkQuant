
def coerce(from_, to, **to_kwargs):
    """
    A preprocessing decorator that coerces inputs of a given type by passing
    them to a callable.

    Parameters
    ----------
    from : type or tuple or types
        Inputs types on which to call ``to``.
    to : function
        Coercion function to call on inputs.
    **to_kwargs
        Additional keywords to forward to every call to ``to``.

    Examples
    --------
    >>> @preprocess(x=coerce(float, int), y=coerce(float, int))
    ... def floordiff(x, y):
    ...     return x - y
    ...
    >>> floordiff(3.2, 2.5)
    1

    >>> @preprocess(x=coerce(str, int, base=2), y=coerce(str, int, base=2))
    ... def add_binary_strings(x, y):
    ...     return bin(x + y)[2:]
    ...
    >>> add_binary_strings('101', '001')
    '110'
    """
    def preprocessor(func, argname, arg):
        if isinstance(arg, from_):
            return to(arg, **to_kwargs)
        return arg
    return preprocessor

def expect_types(__funcname=_qualified_name, **named):
    """
    Preprocessing decorator that verifies inputs have expected types.

    Examples
    --------
    >>> @expect_types(x=int, y=str)
    ... def foo(x, y):
    ...    return x, y
    ...
    >>> foo(2, '3')
    (2, '3')
    >>> foo(2.0, '3')  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    TypeError: ...foo() expected a value of type int for argument 'x',
    but got float instead.

    Notes
    -----
    A special argument, __funcname, can be provided as a string to override the
    function name shown in error messages.  This is most often used on __init__
    or __new__ methods to make errors refer to the class name instead of the
    function name.
    """
    for name, type_ in iteritems(named):
        if not isinstance(type_, (type, tuple)):
            raise TypeError(
                "expect_types() expected a type or tuple of types for "
                "argument '{name}', but got {type_} instead.".format(
                    name=name, type_=type_,
                )
            )

    def _expect_type(type_):
        # Slightly different messages for type and tuple of types.
        _template = (
            "%(funcname)s() expected a value of type {type_or_types} "
            "for argument '%(argname)s', but got %(actual)s instead."
        )
        if isinstance(type_, tuple):
            template = _template.format(
                type_or_types=' or '.join(map(_qualified_name, type_))
            )
        else:
            template = _template.format(type_or_types=_qualified_name(type_))

        return make_check(
            exc_type=TypeError,
            template=template,
            pred=lambda v: not isinstance(v, type_),
            actual=compose(_qualified_name, type),
            funcname=__funcname,
        )

    return preprocess(**valmap(_expect_type, named))