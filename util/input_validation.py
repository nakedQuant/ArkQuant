# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from datetime import tzinfo
from functools import partial
from operator import attrgetter
import toolz.curried.operator as op
from functools import wraps
from numpy import dtype
import pandas as pd
from pytz import timezone
from six import iteritems, string_types
from toolz import valmap, complement, compose

""" 
toolz.functoolz.complement(func)[source]
Convert a predicate function to its logical complement.

In other words, return a function that, for inputs that normally yield True, yields False, and vice-versa.

>>> def iseven(n): return n % 2 == 0
>>> isodd = complement(iseven)
>>> iseven(2)
True
>>> isodd(2)
False
"""

_qualified_name = attrgetter('__qualname__')


def optionally(preprocessor):
    """Modify a preprocessor to explicitly allow `None`.

    Parameters
    ----------
    preprocessor : callable[callable, str, any -> any]
        A preprocessor to delegate to when `arg is not None`.

    Returns
    -------
    optional_preprocessor : callable[callable, str, any -> any]
        A preprocessor that delegates to `preprocessor` when `arg is not None`.

    Examples
    --------
    >>> def preprocessor(func, argname, arg):
    ...     if not isinstance(arg, int):
    ...         raise TypeError('arg must be int')
    ...     return arg
    ...
    >>> @preprocess(a=optionally(preprocessor))
    ... def f(a):
    ...     return a
    ...
    >>> f(1)  # call with int
    1
    >>> f('a')  # call with not int
    Traceback (most recent call last):
       ...
    TypeError: arg must be int
    >>> f(None) is None  # call with explicit None
    True
    """
    @wraps(preprocessor)
    def wrapper(func, argname, arg):
        return arg if arg is None else preprocessor(func, argname, arg)

    return wrapper


def ensure_upper_case(func, argname, arg):
    if isinstance(arg, string_types):
        return arg.upper()
    else:
        raise TypeError(
            "{0}() expected argument '{1}' to"
            " be a string, but got {2} instead.".format(
                func.__name__,
                argname,
                arg,
            ),
        )


def ensure_dtype(func, argname, arg):
    """
    Argument preprocessor that converts the input into a numpy dtype.
    """
    try:
        return dtype(arg)
    except TypeError:
        raise TypeError(
            "{func}() couldn't convert argument "
            "{argname}={arg!r} to a numpy dtype.".format(
                func=_qualified_name(func),
                argname=argname,
                arg=arg,
            ),
        )


def ensure_timezone(func, argname, arg):
    """Argument preprocessor that converts the input into a tzinfo object.
    """
    if isinstance(arg, tzinfo):
        return arg
    if isinstance(arg, string_types):
        return timezone(arg)

    raise TypeError(
        "{func}() couldn't convert argument "
        "{argname}={arg!r} to a timezone.".format(
            func=_qualified_name(func),
            argname=argname,
            arg=arg,
        ),
    )


def ensure_timestamp(func, argname, arg):
    """Argument preprocessor that converts the input into a pandas Timestamp
    object.
    """
    try:
        return pd.Timestamp(arg)
    except ValueError as e:
        raise TypeError(
            "{func}() couldn't convert argument "
            "{argname}={arg!r} to a pandas Timestamp.\n"
            "Original error was: {t}: {e}".format(
                func=_qualified_name(func),
                argname=argname,
                arg=arg,
                t=_qualified_name(type(e)),
                e=e,
            ),
        )


def restrict_to_dtype(dtype, message_template):
    """
    A factory for decorators that restrict Term methods to only be callable on
    Terms with a specific dtype.

    This is conceptually similar to
    zipline.util.input_validation.expect_dtypes, but provides more flexibility
    for providing error messages that are specifically targeting Term methods.

    Parameters
    ----------
    dtype : numpy.dtype
        The dtype on which the decorated method may be called.
    message_template : str
        A template for the error message to be raised.
        `message_template.format` will be called with keyword arguments
        `method_name`, `expected_dtype`, and `received_dtype`.

    Examples
    --------
    @restrict_to_dtype(
        dtype=float64_dtype,
        message_template=(
            "{method_name}() was called on a factor of dtype {received_dtype}."
            "{method_name}() requires factors of dtype{expected_dtype}."

        ),
    )
    def some_factor_method(self, ...):
        self.stuff_that_requires_being_float64(...)
    """
    def processor(term_method, _, term_instance):
        term_dtype = term_instance.dtype
        if term_dtype != dtype:
            raise TypeError(
                message_template.format(
                    method_name=term_method.__name__,
                    expected_dtype=dtype.name,
                    received_dtype=term_dtype,
                )
            )
        return term_instance
    return preprocess(self=processor)


# 扩展 --- 通过字典对不同的参数进行函数限制，即为每一个参数调用preprocess
def expect_dtypes(__funcname=_qualified_name, **named):
    """
    Preprocessing decorator that verifies inputs have expected numpy dtypes.

    Examples
    --------
    >>> from numpy import dtype, arange, int8, float64
    >>> @expect_dtypes(x=dtype(int8))
    ... def foo(x, y):
    ...    return x, y
    ...
    >>> foo(arange(3, dtype=int8), 'foo')
    (array([0, 1, 2], dtype=int8), 'foo')
    >>> foo(arange(3, dtype=float64), 'foo')  # doctest: +NORMALIZE_WHITESPACE
    ...                                       # doctest: +ELLIPSIS
    Traceback (most recent call last):
       ...
    TypeError: ...foo() expected a value with dtype 'int8' for argument 'x',
    but got 'float64' instead.
    """
    for name, type_ in iteritems(named):
        if not isinstance(type_, (dtype, tuple)):
            raise TypeError(
                "expect_dtypes() expected a numpy dtype or tuple of dtypes"
                " for argument {name!r}, but got {dtype} instead.".format(
                    name=name, dtype=dtype,
                )
            )

    if isinstance(__funcname, str):
        def get_funcname(_):
            return __funcname
    else:
        get_funcname = __funcname

    @preprocess(dtypes=call(lambda x: x if isinstance(x, tuple) else (x,)))
    def _expect_dtype(dtypes):
        """
        Factory for dtype-checking functions that work with the @preprocess
        decorator.
        """
        def error_message(func, argname, value):
            # If the bad value has a dtype, but it's wrong, show the dtype
            # name.  Otherwise just show the value.
            try:
                value_to_show = value.dtype.name
            except AttributeError:
                value_to_show = value
            return (
                "{funcname}() expected a value with dtype {dtype_str} "
                "for argument {argname!r}, but got {value!r} instead."
            ).format(
                funcname=get_funcname(func),
                dtype_str=' or '.join(repr(d.name) for d in dtypes),
                argname=argname,
                value=value_to_show,
            )

        def _actual_preprocessor(func, argname, argvalue):
            if getattr(argvalue, 'dtype', object()) not in dtypes:
                raise TypeError(error_message(func, argname, argvalue))
            return argvalue

        return _actual_preprocessor

    return preprocess(**valmap(_expect_dtype, named))


def expect_kinds(**named):
    """
    Preprocessing decorator that verifies inputs have expected dtype kinds.

    Examples
    --------
    >>> from numpy import int64, int32, float32
    >>> @expect_kinds(x='i')
    ... def foo(x):
    ...    return x
    ...
    >>> foo(int64(2))
    2
    >>> foo(int32(2))
    2
    >>> foo(float32(2))  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    TypeError: ...foo() expected a numpy object of kind 'i' for argument 'x',
    but got 'f' instead.
    """
    for name, kind in iteritems(named):
        if not isinstance(kind, (str, tuple)):
            raise TypeError(
                "expect_dtype_kinds() expected a string or tuple of strings"
                " for argument {name!r}, but got {kind} instead.".format(
                    name=name, kind=dtype,
                )
            )

    @preprocess(kinds=call(lambda x: x if isinstance(x, tuple) else (x,)))
    def _expect_kind(kinds):
        """
        Factory for kind-checking functions that work the @preprocess
        decorator.
        """
        def error_message(func, argname, value):
            # If the bad value has a dtype, but it's wrong, show the dtype
            # kind.  Otherwise just show the value.
            try:
                value_to_show = value.dtype.kind
            except AttributeError:
                value_to_show = value
            return (
                "{funcname}() expected a numpy object of kind {kinds} "
                "for argument {argname!r}, but got {value!r} instead."
            ).format(
                funcname=_qualified_name(func),
                kinds=' or '.join(map(repr, kinds)),
                argname=argname,
                value=value_to_show,
            )

        def _actual_preprocessor(func, argname, argvalue):
            if getattr(argvalue, ('dtype', 'kind'), object()) not in kinds:
                raise TypeError(error_message(func, argname, argvalue))
            return argvalue

        return _actual_preprocessor

    return preprocess(**valmap(_expect_kind, named))


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


def optional(type_):
    """
    Helper for use with `expect_types` when an input can be `type_` or `None`.

    Returns an object such that both `None` and instances of `type_` pass
    checks of the form `isinstance(obj, optional(type_))`.

    Parameters
    ----------
    type_ : type
       Type for which to produce an option.

    Examples
    --------
    >>> isinstance({}, optional(dict))
    True
    >>> isinstance(None, optional(dict))
    True
    >>> isinstance(1, optional(dict))
    False
    """
    return (type_, type(None))


def make_check(exc_type, template, pred, actual, funcname):
    """
    Factory for making preprocessing functions that check a predicate on the
    input value.

    Parameters
    ----------
    exc_type : Exception
        The exception type to raise if the predicate fails.
    template : str
        A template string to use to create error messages.
        Should have %-style named template parameters for 'funcname',
        'argname', and 'actual'.
    pred : function[object -> bool]
        A function to call on the argument being preprocessed.  If the
        predicate returns `True`, we raise an instance of `exc_type`.
    actual : function[object -> object]
        A function to call on bad values to produce the value to display in the
        error message.
    funcname : str or callable
        Name to use in error messages, or function to call on decorated
        functions to produce a name.  Passing an explicit name is useful when
        creating checks for __init__ or __new__ methods when you want the error
        to refer to the class name instead of the method name.
    """
    if isinstance(funcname, str):
        def get_funcname(_):
            return funcname
    else:
        get_funcname = funcname

    def _check(func, argname, argvalue):
        if pred(argvalue):
            raise exc_type(
                template % {
                    'funcname': get_funcname(func),
                    'argname': argname,
                    'actual': actual(argvalue),
                },
            )
        return argvalue
    return _check


def expect_element(__funcname=_qualified_name, **named):
    """
    Preprocessing decorator that verifies inputs are elements of some
    expected collection.

    Examples
    --------
    >>> @expect_element(x=('a', 'b'))
    ... def foo(x):
    ...    return x.upper()
    ...
    >>> foo('a')
    'A'
    >>> foo('b')
    'B'
    >>> foo('c')  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    ValueError: ...foo() expected a value in ('a', 'b') for argument 'x',
    but got 'c' instead.

    Notes
    -----
    A special argument, __funcname, can be provided as a string to override the
    function name shown in error messages.  This is most often used on __init__
    or __new__ methods to make errors refer to the class name instead of the
    function name.

    This uses the `in` operator (__contains__) to make the containment check.
    This allows us to use any custom container as long as the object supports
    the container protocol.
    """
    def _expect_element(collection):
        if isinstance(collection, (set, frozenset)):
            # Special case the error message for set and frozen set to make it
            # less verbose.
            collection_for_error_message = tuple(sorted(collection))
        else:
            collection_for_error_message = collection

        template = (
            "%(funcname)s() expected a value in {collection} "
            "for argument '%(argname)s', but got %(actual)s instead."
        ).format(collection=collection_for_error_message)
        return make_check(
            ValueError,
            template,
            complement(op.contains(collection)),
            repr,
            funcname=__funcname,
        )
    return preprocess(**valmap(_expect_element, named))


def expect_bounded(__funcname=_qualified_name, **named):
    """
    Preprocessing decorator verifying that inputs fall INCLUSIVELY between
    bounds.

    Bounds should be passed as a pair of ``(min_value, max_value)``.

    ``None`` may be passed as ``min_value`` or ``max_value`` to signify that
    the input is only bounded above or below.

    Examples
    --------
    >>> @expect_bounded(x=(1, 5))
    ... def foo(x):
    ...    return x + 1
    ...
    >>> foo(1)
    2
    >>> foo(5)
    6
    >>> foo(6)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    ValueError: ...foo() expected a value inclusively between 1 and 5 for
    argument 'x', but got 6 instead.

    >>> @expect_bounded(x=(2, None))
    ... def foo(x):
    ...    return x
    ...
    >>> foo(100000)
    100000
    >>> foo(1)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    ValueError: ...foo() expected a value greater than or equal to 2 for
    argument 'x', but got 1 instead.

    >>> @expect_bounded(x=(None, 5))
    ... def foo(x):
    ...    return x
    ...
    >>> foo(6)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    ValueError: ...foo() expected a value less than or equal to 5 for
    argument 'x', but got 6 instead.
    """
    def _make_bounded_check(bounds):
        (lower, upper) = bounds
        if lower is None:
            def should_fail(value):
                return value > upper
            predicate_descr = "less than or equal to " + str(upper)
        elif upper is None:
            def should_fail(value):
                return value < lower
            predicate_descr = "greater than or equal to " + str(lower)
        else:
            def should_fail(value):
                return not (lower <= value <= upper)
            predicate_descr = "inclusively between %s and %s" % bounds

        template = (
            "%(funcname)s() expected a value {predicate}"
            " for argument '%(argname)s', but got %(actual)s instead."
        ).format(predicate=predicate_descr)

        return make_check(
            exc_type=ValueError,
            template=template,
            pred=should_fail,
            actual=repr,
            funcname=__funcname,
        )

    return _expect_bounded(_make_bounded_check, __funcname=__funcname, **named)


def expect_strictly_bounded(__funcname=_qualified_name, **named):
    """
    Preprocessing decorator verifying that inputs fall EXCLUSIVELY between
    bounds.

    Bounds should be passed as a pair of ``(min_value, max_value)``.

    ``None`` may be passed as ``min_value`` or ``max_value`` to signify that
    the input is only bounded above or below.

    Examples
    --------
    >>> @expect_strictly_bounded(x=(1, 5))
    ... def foo(x):
    ...    return x + 1
    ...
    >>> foo(2)
    3
    >>> foo(4)
    5
    >>> foo(5)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    ValueError: ...foo() expected a value exclusively between 1 and 5 for
    argument 'x', but got 5 instead.

    >>> @expect_strictly_bounded(x=(2, None))
    ... def foo(x):
    ...    return x
    ...
    >>> foo(100000)
    100000
    >>> foo(2)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    ValueError: ...foo() expected a value strictly greater than 2 for
    argument 'x', but got 2 instead.

    >>> @expect_strictly_bounded(x=(None, 5))
    ... def foo(x):
    ...    return x
    ...
    >>> foo(5)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Traceback (most recent call last):
       ...
    ValueError: ...foo() expected a value strictly less than 5 for
    argument 'x', but got 5 instead.
    """
    def _make_bounded_check(bounds):
        (lower, upper) = bounds
        if lower is None:
            def should_fail(value):
                return value >= upper
            predicate_descr = "strictly less than " + str(upper)
        elif upper is None:
            def should_fail(value):
                return value <= lower
            predicate_descr = "strictly greater than " + str(lower)
        else:
            def should_fail(value):
                return not (lower < value < upper)
            predicate_descr = "exclusively between %s and %s" % bounds

        template = (
            "%(funcname)s() expected a value {predicate}"
            " for argument '%(argname)s', but got %(actual)s instead."
        ).format(predicate=predicate_descr)

        return make_check(
            exc_type=ValueError,
            template=template,
            pred=should_fail,
            actual=repr,
            funcname=__funcname,
        )

    return _expect_bounded(_make_bounded_check, __funcname=__funcname, **named)


def _expect_bounded(make_bounded_check, __funcname, **named):
    def valid_bounds(t):
        return (
            isinstance(t, tuple)
            and len(t) == 2
            and t != (None, None)
        )

    for name, bounds in iteritems(named):
        if not valid_bounds(bounds):
            raise TypeError(
                "expect_bounded() expected a tuple of bounds for"
                " argument '{name}', but got {bounds} instead.".format(
                    name=name,
                    bounds=bounds,
                )
            )

    return preprocess(**valmap(make_bounded_check, named))


def expect_dimensions(__funcname=_qualified_name, **dimensions):
    """
    Preprocessing decorator that verifies inputs are numpy arrays with a
    specific dimensionality.

    Examples
    --------
    >>> from numpy import array
    >>> @expect_dimensions(x=1, y=2)
    ... def foo(x, y):
    ...    return x[0] + y[0, 0]
    ...
    >>> foo(array([1, 1]), array([[1, 1], [2, 2]]))
    2
    >>> foo(array([1, 1]), array([1, 1]))  # doctest: +NORMALIZE_WHITESPACE
    ...                                    # doctest: +ELLIPSIS
    Traceback (most recent call last):
       ...
    ValueError: ...foo() expected a 2-D array for argument 'y',
    but got a 1-D array instead.
    """
    if isinstance(__funcname, str):
        def get_funcname(_):
            return __funcname
    else:
        get_funcname = __funcname

    def _expect_dimension(expected_ndim):
        def _check(func, argname, argvalue):
            actual_ndim = argvalue.ndim
            if actual_ndim != expected_ndim:
                if actual_ndim == 0:
                    actual_repr = 'scalar'
                else:
                    actual_repr = "%d-D array" % actual_ndim
                raise ValueError(
                    "{func}() expected a {expected:d}-D array"
                    " for argument {argname!r}, but got a {actual}"
                    " instead.".format(
                        func=get_funcname(func),
                        expected=expected_ndim,
                        argname=argname,
                        actual=actual_repr,
                    )
                )
            return argvalue
        return _check
    return preprocess(**valmap(_expect_dimension, dimensions))


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


def coerce_types(**kwargs):
    """
    Preprocessing decorator that applies type coercions.

    Parameters
    ----------
    **kwargs : dict[str -> (type, callable)]
         Keyword arguments mapping function parameter names to pairs of
         (from_type, to_type).

    Examples
    --------
    >>> @coerce_types(x=(float, int), y=(int, str))
    ... def func(x, y):
    ...     return (x, y)
    ...
    >>> func(1.0, 3)
    (1, '3')
    """
    def _coerce(types):
        return coerce(*types)

    return preprocess(**valmap(_coerce, kwargs))


coerce_string = partial(coerce, string_types)
