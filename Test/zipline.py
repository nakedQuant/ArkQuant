"""
Utilities for validating inputs to user-facing API functions.
"""
from textwrap import dedent
from types import CodeType
from uuid import uuid4

from functools import partial
import inspect,functools

from toolz.curried.operator import getitem
from six import viewkeys, exec_, PY3

# from zipline.Utils.compat import getargspec, wraps

def getargspec(f):
    full_argspec = inspect.getfullargspec(f)
    return inspect.ArgSpec(
        args=full_argspec.args,
        varargs=full_argspec.varargs,
        keywords=full_argspec.varkw,
        defaults=full_argspec.defaults,
    )


def update_wrapper(wrapper,
                   wrapped,
                   assigned=functools.WRAPPER_ASSIGNMENTS,
                   updated=functools.WRAPPER_UPDATES):
    """Backport of Python 3's functools.update_wrapper for __wrapped__.
    """
    for attr in assigned:
        try:
            value = getattr(wrapped, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)
    for attr in updated:
        getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
    # Issue #17482: set __wrapped__ last so we don't inadvertently copy it
    # from the wrapped function when updating __dict__
    wrapper.__wrapped__ = wrapped
    # Return the wrapper so this can be used as a decorator via partial()
    return wrapper


def wraps(wrapped,
          assigned=functools.WRAPPER_ASSIGNMENTS,
          updated=functools.WRAPPER_UPDATES):
    """Decorator factory to apply update_wrapper() to a wrapper function

       Returns a decorator that invokes update_wrapper() with the decorated
       function as the wrapper argument and the arguments to wraps() as the
       remaining arguments. Default arguments are as for update_wrapper().
       This is a convenience function to simplify applying partial() to
       update_wrapper().
    """
    return functools.partial(update_wrapper, wrapped=wrapped,
                         assigned=assigned, updated=updated)


_code_argorder = (
    ('co_argcount', 'co_kwonlyargcount') if PY3 else ('co_argcount',)
) + (
    'co_nlocals',
    'co_stacksize',
    'co_flags',
    'co_code',
    'co_consts',
    'co_names',
    'co_varnames',
    'co_filename',
    'co_name',
    'co_firstlineno',
    'co_lnotab',
    'co_freevars',
    'co_cellvars',
)

NO_DEFAULT = object()


def preprocess(*_unused, **processors):

    if _unused:
        raise TypeError("preprocess() doesn't accept positional arguments")

    def _decorator(f):
        args, varargs, varkw, defaults = argspec = getargspec(f)
        print('default',defaults)
        if defaults is None:
            defaults = ()
        no_defaults = (NO_DEFAULT,) * (len(args) - len(defaults))
        args_defaults = list(zip(args, no_defaults + defaults))
        if varargs:
            args_defaults.append((varargs, NO_DEFAULT))
        if varkw:
            args_defaults.append((varkw, NO_DEFAULT))

        argset = set(args) | {varargs, varkw} - {None}

        # Arguments can be declared as tuples in Python 2.
        if not all(isinstance(arg, str) for arg in args):
            raise TypeError(
                "Can't validate functions using tuple unpacking: %s" %
                (argspec,)
            )

        # Ensure that all processors map to valid names.
        bad_names = viewkeys(processors) - argset
        if bad_names:
            raise TypeError(
                "Got processors for unknown arguments: %s." % bad_names
            )
        print('args_default',args_defaults)
        print('processors',processors)
        print(f, processors, args_defaults, varargs, varkw)
        res = _build_preprocessed_function(
            f, processors, args_defaults, varargs, varkw,
        )
        print('result-------------------',res)
        return _build_preprocessed_function(
            f, processors, args_defaults, varargs, varkw,
        )
    return _decorator


def _build_preprocessed_function(func,
                                 processors,
                                 args_defaults,
                                 varargs,
                                 varkw):
    """
    Build a preprocessed function with the same signature as `func`.

    Uses `exec` internally to build a function that actually has the same
    signature as `func.
    """
    format_kwargs = {'func_name': func.__name__}

    def mangle(name):
        return 'a' + uuid4().hex + name

    format_kwargs['mangled_func'] = mangled_funcname = mangle(func.__name__)

    def make_processor_assignment(arg, processor_name):
        template = "{arg} = {processor}({func}, '{arg}', {arg})"
        return template.format(
            arg=arg,
            processor=processor_name,
            func=mangled_funcname,
        )

    exec_globals = {mangled_funcname: func, 'wraps': wraps}
    defaults_seen = 0
    default_name_template = 'a' + uuid4().hex + '_%d'
    signature = []
    call_args = []
    assignments = []
    star_map = {
        varargs: '*',
        varkw: '**',
    }

    def name_as_arg(arg):
        return star_map.get(arg, '') + arg

    for arg, default in args_defaults:
        if default is NO_DEFAULT:
            signature.append(name_as_arg(arg))
        else:
            default_name = default_name_template % defaults_seen
            exec_globals[default_name] = default
            signature.append('='.join([name_as_arg(arg), default_name]))
            defaults_seen += 1

        if arg in processors:
            procname = mangle('_processor_' + arg)
            exec_globals[procname] = processors[arg]
            code = make_processor_assignment(arg, procname)
            print('code',code)
            assignments.append(make_processor_assignment(arg, procname))

        call_args.append(name_as_arg(arg))

    exec_str = dedent(
        """\
        @wraps({wrapped_funcname})
        def {func_name}({signature}):
            {assignments}
            return {wrapped_funcname}({call_args})
        """
    ).format(
        func_name=func.__name__,
        signature=', '.join(signature),
        assignments='\n    '.join(assignments),
        wrapped_funcname=mangled_funcname,
        call_args=', '.join(call_args),
    )
    print('exec_str',exec_str)
    compiled = compile(
        exec_str,
        func.__code__.co_filename,
        mode='exec',
    )
    print('compiled',compiled)
    exec_locals = {}
    print('exec',exec_(compiled, exec_globals, exec_locals))

    new_func = exec_locals[func.__name__]
    print('new_func',new_func)
    code = new_func.__code__
    args = {
        attr: getattr(code, attr)
        for attr in dir(code)
        if attr.startswith('co_')
    }
    # Copy the firstlineno out of the underlying function so that exceptions
    # get raised with the correct traceback.
    # This also makes dynamic source inspection (like IPython `??` operator)
    # work as intended.
    try:
        # Try to get the pycode object from the underlying function.
        original_code = func.__code__
    except AttributeError:
        try:
            # The underlying callable was not a function, try to grab the
            # `__func__.__code__` which exists on method objects.
            original_code = func.__func__.__code__
        except AttributeError:
            # The underlying callable does not have a `__code__`. There is
            # nothing for us to correct.
            return new_func

    args['co_firstlineno'] = original_code.co_firstlineno
    new_func.__code__ = CodeType(*map(getitem(args), _code_argorder))
    print('new_func---',new_func)
    return new_func


if __name__ == '__main__':

    # # def func(a,b,num = 2):
    # #     c = a + b
    # #     d = c + num
    # #     return d,c
    # #
    # #
    # # NO_DEFAULT = object()
    # #
    # # args, varargs, varkw, defaults = argspec = getargspec(func)
    # # print('argspec',argspec)
    # # if defaults is None:
    # #     defaults = ()
    # # no_defaults = (NO_DEFAULT,) * (len(args) - len(defaults))
    # # args_defaults = list(zip(args, no_defaults + defaults))
    # # if varargs:
    # #     args_defaults.append((varargs, NO_DEFAULT))
    # # if varkw:
    # #     args_defaults.append((varkw, NO_DEFAULT))
    # # argset = set(args) | {varargs, varkw} - {None}
    # # print('argset',argset)
    # # print('default',args_defaults)
    # # print('name',func.__name__)
    from operator import attrgetter
    _qualified_name = attrgetter('__qualname__')
    # # print('_qualified_name',_qualified_name)
    #
    # from numpy import dtype
    #
    # def ensure_dtype(func, argname, arg):
    #     try:
    #         return dtype(arg)
    #     except TypeError:
    #         raise TypeError(
    #             "{func}() couldn't convert argument "
    #             "{argname}={arg!r} to a numpy dtype.".format(
    #                 func=_qualified_name(func),
    #                 argname=argname,
    #                 arg=arg,
    #             ),
    #         )
    # @preprocess(dtype=ensure_dtype)
    # def foo(dtype):
    #     return dtype
    # res = foo(int)
    # print('res',res)

    from toolz import valmap,compose

    def make_check(exc_type, template, pred, actual, funcname):
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

    def expect_types(__funcname=_qualified_name, **named):
        print('__funcname',__funcname)
        for name, type_ in named.items():
            print(name,type_)
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

    @expect_types(x=int, y=str)
    def foo(x, y):
        return x, y

    res = foo(2, '3')
    print(res)