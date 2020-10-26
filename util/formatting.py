# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import re
from toolz import curry


def bulleted_list(items, max_count=None, indent=2):
    """Format a bulleted list of values.
    """
    if max_count is not None and len(items) > max_count:
        item_list = list(items)
        items = item_list[:max_count - 1]
        items.append('...')
        items.append(item_list[-1])

    line_template = (" " * indent) + "- {}"
    return "\n".join(map(line_template.format, items))


def s(word, seq, suffix='s'):
    """Adds a suffix to ``word`` if some sequence has anything other than
    exactly one element.

    Parameters
    ----------
    word : str
        The string to add the suffix to.
    seq : sequence
        The sequence to check the length of.
    suffix : str, optional.
        The suffix to add to ``word``

    Returns
    -------
    maybe_plural : str
        ``word`` with ``suffix`` added if ``len(seq) != 1``.
    """
    if len(seq) == 1:
        return word

    return word + suffix


def plural(singular, plural, seq):
    """Selects a singular or plural word based on the length of a sequence.

    Parameters
    ----------
    singlular : str
        The string to use when ``len(seq) == 1``.
    plural : str
        The string to use when ``len(seq) != 1``.
    seq : sequence
        The sequence to check the length of.

    Returns
    -------
    maybe_plural : str
        Either ``singlular`` or ``plural``.
    """
    if len(seq) == 1:
        return singular

    return plural


def bulleted_list(items, indent=0, bullet_type='-'):
    """Format a bulleted list of values.

    Parameters
    ----------
    items : sequence
        The items to make a list.
    indent : int, optional
        The number of spaces to add before each bullet.
    bullet_type : str, optional
        The bullet type to use.

    Returns
    -------
    formatted_list : str
        The formatted list as a single string.
    """
    format_string = ' ' * indent + bullet_type + ' {}'
    return "\n".join(map(format_string.format, items))


@curry
def copydoc(from_, to):
    """Copies the docstring from one function to another.
    Parameters
    ----------
    from_ : any
        The object to copy the docstring from.
    to : any
        The object to copy the docstring to.
    Returns
    -------
    to : any
        ``to`` with the docstring from ``from_``
    """
    to.__doc__ = from_.__doc__
    return to


def format_docstring(owner_name, docstring, formatters):
    """
    Template ``formatters`` into ``docstring``.

    Parameters
    ----------
    owner_name : str
        The name of the function or class whose docstring is being templated.
        Only used for error messages.
    docstring : str
        The docstring to template.
    formatters : dict[str -> str]
        Parameters for a a str.format() call on ``docstring``.

        Multi-line values in ``formatters`` will have leading whitespace padded
        to match the leading whitespace of the substitution string.
    """
    # Build a dict of parameters to a vanilla format() call by searching for
    # each entry in **formatters and applying any leading whitespace to each
    # line in the desired substitution.
    format_params = {}
    for target, doc_for_target in formatters.items():
        # Search for '{name}', with optional leading whitespace.
        regex = re.compile(r'^(\s*)' + '({' + target + '})$', re.MULTILINE)
        matches = regex.findall(docstring)
        if not matches:
            raise ValueError(
                "Couldn't find template for parameter {!r} in docstring "
                "for {}."
                "\nParameter name must be alone on a line surrounded by "
                "braces.".format(target, owner_name),
            )
        elif len(matches) > 1:
            raise ValueError(
                "Couldn't found multiple templates for parameter {!r}"
                "in docstring for {}."
                "\nParameter should only appear once.".format(
                    target, owner_name
                )
            )

        (leading_whitespace, _) = matches[0]
        format_params[target] = pad_lines_after_first(
            leading_whitespace,
            doc_for_target,
        )

    return docstring.format(**format_params)


def templated_docstring(**docs):
    """
    Decorator allowing the use of templated docstrings.

    Examples
    --------
    >>> @templated_docstring(foo='bar')
    ... def my_func(self, foo):
    ...     '''{foo}'''
    ...
    >>> my_func.__doc__
    'bar'
    """
    def decorator(f):
        f.__doc__ = format_docstring(f.__name__, f.__doc__, docs)
        return f
    return decorator
