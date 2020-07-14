
from threading import Lock
from functools import wraps
from toolz import memoize

def _weak_lru_cache(maxsize=100):
    """
    Users should only access the lru_cache through its public API:
    cache_info, cache_clear
    The internals of the lru_cache are encapsulated for thread safety and
    to allow the implementation to change.
    """
    def decorating_function(user_function):

        hits, misses = [0], [0]
        kwd_mark = (object(),)    # separates positional and keyword args
        lock = Lock()             # needed because OrderedDict isn't threadsafe

        if maxsize is None:
            cache = _WeakArgsDict()  # cache without ordering or size limit

            @wraps(user_function)
            def wrapper(*args, **kwds):
                key = args
                if kwds:
                    key += kwd_mark + tuple(sorted(kwds.items()))
                try:
                    result = cache[key]
                    hits[0] += 1
                    return result
                except KeyError:
                    pass
                result = user_function(*args, **kwds)
                cache[key] = result
                misses[0] += 1
                return result
        else:
            # ordered least recent to most recent
            cache = _WeakArgsOrderedDict()
            cache_popitem = cache.popitem
            cache_renew = cache.move_to_end

            @wraps(user_function)
            def wrapper(*args, **kwds):
                key = args
                if kwds:
                    key += kwd_mark + tuple(sorted(kwds.items()))
                with lock:
                    try:
                        result = cache[key]
                        cache_renew(key)    # record recent use of this key
                        hits[0] += 1
                        return result
                    except KeyError:
                        pass
                result = user_function(*args, **kwds)
                with lock:
                    cache[key] = result     # record recent use of this key
                    misses[0] += 1
                    if len(cache) > maxsize:
                        # purge least recently used cache entry
                        cache_popitem(False)
                return result

        def cache_info():
            """Report cache statistics"""
            with lock:
                return hits[0], misses[0], maxsize, len(cache)

        def cache_clear():
            """Clear the cache and cache statistics"""
            with lock:
                cache.clear()
                hits[0] = misses[0] = 0

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return wrapper

    return decorating_function