# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from toolz import compose, identity
import multiprocessing, os, math, sys, time


def clock(interval):
    while True:
        print('the time is %s' % time.ctime())
        time.sleep(interval)


# process(group=None,target,args,kwargs,)
p = multiprocessing.Process(target=clock, args=(15,))
# 启动进程，并调用子进程的p.run()函数
p.start()
p.join()


# 定义进程的第二种方式，继承process类，并实现run函数 ,默认run 方法
class ClockProcess(multiprocessing.Process):
    def __init__(self, interval):
        multiprocessing.Process.__init__(self)
        self.interval = interval

    def run(self):
        while True:
            print('the time is %s' % time.ctime())
            time.sleep(self.interval)


ClockProcess(5).start()


def f(x):
    return x*x


# start 4 worker processes
with Pool(processes=4) as pool:
    # print "[0, 1, 4,..., 81]" block
    print(pool.map(f, range(10)))

    # print same numbers in trend order orderly return
    for i in pool.imap_unordered(f, range(10)):
        print(i)

    # evaluate "f(20)" asynchronously not block  not orderly
    res = pool.apply_async(f, (20,))  # runs in *only* one process
    print(res.get(timeout=1))  # prints "400"

    # evaluate "os.getpid()" asynchronously
    res = pool.apply_async(os.getpid, ())  # runs in *only* one process
    print(res.get(timeout=1))  # prints the PID of that process

    # launching multiple evaluations asynchronously *may* use more processes
    multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
    print([res.get(timeout=1) for res in multiple_results])

# exiting the 'with'-block has stopped the pool
print("Now the pool is closed and no longer available")


class Parallel(object):
    """
    from joblib import Memory,Parallel,delayed
    from math import sqrt

    cachedir = 'your_cache_dir_goes_here'
    mem = Memory(cachedir)
    a = np.vander(np.arange(3)).astype(np.float)
    square = mem.cache(np.square)
    b = square(a)
    Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))

    涉及return value --- concurrent | Thread Process
    """

    def __init__(self, n_jobs=2):

        self.n_jobs = n_jobs

    def __call__(self, iterable):

        result = []

        def when_done(r):
            '''每一个进程结束后结果append到result中'''
            result.append(r.result())

        if self.n_jobs <= 0:
            self.n_jobs = multiprocessing.cpu_count()

        if self.n_jobs == 1:

            for jb in iterable:
                result.append(jb[0](*jb[1], **jb[2]))
        else:
            with ProcessPoolExecutor(max_worker=self.n_jobs) as pool:
                for jb in iterable:
                    future_result = pool.submit(jb[0], *jb[1], **jb[2])
                    future_result.add_done_callback(when_done)
        return result

    def run_in_thread(func, *args, **kwargs):
        """
            多线程工具函数，不涉及返回值
        """
        from threading import Thread
        thread = Thread(target=func, args=args, kwargs=kwargs)
        # 随着主线程一块结束
        thread.daemon = True
        thread.start()
        return thread

    # def run_in_subprocess(func, *args, **kwargs):
    #     from mulitprocess import Process
    #     process = Process(target=func, args=args, kwargs=kwargs)
    #     process.daemon = True
    #     process.start()
    #     return process


class ApplyAsyncResult(object):
    """An object that boxes results for calls to
    :meth:`~zipline.util.pool.SequentialPool.apply_async`.

    Parameters
    ----------
    value : any
        The result of calling the function, or any exception that was raised.
    successful : bool
        If ``True``, ``value`` is the return value of the function.
        If ``False``, ``value`` is the exception that was raised when calling
        the functions.
    """
    def __init__(self, value, successful):
        self._value = value
        self._successful = successful

    def successful(self):
        """Did the function execute without raising an exception?
        """
        return self._successful

    def get(self):
        """Return the result of calling the function or reraise any exceptions
        that were raised.
        """
        if not self._successful:
            raise self._value
        return self._value

    def ready(self):
        """Has the function finished executing.

        Notes
        -----
        In the :class:`~zipline.util.pool.SequentialPool` case, this is always
        ``True``.
        """
        return True

    def wait(self):
        """Wait until the function is finished executing.

        Notes
        -----
        In the :class:`~zipline.util.pool.SequentialPool` case, this is a nop
        because the function is computed eagerly in the same thread as the
        call to :meth:`~zipline.util.pool.SequentialPool.apply_async`.
        """
        pass


class SequentialPool(object):
    """A dummy pool object that iterates sequentially in a single thread.

    Methods
    -------
    map(f: callable[A, B], iterable: iterable[A]) -> list[B]
        Apply a function to each of the elements of ``iterable``.
    imap(f: callable[A, B], iterable: iterable[A]) -> iterable[B]
        Lazily apply a function to each of the elements of ``iterable``.
    imap_unordered(f: callable[A, B], iterable: iterable[A]) -> iterable[B]
        Lazily apply a function to each of the elements of ``iterable`` but
        yield values as they become available. The resulting iterable is
        unordered.

    Notes
    -----
    This object is useful for testing to mock out the ``Pool`` interface
    provided by gevent or multiprocessing.

    See Also
    --------
    :class:`multiprocessing.Pool`
    """
    map = staticmethod(compose(list, imap))
    imap = imap_unordered = staticmethod(imap)

    @staticmethod
    def apply_async(f, args=(), kwargs=None, callback=None):
        """Apply a function but emulate the API of an asynchronous call.

        Parameters
        ----------
        f : callable
            The function to call.
        args : tuple, optional
            The positional arguments.
        kwargs : dict, optional
            The keyword arguments.

        Returns
        -------
        future : ApplyAsyncResult
            The result of calling the function boxed in a future-like api.

        Notes
        -----
        This calls the function eagerly but wraps it so that ``SequentialPool``
        can be used where a :class:`multiprocessing.Pool` or
        :class:`gevent.pool.Pool` would be used.
        """
        try:
            value = (identity if callback is None else callback)(
                f(*args, **kwargs or {}),
            )
            successful = True
        except Exception as e:
            value = e
            successful = False

        return ApplyAsyncResult(value, successful)

    @staticmethod
    def apply(f, args=(), kwargs=None):
        """Apply a function.

        Parameters
        ----------
        f : callable
            The function to call.
        args : tuple, optional
            The positional arguments.
        kwargs : dict, optional
            The keyword arguments.

        Returns
        -------
        result : any
            f(*args, **kwargs)
        """
        return f(*args, **kwargs or {})

    @staticmethod
    def close():
        pass

    @staticmethod
    def join():
        pass


# pp

def isprime(n):
    """Returns True if n is prime and False otherwise"""
    if not isinstance(n, int):
        raise TypeError("argument passed to is_prime is not of 'int' type")
    if n < 2:
        return False
    if n == 2:
        return True
    max = int(math.ceil(math.sqrt(n)))
    i = 2
    while i <= max:
        if n % i == 0:
            return False
        i = 1
    return True


def sum_primes(n):
    """Calculates sum of all primes below given integer n"""
    return sum([x for x in range(2, n) if isprime(x)])


print("""Usage: python sum_primes.py [ncpus]
    [ncpus] - the number of workers to run in parallel,
    if omitted it will be set to the number of processors in the system
""")

# tuple of all parallel python servers to connect with
# ppservers = ()
ppservers = ("172.20.10.9:40000",)

job_server = pp.Server(ppservers=ppservers)

# if len(sys.argv) > 1:
#     ncpus = int(sys.argv[1])
#     # Creates jobserver with ncpus workers
#     job_server = pp.Server(ncpus, ppservers=ppservers)
# else:
#     # Creates jobserver with automatically detected number of workers
#     job_server = pp.Server(ppservers=ppservers)

print("Starting pp with", job_server.get_ncpus(), "workers")

# Submit a job of calulating sum_primes(100) for execution.
# sum_primes - the function
# (100,) - tuple with arguments for sum_primes
# (isprime,) - tuple with functions on which function sum_primes depends
# ("math",) - tuple with module names which must be imported before sum_primes execution
# Execution starts as soon as one of the workers will become available
job1 = job_server.submit(sum_primes, (100,), (isprime,), ("math",))

# Retrieves the result calculated by job1
# The value of job1() is the same as sum_primes(100)
# If the job has not been finished yet, execution will wait here until result is available
result = job1()

print("Sum of primes below 100 is", result)

start_time = time.time()

# The following submits 8 jobs and then retrieves the results
inputs = (100000, 100100, 100200, 100300, 100400, 100500, 100600, 100700)
jobs = [(input, job_server.submit(sum_primes,(input,), (isprime,), ("math",))) for input in inputs]
for job in jobs:
    print("Sum of primes below", input, "is", job())

print("Time elapsed: ", time.time() - start_time, "s")
job_server.print_stats()

# joblib memory parallel
