# # -*- coding: utf-8  -*-
import multiprocessing,time,os


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


# start 4 worker processes
from multiprocessing import Pool

def f(x):
    return x*x

with Pool(processes=4) as pool:
    # print "[0, 1, 4,..., 81]" block
    print(pool.map(f, range(10)))

    # print same numbers in arbitrary order orderly return
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

from concurrent.futures import ProcessPoolExecutor

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
        '''
           多线程工具函数，不涉及返回值等'''
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