# GIL，在标准库中，所有执行的阻塞型i/o操作函数，等待操作系统返回结果时，time.sleep也会释放GIL，在这个程度上使用多线程
# multithread run the method
# 使用工作的线程数实例化，executor.__exit__,executor.shutdown(wait=True),它会在所有线程都在所有线程都执行前阻塞线程
# with futures.ThreadPoolExecutor(workers) as executors:
# map方法作用与内置的map相似，download_one会在多个线程中并发调用，返回一个生产器，迭代返回各个函数的值
# map函数使用future，返回迭代器，__next__调用future的result方法
# #不同线程尝试向一个文件写入数据，互斥锁来同步他们的操作
# from threading impport Lock
# write_lock = Lock()
# write_lock.acquire()
# f.write('here is some data.\n')
# write_lock.release()
# from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
# with futures.ThreadPoolExecutor(4) as executor:
#     to_do = []
#     # future 将要发生的，需要排定，submit参数为调用方法，返回一个对象排期
#     for parse_args in sorted(args_list):
#         future = executor.submit(self.urllib_parse, parse_args)
#         to_do.append(future)
#         msg = 'Schedual for {}:{}'
#         print(msg.format(cc, future))
#     # as_completed,参数为一个future列表，返回一个迭代器，在future运行结束后，产出future
#     results = []
#     for future in futures.as_completed(to_do):
#         # result返回可调用对象的结果或者异常，如果future没有运行结束，result会阻塞所在线程直至结果返回，timeout参数
#         res = future.result()
#         msg = '{} result"{}'
#         print(msg.format(future, res))
#         results.append(res)