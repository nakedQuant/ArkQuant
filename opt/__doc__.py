"""
memory : joblib.Memory interface used to cache the fitted transformers of
    the Pipeline. By default,no caching is performed. If a string is given,
    it is the path to the caching directory. Enabling caching triggers a clone
    of the transformers before fitting.Caching the transformers is advantageous
    when fitting is time consuming.

3. from multiprocessing import Pool
     pool = Pool(4)
     res = [pool.apply_async(func,(i,)) for i in range(10)]
     for r in res:
         print(r.get())
func --- main()  --- 单独的函数（以def 开始的）, 类里面的函数不可以apply_async, get()
super().method ; super().__init__()
生成器：基于yield方法将函数转化为迭代器，next方法，每次执行到yield停止；而iter（迭代器将非可迭代对象强制转化为对象）
from interface import Interface,implements
sparse.hstack(Xs).tocsr() # Compressed Sparse Row format
关于精度 ---- float(2进制）, decimal ---- (10进制)
from decimal import Decimal , getcontext
decimal.getcontext().prec = 3
decimal.Decimal
# 堆队列(数值小，优先权高)
from heapq import heappush
# from collections import deque，defaultdict
# dqueue 双向队列

# from collections import ChainMap
# 有多个字典或者映射，想把它们合并成为一个单独的映射
c=ChainMap()
d=c.new_child()
e=c.new_child()
e.parents
# os.path.expandvars(path）
# 根据环境变量的值替换path中包含的
# "$name"
"""