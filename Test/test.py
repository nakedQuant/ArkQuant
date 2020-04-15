# class Parent:
#     def __init__(self,p_name='parent',type='super'):
#         self.p_name = p_name
#         self.type = type
#
#     def run(self):
#         print('class Parent',self.p_name)
#
# class Child(Parent):
#     def __init__(self,c_name = 'Child'):
#         super().__init__(p_name = 'test',type='test')
#         self._type = c_name
#         #self.p_name = 'test'
#
#     def private(self):
#         super().run()
#         print('Child',self.p_name,self.type)
#
# def func(x):
#     return x+2
#
# if __name__=='__main__':
#     # child = Child()
#     # #child.run()
#     # child.private()
#     # #parent = Parent()
#     # #parent.run()
#     from multiprocessing import Pool
#     pool = Pool(4)
#     res = [pool.apply_async(func,(i,)) for i in range(10)]
#     for r in res:
#         print(r.get())
#
#     test = dict()
#     test[Child] = 3
#     print(test.keys())
#     print(type(test.values()))

#生成器：基于yield方法将函数转化为迭代器，next方法，每次执行到yield停止；而iter（迭代器将非可迭代对象强制转化为对象）
# def test(n):
#     for id_ in range(n):
#         yield id_
#         print('----------yield:',id_)
#
# iter = test(5)
# #print(iter)
# # for i in iter:
# #     pass
# next(iter)

# from itertools import chain,repeat
#
# chunks = chain([5], repeat(126))
# print(chunks)
# print(next(chunks))
# if not Xs:
#     # All transformers are None
#     return np.zeros((X.shape[0], 0))
# if any(sparse.issparse(f) for f in Xs):
#     Xs = sparse.hstack(Xs).tocsr() Compressed Sparse Row format
# else:
#     Xs = np.hstack(Xs)

# import functools
# from interface import Interface,Implements
# class A(Interface):
#
#     def test(self,a,b):
#         print(a,b)
#
# class B(implements(A)):
#
#     def test(self,a,c):
#         b = a+c
#         print(b)

#if __name__ =='__main__':
    # print(type(A))
    # a = A(3)
    # b = A(4)
    # print('-------------')
    # print(a==b)
    # print(a.__name__)
    # print(b.__name__)
    # # flag_c = isinstance(a,A)
    # # print(flag_c)
    # # flag_d = isinstance(A,A)
    # # print(flag_d)
#
# class A:
#
#     '''test A '''
#
#     def __init__(self,a):
#         self.__name__ = 'passthrough'
#
#     def test(self,c):
#         print(c)
#
#     def __call__(self,b):
#         print(b)
#
# print(A.__doc__)
# print(A.__name__)
# A().test('a')
# print(callable(A))
#
# a = A(3)
# a(5)

# print(a.__dir__())
# print(a.__doc__)
# print(type(A.__name__))
# print(A.__dict__)


# from functools import partial,wraps
# #wraps 单纯的修饰函数
# def _validate_type(_type):
#     def decorate(func):
#         def wrap(*args):
#             res = func(*args)
#             print('----------------------1')
#             if not isinstance(res,_type):
#                 print('--------------------------2')
#                 try:
#                     res = _type(res)
#                 except:
#                     print('can not Algorithm type:%s'%_type)
#             return res
#         return wrap
#     return decorate
#
# @_validate_type(int)
# def func(a,b):
#     c= a+b
#     return c
#
#
# res = func('5','4')
# print(type(res))
# raise TypeError('------------------------------2')

# data = None
# for file in os.listdir(folderpath):
#     if '.csv' not in file:
#         continue
#     raw = load_prices_from_csv(os.path.join(folderpath, file),
#                                identifier_col, tz)
#     if data is None:
#         data = raw
#     else:
#         data = pd.concat([data, raw], axis=1)

# from collections import ChainMap
# #有多个字典或者映射，想把它们合并成为一个单独的映射
# c=ChainMap()
# d=c.new_child()
# e=c.new_child()
# e.parents

# from  decimal import Decimal
# from  decimal import getcontext
# #整数、字符串或者元组构建decimal.Decimal，对于浮点数需要先将其转换为字符串
# d_context = getcontext()
# d_context.prec = 6
#
# d = Decimal(1) / Decimal(3)
# print(type(d), d)
# 两分查找,复杂度对数级别的
# from bisect import bisect_left

# 堆队列(数值小，优先权高)
# from heapq import heappush
# a=[]
# heappush(a,5)
# heappush(a,3)
# heappush(a,9)
# heappush(a,15)

# from collections import deque，defaultdict
# dqueue 双向队列
# fifo=deque('ab')
# fifo.appendleft('c')
# fifo.append('d')
# fifo.popleft()

# 默认为0
# stats=defaultdict(int)
# os.path.expandvars(path
# 根据环境变量的值替换path中包含的
# "$name"
# expanduser('~') - -- 用户目录
# os.path.basename | os.path.dirname
# np.fmax(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', subok=True)
# 通过scipy.interpolate.interp1d插值形成的模型，通过sco.fmin_bfgs计算min
# param
# find_min_pos: 寻找min的点位值
# param
# linear_interp: scipy.interpolate.interp1d插值形成的模型
# local_min_pos = sco.fmin_bfgs(linear_interp, find_min_pos, disp=False)[0]
# scipy.interpolate.interp1d插值形成模型

# import sympy as sy
# 符号计算（在交换式金融分析，比较有效）
# sy.Symbol('x') sy.sqrt(x)
# sy.solve(x ** 2 -1 ) 方程式右边为0
# #打印出符号积分
# sy.pretty(sy.Intergal(sy.sin(x) + 0.5 * x,x))
# #积分,求出反导数
# init_func = sy.intergrate(sy.sin(x) + 0.5 * x,x)
# #求导
# init_func.diff()
# #偏导数
# init_func.diff(,x)
# #subs代入数值，evalf求值
# init_func.subs(x,0.5).evalf()
# #nsolve与solve
# from sympy.solvers import nsolve
# solve 处理等式右边为0的表达式；而nsolve处理表达式（范围更加广）
#from pyalgotrade import dataseries

#
# def datetime_aligned(ds1, ds2, maxLen=None):
#     """
#     Returns two dataseries that exhibit only those values whose datetimes are in both dataseries.
#
#     :param ds1: A DataSeries instance.
#     :type ds1: :class:`DataSeries`.
#     :param ds2: A DataSeries instance.
#     :type ds2: :class:`DataSeries`.
#     :param maxLen: The maximum number of values to hold for the returned :class:`DataSeries`.
#         Once a bounded length is full, when new items are added, a corresponding number of items are discarded from the
#         opposite end. If None then dataseries.DEFAULT_MAX_LEN is used.
#     :type maxLen: int.
#     """
#     aligned1 = dataseries.SequenceDataSeries(maxLen)
#     aligned2 = dataseries.SequenceDataSeries(maxLen)
#     Syncer(ds1, ds2, aligned1, aligned2)
#     return (aligned1, aligned2)
#
#
# # This class is responsible for filling 2 dataseries when 2 other dataseries get new values.
# class Syncer(object):
#     def __init__(self, sourceDS1, sourceDS2, destDS1, destDS2):
#         self.__values1 = []  # (datetime, value)
#         self.__values2 = []  # (datetime, value)
#         self.__destDS1 = destDS1
#         self.__destDS2 = destDS2
#         sourceDS1.getNewValueEvent().subscribe(self.__onNewValue1)
#         sourceDS2.getNewValueEvent().subscribe(self.__onNewValue2)
#         # Source dataseries will keep a reference to self and that will prevent from getting this destroyed.
#
#     # Scan backwards for the position of dateTime in ds.
#     def __findPosForDateTime(self, values, dateTime):
#         ret = None
#         i = len(values) - 1
#         while i >= 0:
#             if values[i][0] == dateTime:
#                 ret = i
#                 break
#             elif values[i][0] < dateTime:
#                 break
#             i -= 1
#         return ret
#
#     def __onNewValue1(self, dataSeries, dateTime, value):
#         pos2 = self.__findPosForDateTime(self.__values2, dateTime)
#         # If a value for dateTime was added to first dataseries, and a value for that same datetime is also in the second one
#         # then append to both destination dataseries.
#         if pos2 is not None:
#             self.__append(dateTime, value, self.__values2[pos2][1])
#             # Reset buffers.
#             self.__values1 = []
#             self.__values2 = self.__values2[pos2+1:]
#         else:
#             # Since source dataseries may not hold all the values we need, we need to buffer manually.
#             self.__values1.append((dateTime, value))
#
#     def __onNewValue2(self, dataSeries, dateTime, value):
#         pos1 = self.__findPosForDateTime(self.__values1, dateTime)
#         # If a value for dateTime was added to second dataseries, and a value for that same datetime is also in the first one
#         # then append to both destination dataseries.
#         if pos1 is not None:
#             self.__append(dateTime, self.__values1[pos1][1], value)
#             # Reset buffers.
#             self.__values1 = self.__values1[pos1+1:]
#             self.__values2 = []
#         else:
#             # Since source dataseries may not hold all the values we need, we need to buffer manually.
#             self.__values2.append((dateTime, value))
#
#     def __append(self, dateTime, value1, value2):
#         self.__destDS1.appendWithDateTime(dateTime, value1)
#         self.__destDS2.appendWithDateTime(dateTime, value2)
#
#
#
#
#
# import abc
#
# import six
#
# from pyalgotrade import dataseries
# from pyalgotrade.dataseries import bards
# from pyalgotrade import bar
# from pyalgotrade import resamplebase
#
#
# class AggFunGrouper(resamplebase.Grouper):
#     def __init__(self, groupDateTime, value, aggfun):
#         super(AggFunGrouper, self).__init__(groupDateTime)
#         self.__values = [value]
#         self.__aggfun = aggfun
#
#     def addValue(self, value):
#         self.__values.append(value)
#
#     def getGrouped(self):
#         return self.__aggfun(self.__values)
#
#
# class BarGrouper(resamplebase.Grouper):
#     def __init__(self, groupDateTime, bar_, frequency):
#         super(BarGrouper, self).__init__(groupDateTime)
#         self.__open = bar_.getOpen()
#         self.__high = bar_.getHigh()
#         self.__low = bar_.getLow()
#         self.__close = bar_.getClose()
#         self.__volume = bar_.getVolume()
#         self.__adjClose = bar_.getAdjClose()
#         self.__useAdjValue = bar_.getUseAdjValue()
#         self.__frequency = frequency
#
#     def addValue(self, value):
#         self.__high = max(self.__high, value.getHigh())
#         self.__low = min(self.__low, value.getLow())
#         self.__close = value.getClose()
#         self.__adjClose = value.getAdjClose()
#         self.__volume += value.getVolume()
#
#     def getGrouped(self):
#         """Return the grouped value."""
#         ret = bar.BasicBar(
#             self.getDateTime(),
#             self.__open, self.__high, self.__low, self.__close, self.__volume, self.__adjClose,
#             self.__frequency
#         )
#         ret.setUseAdjustedValue(self.__useAdjValue)
#         return ret
#
#
# @six.add_metaclass(abc.ABCMeta)
# class DSResampler(object):
#
#     def initDSResampler(self, dataSeries, frequency):
#         if not resamplebase.is_valid_frequency(frequency):
#             raise Exception("Unsupported frequency")
#
#         self.__frequency = frequency
#         self.__grouper = None
#         self.__range = None
#
#         dataSeries.getNewValueEvent().subscribe(self.__onNewValue)
#
#     @abc.abstractmethod
#     def buildGrouper(self, range_, value, frequency):
#         raise NotImplementedError()
#
#     def __onNewValue(self, dataSeries, dateTime, value):
#         if self.__range is None:
#             self.__range = resamplebase.build_range(dateTime, self.__frequency)
#             self.__grouper = self.buildGrouper(self.__range, value, self.__frequency)
#         elif self.__range.belongs(dateTime):
#             self.__grouper.addValue(value)
#         else:
#             self.appendWithDateTime(self.__grouper.getDateTime(), self.__grouper.getGrouped())
#             self.__range = resamplebase.build_range(dateTime, self.__frequency)
#             self.__grouper = self.buildGrouper(self.__range, value, self.__frequency)
#
#     def pushLast(self):
#         if self.__grouper is not None:
#             self.appendWithDateTime(self.__grouper.getDateTime(), self.__grouper.getGrouped())
#             self.__grouper = None
#             self.__range = None
#
#     def checkNow(self, dateTime):
#         if self.__range is not None and not self.__range.belongs(dateTime):
#             self.appendWithDateTime(self.__grouper.getDateTime(), self.__grouper.getGrouped())
#             self.__grouper = None
#             self.__range = None
#
#
# class ResampledBarDataSeries(bards.BarDataSeries, DSResampler):
#     """A BarDataSeries that will build on top of another, higher frequency, BarDataSeries.
#     Resampling will take place as new values get pushed into the dataseries being resampled.
#
#     :param dataSeries: The DataSeries instance being resampled.
#     :type dataSeries: :class:`pyalgotrade.dataseries.bards.BarDataSeries`
#     :param frequency: The grouping frequency in seconds. Must be > 0.
#     :param maxLen: The maximum number of values to hold.
#         Once a bounded length is full, when new items are added, a corresponding number of items are discarded
#         from the opposite end.
#     :type maxLen: int.
#
#     .. note::
#         * Supported resampling frequencies are:
#             * Less than bar.Frequency.DAY
#             * bar.Frequency.DAY
#             * bar.Frequency.MONTH
#     """
#
#     def __init__(self, dataSeries, frequency, maxLen=None):
#         if not isinstance(dataSeries, bards.BarDataSeries):
#             raise Exception("dataSeries must be a dataseries.bards.BarDataSeries instance")
#
#         super(ResampledBarDataSeries, self).__init__(maxLen)
#         self.initDSResampler(dataSeries, frequency)
#
#     def checkNow(self, dateTime):
#         """Forces a resample check. Depending on the resample frequency, and the current datetime, a new
#         value may be generated.
#
#        :param dateTime: The current datetime.
#        :type dateTime: :class:`datetime.datetime`
#         """
#
#         return super(ResampledBarDataSeries, self).checkNow(dateTime)
#
#     def buildGrouper(self, range_, value, frequency):
#         return BarGrouper(range_.getBeginning(), value, frequency)
#
#
# class ResampledDataSeries(dataseries.SequenceDataSeries, DSResampler):
#     def __init__(self, dataSeries, frequency, aggfun, maxLen=None):
#         super(ResampledDataSeries, self).__init__(maxLen)
#         self.initDSResampler(dataSeries, frequency)
#         self.__aggfun = aggfun
#
#     def buildGrouper(self, range_, value, frequency):
#         return AggFunGrouper(range_.getBeginning(), value, self.__aggfun)
#
#
#
# class Dispatcher(object):
#     def __init__(self):
#         self.__subjects = []
#         self.__stop = False
#         self.__startEvent = observer.Event()
#         self.__idleEvent = observer.Event()
#         self.__currDateTime = None
#
#     # Returns the current event datetime. It may be None for events from realtime subjects.
#     def getCurrentDateTime(self):
#         return self.__currDateTime
#
#     def getStartEvent(self):
#         return self.__startEvent
#
#     def getIdleEvent(self):
#         return self.__idleEvent
#
#     def stop(self):
#         self.__stop = True
#
#     def getSubjects(self):
#         return self.__subjects
#
#     def addSubject(self, subject):
#         # Skip the subject if it was already added.
#         if subject in self.__subjects:
#             return
#
#         # If the subject has no specific dispatch priority put it right at the end.
#         if subject.getDispatchPriority() is dispatchprio.LAST:
#             self.__subjects.append(subject)
#         else:
#             # Find the position according to the subject's priority.
#             pos = 0
#             for s in self.__subjects:
#                 if s.getDispatchPriority() is dispatchprio.LAST or subject.getDispatchPriority() < s.getDispatchPriority():
#                     break
#                 pos += 1
#             self.__subjects.insert(pos, subject)
#
#         subject.onDispatcherRegistered(self)
#
#     # Return True if events were dispatched.
#     def __dispatchSubject(self, subject, currEventDateTime):
#         ret = False
#         # Dispatch if the datetime is currEventDateTime of if its a realtime subject.
#         if not subject.eof() and subject.peekDateTime() in (None, currEventDateTime):
#             ret = subject.dispatch() is True
#         return ret
#
#     # Returns a tuple with booleans
#     # 1: True if all subjects hit eof
#     # 2: True if at least one subject dispatched events.
#     def __dispatch(self):
#         smallestDateTime = None
#         eof = True
#         eventsDispatched = False
#
#         # Scan for the lowest datetime.
#         for subject in self.__subjects:
#             if not subject.eof():
#                 eof = False
#                 smallestDateTime = Utils.safe_min(smallestDateTime, subject.peekDateTime())
#
#         # Dispatch realtime subjects and those subjects with the lowest datetime.
#         if not eof:
#             self.__currDateTime = smallestDateTime
#
#             for subject in self.__subjects:
#                 if self.__dispatchSubject(subject, smallestDateTime):
#                     eventsDispatched = True
#         return eof, eventsDispatched
#
#     def run(self):
#         try:
#             for subject in self.__subjects:
#                 subject.start()
#
#             self.__startEvent.emit()
#
#             while not self.__stop:
#                 eof, eventsDispatched = self.__dispatch()
#                 if eof:
#                     self.__stop = True
#                 elif not eventsDispatched:
#                     self.__idleEvent.emit()
#         finally:
#             # There are no more events.
#             self.__currDateTime = None
#
#             for subject in self.__subjects:
#                 subject.stop()
#             for subject in self.__subjects:
#                 subject.join()
#
# import numpy as np
# import matplotlib.pyplot as plt
# from six.moves import xrange
#
# from pyalgotrade.technical import roc
# from pyalgotrade import dispatcher
#
#
# class Results(object):
#     """Results from the profiler."""
#     def __init__(self, eventsDict, lookBack, lookForward):
#         assert(lookBack > 0)
#         assert(lookForward > 0)
#         self.__lookBack = lookBack
#         self.__lookForward = lookForward
#         self.__values = [[] for i in xrange(lookBack+lookForward+1)]
#         self.__eventCount = 0
#
#         # Process events.
#         for instrument, events in eventsDict.items():
#             for event in events:
#                 # Skip events which are on the boundary or for some reason are not complete.
#                 if event.isComplete():
#                     self.__eventCount += 1
#                     # Compute cumulative returns: (1 + R1)*(1 + R2)*...*(1 + Rn)
#                     values = np.cumprod(event.getValues() + 1)
#                     # Normalize everything to the time of the event
#                     values = values / values[event.getLookBack()]
#                     for t in range(event.getLookBack()*-1, event.getLookForward()+1):
#                         self.setValue(t, values[t+event.getLookBack()])
#
#     def __mapPos(self, t):
#         assert(t >= -1*self.__lookBack and t <= self.__lookForward)
#         return t + self.__lookBack
#
#     def setValue(self, t, value):
#         if value is None:
#             raise Exception("Invalid value at time %d" % (t))
#         pos = self.__mapPos(t)
#         self.__values[pos].append(value)
#
#     def getValues(self, t):
#         pos = self.__mapPos(t)
#         return self.__values[pos]
#
#     def getLookBack(self):
#         return self.__lookBack
#
#     def getLookForward(self):
#         return self.__lookForward
#
#     def getEventCount(self):
#         """Returns the number of events occurred. Events that are on the boundary are skipped."""
#         return self.__eventCount
#
#
# class Predicate(object):
#     """Base class for event identification. You should subclass this to implement
#     the event identification logic."""
#
#     def eventOccurred(self, instrument, bards):
#         """Override (**mandatory**) to determine if an event took place in the last bar (bards[-1]).
#
#         :param instrument: Instrument identifier.
#         :type instrument: string.
#         :param bards: The BarDataSeries for the given instrument.
#         :type bards: :class:`pyalgotrade.dataseries.bards.BarDataSeries`.
#         :rtype: boolean.
#         """
#         raise NotImplementedError()
#
#
# class Event(object):
#     def __init__(self, lookBack, lookForward):
#         assert(lookBack > 0)
#         assert(lookForward > 0)
#         self.__lookBack = lookBack
#         self.__lookForward = lookForward
#         self.__values = np.empty((lookBack + lookForward + 1))
#         self.__values[:] = np.NAN
#
#     def __mapPos(self, t):
#         assert(t >= -1*self.__lookBack and t <= self.__lookForward)
#         return t + self.__lookBack
#
#     def isComplete(self):
#         return not any(np.isnan(self.__values))
#
#     def getLookBack(self):
#         return self.__lookBack
#
#     def getLookForward(self):
#         return self.__lookForward
#
#     def setValue(self, t, value):
#         if value is not None:
#             pos = self.__mapPos(t)
#             self.__values[pos] = value
#
#     def getValue(self, t):
#         pos = self.__mapPos(t)
#         return self.__values[pos]
#
#     def getValues(self):
#         return self.__values
#
#
# class Profiler(object):
#     """This class is responsible for scanning over historical data and analyzing returns before
#     and after the events.
#
#     :param predicate: A :class:`Predicate` subclass responsible for identifying events.
#     :type predicate: :class:`Predicate`.
#     :param lookBack: The number of bars before the event to analyze. Must be > 0.
#     :type lookBack: int.
#     :param lookForward: The number of bars after the event to analyze. Must be > 0.
#     :type lookForward: int.
#     """
#
#     def __init__(self, predicate, lookBack, lookForward):
#         assert(lookBack > 0)
#         assert(lookForward > 0)
#         self.__predicate = predicate
#         self.__lookBack = lookBack
#         self.__lookForward = lookForward
#         self.__feed = None
#         self.__rets = {}
#         self.__futureRets = {}
#         self.__events = {}
#
#     def __addPastReturns(self, instrument, event):
#         begin = (event.getLookBack() + 1) * -1
#         for t in xrange(begin, 0):
#             try:
#                 ret = self.__rets[instrument][t]
#                 if ret is not None:
#                     event.setValue(t+1, ret)
#             except IndexError:
#                 pass
#
#     def __addCurrentReturns(self, instrument):
#         nextTs = []
#         for event, t in self.__futureRets[instrument]:
#             event.setValue(t, self.__rets[instrument][-1])
#             if t < event.getLookForward():
#                 t += 1
#                 nextTs.append((event, t))
#         self.__futureRets[instrument] = nextTs
#
#     def __onBars(self, dateTime, bars):
#         for instrument in bars.getInstruments():
#             self.__addCurrentReturns(instrument)
#             eventOccurred = self.__predicate.eventOccurred(instrument, self.__feed[instrument])
#             if eventOccurred:
#                 event = Event(self.__lookBack, self.__lookForward)
#                 self.__events[instrument].append(event)
#                 self.__addPastReturns(instrument, event)
#                 # Add next return for this instrument at t=1.
#                 self.__futureRets[instrument].append((event, 1))
#
#     def getResults(self):
#         """Returns the results of the analysis.
#
#         :rtype: :class:`Results`.
#         """
#         return Results(self.__events, self.__lookBack, self.__lookForward)
#
#     def run(self, feed, useAdjustedCloseForReturns=True):
#         """Runs the analysis using the bars supplied by the feed.
#
#         :param barFeed: The bar feed to use to run the analysis.
#         :type barFeed: :class:`pyalgotrade.barfeed.BarFeed`.
#         :param useAdjustedCloseForReturns: True if adjusted close values should be used to calculate returns.
#         :type useAdjustedCloseForReturns: boolean.
#         """
#
#         if useAdjustedCloseForReturns:
#             assert feed.barsHaveAdjClose(), "Feed doesn't have adjusted close values"
#
#         try:
#             self.__feed = feed
#             self.__rets = {}
#             self.__futureRets = {}
#             for instrument in feed.getRegisteredInstruments():
#                 self.__events.setdefault(instrument, [])
#                 self.__futureRets[instrument] = []
#                 if useAdjustedCloseForReturns:
#                     ds = feed[instrument].getAdjCloseDataSeries()
#                 else:
#                     ds = feed[instrument].getCloseDataSeries()
#                 self.__rets[instrument] = roc.RateOfChange(ds, 1)
#
#             feed.getNewValuesEvent().subscribe(self.__onBars)
#             disp = dispatcher.Dispatcher()
#             disp.addSubject(feed)
#             disp.run()
#         finally:
#             feed.getNewValuesEvent().unsubscribe(self.__onBars)
#
#
# import abc
#
# import six
#
#
# class Frequency(object):
#
#     """Enum like class for bar frequencies. Valid values are:
#
#     * **Frequency.TRADE**: The bar represents a single trade.
#     * **Frequency.SECOND**: The bar summarizes the trading activity during 1 second.
#     * **Frequency.MINUTE**: The bar summarizes the trading activity during 1 minute.
#     * **Frequency.HOUR**: The bar summarizes the trading activity during 1 hour.
#     * **Frequency.DAY**: The bar summarizes the trading activity during 1 day.
#     * **Frequency.WEEK**: The bar summarizes the trading activity during 1 week.
#     * **Frequency.MONTH**: The bar summarizes the trading activity during 1 month.
#     """
#
#     # It is important for frequency values to get bigger for bigger windows.
#     TRADE = -1
#     SECOND = 1
#     MINUTE = 60
#     HOUR = 60*60
#     DAY = 24*60*60
#     WEEK = 24*60*60*7
#     MONTH = 24*60*60*31
#
#
# @six.add_metaclass(abc.ABCMeta)
# class Bar(object):
#
#     """A Bar is a summary of the trading activity for a security in a given period.
#
#     .. note::
#         This is a base class and should not be used directly.
#     """
#
#     @abc.abstractmethod
#     def setUseAdjustedValue(self, useAdjusted):
#         raise NotImplementedError()
#
#     @abc.abstractmethod
#     def getUseAdjValue(self):
#         raise NotImplementedError()
#
#     @abc.abstractmethod
#     def getDateTime(self):
#         """Returns the :class:`datetime.datetime`."""
#         raise NotImplementedError()
#
#     @abc.abstractmethod
#     def getOpen(self, adjusted=False):
#         """Returns the opening price."""
#         raise NotImplementedError()
#
#     @abc.abstractmethod
#     def getHigh(self, adjusted=False):
#         """Returns the highest price."""
#         raise NotImplementedError()
#
#     @abc.abstractmethod
#     def getLow(self, adjusted=False):
#         """Returns the lowest price."""
#         raise NotImplementedError()
#
#     @abc.abstractmethod
#     def getClose(self, adjusted=False):
#         """Returns the closing price."""
#         raise NotImplementedError()
#
#     @abc.abstractmethod
#     def getVolume(self):
#         """Returns the volume."""
#         raise NotImplementedError()
#
#     @abc.abstractmethod
#     def getAdjClose(self):
#         """Returns the adjusted closing price."""
#         raise NotImplementedError()
#
#     @abc.abstractmethod
#     def getFrequency(self):
#         """The bar's period."""
#         raise NotImplementedError()
#
#     def getTypicalPrice(self):
#         """Returns the typical price."""
#         return (self.getHigh() + self.getLow() + self.getClose()) / 3.0
#
#     @abc.abstractmethod
#     def getPrice(self):
#         """Returns the closing or adjusted closing price."""
#         raise NotImplementedError()
#
#     def getExtraColumns(self):
#         return {}
#
#
# class BasicBar(Bar):
#     # Optimization to reduce memory footprint.
#     __slots__ = (
#         '__dateTime',
#         '__open',
#         '__close',
#         '__high',
#         '__low',
#         '__volume',
#         '__adjClose',
#         '__frequency',
#         '__useAdjustedValue',
#         '__extra',
#     )
#
#     def __init__(self, dateTime, open_, high, low, close, volume, adjClose, frequency, extra={}):
#         if high < low:
#             raise Exception("high < low on %s" % (dateTime))
#         elif high < open_:
#             raise Exception("high < open on %s" % (dateTime))
#         elif high < close:
#             raise Exception("high < close on %s" % (dateTime))
#         elif low > open_:
#             raise Exception("low > open on %s" % (dateTime))
#         elif low > close:
#             raise Exception("low > close on %s" % (dateTime))
#
#         self.__dateTime = dateTime
#         self.__open = open_
#         self.__close = close
#         self.__high = high
#         self.__low = low
#         self.__volume = volume
#         self.__adjClose = adjClose
#         self.__frequency = frequency
#         self.__useAdjustedValue = False
#         self.__extra = extra
#
#     def __setstate__(self, state):
#         (self.__dateTime,
#             self.__open,
#             self.__close,
#             self.__high,
#             self.__low,
#             self.__volume,
#             self.__adjClose,
#             self.__frequency,
#             self.__useAdjustedValue,
#             self.__extra) = state
#
#     def __getstate__(self):
#         return (
#             self.__dateTime,
#             self.__open,
#             self.__close,
#             self.__high,
#             self.__low,
#             self.__volume,
#             self.__adjClose,
#             self.__frequency,
#             self.__useAdjustedValue,
#             self.__extra
#         )
#
#     def setUseAdjustedValue(self, useAdjusted):
#         if useAdjusted and self.__adjClose is None:
#             raise Exception("Adjusted close is not available")
#         self.__useAdjustedValue = useAdjusted
#
#     def getUseAdjValue(self):
#         return self.__useAdjustedValue
#
#     def getDateTime(self):
#         return self.__dateTime
#
#     def getOpen(self, adjusted=False):
#         if adjusted:
#             if self.__adjClose is None:
#                 raise Exception("Adjusted close is missing")
#             return self.__adjClose * self.__open / float(self.__close)
#         else:
#             return self.__open
#
#     def getHigh(self, adjusted=False):
#         if adjusted:
#             if self.__adjClose is None:
#                 raise Exception("Adjusted close is missing")
#             return self.__adjClose * self.__high / float(self.__close)
#         else:
#             return self.__high
#
#     def getLow(self, adjusted=False):
#         if adjusted:
#             if self.__adjClose is None:
#                 raise Exception("Adjusted close is missing")
#             return self.__adjClose * self.__low / float(self.__close)
#         else:
#             return self.__low
#
#     def getClose(self, adjusted=False):
#         if adjusted:
#             if self.__adjClose is None:
#                 raise Exception("Adjusted close is missing")
#             return self.__adjClose
#         else:
#             return self.__close
#
#     def getVolume(self):
#         return self.__volume
#
#     def getAdjClose(self):
#         return self.__adjClose
#
#     def getFrequency(self):
#         return self.__frequency
#
#     def getPrice(self):
#         if self.__useAdjustedValue:
#             return self.__adjClose
#         else:
#             return self.__close
#
#     def getExtraColumns(self):
#         return self.__extra
#
# @six.add_metaclass(abc.ABCMeta)
# class Subject(object):
#
#     def __init__(self):
#         self.__dispatchPrio = dispatchprio.LAST
#
#     # This may raise.
#     @abc.abstractmethod
#     def start(self):
#         pass
#
#     # This should not raise.
#     @abc.abstractmethod
#     def stop(self):
#         raise NotImplementedError()
#
#     # This should not raise.
#     @abc.abstractmethod
#     def join(self):
#         raise NotImplementedError()
#
#     # Return True if there are not more events to dispatch.
#     @abc.abstractmethod
#     def eof(self):
#         raise NotImplementedError()
#
#     # Dispatch events. If True is returned, it means that at least one event was dispatched.
#     @abc.abstractmethod
#     def dispatch(self):
#         raise NotImplementedError()
#
#     @abc.abstractmethod
#     def peekDateTime(self):
#         # Return the datetime for the next event.
#         # This is needed to properly synchronize non-realtime subjects.
#         # Return None since this is a realtime subject.
#         raise NotImplementedError()
#
#     def getDispatchPriority(self):
#         # Returns a priority used to sort subjects within the dispatch queue.
#         # The return value should never change once this subject is added to the dispatcher.
#         return self.__dispatchPrio
#
#     def setDispatchPriority(self, dispatchPrio):
#         self.__dispatchPrio = dispatchPrio
#
#     def onDispatcherRegistered(self, dispatcher):
#         # Called when the subject is registered with a dispatcher.
#         pass
#
#
# class Bars(object):
#
#     """A group of :class:`Bar` objects.
#
#     :param barDict: A map of instrument to :class:`Bar` objects.
#     :type barDict: map.
#
#     .. note::
#         All bars must have the same datetime.
#     """
#
#     def __init__(self, barDict):
#         if len(barDict) == 0:
#             raise Exception("No bars supplied")
#
#         # Check that bar datetimes are in sync
#         firstDateTime = None
#         firstInstrument = None
#         for instrument, currentBar in six.iteritems(barDict):
#             if firstDateTime is None:
#                 firstDateTime = currentBar.getDateTime()
#                 firstInstrument = instrument
#             elif currentBar.getDateTime() != firstDateTime:
#                 raise Exception("Bar data times are not in sync. %s %s != %s %s" % (
#                     instrument,
#                     currentBar.getDateTime(),
#                     firstInstrument,
#                     firstDateTime
#                 ))
#
#         self.__barDict = barDict
#         self.__dateTime = firstDateTime
#
#     def __getitem__(self, instrument):
#         """Returns the :class:`pyalgotrade.bar.Bar` for the given instrument.
#         If the instrument is not found an exception is raised."""
#         return self.__barDict[instrument]
#
#     def __contains__(self, instrument):
#         """Returns True if a :class:`pyalgotrade.bar.Bar` for the given instrument is available."""
#         return instrument in self.__barDict
#
#     def items(self):
#         return list(self.__barDict.items())
#
#     def keys(self):
#         return list(self.__barDict.keys())
#
#     def getInstruments(self):
#         """Returns the instrument symbols."""
#         return list(self.__barDict.keys())
#
#     def getDateTime(self):
#         """Returns the :class:`datetime.datetime` for this set of bars."""
#         return self.__dateTime
#
#     def getBar(self, instrument):
#         """Returns the :class:`pyalgotrade.bar.Bar` for the given instrument or None if the instrument is not found."""
#         return self.__barDict.get(instrument, None)

# from functools import partial
#
# def reduction(x,y,wgt):
#     res = x * (1- wgt) + y * wgt
#     return res
#
#
# if __name__ == '__main__':
#     frozen = partial(reduction,wgt = 0.7)
#     result = frozen(5,3)
#     print(result)

# test = "192.0.0.1?!289.0.0.1!0.0.0.0!192.163.10.28?192.0.0.1"
# test_replace = test.replace('?','!')
# test_tuple = test_replace.split('!')
# test_sorted = sorted(test_tuple,key = lambda x : x.split('.')[-1])[1:]
# print(test_sorted)
#
# a=' '.join(sorted(test.replace('?','!').split('!'),key=lambda x:x.split('.')[-1])).split()
# print(a)
#numpy memmap
#判断是否有非法字符
# def check_validate(List):
#     if len(List) == 0:
#         raise ValueError('the output of Pipeline must be not null')
#     pattern = re.compile('^(6|0|3)(\d){5}.(SZ|SH)$')
#     for idx,item in enumerate(List):
#         match = pattern.match(item.upper())
#         if not match :
#             raise ValueError('invalid stockCode : %s in prediction'%match.group())
#
# class EventEngine():
#     """
#         定义策略引擎将不同的算法按照顺序放到一个队列里面依次进行执行，算法对应事件，可以注册、剔除事件
#     """
#     def __init__(self):
#
#         self._queue = Queue()
#         self._thread = Thread(target=self._run)
#         # 以计时模块为例，换成其他的需求添加都队列里面
#         self._timer = Thread(target=self._run_timer)
#         self._handlers = defaultdict(list)
#         self._general = []
#
#     def _run(self):
#         while self._active:
#             try:
#                 algo = self._queue.get(block=True, timeout=1)
#                 self._process(algo)
#             except Empty:
#                 pass
#
#     def _process(self, algo):
#         if algo._type in self._handlers:
#             [handler(algo) for handler in self._handlers[algo._type]]
#
#         if self._general:
#             [handler(algo) for handler in self._general]
#
#     def _run_timer(self):
#
#         while self._active:
#             # sleep(self._interval)
#             event = Event('timer')
#             self.put(event)
#
#     def start(self):
#
#         self._active = True
#         self._thread.start()
#         self._timer.start()
#
#     def stop(self):
#         self._active = False
#         self._timer.join()
#         self._thread.join()
#
#     def put(self, event):
#         self._queue.put(event)
#from collections import defaultdict
# from urllib.request import urlopen
# from selenium.webdriver import Chrome
# from bs4 import BeautifulSoup
# import requests
# #存在反爬虫
# ths = 'http://d.10jqka.com.cn/v6/line/hs_002570/01/all.js'
# #ths = 'http://stockpage.10jqka.com.cn/HQ_v4.html'
# #obj = urlopen(ths)
# headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36'}
# obj = requests.get(url=ths,headers = headers)
# # drive = Chrome()
# # obj = drive.get(ths)
# # input = drive.find_element_by_link_text('002570')
# # search = drive.find_element_by_id('su')
# # input.send_keys('002570.SZ')
# # search.click()
# #ths = 'https://link.jianshu.com/?t=http://stockpage.10jqka.com.cn/600196/finance/#view'
# #obj = urlopen(ths)
# print(obj)
# import pymysql
# from DBUtils.PooledDB import PooledDB
#
# class Ora():
#     """
#         分为 simplepooleddb steadydb persistentdb pooleddb
#         from DBUtils.PersistentDB import PersistentDB
#         @property @staticmethod @classmethod(cls,)
#         db_oracle={'user':'factor_factory','password':'htfactor123','host':,'port':,'sid':}
#         pool_name: 连接池的名称，多种连接参数对应多个不同的连接池对象，多单例模式；
#         host: 数据库地址
#         user: 数据库服务器用户名
#         password: 用户密码
#         database: 默认选择的数据库
#         port: 数据库服务器的端口
#         charset: 字符集，默认为 ‘utf8'
#         use_dict_cursor: 使用字典格式或者元组返回数据；
#         max_pool_size: 连接池优先最大连接数；
#         step_size: 连接池动态增加连接数大小；
#         enable_auto_resize: 是否动态扩展连接池，即当超过 max_pool_size 时，自动扩展 max_pool_size；
#         pool_resize_boundary: 该配置为连接池最终可以增加的上上限大小，即时扩展也不可超过该值；
#         auto_resize_scale: 自动扩展 max_pool_size 的增益，默认为 1.5 倍扩展；
#         wait_timeout: 在排队等候连接对象时，最多等待多久，当超时时连接池尝试自动扩展当前连接数；
#         kwargs: 其他配置参数将会在创建连接对象时传递给
#
#         frame.to_sql(tablename,conn,if_exists='append',chunksize=50000)
#
#         result = pd.read_sql('select * from "{}"'.format(table_name),conn,index_col='index',**kwargs).rename_axis(None)


# 该软件包中的功能要求子项可以导入 __main__ 模块。这包含在 编程指导，
# 例如 multiprocessing.pool.Pool 示例在交互式解释器中不起作用
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Sat Feb 16 13:56:19 2019
#
# @author: python
# """
#
#
# class RpcGateway(BaseGateway):
#     """
#     VN Trader Gateway for RPC service.
#     """
#
#     default_setting = {
#         "主动请求地址": "tcp://127.0.0.1:2014",
#         "推送订阅地址": "tcp://127.0.0.1:4102"
#     }
#
#
#     def __init__(self, event_engine):
#         """Constructor"""
#         super().__init__(event_engine, "RPC")
#
#         self.symbol_gateway_map = {}
#
#         self.client = RpcClient()
#         self.client.callback = self.client_callback
#
#     def connect(self, setting: dict):
#         """"""
#         req_address = setting["主动请求地址"]
#         pub_address = setting["推送订阅地址"]
#
#         self.client.subscribe_topic("")
#         self.client.start(req_address, pub_address)
#
#         self.write_log("服务器连接成功，开始初始化查询")
#
#         self.query_all()
#
#     def subscribe(self, req: SubscribeRequest):
#         """"""
#         gateway_name = self.symbol_gateway_map.get(req.vt_symbol, "")
#         self.client.subscribe(req, gateway_name)
#
#     def send_order(self, req: OrderRequest):
#         """"""
#         gateway_name = self.symbol_gateway_map.get(req.vt_symbol, "")
#         self.client.send_order(req, gateway_name)
#
#     def cancel_order(self, req: CancelRequest):
#         """"""
#         gateway_name = self.symbol_gateway_map.get(req.vt_symbol, "")
#         self.client.cancel_order(req, gateway_name)
#
#     def query_account(self):
#         """"""
#         pass
#
#     def query_position(self):
#         """"""
#         pass
#
#     def query_all(self):
#         """"""
#         contracts = self.client.get_all_contracts()
#         for contract in contracts:
#             self.symbol_gateway_map[contract.vt_symbol] = contract.gateway_name
#             contract.gateway_name = self.gateway_name
#             self.on_contract(contract)
#         self.write_log("合约信息查询成功")
#
#         accounts = self.client.get_all_accounts()
#         for account in accounts:
#             account.gateway_name = self.gateway_name
#             self.on_account(account)
#         self.write_log("资金信息查询成功")
#
#         positions = self.client.get_all_positions()
#         for position in positions:
#             position.gateway_name = self.gateway_name
#             self.on_position(position)
#         self.write_log("持仓信息查询成功")
#
#         orders = self.client.get_all_orders()
#         for order in orders:
#             order.gateway_name = self.gateway_name
#             self.on_order(order)
#         self.write_log("委托信息查询成功")
#
#         trades = self.client.get_all_trades()
#         for trade in trades:
#             trade.gateway_name = self.gateway_name
#             self.on_trade(trade)
#         self.write_log("成交信息查询成功")
#
#     def close(self):
#         """"""
#         self.client.stop()
#
#     def client_callback(self, topic: str, event: Event):
#         """"""
#         if event is None:
#             print("none event", topic, event)
#             return
#
#         data = event.data
#
#         if hasattr(data, "gateway_name"):
#             data.gateway_name = self.gateway_name
#
#         self.event_engine.put(event)

# subprocess replace os.system(),os.spawnv(),os , popen2,command
# subprocess -- run call check_call check_output
# subprocess.run(['ls'])
# call("pip install --upgrade h5py",shell=True)
from dateutil.relativedelta import relativedelta as timedelta
from datetime import timedelta
from glob import glob
from textwrap import dedent
from collections import namedtuple
from numpy import full, nan, int64, zeros
from inspect import signature, Parameter
import csv

# with open(filepath, 'w', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=' ',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     for r in kline.values.tolist():
#         print(r)
#         spamwriter.writerow(r)
#         spamwriter.writerow('\n')

# test = '{page:test,data:["test":3]}'
# import re
# match = re.search('\[(.*.)\]',test)
# print(match.group())
# https://pypi.tuna.tsinghua.edu.cn/simple
#apply --- dataframe -行或者一列
#applymap --- dataframe 每一个元素
#map --- series

class A(object):

    def __init__(self,a):
        self.a = a

    def trans(self):
        b = self.a
        if b >0 :
            b = 2
        else:
            b = 3
        print('b',b)
        print('------',self.a)

# b = A(4)
# b.trans()
# print(b.a)