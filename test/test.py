# # class Parent:
# #     def __init__(self,p_name='parent',type='super'):
# #         self.p_name = p_name
# #         self.type = type
# #
# #     def run(self):
# #         print('class Parent',self.p_name)
# #
# # class Child(Parent):
# #     def __init__(self,c_name = 'Child'):
# #         super().__init__(p_name = 'test',type='test')
# #         self._type = c_name
# #         #self.p_name = 'test'
# #
# #     def private(self):
# #         super().run()
# #         print('Child',self.p_name,self.type)
# #
# # def func(x):
# #     return x+2
# #
# # if __name__=='__main__':
# #     # child = Child()
# #     # #child.run()
# #     # child.private()
# #     # #parent = Parent()
# #     # #parent.run()
# #     from multiprocessing import Pool
# #     pool = Pool(4)
# #     res = [pool.apply_async(func,(i,)) for i in range(10)]
# #     for r in res:
# #         print(r.get())
# #
# #     test = dict()
# #     test[Child] = 3
# #     print(test.keys())
# #     print(type(test.values()))
#
# #生成器：基于yield方法将函数转化为迭代器，next方法，每次执行到yield停止；而iter（迭代器将非可迭代对象强制转化为对象）
# # def test(n):
# #     for id_ in range(n):
# #         yield id_
# #         print('----------yield:',id_)
# #
# # iter = test(5)
# # #print(iter)
# # # for i in iter:
# # #     pass
# # next(iter)
#
# # from itertools import chain,repeat
# #
# # chunks = chain([5], repeat(126))
# # print(chunks)
# # print(next(chunks))
# # if not Xs:
# #     # All transformers are None
# #     return np.zeros((X.shape[0], 0))
# # if any(sparse.issparse(f) for f in Xs):
# #     Xs = sparse.hstack(Xs).tocsr() Compressed Sparse Row format
# # else:
# #     Xs = np.hstack(Xs)
#
# # import functools
# # from interface import Interface,Implements
# # class A(Interface):
# #
# #     def test(self,a,b):
# #         print(a,b)
# #
# # class B(implements(A)):
# #
# #     def test(self,a,c):
# #         b = a+c
# #         print(b)
#
# #if __name__ =='__main__':
#     # print(type(A))
#     # a = A(3)
#     # b = A(4)
#     # print('-------------')
#     # print(a==b)
#     # print(a.__name__)
#     # print(b.__name__)
#     # # flag_c = isinstance(a,A)
#     # # print(flag_c)
#     # # flag_d = isinstance(A,A)
#     # # print(flag_d)
# #
# # class A:
# #
# #     '''test A '''
# #
# #     def __init__(self,a):
# #         self.__name__ = 'passthrough'
# #
# #     def test(self,c):
# #         print(c)
# #
# #     def __call__(self,b):
# #         print(b)
# #
# # print(A.__doc__)
# # print(A.__name__)
# # A().test('a')
# # print(callable(A))
# #
# # a = A(3)
# # a(5)
#
# # print(a.__dir__())
# # print(a.__doc__)
# # print(type(A.__name__))
# # print(A.__dict__)
#
#
# # from functools import partial,wraps
# # #wraps 单纯的修饰函数
# # def _validate_type(_type):
# #     def decorate(func):
# #         def wrap(*args):
# #             res = func(*args)
# #             print('----------------------1')
# #             if not isinstance(res,_type):
# #                 print('--------------------------2')
# #                 try:
# #                     res = _type(res)
# #                 except:
# #                     print('can not algorithm type:%s'%_type)
# #             return res
# #         return wrap
# #     return decorate
# #
# # @_validate_type(int)
# # def func(a,b):
# #     c= a+b
# #     return c
# #
# #
# # res = func('5','4')
# # print(type(res))
# # raise TypeError('------------------------------2')
#
# # data = None
# # for file in os.listdir(folderpath):
# #     if '.csv' not in file:
# #         continue
# #     raw = load_prices_from_csv(os.path.join(folderpath, file),
# #                                identifier_col, tz)
# #     if data is None:
# #         data = raw
# #     else:
# #         data = pd.concat([data, raw], axis=1)
#
# # from collections import ChainMap
# # #有多个字典或者映射，想把它们合并成为一个单独的映射
# # c=ChainMap()
# # d=c.new_child()
# # e=c.new_child()
# # e.parents
#
# # from  decimal import Decimal
# # from  decimal import getcontext
# # #整数、字符串或者元组构建decimal.Decimal，对于浮点数需要先将其转换为字符串
# # d_context = getcontext()
# # d_context.prec = 6
# #
# # d = Decimal(1) / Decimal(3)
# # print(type(d), d)
# # 两分查找,复杂度对数级别的
# # from bisect import bisect_left
#
# # 堆队列(数值小，优先权高)
# # from heapq import heappush
# # a=[]
# # heappush(a,5)
# # heappush(a,3)
# # heappush(a,9)
# # heappush(a,15)
#
# # from collections import deque，defaultdict
# # dqueue 双向队列
# # fifo=deque('ab')
# # fifo.appendleft('c')
# # fifo.append('d')
# # fifo.popleft()
#
# # 默认为0
# # stats=defaultdict(int)
# # os.path.expandvars(path
# # 根据环境变量的值替换path中包含的
# # "$name"
# # expanduser('~') - -- 用户目录
# # os.path.basename | os.path.dirname
# # np.fmax(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', subok=True)
# # 通过scipy.interpolate.interp1d插值形成的模型，通过sco.fmin_bfgs计算min
# # param
# # find_min_pos: 寻找min的点位值
# # param
# # linear_interp: scipy.interpolate.interp1d插值形成的模型
# # local_min_pos = sco.fmin_bfgs(linear_interp, find_min_pos, disp=False)[0]
# # scipy.interpolate.interp1d插值形成模型
#
# # import sympy as sy
# # 符号计算（在交换式金融分析，比较有效）
# # sy.Symbol('x') sy.sqrt(x)
# # sy.solve(x ** 2 -1 ) 方程式右边为0
# # #打印出符号积分
# # sy.pretty(sy.Intergal(sy.sin(x) + 0.5 * x,x))
# # #积分,求出反导数
# # init_func = sy.intergrate(sy.sin(x) + 0.5 * x,x)
# # #求导
# # init_func.diff()
# # #偏导数
# # init_func.diff(,x)
# # #subs代入数值，evalf求值
# # init_func.subs(x,0.5).evalf()
# # #nsolve与solve
# # from sympy.solvers import nsolve
# # solve 处理等式右边为0的表达式；而nsolve处理表达式（范围更加广）
# #from pyalgotrade import dataseries
#
# #
# # def datetime_aligned(ds1, ds2, maxLen=None):
# #     """
# #     Returns two dataseries that exhibit only those values whose datetimes are in both dataseries.
# #
# #     :param ds1: A DataSeries instance.
# #     :type ds1: :class:`DataSeries`.
# #     :param ds2: A DataSeries instance.
# #     :type ds2: :class:`DataSeries`.
# #     :param maxLen: The maximum number of values to hold for the returned :class:`DataSeries`.
# #         Once a bounded length is full, when new items are added, a corresponding number of items are discarded from the
# #         opposite end. If None then dataseries.DEFAULT_MAX_LEN is used.
# #     :type maxLen: int.
# #     """
# #     aligned1 = dataseries.SequenceDataSeries(maxLen)
# #     aligned2 = dataseries.SequenceDataSeries(maxLen)
# #     Syncer(ds1, ds2, aligned1, aligned2)
# #     return (aligned1, aligned2)
# #
# #
# # # This class is responsible for filling 2 dataseries when 2 other dataseries get new values.
# # class Syncer(object):
# #     def __init__(self, sourceDS1, sourceDS2, destDS1, destDS2):
# #         self.__values1 = []  # (datetime, value)
# #         self.__values2 = []  # (datetime, value)
# #         self.__destDS1 = destDS1
# #         self.__destDS2 = destDS2
# #         sourceDS1.getNewValueEvent().subscribe(self.__onNewValue1)
# #         sourceDS2.getNewValueEvent().subscribe(self.__onNewValue2)
# #         # Source dataseries will keep a reference to self and that will prevent from getting this destroyed.
# #
# #     # Scan backwards for the position of dateTime in ds.
# #     def __findPosForDateTime(self, values, dateTime):
# #         ret = None
# #         i = len(values) - 1
# #         while i >= 0:
# #             if values[i][0] == dateTime:
# #                 ret = i
# #                 break
# #             elif values[i][0] < dateTime:
# #                 break
# #             i -= 1
# #         return ret
# #
# #     def __onNewValue1(self, dataSeries, dateTime, value):
# #         pos2 = self.__findPosForDateTime(self.__values2, dateTime)
# #         # If a value for dateTime was added to first dataseries, and a value for that same datetime is also in the second one
# #         # then append to both destination dataseries.
# #         if pos2 is not None:
# #             self.__append(dateTime, value, self.__values2[pos2][1])
# #             # Reset buffers.
# #             self.__values1 = []
# #             self.__values2 = self.__values2[pos2+1:]
# #         else:
# #             # Since source dataseries may not hold all the values we need, we need to buffer manually.
# #             self.__values1.append((dateTime, value))
# #
# #     def __onNewValue2(self, dataSeries, dateTime, value):
# #         pos1 = self.__findPosForDateTime(self.__values1, dateTime)
# #         # If a value for dateTime was added to second dataseries, and a value for that same datetime is also in the first one
# #         # then append to both destination dataseries.
# #         if pos1 is not None:
# #             self.__append(dateTime, self.__values1[pos1][1], value)
# #             # Reset buffers.
# #             self.__values1 = self.__values1[pos1+1:]
# #             self.__values2 = []
# #         else:
# #             # Since source dataseries may not hold all the values we need, we need to buffer manually.
# #             self.__values2.append((dateTime, value))
# #
# #     def __append(self, dateTime, value1, value2):
# #         self.__destDS1.appendWithDateTime(dateTime, value1)
# #         self.__destDS2.appendWithDateTime(dateTime, value2)
# #
# #
# #
# #
# #
# # import abc
# #
# # import six
# #
# # from pyalgotrade import dataseries
# # from pyalgotrade.dataseries import bards
# # from pyalgotrade import bar
# # from pyalgotrade import resamplebase
# #
# #
# # class AggFunGrouper(resamplebase.Grouper):
# #     def __init__(self, groupDateTime, value, aggfun):
# #         super(AggFunGrouper, self).__init__(groupDateTime)
# #         self.__values = [value]
# #         self.__aggfun = aggfun
# #
# #     def addValue(self, value):
# #         self.__values.append(value)
# #
# #     def getGrouped(self):
# #         return self.__aggfun(self.__values)
# #
# #
# # class BarGrouper(resamplebase.Grouper):
# #     def __init__(self, groupDateTime, bar_, frequency):
# #         super(BarGrouper, self).__init__(groupDateTime)
# #         self.__open = bar_.getOpen()
# #         self.__high = bar_.getHigh()
# #         self.__low = bar_.getLow()
# #         self.__close = bar_.getClose()
# #         self.__volume = bar_.getVolume()
# #         self.__adjClose = bar_.getAdjClose()
# #         self.__useAdjValue = bar_.getUseAdjValue()
# #         self.__frequency = frequency
# #
# #     def addValue(self, value):
# #         self.__high = max(self.__high, value.getHigh())
# #         self.__low = min(self.__low, value.getLow())
# #         self.__close = value.getClose()
# #         self.__adjClose = value.getAdjClose()
# #         self.__volume += value.getVolume()
# #
# #     def getGrouped(self):
# #         """Return the grouped value."""
# #         ret = bar.BasicBar(
# #             self.getDateTime(),
# #             self.__open, self.__high, self.__low, self.__close, self.__volume, self.__adjClose,
# #             self.__frequency
# #         )
# #         ret.setUseAdjustedValue(self.__useAdjValue)
# #         return ret
# #
# #
# # @six.add_metaclass(abc.ABCMeta)
# # class DSResampler(object):
# #
# #     def initDSResampler(self, dataSeries, frequency):
# #         if not resamplebase.is_valid_frequency(frequency):
# #             raise Exception("Unsupported frequency")
# #
# #         self.__frequency = frequency
# #         self.__grouper = None
# #         self.__range = None
# #
# #         dataSeries.getNewValueEvent().subscribe(self.__onNewValue)
# #
# #     @abc.abstractmethod
# #     def buildGrouper(self, range_, value, frequency):
# #         raise NotImplementedError()
# #
# #     def __onNewValue(self, dataSeries, dateTime, value):
# #         if self.__range is None:
# #             self.__range = resamplebase.build_range(dateTime, self.__frequency)
# #             self.__grouper = self.buildGrouper(self.__range, value, self.__frequency)
# #         elif self.__range.belongs(dateTime):
# #             self.__grouper.addValue(value)
# #         else:
# #             self.appendWithDateTime(self.__grouper.getDateTime(), self.__grouper.getGrouped())
# #             self.__range = resamplebase.build_range(dateTime, self.__frequency)
# #             self.__grouper = self.buildGrouper(self.__range, value, self.__frequency)
# #
# #     def pushLast(self):
# #         if self.__grouper is not None:
# #             self.appendWithDateTime(self.__grouper.getDateTime(), self.__grouper.getGrouped())
# #             self.__grouper = None
# #             self.__range = None
# #
# #     def checkNow(self, dateTime):
# #         if self.__range is not None and not self.__range.belongs(dateTime):
# #             self.appendWithDateTime(self.__grouper.getDateTime(), self.__grouper.getGrouped())
# #             self.__grouper = None
# #             self.__range = None
# #
# #
# # class ResampledBarDataSeries(bards.BarDataSeries, DSResampler):
# #     """A BarDataSeries that will build on top of another, higher frequency, BarDataSeries.
# #     Resampling will take place as new values get pushed into the dataseries being resampled.
# #
# #     :param dataSeries: The DataSeries instance being resampled.
# #     :type dataSeries: :class:`pyalgotrade.dataseries.bards.BarDataSeries`
# #     :param frequency: The grouping frequency in seconds. Must be > 0.
# #     :param maxLen: The maximum number of values to hold.
# #         Once a bounded length is full, when new items are added, a corresponding number of items are discarded
# #         from the opposite end.
# #     :type maxLen: int.
# #
# #     .. note::
# #         * Supported resampling frequencies are:
# #             * Less than bar.Frequency.DAY
# #             * bar.Frequency.DAY
# #             * bar.Frequency.MONTH
# #     """
# #
# #     def __init__(self, dataSeries, frequency, maxLen=None):
# #         if not isinstance(dataSeries, bards.BarDataSeries):
# #             raise Exception("dataSeries must be a dataseries.bards.BarDataSeries instance")
# #
# #         super(ResampledBarDataSeries, self).__init__(maxLen)
# #         self.initDSResampler(dataSeries, frequency)
# #
# #     def checkNow(self, dateTime):
# #         """Forces a resample check. Depending on the resample frequency, and the current datetime, a new
# #         value may be generated.
# #
# #        :param dateTime: The current datetime.
# #        :type dateTime: :class:`datetime.datetime`
# #         """
# #
# #         return super(ResampledBarDataSeries, self).checkNow(dateTime)
# #
# #     def buildGrouper(self, range_, value, frequency):
# #         return BarGrouper(range_.getBeginning(), value, frequency)
# #
# #
# # class ResampledDataSeries(dataseries.SequenceDataSeries, DSResampler):
# #     def __init__(self, dataSeries, frequency, aggfun, maxLen=None):
# #         super(ResampledDataSeries, self).__init__(maxLen)
# #         self.initDSResampler(dataSeries, frequency)
# #         self.__aggfun = aggfun
# #
# #     def buildGrouper(self, range_, value, frequency):
# #         return AggFunGrouper(range_.getBeginning(), value, self.__aggfun)
# #
# #
# #
# # class Dispatcher(object):
# #     def __init__(self):
# #         self.__subjects = []
# #         self.__stop = False
# #         self.__startEvent = observer.Event()
# #         self.__idleEvent = observer.Event()
# #         self.__currDateTime = None
# #
# #     # Returns the current event datetime. It may be None for events from realtime subjects.
# #     def getCurrentDateTime(self):
# #         return self.__currDateTime
# #
# #     def getStartEvent(self):
# #         return self.__startEvent
# #
# #     def getIdleEvent(self):
# #         return self.__idleEvent
# #
# #     def stop(self):
# #         self.__stop = True
# #
# #     def getSubjects(self):
# #         return self.__subjects
# #
# #     def addSubject(self, subject):
# #         # Skip the subject if it was already added.
# #         if subject in self.__subjects:
# #             return
# #
# #         # If the subject has no specific dispatch priority put it right at the end.
# #         if subject.getDispatchPriority() is dispatchprio.LAST:
# #             self.__subjects.append(subject)
# #         else:
# #             # Find the position according to the subject's priority.
# #             pos = 0
# #             for s in self.__subjects:
# #                 if s.getDispatchPriority() is dispatchprio.LAST or subject.getDispatchPriority() < s.getDispatchPriority():
# #                     break
# #                 pos += 1
# #             self.__subjects.insert(pos, subject)
# #
# #         subject.onDispatcherRegistered(self)
# #
# #     # Return True if events were dispatched.
# #     def __dispatchSubject(self, subject, currEventDateTime):
# #         ret = False
# #         # Dispatch if the datetime is currEventDateTime of if its a realtime subject.
# #         if not subject.eof() and subject.peekDateTime() in (None, currEventDateTime):
# #             ret = subject.dispatch() is True
# #         return ret
# #
# #     # Returns a tuple with booleans
# #     # 1: True if all subjects hit eof
# #     # 2: True if at least one subject dispatched events.
# #     def __dispatch(self):
# #         smallestDateTime = None
# #         eof = True
# #         eventsDispatched = False
# #
# #         # Scan for the lowest datetime.
# #         for subject in self.__subjects:
# #             if not subject.eof():
# #                 eof = False
# #                 smallestDateTime = util.safe_min(smallestDateTime, subject.peekDateTime())
# #
# #         # Dispatch realtime subjects and those subjects with the lowest datetime.
# #         if not eof:
# #             self.__currDateTime = smallestDateTime
# #
# #             for subject in self.__subjects:
# #                 if self.__dispatchSubject(subject, smallestDateTime):
# #                     eventsDispatched = True
# #         return eof, eventsDispatched
# #
# #     def run(self):
# #         try:
# #             for subject in self.__subjects:
# #                 subject.start()
# #
# #             self.__startEvent.emit()
# #
# #             while not self.__stop:
# #                 eof, eventsDispatched = self.__dispatch()
# #                 if eof:
# #                     self.__stop = True
# #                 elif not eventsDispatched:
# #                     self.__idleEvent.emit()
# #         finally:
# #             # There are no more events.
# #             self.__currDateTime = None
# #
# #             for subject in self.__subjects:
# #                 subject.stop()
# #             for subject in self.__subjects:
# #                 subject.join()
# #
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from six.moves import xrange
# #
# # from pyalgotrade.technical import roc
# # from pyalgotrade import dispatcher
# #
# #
# # class Results(object):
# #     """Results from the profiler."""
# #     def __init__(self, eventsDict, lookBack, lookForward):
# #         assert(lookBack > 0)
# #         assert(lookForward > 0)
# #         self.__lookBack = lookBack
# #         self.__lookForward = lookForward
# #         self.__values = [[] for i in xrange(lookBack+lookForward+1)]
# #         self.__eventCount = 0
# #
# #         # Process events.
# #         for instrument, events in eventsDict.items():
# #             for event in events:
# #                 # Skip events which are on the boundary or for some reason are not complete.
# #                 if event.isComplete():
# #                     self.__eventCount += 1
# #                     # Compute cumulative returns: (1 + R1)*(1 + R2)*...*(1 + Rn)
# #                     values = np.cumprod(event.getValues() + 1)
# #                     # Normalize everything to the time of the event
# #                     values = values / values[event.getLookBack()]
# #                     for t in range(event.getLookBack()*-1, event.getLookForward()+1):
# #                         self.setValue(t, values[t+event.getLookBack()])
# #
# #     def __mapPos(self, t):
# #         assert(t >= -1*self.__lookBack and t <= self.__lookForward)
# #         return t + self.__lookBack
# #
# #     def setValue(self, t, value):
# #         if value is None:
# #             raise Exception("Invalid value at time %d" % (t))
# #         pos = self.__mapPos(t)
# #         self.__values[pos].append(value)
# #
# #     def getValues(self, t):
# #         pos = self.__mapPos(t)
# #         return self.__values[pos]
# #
# #     def getLookBack(self):
# #         return self.__lookBack
# #
# #     def getLookForward(self):
# #         return self.__lookForward
# #
# #     def getEventCount(self):
# #         """Returns the number of events occurred. Events that are on the boundary are skipped."""
# #         return self.__eventCount
# #
# #
# # class Predicate(object):
# #     """Base class for event identification. You should subclass this to implement
# #     the event identification logic."""
# #
# #     def eventOccurred(self, instrument, bards):
# #         """Override (**mandatory**) to determine if an event took place in the last bar (bards[-1]).
# #
# #         :param instrument: Instrument identifier.
# #         :type instrument: string.
# #         :param bards: The BarDataSeries for the given instrument.
# #         :type bards: :class:`pyalgotrade.dataseries.bards.BarDataSeries`.
# #         :rtype: boolean.
# #         """
# #         raise NotImplementedError()
# #
# #
# # class Event(object):
# #     def __init__(self, lookBack, lookForward):
# #         assert(lookBack > 0)
# #         assert(lookForward > 0)
# #         self.__lookBack = lookBack
# #         self.__lookForward = lookForward
# #         self.__values = np.empty((lookBack + lookForward + 1))
# #         self.__values[:] = np.NAN
# #
# #     def __mapPos(self, t):
# #         assert(t >= -1*self.__lookBack and t <= self.__lookForward)
# #         return t + self.__lookBack
# #
# #     def isComplete(self):
# #         return not any(np.isnan(self.__values))
# #
# #     def getLookBack(self):
# #         return self.__lookBack
# #
# #     def getLookForward(self):
# #         return self.__lookForward
# #
# #     def setValue(self, t, value):
# #         if value is not None:
# #             pos = self.__mapPos(t)
# #             self.__values[pos] = value
# #
# #     def getValue(self, t):
# #         pos = self.__mapPos(t)
# #         return self.__values[pos]
# #
# #     def getValues(self):
# #         return self.__values
# #
# #
# # class Profiler(object):
# #     """This class is responsible for scanning over historical data and analyzing returns before
# #     and after the events.
# #
# #     :param predicate: A :class:`Predicate` subclass responsible for identifying events.
# #     :type predicate: :class:`Predicate`.
# #     :param lookBack: The number of bars before the event to analyze. Must be > 0.
# #     :type lookBack: int.
# #     :param lookForward: The number of bars after the event to analyze. Must be > 0.
# #     :type lookForward: int.
# #     """
# #
# #     def __init__(self, predicate, lookBack, lookForward):
# #         assert(lookBack > 0)
# #         assert(lookForward > 0)
# #         self.__predicate = predicate
# #         self.__lookBack = lookBack
# #         self.__lookForward = lookForward
# #         self.__feed = None
# #         self.__rets = {}
# #         self.__futureRets = {}
# #         self.__events = {}
# #
# #     def __addPastReturns(self, instrument, event):
# #         begin = (event.getLookBack() + 1) * -1
# #         for t in xrange(begin, 0):
# #             try:
# #                 ret = self.__rets[instrument][t]
# #                 if ret is not None:
# #                     event.setValue(t+1, ret)
# #             except IndexError:
# #                 pass
# #
# #     def __addCurrentReturns(self, instrument):
# #         nextTs = []
# #         for event, t in self.__futureRets[instrument]:
# #             event.setValue(t, self.__rets[instrument][-1])
# #             if t < event.getLookForward():
# #                 t += 1
# #                 nextTs.append((event, t))
# #         self.__futureRets[instrument] = nextTs
# #
# #     def __onBars(self, dateTime, bars):
# #         for instrument in bars.getInstruments():
# #             self.__addCurrentReturns(instrument)
# #             eventOccurred = self.__predicate.eventOccurred(instrument, self.__feed[instrument])
# #             if eventOccurred:
# #                 event = Event(self.__lookBack, self.__lookForward)
# #                 self.__events[instrument].append(event)
# #                 self.__addPastReturns(instrument, event)
# #                 # Add next return for this instrument at t=1.
# #                 self.__futureRets[instrument].append((event, 1))
# #
# #     def getResults(self):
# #         """Returns the results of the analysis.
# #
# #         :rtype: :class:`Results`.
# #         """
# #         return Results(self.__events, self.__lookBack, self.__lookForward)
# #
# #     def run(self, feed, useAdjustedCloseForReturns=True):
# #         """Runs the analysis using the bars supplied by the feed.
# #
# #         :param barFeed: The bar feed to use to run the analysis.
# #         :type barFeed: :class:`pyalgotrade.barfeed.BarFeed`.
# #         :param useAdjustedCloseForReturns: True if adjusted close values should be used to calculate returns.
# #         :type useAdjustedCloseForReturns: boolean.
# #         """
# #
# #         if useAdjustedCloseForReturns:
# #             assert feed.barsHaveAdjClose(), "Feed doesn't have adjusted close values"
# #
# #         try:
# #             self.__feed = feed
# #             self.__rets = {}
# #             self.__futureRets = {}
# #             for instrument in feed.getRegisteredInstruments():
# #                 self.__events.setdefault(instrument, [])
# #                 self.__futureRets[instrument] = []
# #                 if useAdjustedCloseForReturns:
# #                     ds = feed[instrument].getAdjCloseDataSeries()
# #                 else:
# #                     ds = feed[instrument].getCloseDataSeries()
# #                 self.__rets[instrument] = roc.RateOfChange(ds, 1)
# #
# #             feed.getNewValuesEvent().subscribe(self.__onBars)
# #             disp = dispatcher.Dispatcher()
# #             disp.addSubject(feed)
# #             disp.run()
# #         finally:
# #             feed.getNewValuesEvent().unsubscribe(self.__onBars)
# #
# #
# # import abc
# #
# # import six
# #
# #
# # class Frequency(object):
# #
# #     """Enum like class for bar frequencies. Valid values are:
# #
# #     * **Frequency.TRADE**: The bar represents a single trade.
# #     * **Frequency.SECOND**: The bar summarizes the trading activity during 1 second.
# #     * **Frequency.MINUTE**: The bar summarizes the trading activity during 1 minute.
# #     * **Frequency.HOUR**: The bar summarizes the trading activity during 1 hour.
# #     * **Frequency.DAY**: The bar summarizes the trading activity during 1 day.
# #     * **Frequency.WEEK**: The bar summarizes the trading activity during 1 week.
# #     * **Frequency.MONTH**: The bar summarizes the trading activity during 1 month.
# #     """
# #
# #     # It is important for frequency values to get bigger for bigger windows.
# #     TRADE = -1
# #     SECOND = 1
# #     MINUTE = 60
# #     HOUR = 60*60
# #     DAY = 24*60*60
# #     WEEK = 24*60*60*7
# #     MONTH = 24*60*60*31
# #
# #
# # @six.add_metaclass(abc.ABCMeta)
# # class Bar(object):
# #
# #     """A Bar is a summary of the trading activity for a security in a given period.
# #
# #     .. note::
# #         This is a base class and should not be used directly.
# #     """
# #
# #     @abc.abstractmethod
# #     def setUseAdjustedValue(self, useAdjusted):
# #         raise NotImplementedError()
# #
# #     @abc.abstractmethod
# #     def getUseAdjValue(self):
# #         raise NotImplementedError()
# #
# #     @abc.abstractmethod
# #     def getDateTime(self):
# #         """Returns the :class:`datetime.datetime`."""
# #         raise NotImplementedError()
# #
# #     @abc.abstractmethod
# #     def getOpen(self, adjusted=False):
# #         """Returns the opening price."""
# #         raise NotImplementedError()
# #
# #     @abc.abstractmethod
# #     def getHigh(self, adjusted=False):
# #         """Returns the highest price."""
# #         raise NotImplementedError()
# #
# #     @abc.abstractmethod
# #     def getLow(self, adjusted=False):
# #         """Returns the lowest price."""
# #         raise NotImplementedError()
# #
# #     @abc.abstractmethod
# #     def getClose(self, adjusted=False):
# #         """Returns the closing price."""
# #         raise NotImplementedError()
# #
# #     @abc.abstractmethod
# #     def getVolume(self):
# #         """Returns the volume."""
# #         raise NotImplementedError()
# #
# #     @abc.abstractmethod
# #     def getAdjClose(self):
# #         """Returns the adjusted closing price."""
# #         raise NotImplementedError()
# #
# #     @abc.abstractmethod
# #     def getFrequency(self):
# #         """The bar's period."""
# #         raise NotImplementedError()
# #
# #     def getTypicalPrice(self):
# #         """Returns the typical price."""
# #         return (self.getHigh() + self.getLow() + self.getClose()) / 3.0
# #
# #     @abc.abstractmethod
# #     def getPrice(self):
# #         """Returns the closing or adjusted closing price."""
# #         raise NotImplementedError()
# #
# #     def getExtraColumns(self):
# #         return {}
# #
# #
# # class BasicBar(Bar):
# #     # Optimization to reduce memory footprint.
# #     __slots__ = (
# #         '__dateTime',
# #         '__open',
# #         '__close',
# #         '__high',
# #         '__low',
# #         '__volume',
# #         '__adjClose',
# #         '__frequency',
# #         '__useAdjustedValue',
# #         '__extra',
# #     )
# #
# #     def __init__(self, dateTime, open_, high, low, close, volume, adjClose, frequency, extra={}):
# #         if high < low:
# #             raise Exception("high < low on %s" % (dateTime))
# #         elif high < open_:
# #             raise Exception("high < open on %s" % (dateTime))
# #         elif high < close:
# #             raise Exception("high < close on %s" % (dateTime))
# #         elif low > open_:
# #             raise Exception("low > open on %s" % (dateTime))
# #         elif low > close:
# #             raise Exception("low > close on %s" % (dateTime))
# #
# #         self.__dateTime = dateTime
# #         self.__open = open_
# #         self.__close = close
# #         self.__high = high
# #         self.__low = low
# #         self.__volume = volume
# #         self.__adjClose = adjClose
# #         self.__frequency = frequency
# #         self.__useAdjustedValue = False
# #         self.__extra = extra
# #
# #     def __setstate__(self, state):
# #         (self.__dateTime,
# #             self.__open,
# #             self.__close,
# #             self.__high,
# #             self.__low,
# #             self.__volume,
# #             self.__adjClose,
# #             self.__frequency,
# #             self.__useAdjustedValue,
# #             self.__extra) = state
# #
# #     def __getstate__(self):
# #         return (
# #             self.__dateTime,
# #             self.__open,
# #             self.__close,
# #             self.__high,
# #             self.__low,
# #             self.__volume,
# #             self.__adjClose,
# #             self.__frequency,
# #             self.__useAdjustedValue,
# #             self.__extra
# #         )
# #
# #     def setUseAdjustedValue(self, useAdjusted):
# #         if useAdjusted and self.__adjClose is None:
# #             raise Exception("Adjusted close is not available")
# #         self.__useAdjustedValue = useAdjusted
# #
# #     def getUseAdjValue(self):
# #         return self.__useAdjustedValue
# #
# #     def getDateTime(self):
# #         return self.__dateTime
# #
# #     def getOpen(self, adjusted=False):
# #         if adjusted:
# #             if self.__adjClose is None:
# #                 raise Exception("Adjusted close is missing")
# #             return self.__adjClose * self.__open / float(self.__close)
# #         else:
# #             return self.__open
# #
# #     def getHigh(self, adjusted=False):
# #         if adjusted:
# #             if self.__adjClose is None:
# #                 raise Exception("Adjusted close is missing")
# #             return self.__adjClose * self.__high / float(self.__close)
# #         else:
# #             return self.__high
# #
# #     def getLow(self, adjusted=False):
# #         if adjusted:
# #             if self.__adjClose is None:
# #                 raise Exception("Adjusted close is missing")
# #             return self.__adjClose * self.__low / float(self.__close)
# #         else:
# #             return self.__low
# #
# #     def getClose(self, adjusted=False):
# #         if adjusted:
# #             if self.__adjClose is None:
# #                 raise Exception("Adjusted close is missing")
# #             return self.__adjClose
# #         else:
# #             return self.__close
# #
# #     def getVolume(self):
# #         return self.__volume
# #
# #     def getAdjClose(self):
# #         return self.__adjClose
# #
# #     def getFrequency(self):
# #         return self.__frequency
# #
# #     def getPrice(self):
# #         if self.__useAdjustedValue:
# #             return self.__adjClose
# #         else:
# #             return self.__close
# #
# #     def getExtraColumns(self):
# #         return self.__extra
# #
# # @six.add_metaclass(abc.ABCMeta)
# # class Subject(object):
# #
# #     def __init__(self):
# #         self.__dispatchPrio = dispatchprio.LAST
# #
# #     # This may raise.
# #     @abc.abstractmethod
# #     def start(self):
# #         pass
# #
# #     # This should not raise.
# #     @abc.abstractmethod
# #     def stop(self):
# #         raise NotImplementedError()
# #
# #     # This should not raise.
# #     @abc.abstractmethod
# #     def join(self):
# #         raise NotImplementedError()
# #
# #     # Return True if there are not more events to dispatch.
# #     @abc.abstractmethod
# #     def eof(self):
# #         raise NotImplementedError()
# #
# #     # Dispatch events. If True is returned, it means that at least one event was dispatched.
# #     @abc.abstractmethod
# #     def dispatch(self):
# #         raise NotImplementedError()
# #
# #     @abc.abstractmethod
# #     def peekDateTime(self):
# #         # Return the datetime for the next event.
# #         # This is needed to properly synchronize non-realtime subjects.
# #         # Return None since this is a realtime subject.
# #         raise NotImplementedError()
# #
# #     def getDispatchPriority(self):
# #         # Returns a priority used to sort subjects within the dispatch queue.
# #         # The return value should never change once this subject is added to the dispatcher.
# #         return self.__dispatchPrio
# #
# #     def setDispatchPriority(self, dispatchPrio):
# #         self.__dispatchPrio = dispatchPrio
# #
# #     def onDispatcherRegistered(self, dispatcher):
# #         # Called when the subject is registered with a dispatcher.
# #         pass
# #
# #
# # class Bars(object):
# #
# #     """A group of :class:`Bar` objects.
# #
# #     :param barDict: A map of instrument to :class:`Bar` objects.
# #     :type barDict: map.
# #
# #     .. note::
# #         All bars must have the same datetime.
# #     """
# #
# #     def __init__(self, barDict):
# #         if len(barDict) == 0:
# #             raise Exception("No bars supplied")
# #
# #         # Check that bar datetimes are in sync
# #         firstDateTime = None
# #         firstInstrument = None
# #         for instrument, currentBar in six.iteritems(barDict):
# #             if firstDateTime is None:
# #                 firstDateTime = currentBar.getDateTime()
# #                 firstInstrument = instrument
# #             elif currentBar.getDateTime() != firstDateTime:
# #                 raise Exception("Bar data times are not in sync. %s %s != %s %s" % (
# #                     instrument,
# #                     currentBar.getDateTime(),
# #                     firstInstrument,
# #                     firstDateTime
# #                 ))
# #
# #         self.__barDict = barDict
# #         self.__dateTime = firstDateTime
# #
# #     def __getitem__(self, instrument):
# #         """Returns the :class:`pyalgotrade.bar.Bar` for the given instrument.
# #         If the instrument is not found an exception is raised."""
# #         return self.__barDict[instrument]
# #
# #     def __contains__(self, instrument):
# #         """Returns True if a :class:`pyalgotrade.bar.Bar` for the given instrument is available."""
# #         return instrument in self.__barDict
# #
# #     def items(self):
# #         return list(self.__barDict.items())
# #
# #     def keys(self):
# #         return list(self.__barDict.keys())
# #
# #     def getInstruments(self):
# #         """Returns the instrument symbols."""
# #         return list(self.__barDict.keys())
# #
# #     def getDateTime(self):
# #         """Returns the :class:`datetime.datetime` for this set of bars."""
# #         return self.__dateTime
# #
# #     def getBar(self, instrument):
# #         """Returns the :class:`pyalgotrade.bar.Bar` for the given instrument or None if the instrument is not found."""
# #         return self.__barDict.get(instrument, None)
#
# # from functools import partial
# #
# # def reduction(x,y,wgt):
# #     res = x * (1- wgt) + y * wgt
# #     return res
# #
# #
# # if __name__ == '__main__':
# #     frozen = partial(reduction,wgt = 0.7)
# #     result = frozen(5,3)
# #     print(result)
#
# # test = "192.0.0.1?!289.0.0.1!0.0.0.0!192.163.10.28?192.0.0.1"
# # test_replace = test.replace('?','!')
# # test_tuple = test_replace.split('!')
# # test_sorted = sorted(test_tuple,key = lambda x : x.split('.')[-1])[1:]
# # print(test_sorted)
# #
# # a=' '.join(sorted(test.replace('?','!').split('!'),key=lambda x:x.split('.')[-1])).split()
# # print(a)
# #numpy memmap
# #判断是否有非法字符
# # def check_validate(List):
# #     if len(List) == 0:
# #         raise ValueError('the output of Pipeline must be not null')
# #     pattern = re.compile('^(6|0|3)(\d){5}.(SZ|SH)$')
# #     for idx,item in enumerate(List):
# #         match = pattern.match(item.upper())
# #         if not match :
# #             raise ValueError('invalid stockCode : %s in prediction'%match.group())
# #
# # class EventEngine():
# #     """
# #         定义策略引擎将不同的算法按照顺序放到一个队列里面依次进行执行，算法对应事件，可以注册、剔除事件
# #     """
# #     def __init__(self):
# #
# #         self._queue = Queue()
# #         self._thread = Thread(target=self._run)
# #         # 以计时模块为例，换成其他的需求添加都队列里面
# #         self._timer = Thread(target=self._run_timer)
# #         self._handlers = defaultdict(list)
# #         self._general = []
# #
# #     def _run(self):
# #         while self._active:
# #             try:
# #                 algo = self._queue.get(block=True, timeout=1)
# #                 self._process(algo)
# #             except Empty:
# #                 pass
# #
# #     def _process(self, algo):
# #         if algo._type in self._handlers:
# #             [handler(algo) for handler in self._handlers[algo._type]]
# #
# #         if self._general:
# #             [handler(algo) for handler in self._general]
# #
# #     def _run_timer(self):
# #
# #         while self._active:
# #             # sleep(self._interval)
# #             event = Event('timer')
# #             self.put(event)
# #
# #     def start(self):
# #
# #         self._active = True
# #         self._thread.start()
# #         self._timer.start()
# #
# #     def stop(self):
# #         self._active = False
# #         self._timer.join()
# #         self._thread.join()
# #
# #     def put(self, event):
# #         self._queue.put(event)
# #from collections import defaultdict
# # from urllib.request import urlopen
# # from selenium.webdriver import Chrome
# # from bs4 import BeautifulSoup
# # import requests
# # #存在反爬虫
# # ths = 'http://d.10jqka.com.cn/v6/line/hs_002570/01/all.js'
# # #ths = 'http://stockpage.10jqka.com.cn/HQ_v4.html'
# # #obj = urlopen(ths)
# # headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.117 Safari/537.36'}
# # obj = requests.get(url=ths,headers = headers)
# # # drive = Chrome()
# # # obj = drive.get(ths)
# # # input = drive.find_element_by_link_text('002570')
# # # search = drive.find_element_by_id('su')
# # # input.send_keys('002570.SZ')
# # # search.click()
# # #ths = 'https://link.jianshu.com/?t=http://stockpage.10jqka.com.cn/600196/finance/#view'
# # #obj = urlopen(ths)
# # print(obj)
# # import pymysql
# # from DBUtils.PooledDB import PooledDB
# #
# # class Ora():
# #     """
# #         分为 simplepooleddb steadydb persistentdb pooleddb
# #         from DBUtils.PersistentDB import PersistentDB
# #         @property @staticmethod @classmethod(cls,)
# #         db_oracle={'user':'factor_factory','password':'htfactor123','host':,'port':,'sid':}
# #         pool_name: 连接池的名称，多种连接参数对应多个不同的连接池对象，多单例模式；
# #         host: 数据库地址
# #         user: 数据库服务器用户名
# #         password: 用户密码
# #         database: 默认选择的数据库
# #         port: 数据库服务器的端口
# #         charset: 字符集，默认为 ‘utf8'
# #         use_dict_cursor: 使用字典格式或者元组返回数据；
# #         max_pool_size: 连接池优先最大连接数；
# #         step_size: 连接池动态增加连接数大小；
# #         enable_auto_resize: 是否动态扩展连接池，即当超过 max_pool_size 时，自动扩展 max_pool_size；
# #         pool_resize_boundary: 该配置为连接池最终可以增加的上上限大小，即时扩展也不可超过该值；
# #         auto_resize_scale: 自动扩展 max_pool_size 的增益，默认为 1.5 倍扩展；
# #         wait_timeout: 在排队等候连接对象时，最多等待多久，当超时时连接池尝试自动扩展当前连接数；
# #         kwargs: 其他配置参数将会在创建连接对象时传递给
# #
# #         frame.to_sql(tablename,conn,if_exists='append',chunksize=50000)
# #
# #         result = pd.read_sql('select * from "{}"'.format(table_name),conn,index_col='index',**kwargs).rename_axis(None)
#
#
# # 该软件包中的功能要求子项可以导入 __main__ 模块。这包含在 编程指导，
# # 例如 multiprocessing.pool.Pool 示例在交互式解释器中不起作用
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # """
# # Created on Sat Feb 16 13:56:19 2019
# #
# # @author: python
# # """
# #
# #
# # class RpcGateway(BaseGateway):
# #     """
# #     VN Trader Gateway for RPC service.
# #     """
# #
# #     default_setting = {
# #         "主动请求地址": "tcp://127.0.0.1:2014",
# #         "推送订阅地址": "tcp://127.0.0.1:4102"
# #     }
# #
# #
# #     def __init__(self, event_engine):
# #         """Constructor"""
# #         super().__init__(event_engine, "RPC")
# #
# #         self.symbol_gateway_map = {}
# #
# #         self.client = RpcClient()
# #         self.client.callback = self.client_callback
# #
# #     def connect(self, setting: dict):
# #         """"""
# #         req_address = setting["主动请求地址"]
# #         pub_address = setting["推送订阅地址"]
# #
# #         self.client.subscribe_topic("")
# #         self.client.start(req_address, pub_address)
# #
# #         self.write_log("服务器连接成功，开始初始化查询")
# #
# #         self.query_all()
# #
# #     def subscribe(self, req: SubscribeRequest):
# #         """"""
# #         gateway_name = self.symbol_gateway_map.get(req.vt_symbol, "")
# #         self.client.subscribe(req, gateway_name)
# #
# #     def send_order(self, req: OrderRequest):
# #         """"""
# #         gateway_name = self.symbol_gateway_map.get(req.vt_symbol, "")
# #         self.client.send_order(req, gateway_name)
# #
# #     def cancel_order(self, req: CancelRequest):
# #         """"""
# #         gateway_name = self.symbol_gateway_map.get(req.vt_symbol, "")
# #         self.client.cancel_order(req, gateway_name)
# #
# #     def query_account(self):
# #         """"""
# #         pass
# #
# #     def query_position(self):
# #         """"""
# #         pass
# #
# #     def query_all(self):
# #         """"""
# #         contracts = self.client.get_all_contracts()
# #         for contract in contracts:
# #             self.symbol_gateway_map[contract.vt_symbol] = contract.gateway_name
# #             contract.gateway_name = self.gateway_name
# #             self.on_contract(contract)
# #         self.write_log("合约信息查询成功")
# #
# #         accounts = self.client.get_all_accounts()
# #         for account in accounts:
# #             account.gateway_name = self.gateway_name
# #             self.on_account(account)
# #         self.write_log("资金信息查询成功")
# #
# #         positions = self.client.get_all_positions()
# #         for position in positions:
# #             position.gateway_name = self.gateway_name
# #             self.on_position(position)
# #         self.write_log("持仓信息查询成功")
# #
# #         orders = self.client.get_all_orders()
# #         for order in orders:
# #             order.gateway_name = self.gateway_name
# #             self.on_order(order)
# #         self.write_log("委托信息查询成功")
# #
# #         trades = self.client.get_all_trades()
# #         for trade in trades:
# #             trade.gateway_name = self.gateway_name
# #             self.on_trade(trade)
# #         self.write_log("成交信息查询成功")
# #
# #     def close(self):
# #         """"""
# #         self.client.stop()
# #
# #     def client_callback(self, topic: str, event: Event):
# #         """"""
# #         if event is None:
# #             print("none event", topic, event)
# #             return
# #
# #         data = event.data
# #
# #         if hasattr(data, "gateway_name"):
# #             data.gateway_name = self.gateway_name
# #
# #         self.event_engine.put(event)
#
# # subprocess replace os.system(),os.spawnv(),os , popen2,command
# # subprocess -- run call check_call check_output
# # subprocess.run(['ls'])
# # call("pip install --upgrade h5py",shell=True)
# from dateutil.relativedelta import relativedelta as timedelta
# from datetime import timedelta
# from glob import glob
# from textwrap import dedent
# from collections import namedtuple
# from numpy import full, nan, int64, zeros
# from inspect import signature, Parameter
# import csv,re
#
# # with open(filepath, 'w', newline='') as csvfile:
# #     spamwriter = csv.writer(csvfile, delimiter=' ',
# #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
# #     for r in kline.values.tolist():
# #         print(r)
# #         spamwriter.writerow(r)
# #         spamwriter.writerow('\n')
#
# # test = '{page:test,data:["test":3]}'
# # import re
# # match = re.search('\[(.*.)\]',test)
# # print(match.group())
# # https://pypi.tuna.tsinghua.edu.cn/simple
# #apply --- dataframe -行或者一列
# #applymap --- dataframe 每一个元素
# #map --- series
#
# # class A(object):
# #
# #     def __init__(self,a):
# #         self.a = a
# #
# #     def trans(self):
# #         b = self.a
# #         if b >0 :
# #             b = 2
# #         else:
# #             b = 3
# #         print('b',b)
# #         print('------',self.a)
# #
# # print('------',A.__name__)
#
# # b = A(4)
# # # b.trans()
# # # print(b.a)
#
# # from numpy import (
# #     array,
# #     full,
# #     recarray,
# #     vstack,
# # )
# # from abc import ABC
# # from bisect import insort
# # from collections import Mapping
# # from pandas import NaT as pd_NaT
# # from numpy import array,dtype as dtype_class , ndarray,searchsorted
# # import datetime
# # from textwrap import dedent
#
# # from functools import reduce
# #
# # inputs = {'a':[1,2,3,5],'b':[2,3,4,5,6,7],'c':[4,5,6,7,8]}
# #
# # term_input = reduce(lambda x, y: set(x) & set(y), inputs.values())
# #
# # print(term_input)
# # idx = trading_days.searchsorted(dt)
# # start_ix, end_ix = sessions.slice_locs(start_date, end_date)
# # return (
# #     (r[0], r[-1]) for r in partition_all(
# #     chunksize, sessions[start_ix:end_ix]
# # )
# # )
# # def categorical_df_concat(df_list, inplace=False):
# #     """
# #     Prepare list of pandas DataFrames to be used as input to pd.concat.
# #     Ensure any columns of type 'category' have the same categories across each
# #     dataframe.
# #
# #     Parameters
# #     ----------
# #     df_list : list
# #         List of dataframes with same columns.
# #     inplace : bool
# #         True if input list can be modified. Default is False.
# #
# #     Returns
# #     -------
# #     concatenated : df
# #         Dataframe of concatenated list.
# #     """
# #
# #     if not inplace:
# #         df_list = copy.deepcopy(df_list)
# #
# #     # Assert each dataframe has the same columns/dtypes
# #     df = df_list[0]
# #     if not all([(df.dtypes.equals(df_i.dtypes)) for df_i in df_list[1:]]):
# #         raise ValueError("Input DataFrames must have the same columns/dtypes.")
# #
# #     categorical_columns = df.columns[df.dtypes == 'category']
# #
# #     for col in categorical_columns:
# #         new_categories = sorted(
# #             set().union(
# #                 *(frame[col].cat.categories for frame in df_list)
# #             )
# #         )
# #
# #         with ignore_pandas_nan_categorical_warning():
# #             for df in df_list:
# #                 df[col].cat.set_categories(new_categories, inplace=True)
# #
# #     return pd.concat(df_list)
# # tolerant_equals
# # from abc import ABC
# # from collections import deque, namedtuple
# # from numbers import Integral
# # from operator import itemgetter, attrgetter
# # # import numpy as np
# # import pandas as pd
# # from pandas import isnull
# # from six import with_metaclass, string_types, viewkeys, iteritems
# # from toolz import (
# #     compose,
# #     concat,
# #     # vertical itertools.chain
# #     concatv,
# #     curry,
# #     groupby,
# #     merge,
# #     partition_all,
# #     sliding_window,
# #     valmap,
# # )
#
# # self.conn.execute(
# #     "CREATE INDEX IF NOT EXISTS stock_dividends_payouts_ex_date "
# #     "ON stock_dividend_payouts(ex_date)"
# # frame['effective_date'] = frame['effective_date'].values.astype(
# #     'datetime64[s]',
# # ).astype('int64')
# # actual_dtypes = frame.dtypes
# # for colname, expected in six.iteritems(expected_dtypes):
# #     actual = actual_dtypes[colname]
# #     if not np.issubdtype(actual, expected):
# #         raise TypeError(
# #             "Expected data of type {expected} for column"
# #             " '{colname}', but got '{actual}'.".format(
# #                 expected=expected,
# #                 colname=colname,
# #                 actual=actual,
# #             ),
# #         )
# # from sqlalchemy import join
# #
# # j = user_table.join(address_table,
# #                 user_table.c.id == address_table.c.user_id)
# # stmt = select([user_table]).select_from(j)
# # """
# # READ COMMITTED
# # READ UNCOMMITTED
# # REPEATABLE READ
# # SERIALIZABLE
# # AUTOCOMMIT
# # """
# # engine = create_engine('mysql+pymysql://root:macpython@localhost:3306/test',
# #                        isolation_level="READ UNCOMMITTED")
# # engine = create_engine('mysql+pymysql://root:macpython@localhost:3306/test',
# #                        pool_size=50, max_overflow=100, pool_timeout=-1)
# # db = 'db'
# # with engine.connect() as conn:
# #     conn.execute('create database %s'%db)
# #
# # metadata.create_all(bind = engine)
# #
# # # engine.execution_options()
# # print(metadata.clear())
# # tbls = engine.table_names()
# # print(tbls)
# # conn = engine.connect()
# # res = conn.execution_options(isolation_level="READ COMMITTED")
# # print(res.get_execution_options())
# # # engine.execution_options(isolation_level="READ COMMITTED")
# # # print(engine.get_execution_options())
# # #代理
# # from sqlalchemy import inspect
# # insp = inspect(engine)
# # print(insp.get_table_names())
# # print(insp.get_columns('asharePrice'))
# # print(insp.get_schema_names())
# # # get_pk_constraint get_primary_keys get_foreign_keys get_indexes
# # sa.CheckConstraint('id <= 1')
# # ins = ins.order_by(table.c.trade_dt)
# # def canonicalize_datetime(dt):
# #     # Strip out any HHMMSS or timezone info in the user's datetime, so that
# #     # all the datetimes we return will be 00:00:00 UTC.
# #     return datetime(dt.year, dt.month, dt.day, tzinfo=pytz.utc)
# # pd.date_range(start=start.date(),end=end.date(),freq=trading_day).tz_localize('UTC')
# # end = end_base + pd.Timedelta(days=365)
# # if __name__ == '__main__':
# #
# #     # start = pd.Timestamp('1990-01-01', tz='UTC')
# #     # end_base = pd.Timestamp('today', tz='UTC')
# #     tz = pytz.timezone('Asia/Shanghai')
# #     start = pd.Timestamp('19900101',tz = tz)
# #     end_base = pd.Timestamp('today',tz = tz)
# #     end = end_base + pd.Timedelta(days=365)
# #
# #     new_year = rrule.rrule(
# #         rrule.YEARLY,
# #         byyearday=1,
# #         cache=True,
# #         dtstart=start,
# #         until=end
# #     )
# #
# #     print(list(new_year))
# #
# #     qing_ming = rrule.rrule(
# #         rrule.YEARLY,
# #         bymonth=4,
# #         bymonthday=4,
# #         cache=True,
# #         dtstart=start,
# #         until=end
# #     )
# #
# #     # print(list(qing_ming))
# #
# #     labour_day = rrule.rrule(
# #         rrule.YEARLY,
# #         bymonth=5,
# #         bymonthday=1,
# #         cache=True,
# #         dtstart=start,
# #         until=end
# #     )
# #
# #     # print(list(labour_day))
# #
# #     national_day = rrule.rrule(
# #         rrule.YEARLY,
# #         bymonth=10,
# #         bymonthday=1,
# #         cache=True,
# #         dtstart=start,
# #         until=end
# #     )
# # from operator import methodcaller
# # import sys
# # class classproperty(object):
# #     """Class property
# #     """
# #     def __init__(self, fget):
# #         self.fget = fget
# #
# #     def __get__(self, instance, owner):
# #         return self.fget(owner)
# #
# # class DummyMapping(object):
# #     """
# #     Dummy object used to provide a mapping interface for singular values.
# #     """
# #     def __init__(self, value):
# #         self._value = value
# #
# #     def __getitem__(self, key):
# #         return self._value
# #
# # class IDBox(object):
# #     """A wrapper that hashs to the id of the underlying object and compares
# #     equality on the id of the underlying.
# #
# #     Parameters
# #     ----------
# #     ob : any
# #         The object to wrap.
# #
# #     Attributes
# #     ----------
# #     ob : any
# #         The object being wrapped.
# #
# #     Notes
# #     -----
# #     This is useful for storing non-hashable values in a set or dict.
# #     """
# #     def __init__(self, ob):
# #         self.ob = ob
# #
# #     def __hash__(self):
# #         return id(self)
# #
# #     def __eq__(self, other):
# #         if not isinstance(other, IDBox):
# #             return NotImplemented
# #
# #         return id(self.ob) == id(other.ob)
# #
# #
# # class NamedExplodingObject(object):
# #     """An object which has no attributes but produces a more informative
# #     error message when accessed.
# #
# #     Parameters
# #     ----------
# #     name : str
# #         The name of the object. This will appear in the error messages.
# #
# #     Notes
# #     -----
# #     One common use for this object is so ensure that an attribute always exists
# #     even if sometimes it should not be used.
# #     """
# #     def __init__(self, name, extra_message=None):
# #         self._name = name
# #         self._extra_message = extra_message
# #
# #     def __getattr__(self, attr):
# #         extra_message = self._extra_message
# #         raise AttributeError(
# #             'attempted to access attribute %r of ExplodingObject %r%s' % (
# #                 attr,
# #                 self._name,
# #             ),
# #             ' ' + extra_message if extra_message is not None else '',
# #         )
# #
# #     def __repr__(self):
# #         return '%s(%r%s)' % (
# #             type(self).__name__,
# #             self._name,
# #             # show that there is an extra message but truncate it to be
# #             # more readable when debugging
# #             ', extra_message=...' if self._extra_message is not None else '',
# #         )
# #
# # try:
# #     # fast versions
# #     import bottleneck as bn
# #     nanmean = bn.nanmean
# #     nanstd = bn.nanstd
# #     nansum = bn.nansum
# #     nanmax = bn.nanmax
# #     nanmin = bn.nanmin
# #     nanargmax = bn.nanargmax
# #     nanargmin = bn.nanargmin
# # except ImportError:
# #     # slower numpy
# #     import numpy as np
# #     nanmean = np.nanmean
# #     nanstd = np.nanstd
# #     nansum = np.nansum
# #     nanmax = np.nanmax
# #     nanmin = np.nanmin
# #     nanargmax = np.nanargmax
# #     nanargmin = np.nanargmin
# # # numpy.flatnonzero --- 非0的indice
# # n = self.start
# # stop = self.stop
# # step = self.step
# # cmp_ = op.lt if step > 0 else op.gt
# # while cmp_(n, stop):
# #     yield n
# #     n += step
# #
# #
# # from io import StringIO
# #
# # output = StringIO()
# # output.write('First line.\n')
# # contents = output.getvalue()
# # output.close()
# # fd = StringIO()
# # fd.tell()
# # fd.seek(0)
# # fd.close()
# # fd = StringIO()
# # if isinstance(data, str):
# #     fd.write(data)
# # else:
# #     for chunk in data:
# #         fd.write(chunk)
# # self.fetch_size = fd.tell()
# # fd.seek(0)
#
# # from abc import ABCMeta, abstractmethod
# # from collections import namedtuple
# # import hashlib
# # from textwrap import dedent
# # import warnings
# #
# # import numpy
# # import pandas as pd
# # from pandas import read_csv
# # import pytz
# # import requests
#
#
# # def __iter__(self):
# #     warnings.warn(
# #         'Iterating over security_lists is deprecated. Use '
# #         '`for sid in <security_list>.current_securities(dt)` instead.',
# #         category=ZiplineDeprecationWarning,
# #         stacklevel=2
# #     )
# #     return iter(self.current_securities(self.current_date()))
# #
# # def __contains__(self, item):
# #     warnings.warn(
# #         'Evaluating inclusion in security_lists is deprecated. Use '
# #         '`sid in <security_list>.current_securities(dt)` instead.',
# #         category=ZiplineDeprecationWarning,
# #         stacklevel=2
# #     )
# #     return item in self.current_securities(self.current_date())
# #
# #
# # def __new__(cls):
# #     raise TypeError('cannot create %r instances' % name)
# # index_lookup = {'上证指数':'000001' ,
# #                 '深证成指':'399001' ,
# #                 'Ｂ股指数':'000003',
# #                 '深成指R':'399002',
# #                 '成份Ｂ指':'399003',
# #                 '深证综指':'399106',
# #                 '上证180':'000010',
# #                 '基金指数':'000011',
# #                 '深证100R':'399004',
# #                 '国债指数':'000012',
# #                 '企债指数':'000013',
# #                 '上证50':'000016',
# #                 '上证380':'000009',
# #                 '沪深300':'000300',
# #                 '中证500':'000905',
# #                 '中小板指':'399005',
# #                 '新指数':'399100',
# #                 '中证100':'000903',
# #                 '中证800':'000906',
# #                 '深证300':'399007',
# #                 '中小300':'399008',
# #                 '创业板指':'399006',
# #                 '上证100':'000132',
# #                 '上证150':'000133',
# #                 '央视50':'399550',
# #                 '创业大盘':'399293',
# #                 '道琼斯':'us.DJI',
# #                 '纳斯达克':'us.IXIC',
# #                 '标普500':'us.INX',
# #                 '香港恒生指数':'hkHSI',
# #                 '香港国企指数':'hkHSCEI',
# #                 '香港红筹指数':'hkHSCCI'}
# # from bs4 import BeautifulSoup
# # # import json,re
#
# # def _parse_url(url,encoding='gbk', bs=True,*args,**kwargs):
# #     Header = {
# #         'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36(KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'}
# #     req = requests.get(url, headers=Header, timeout=1)
# #     if encoding:
# #         req.encoding = encoding
# #     if bs:
# #         raw = BeautifulSoup(req.text, features='lxml')
# #     else:
# #         raw = req.text
# #     return raw
# #
# # EVENT_REQUEST_URL = {'massive':'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=DZJYXQ'
# #                         '&token=70f12f2f4f091e459a279469fe49eca5&cmd=&st=SECUCODE&sr=1&p=%d&ps=50&',
# #                     'release':'http://dcfm.eastmoney.com/EM_MutiSvcExpandInterface/api/js/get?type=XSJJ_NJ_PC'
# #                         '&token=70f12f2f4f091e459a279469fe49eca5&st=kjjsl&sr=-1&p=%d&ps=10&filter=(mkt=)'}
# #
# # sdate = '2020-05-10'
# # edate = '2020-05-17'
# # massive = pd.DataFrame()
# # count = 1
# # prefix = 'js={"data":(x)}&filter=(Stype=%27EQA%27)' + \
# #          '(TDATE%3E=^{}^%20and%20TDATE%3C=^{}^)'.format(sdate, edate)
# # # while True:
# #     url = EVENT_REQUEST_URL['massive'] % count + prefix
# #     raw = _parse_url(url, bs=False, encoding=None)
# #     raw = json.loads(raw)
# #     if raw['data'] and len(raw['data']):
# #         mass = pd.DataFrame(raw['data'])
# #         newcol =['trade_dt', 'sid', 'cname', 'price', 'volume', 'amount', 'buyer_code',
# #                  'buyer','seller_code', 'seller', 'type', 'unit', 'pct', 'close', 'YSSLTAG',
# #                  'discount','cjeltszb','1_pct', '5_pct', '10_pct', '20_pct', 'TEXCH']
# #         mass.columns = newcol
# #         massive = massive.append(mass)
# #         count = count + 1
# #     else:
# #         break
# #
# # print('massive',massive)
# # print(massive.columns)
# # print(massive.iloc[0,:])
# #
# # page = 1
# # url = "http://data.eastmoney.com/DataCenter_V3/gdzjc.ashx?pagesize=50&page=%d&param=&sortRule=-1&sortType=BDJZ" % page
# # raw = _parse_url(url, bs=False)
# # match = re.search('\[(.*.)\]', raw)
# # data = json.loads(match.group())
# # data = [item.split(',')[:-1] for item in data]
# # print('share_pct',data)
# # import datetime
# # margin = pd.DataFrame()
# #
# # while True:
# #     url = 'http://api.dataide.eastmoney.com/data/get_rzrq_lshj?orderby=dim_date&order=desc&pageindex=%d&pagesize=50' % page
# #     raw = _parse_url(url, bs=False)
# #     raw = json.loads(raw)
# #     raw = [[item['dim_date'], item['rzye'], item['rqye'], item['rzrqye'], item['rzrqyecz'], item['new'], item['zdf']]
# #            for item in raw['data']]
# #     data = pd.DataFrame(raw, columns=['trade_dt', 'rzye', 'rqye', 'rzrqze', 'rzrqce', 'hs300', 'pct'])
# #     data.loc[:, 'trade_dt'] = [datetime.datetime.fromtimestamp(dt / 1000) for dt in data['trade_dt']]
# #     data.loc[:, 'trade_dt'] = [datetime.datetime.strftime(t, '%Y-%m-%d') for t in data['trade_dt']]
# #     # filter_data = data[data['trade_dt'] > self.deadline['market_marign']]
# #     if len(data) == 0:
# #         break
# #     margin = margin.append(data)
# #     page = page + 1
# # margin.set_index('trade_dt',inplace= True)
# # print('margin',margin)
#
# #'000571', '*ST大洲', '2.75', '0.73', '大连和升控股集团有限公司', '增持', '824.0719', '1.02', '二级市场', '8506.4405', '10.45', '8506.4405', '10.53', '2020-05-12', '2020-05-12', '2020-05-13'
#
# # import inspect ,uuid
# # from functools import wraps
# #
# # def getargspec(f):
# #     full_argspec = inspect.getfullargspec(f)
# #     return inspect.ArgSpec(
# #         args=full_argspec.args,
# #         varargs=full_argspec.varargs,
# #         keywords=full_argspec.varkw,
# #         defaults=full_argspec.defaults,
# #     )
# # #
# # # print(getargspec(_parse_url))
# # NO_DEFAULT = object()
# # #
# # #
# # args, varargs, varkw, defaults = argspec = getargspec(_parse_url)
# # print('varargs',varargs)
# # print('varkw',varkw)
# # if defaults is None:
# #     defaults = ()
# # no_defaults = (NO_DEFAULT,) * (len(args) - len(defaults))
# # print('args',args)
# # print('no_defaults',no_defaults)
# # args_defaults = list(zip(args, no_defaults + defaults))
# # if varargs:
# #     args_defaults.append((varargs, NO_DEFAULT))
# # if varkw:
# #     args_defaults.append((varkw, NO_DEFAULT))
# # print('args_defaults',args_defaults)
# # print('varargs',varargs)
# # print('varkw',varkw)
# #
# # # argset = set(args) | {varargs, varkw} - {None}
# # argset = set(args) | {varargs, varkw}
# # #
# # print('argset',argset)
#
# # __qualname__
#
# # def _get_prefix(code, exchange='hk'):
# #     if exchange == 'us':
# #         code = exchange + '.' + code
# #     elif exchange == 'hk':
# #         code = exchange + code
# #     else:
# #         raise NotImplementedError
# #     return code
# #
# # def load_daily_dual(asset, sdate, edate,mode = 'qfq'):
# #     """
# #         获取港股Kline , 针对于同时在A股上市的 , AH
# #     """
# #     tencent = 'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=%s,day,%s,%s,10,%s' % (
# #     _get_prefix(asset), sdate, edate, mode)
# #     raw = _parse_url(tencent, bs=False, encoding=None)
# #     raw = json.loads(raw)
# #     data = raw['data']
# #     print('hkline',data)
#
# # load_daily_dual('00168','2011-01-01','2012-01-06')
# # count = 1
# # html = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=DZJYXQ&' \
# #        'token=70f12f2f4f091e459a279469fe49eca5&cmd=&st=SECUCODE&sr=1&p=%d&ps=50&js='%count \
# #        +'{"data":(x)}&filter=(Stype=%27EQA%27)'+'(TDATE%3E=^{}^%20and%20TDATE%3C=^{}^)'.format('2020-04-10', '2020-04-30')
# #
# # # html = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=DZJYXQ&' \
# # #                    'token=70f12f2f4f091e459a279469fe49eca5&cmd=&st=SECUCODE&sr=1&p=%d&ps=50&'%count +\
# # #                    'js={"data":(x)}&filter=(Stype=%27EQA%27)'+'(TDATE%3E=^{}^%20and%20TDATE%3C=^{}^)'.format(sdate,edate)
# #
# # print(html)
#
# # url = 'https://www.jisilu.cn/data/cbnew/cb_list/?'
# # text = _parse_url(url, bs=False, encoding=None)
# # text = json.loads(text)
# # print(text['rows'])
#
# # raw = _parse_url(html, bs=False, encoding=None)
# # raw = json.loads(raw)
# # print('raw',raw)
# # if raw['data'] and len(raw['data']):
# #     mass = pd.DataFrame(raw['data'])
# #     print('mass',mass)
# #
# # assets_with_leading_nan = np.where(isnull(df.iloc[0]))[0]
# # normed_index = df.index.normalize()
# # _code_argorder = (
# #                      ('co_argcount', 'co_kwonlyargcount') if PY3 else ('co_argcount',)
# #                  ) + (
# #                      'co_nlocals',
# #                      'co_stacksize',
# #                      'co_flags',
# #                      'co_code',
# #                      'co_consts',
# #                      'co_names',
# #                      'co_varnames',
# #                      'co_filename',
# #                      'co_name',
# #                      'co_firstlineno',
# #                      'co_lnotab',
# #                      'co_freevars',
# #                      'co_cellvars',
# #                  )
# # code = new_func.__code__
# # args = {
# #     attr: getattr(code, attr)
# #     for attr in dir(code)
# #     if attr.startswith('co_')
# # }
# # # Copy the firstlineno out of the underlying function so that exceptions
# # # get raised with the correct traceback.
# # # This also makes dynamic source inspection (like IPython `??` operator)
# # # work as intended.
# # try:
# #     # Try to get the pycode object from the underlying function.
# #     original_code = func.__code__
# # except AttributeError:
# #     try:
# #         # The underlying callable was not a function, try to grab the
# #         # `__func__.__code__` which exists on method objects.
# #         original_code = func.__func__.__code__
# #     except AttributeError:
# #         # The underlying callable does not have a `__code__`. There is
# #         # nothing for us to correct.
# #         return new_func
# #
# # args['co_firstlineno'] = original_code.co_firstlineno
# # new_func.__code__ = CodeType(*map(getitem(args), _code_argorder))
# #
# # if PY3:
# #     _qualified_name = attrgetter('__qualname__')
# # else:
# #     def _qualified_name(obj):
# #         """
# #         Return the fully-qualified name (ignoring inner classes) of a type.
# #         """
# #         # If the obj has an explicitly-set __qualname__, use it.
# #         try:
# #             return getattr(obj, '__qualname__')
# #         except AttributeError:
# #             pass
# #
# #         # If not, build our own __qualname__ as best we can.
# #         module = obj.__module__
# #         if module in ('__builtin__', '__main__', 'builtins'):
# #             return obj.__name__
# #         return '.'.join([module, obj.__name__])
# # from collections import OrderedDict
# # from datetime import datetime
# # from distutils.version import StrictVersion
# #
# # import numpy as np
# # from numpy import (
# #     array_equal,
# #     broadcast,
# #     busday_count,
# #     datetime64,
# #     diff,
# #     dtype,
# #     empty,
# #     flatnonzero,
# #     hstack,
# #     isnan,
# #     nan,
# #     vectorize,
# #     where
# # )
# # flip --- 反转参数
# from toolz import flip
#
# # def load_from_directory(list_name):
# #     """
# #     To resolve the symbol in the LEVERAGED_ETF list,
# #     the date on which the symbol was in effect is needed.
# #
# #     Furthermore, to maintain a point in time record of our own maintenance
# #     of the restricted list, we need a knowledge date. Thus, restricted lists
# #     are dictionaries of datetime->symbol lists.
# #     new symbols should be entered as a new knowledge date entry.
# #
# #     This method assumes a directory structure of:
# #     SECURITY_LISTS_DIR/listname/knowledge_date/lookup_date/add.txt
# #     SECURITY_LISTS_DIR/listname/knowledge_date/lookup_date/delete.txt
# #
# #     The return value is a dictionary with:
# #     knowledge_date -> lookup_date ->
# #        {add: [symbol list], 'delete': [symbol list]}
# #     """
# #     data = {}
# #     dir_path = os.path.join(SECURITY_LISTS_DIR, list_name)
# #     for kd_name in listdir(dir_path):
# #         kd = datetime.strptime(kd_name, DATE_FORMAT).replace(
# #             tzinfo=pytz.utc)
# #         data[kd] = {}
# #         kd_path = os.path.join(dir_path, kd_name)
# #         for ld_name in listdir(kd_path):
# #             ld = datetime.strptime(ld_name, DATE_FORMAT).replace(
# #                 tzinfo=pytz.utc)
# #             data[kd][ld] = {}
# #             ld_path = os.path.join(kd_path, ld_name)
# #             for fname in listdir(ld_path):
# #                 fpath = os.path.join(ld_path, fname)
# #                 with open(fpath) as f:
# #                     symbols = f.read().splitlines()
# #                     data[kd][ld][fname] = symbols
# #
# #     return data
# # def inspect(self):
# #     """
# #     Return a string representation of the data stored in this array.
# #     """
# #     return dedent(
# #         """\
# #         Adjusted Array ({dtype}):
# #
# #         Data:
# #         {data!r}
# #
# #         Adjustments:
# #         {adjustments}
# #         """
# #     ).format(
# #         dtype=self.dtype.name,
# #         data=self.data,
# #         adjustments=self.adjustments,
# #     )
# # def last_modified_time(path):
# #     """
# #     Get the last modified time of path as a Timestamp.
# #     """
# #     return pd.Timestamp(os.path.getmtime(path), unit='s', tz='UTC')
# #
# # def load_prices_from_csv(filepath, identifier_col, tz='UTC'):
# #     data = pd.read_csv(filepath, index_col=identifier_col)
# #     data.index = pd.DatetimeIndex(data.index, tz=tz)
# #     data.sort_index(inplace=True)
# #     return data
# #
# #
# # def load_prices_from_csv_folder(folderpath, identifier_col, tz='UTC'):
# #     data = None
# #     for file in os.listdir(folderpath):
# #         if '.csv' not in file:
# #             continue
# #         raw = load_prices_from_csv(os.path.join(folderpath, file),
# #                                    identifier_col, tz)
# #         if data is None:
# #             data = raw
# #         else:
# #             data = pd.concat([data, raw], axis=1)
# #     return data
# #
# # def has_data_for_dates(series_or_df, first_date, last_date):
# #     """
# #     Does `series_or_df` have data on or before first_date and on or after
# #     last_date?
# #     """
# #     dts = series_or_df.index
# #     if not isinstance(dts, pd.DatetimeIndex):
# #         raise TypeError("Expected a DatetimeIndex, but got %s." % type(dts))
# #     first, last = dts[[0, -1]]
# #     return (first <= first_date) and (last >= last_date)
#
# # adjustments = adjustments.reindex_axis(ADJUSTMENT_COLUMNS, axis=1)
# # row_loc = dates.get_loc(apply_date, method='bfill')
# # date_ix = np.searchsorted(dates, dividends.ex_date.values)
# # date_ix = np.searchsorted(dates, dividends.ex_date.values)
# # mask = date_ix > 0
# #
# # date_ix = date_ix[mask]
# # sids_ix = sids_ix[mask]
# # input_dates = dividends.ex_date.values[mask]
# #
# # # subtract one day to get the close on the day prior to the merger
# # previous_close = close[date_ix - 1, sids_ix]
# # input_sids = input_sids[mask]
# #
# # amount = dividends.amount.values[mask]
# # ratio = 1.0 - amount / previous_close
# # class AjustmentsWriter(object):
# #
# #     def __init__(self, engine):
# #         self.conn = engine.conncect()
# #         self.tables = sa.MetaData(bind = engine).tables
# #         self._init_declared_date()
# #         self.trading_days = Calendar(self.conn).trading_days
# #
# #     def _get_max_declared_date_from_sqlite(self, tbl):
# #         table = self.tables[tbl]
# #         sql = sa.select([table.c.sid, sa.func.max(table.c.declared_date)])
# #         sql = sql.group_by(table.c.sid)
# #         rp = self.conn.execute(sql)
# #         res = {r.sid:r.declared_date for r in rp.fetchall()}
# #         return res
# #
# #     def _init_declared_date(self):
# #         self._declared_date  = dict()
# #         for tbl in frozenset(('symbol_rights','symbol_splits')):
# #             self._declared_date[tbl] = self._get_max_declared_date_from_sqlite(tbl)
# #
# #     def request_shareBonus(self, code):
# #         url = ADJUSTMENT_URL['shareBonus']%code
# #         obj = _parse_url(url)
# #         self._download_splits_divdend(obj, code)
# #         self._download_issues(obj, code)
# #
# #
# # market_marign = sa.Table(
# #     'market_marign',
# #     metadata,
# #     sa.Column(
# #         'trade_dt',
# #         sa.String(10),
# #         unique=True,
# #         nullable=False,
# #         primary_key=True,
# #     ),
# #     sa.Column('rzye', sa.String(20)),
# #     sa.Column('rqye', sa.String(20)),
# #     sa.Column('rzrqze', sa.String(20)),
# #     sa.Column('rzrqce', sa.String(20)),
# # )
# # dual_symbol_price = sa.Table(
# #     'dual_symbol_price',
# #     metadata,
# #     sa.Column(
# #         'sid',
# #         sa.String(10),
# #         sa.ForeignKey(symbol_price.c.sid),
# #         unique=True,
# #         nullable=False,
# #         primary_key=True,
# #     ),
# #     sa.Column(
# #         'sid_hk',
# #         sa.String(10),
# #         unique=True,
# #         nullable=False,
# #         primary_key=True,
# #     ),
# #     sa.Column('trade_dt', sa.String(10),nullable=False),
# #     sa.Column('open', sa.Numeric(10,2),nullable=False),
# #     sa.Column('high', sa.Numeric(10,2),nullable=False),
# #     sa.Column('low', sa.Numeric(10,2),nullable=False),
# #     sa.Column('close', sa.Numeric(10,2),nullable=False),
# #     sa.Column('volume', sa.Numeric(20,0),nullable=False),
# # )
# # index_price = sa.Table(
# #     'index_price',
# #     metadata,
# #     sa.Column(
# #         'id',
# #         sa.Integer,
# #         unique=True,
# #         nullable=False,
# #         primary_key=True,
# #     ),
# #     sa.Column('sid',sa.String(10)),
# #     sa.Column('cname',sa.Text),
# #     sa.Column('open', sa.Numeric(10,2)),
# #     sa.Column('high', sa.Numeric(10,2)),
# #     sa.Column('low', sa.Numeric(10,2)),
# #     sa.Column('close', sa.Numeric(10,2)),
# #     sa.Column('volume', sa.Numeric(10, 2)),
# #     sa.Column('amount', sa.Numeric(10,2)),
# #     sa.Column('pct', sa.Numeric(10,2)),
# # )
# # def _init(self, min_percentile, max_percentile, *args, **kwargs):
# #     self._min_percentile = min_percentile
# #     self._max_percentile = max_percentile
# #     return super(PercentileFilter, self)._init(*args, **kwargs)
# # XXX: This docstring was mostly written when the abstraction here was
# # "MultiDimensionalDataSet". It probably needs some rewriting.
# # class DataSetFamily(with_metaclass(DataSetFamilyMeta)):
# #     """
# #     Base class for Pipeline dataset families.
# #
# #     Dataset families are used to represent data where the unique identifier for
# #     a row requires more than just asset and date coordinates. A
# #     :class:`DataSetFamily` can also be thought of as a collection of
# #     :class:`~zipline.pipe.data.DataSet` objects, each of which has the same
# #     columns, domain, and ndim.
# #
# #     :class:`DataSetFamily` objects are defined with by one or more
# #     :class:`~zipline.pipe.data.Column` objects, plus one additional field:
# #     ``extra_dims``.
# #
# #     The ``extra_dims`` field defines coordinates other than asset and date that
# #     must be fixed to produce a logical timeseries. The column objects determine
# #     columns that will be shared by slices of the family.
# #
# #     ``extra_dims`` are represented as an ordered dictionary where the keys are
# #     the dimension name, and the values are a set of unique values along that
# #     dimension.
# #
# #     To work with a :class:`DataSetFamily` in a pipe expression, one must
# #     choose a specific value for each of the extra dimensions using the
# #     :meth:`~zipline.pipe.data.DataSetFamily.slice` method.
# #     For example, given a :class:`DataSetFamily`:
# #
# #     .. code-block:: python
# #
# #        class SomeDataSet(DataSetFamily):
# #            extra_dims = [
# #                ('dimension_0', {'a', 'b', 'c'}),
# #                ('dimension_1', {'d', 'e', 'f'}),
# #            ]
# #
# #            column_0 = Column(float)
# #            column_1 = Column(bool)
# #
# #     This dataset might represent a table with the following columns:
# #
# #     ::
# #
# #       sid :: int64
# #       asof_date :: datetime64[ns]
# #       timestamp :: datetime64[ns]
# #       dimension_0 :: str
# #       dimension_1 :: str
# #       column_0 :: float64
# #       column_1 :: bool
# #
# #     Here we see the implicit ``sid``, ``asof_date`` and ``timestamp`` columns
# #     as well as the extra dimensions columns.
# #
# #     This :class:`DataSetFamily` can be converted to a regular :class:`DataSet`
# #     with:
# #
# #     .. code-block:: python
# #
# #        DataSetSlice = SomeDataSet.slice(dimension_0='a', dimension_1='e')
# #
# #     This sliced dataset represents the rows from the higher dimensional dataset
# #     where ``(dimension_0 == 'a') & (dimension_1 == 'e')``.
# #     """
# #     _abstract = True  # Removed by metaclass
# #
# #     domain = GENERIC
# #     slice_ndim = 2
# #
# #     _SliceType = DataSetFamilySlice
# #
# #     @type.__call__
# #     class extra_dims(object):
# #         """OrderedDict[str, frozenset] of dimension name -> unique values
# #
# #         May be defined on subclasses as an iterable of pairs: the
# #         metaclass converts this attribute to an OrderedDict.
# #         """
# #         __isabstractmethod__ = True
# #
# #         def __get__(self, instance, owner):
# #             return []
# #
# #     @classmethod
# #     def _canonical_key(cls, args, kwargs):
# #         extra_dims = cls.extra_dims
# #         dimensions_set = set(extra_dims)
# #         if not set(kwargs) <= dimensions_set:
# #             extra = sorted(set(kwargs) - dimensions_set)
# #             raise TypeError(
# #                 '%s does not have the following %s: %s\n'
# #                 'Valid dimensions are: %s' % (
# #                     cls.__name__,
# #                     s('dimension', extra),
# #                     ', '.join(extra),
# #                     ', '.join(extra_dims),
# #                 ),
# #             )
# #
# #         if len(args) > len(extra_dims):
# #             raise TypeError(
# #                 '%s has %d extra %s but %d %s given' % (
# #                     cls.__name__,
# #                     len(extra_dims),
# #                     s('dimension', extra_dims),
# #                     len(args),
# #                     plural('was', 'were', args),
# #                 ),
# #             )
# #
# #         missing = object()
# #         coords = OrderedDict(zip(extra_dims, repeat(missing)))
# #         to_add = dict(zip(extra_dims, args))
# #         coords.update(to_add)
# #         added = set(to_add)
# #
# #         for key, value in kwargs.items():
# #             if key in added:
# #                 raise TypeError(
# #                     '%s got multiple values for dimension %r' % (
# #                         cls.__name__,
# #                         coords,
# #                     ),
# #                 )
# #             coords[key] = value
# #             added.add(key)
# #
# #         missing = {k for k, v in coords.items() if v is missing}
# #         if missing:
# #             missing = sorted(missing)
# #             raise TypeError(
# #                 'no coordinate provided to %s for the following %s: %s' % (
# #                     cls.__name__,
# #                     s('dimension', missing),
# #                     ', '.join(missing),
# #                 ),
# #             )
# #
# #         # validate that all of the provided values exist along their given
# #         # dimensions
# #         for key, value in coords.items():
# #             if value not in cls.extra_dims[key]:
# #                 raise ValueError(
# #                     '%r is not a value along the %s dimension of %s' % (
# #                         value,
# #                         key,
# #                         cls.__name__,
# #                     ),
# #                 )
# #
# #         return coords, tuple(coords.items())
# #
# #     @classmethod
# #     def _make_dataset(cls, coords):
# #         """Construct a new dataset given the coordinates.
# #         """
# #         class Slice(cls._SliceType):
# #             extra_coords = coords
# #
# #         Slice.__name__ = '%s.slice(%s)' % (
# #             cls.__name__,
# #             ', '.join('%s=%r' % item for item in coords.items()),
# #         )
# #         return Slice
# #
# #     @classmethod
# #     def slice(cls, *args, **kwargs):
# #         """Take a slice of a DataSetFamily to produce a dataset
# #         indexed by asset and date.
# #
# #         Parameters
# #         ----------
# #         *args
# #         **kwargs
# #             The coordinates to fix along each extra dimension.
# #
# #         Returns
# #         -------
# #         dataset : DataSet
# #             A regular pipe dataset indexed by asset and date.
# #
# #         Notes
# #         -----
# #         The extra dimensions coords used to produce the result are available
# #         under the ``extra_coords`` attribute.
# #         """
# #         coords, hash_key = cls._canonical_key(args, kwargs)
# #         try:
# #             return cls._slice_cache[hash_key]
# #         except KeyError:
# #             pass
# #
# #         Slice = cls._make_dataset(coords)
# #         cls._slice_cache[hash_key] = Slice
# #         return Slice
# # class DataSetFamilyMeta(abc.ABCMeta):
# #
# #     def __new__(cls, name, bases, dict_):
# #         columns = {}
# #         for k, v in dict_.items():
# #             if isinstance(v, Column):
# #                 # capture all the columns off the DataSetFamily class
# #                 # and replace them with a descriptor that will raise a helpful
# #                 # error message. The columns will get added to the BaseSlice
# #                 # for this type.
# #                 columns[k] = v
# #                 dict_[k] = _DataSetFamilyColumn(k)
# #
# #         is_abstract = dict_.pop('_abstract', False)
# #
# #         self = super(DataSetFamilyMeta, cls).__new__(
# #             cls,
# #             name,
# #             bases,
# #             dict_,
# #         )
# #
# #         if not is_abstract:
# #             self.extra_dims = extra_dims = OrderedDict([
# #                 (k, frozenset(v))
# #                 for k, v in OrderedDict(self.extra_dims).items()
# #             ])
# #             if not extra_dims:
# #                 raise ValueError(
# #                     'DataSetFamily must be defined with non-empty'
# #                     ' extra_dims, or with `_abstract = True`',
# #                 )
# #
# #             class BaseSlice(self._SliceType):
# #                 dataset_family = self
# #
# #                 ndim = self.slice_ndim
# #                 domain = self.domain
# #
# #                 locals().update(columns)
# #
# #             BaseSlice.__name__ = '%sBaseSlice' % self.__name__
# #             self._SliceType = BaseSlice
# #
# #         # each type gets a unique cache
# #         self._slice_cache = {}
# #         return self
# #
# #     def __repr__(self):
# #         return '<DataSetFamily: %r, extra_dims=%r>' % (
# #             self.__name__,
# #             list(self.extra_dims),
# #         )
# #
#
# # self.estimates = estimates[
# #     estimates[EVENT_DATE_FIELD_NAME].notnull() &
# #     estimates[FISCAL_QUARTER_FIELD_NAME].notnull() &
# #     estimates[FISCAL_YEAR_FIELD_NAME].notnull()
# #     ]
# # self.estimates[NORMALIZED_QUARTERS] = normalize_quarters(
# #     self.estimates[FISCAL_YEAR_FIELD_NAME],
# #     self.estimates[FISCAL_QUARTER_FIELD_NAME],
# # )
# #
# # self.array_overwrites_dict = {
# #     datetime64ns_dtype: Datetime641DArrayOverwrite,
# #     float64_dtype: Float641DArrayOverwrite,
# # }
# # self.scalar_overwrites_dict = {
# #     datetime64ns_dtype: Datetime64Overwrite,
# #     float64_dtype: Float64Overwrite,
# # }
# #
# # self.name_map = name_map
# # values = coerce(list, partial(np.asarray, dtype=object))
# # toolz.functoolz.flip[source]
# # Call the function call with the arguments flipped
# # NamedTemporaryFile has a visble name in file system can be retrieved from the name attribute , delete --- True means delete as file closed
# # def element_of(self, container):
# #     """
# #     Check if each element of self is an of ``container``.
# #
# #     Parameters
# #     ----------
# #     container : object
# #         An object implementing a __contains__ to call on each element of
# #         ``self``.
# #
# #     Returns
# #     -------
# #     is_contained : np.ndarray[bool]
# #         An array with the same shape as self indicating whether each
# #         element of self was an element of ``container``.
# #     """
# #     return self.map_predicate(container.__contains__)
# # functools.total_ordering(cls)
# # Given a class defining one or more rich comparison ordering methods,
# # this class decorator supplies the rest. This simplifies the effort involved in specifying all of the possible rich comparison operations:
# # The class must define one of __lt__(), __le__(), __gt__(), or __ge__(). In addition, the class should supply an __eq__() method.
# import warnings
# #
# # def _deprecated_getitem_method(name, attrs):
# #     """Create a deprecated ``__getitem__`` method that tells users to use
# #     getattr instead.
# #
# #     Parameters
# #     ----------
# #     name : str
# #         The name of the object in the warning message.
# #     attrs : iterable[str]
# #         The set of allowed attributes.
# #
# #     Returns
# #     -------
# #     __getitem__ : callable[any, str]
# #         The ``__getitem__`` method to put in the class dict.
# #     """
# #     attrs = frozenset(attrs)
# #     msg = (
# #         "'{name}[{attr!r}]' is deprecated, please use"
# #         " '{name}.{attr}' instead"
# #     )
# #
# #     def __getitem__(self, key):
# #         """``__getitem__`` is deprecated, please use attribute access instead.
# #         """
# #         warnings(msg.format(name=name, attr=key), DeprecationWarning, stacklevel=2)
# #         if key in attrs:
# #             return getattr(self, key)
# #         raise KeyError(key)
# #
# #     return __getitem__
#
# # @property
# # def first_trading_day(self,sid):
# #     """
# #     Returns
# #     -------
# #     dt : pd.Timestamp
# #         The first trading day (session) for which the reader can provide
# #         data.
# #     """
# #     orm = select([self.equity_basics.c.initial_date]).where(self.equity_basics.c.sid == sid)
# #     first_dt = self.conn.execute(orm).scalar()
# #     return first_dt
# #
# # @property
# # def get_last_traded_dt(self, asset):
# #     """
# #     Get the latest minute on or before ``dt`` in which ``asset`` traded.
# #
# #     If there are no trades on or before ``dt``, returns ``pd.NaT``.
# #
# #     Parameters
# #     ----------
# #     asset : zipline.asset.Asset
# #         The asset for which to get the last traded minute.
# #     dt : pd.Timestamp
# #         The minute at which to start searching for the last traded minute.
# #
# #     Returns
# #     -------
# #     last_traded : pd.Timestamp
# #         The dt of the last trade for the given asset, using the input
# #         dt as a vantage point.
# #     """
# #     orm = select([self.symbol_delist.c.delist_date]).where(self.symbol_delist.c.sid == asset)
# #     rp = self.conn.execute(orm)
# #     dead_date = rp.scalar()
# #     return dead_date
#
# # def shift_calendar(self,dt,window):
# #     window = - abs(window)
# #     index = np.searchsorted(self.all_sessions,dt)
# #     loc = index if self.all_sessions[index] == dt else index -1
# #     if loc + window < 0:
# #         raise ValueError('out of trading_calendar range')
# #     return self.all_sessions[loc + window]
#
# def cartesian(arrays, out=None):
#     """
#         参数组合 ，不同于product
#     """
#     arrays = [np.asarray(x) for x in arrays]
#     print('arrays',arrays)
#     shape = (len(x) for x in arrays)
#     print('shape',shape)
#     dtype = arrays[0].dtype
#
#     ix = np.indices(shape)
#     print('ix',ix)
#     ix = ix.reshape(len(arrays), -1).T
#     print('ix_:',ix)
#
#     if out is None:
#         out = np.empty_like(ix, dtype=dtype)
#         print('out',out.shape)
#
#     for n, arr in enumerate(arrays):
#         print('array',arrays[n])
#         print(ix[:,n])
#         out[:, n] = arrays[n][ix[:, n]]
#         print(out[:,n])
#
#     return out


# class test(object):

# def initialize(self):
#     pass
#
# def handle_data(self):
#     pass
#
# def before_trading_start(self):
#     pass

# class Algorithm(ABC):
#
#     def __init__(self,algo_params,data_portal):
#
#         self.algo_params = algo_params
#         self.data_portal = data_portal
#
#     def handle_data(self):
#         """
#             handle_data to run algorithm
#         """
#     @abstractmethod
#     def before_trading_start(self,dt,asset):
#         """
#             计算持仓股票的卖出信号
#         """
#
#     @abstractmethod
#     def initialzie(self,dt):
#         """
#            run algorithm on dt
#         """
#         pass
#
#
# # asset = namedtuple('Asset',['dt','sid','reason','auto_close_date'])

#
# class UnionEngine(object):
#     """
#         组合不同算法---策略
#         返回 --- Order对象
#     """
#     def __init__(self,algo_mappings,data_portal,broker,assign_policy):
#         self.data_portal = data_portal
#         self.postion_allocation = assign_policy
#         self.broker = broker
#         self.loaders = [self.get_loader_class(key,args) for key,args in algo_mappings.items()]
#
#     @staticmethod
#     def get_loader_class(key,args):
#         """
#         :param key: algo_name or algo_path
#         :param args: algo_params
#         :return: dict -- __name__ : instance
#         """
#
#     # @lru_cache(maxsize=32)
#     def compute_withdraw(self,dt):
#         def run(ins):
#             result = ins.before_trading_start(dt)
#             return result
#
#         with Pool(processes = len(self.loaders)) as pool:
#             exit_assets = [pool.apply_async(run,instance)
#                             for instance in self.loaders.values]
#         return exit_assets
#
#     # @lru_cache(maxsize=32)
#     def compute_algorithm(self,dt,metrics_tracker):
#         unprocessed_loaders = self.tracker_algorithm(metrics_tracker)
#         def run(algo):
#             ins = self.loaders[algo]
#             result = ins.initialize(dt)
#             return result
#
#         with Pool(processes=len(self.loaders)) as pool:
#             exit_assets = [pool.apply_async(run, algo)
#                            for algo in unprocessed_loaders]
#         return exit_assets
#
#     def tracker_algorithm(self,metrics_tracker):
#         unprocessed_algo = set(self.algorithm_mappings.keys()) - \
#                            set(map(lambda x : x.reason ,metrics_tracker.positions.asset))
#         return unprocessed_algo
#
#     def position_allocation(self):
#         return self.assign_policy.map_allocation(self.tracker_algorithm)
#
#     def _calculate_order_amount(self,asset,dt,total_value):
#         """
#             calculate how many shares to order based on the position managment
#             and price where is assigned to 10% limit in order to carry out order max amount
#         """
#         preclose = self.data_portal.get_preclose(asset,dt)
#         porportion = self.postion_allocation.compute_pos_placing(asset)
#         amount = np.floor(porportion * total_value / (preclose * 1.1))
#         return amount
#
#     def get_payout(self, dt,metrics_tracker):
#         """
#         :param metrics_tracker: to get the position
#         :return: sell_orders
#         """
#         assets_of_exit = self.compute_withdraw(dt)
#         positions = metrics_tracker.positions
#         if assets_of_exit:
#             [self.broker.order(asset,
#                                 positions[asset].amount)
#                                 for asset in assets_of_exit]
#             cleanup_transactions,additional_commissions = self.broker.get_transaction(self.data_portal)
#             return cleanup_transactions,additional_commissions
#
#     def get_layout(self,dt,metrics_tracker):
#         asset = self.compute_algorithm(dt,metrics_tracker)
#         avaiable_cash = metrics_tracker.portfolio.cash
#         [self.broker.order(asset,
#                             self._calculate_order_amount(asset,dt,avaiable_cash))
#                             for asset in asset]
#         transactions,new_commissions = self.broker.get_transaction(self.data_portal)
#         return transactions,new_commissions
#
#
#     def _pop_params(cls, kwargs):
#         """
#         Pop entries from the `kwargs` passed to cls.__new__ based on the values
#         in `cls.params`.
#
#         Parameters
#         ----------
#         kwargs : dict
#             The kwargs passed to cls.__new__.
#
#         Returns
#         -------
#         params : list[(str, object)]
#             A list of string, value pairs containing the entries in cls.params.
#
#         Raises
#         ------
#         TypeError
#             Raised if any parameter values are not passed or not hashable.
#         """
#         params = cls.params
#         if not isinstance(params, Mapping):
#             params = {k: NotSpecified for k in params}
#         param_values = []
#         for key, default_value in params.items():
#             try:
#                 value = kwargs.pop(key, default_value)
#                 if value is NotSpecified:
#                     raise KeyError(key)
#
#                 # Check here that the value is hashable so that we fail here
#                 # instead of trying to hash the param values tuple later.
#                 hash(value)
#             except KeyError:
#                 raise TypeError(
#                     "{typename} expected a keyword parameter {name!r}.".format(
#                         typename=cls.__name__,
#                         name=key
#                     )
#                 )
#             except TypeError:
#                 # Value wasn't hashable.
#                 raise TypeError(
#                     "{typename} expected a hashable value for parameter "
#                     "{name!r}, but got {value!r} instead.".format(
#                         typename=cls.__name__,
#                         name=key,
#                         value=value,
#                     )
#                 )
#
#             param_values.append((key, value))
#         return tuple(param_values)
#
#
# def validate_dtype(termname, dtype, missing_value):
#     """
#     Validate a `dtype` and `missing_value` passed to Term.__new__.
#
#     Ensures that we know how to represent ``dtype``, and that missing_value
#     is specified for types without default missing values.
#
#     Returns
#     -------
#     validated_dtype, validated_missing_value : np.dtype, any
#         The dtype and missing_value to use for the new term.
#
#     Raises
#     ------
#     DTypeNotSpecified
#         When no dtype was passed to the instance, and the class doesn't
#         provide a default.
#     NotDType
#         When either the class or the instance provides a value not
#         coercible to a numpy dtype.
#     NoDefaultMissingValue
#         When dtype requires an explicit missing_value, but
#         ``missing_value`` is NotSpecified.
#     """
#     if dtype is NotSpecified:
#         raise DTypeNotSpecified(termname=termname)
#
#     try:
#         dtype = dtype_class(dtype)
#     except TypeError:
#         raise NotDType(dtype=dtype, termname=termname)
#
#     if not can_represent_dtype(dtype):
#         raise UnsupportedDType(dtype=dtype, termname=termname)
#
#     if missing_value is NotSpecified:
#         missing_value = default_missing_value_for_dtype(dtype)
#
#     try:
#         if (dtype == categorical_dtype):
#             # This check is necessary because we use object dtype for
#             # categoricals, and numpy will allow us to promote numerical
#             # values to object even though we don't support them.
#             _assert_valid_categorical_missing_value(missing_value)
#
#         # For any other type, we can check if the missing_value is safe by
#         # making an array of that value and trying to safely convert it to
#         # the desired type.
#         # 'same_kind' allows casting between things like float32 and
#         # float64, but not str and int.
#         array([missing_value]).astype(dtype=dtype, casting='same_kind')
#     except TypeError as e:
#         raise TypeError(
#             "Missing value {value!r} is not a valid choice "
#             "for term {termname} with dtype {dtype}.\n\n"
#             "Coercion attempt failed with: {error}".format(
#                 termname=termname,
#                 value=missing_value,
#                 dtype=dtype,
#                 error=e,
#             )
#         )
#
#     return dtype, missing_value
#
# def initial_refcounts(self, initial_terms):
#     """
#     Calculate initial refcounts for execution of this graph.
#
#     Parameters
#     ----------
#     initial_terms : iterable[Term]
#         An iterable of terms that were pre-computed before graph execution.
#
#     Each node starts with a refcount equal to its outdegree, and output
#     nodes get one extra reference to ensure that they're still in the graph
#     at the end of execution.
#     """
#     refcounts = self.graph.out_degree()
#     for t in self.outputs.values():
#         refcounts[t] += 1
#
#     for t in initial_terms:
#         self._decref_dependencies_recursive(t, refcounts, set())
#
#     return refcounts
#
# def _decref_dependencies_recursive(self, term, refcounts, garbage):
#     """
#     Decrement terms recursively.
#     Notes
#     -----
#     This should only be used to build the initial workspace, after that we
#     should use:
#     :meth:`~zipline.pipe.graph.TermGraph.decref_dependencies`
#     """
#     # Edges are tuple of (from, to).
#     for parent, _ in self.graph.in_edges([term]):
#         refcounts[parent] -= 1
#         # No one else depends on this term. Remove it from the
#         # workspace to conserve memory.
#         if refcounts[parent] == 0:
#             garbage.add(parent)
#             self._decref_dependencies_recursive(parent, refcounts, garbage)
#
# def execution_order(self, workspace, refcounts):
#     """
#     Return a topologically-sorted list of the terms in ``self`` which
#     need to be computed.
#
#     Filters out any terms that are already present in ``workspace``, as
#     well as any terms with refcounts of 0.
#
#     Parameters
#     ----------
#     workspace : dict[Term, np.ndarray]
#         Initial state of workspace for a pipe execution. May contain
#         pre-computed values provided by ``populate_initial_workspace``.
#     refcounts : dict[Term, int]
#         Reference counts for terms to be computed. Terms with reference
#         counts of 0 do not need to be computed.
#     """
#     return list(nx.topological_sort(
#         self.graph.subgraph(
#             {
#                 term for term, refcount in refcounts.items()
#                 if refcount > 0 and term not in workspace
#             },
#         ),
#     ))
#
#
# class ExecutionPlan(TermGraph):
#     """
#     Graph represention of Pipeline Term dependencies that includes metadata
#     about extra rows required to perform computations.
#
#     Each node in the graph has an `extra_rows` attribute, indicating how many,
#     if any, extra rows we should compute for the node.  Extra rows are most
#     often needed when a term is an input to a rolling window computation.  For
#     example, if we compute a 30 day moving average of price from day X to day
#     Y, we need to load price data for the range from day (X - 29) to day Y.
#
#     Parameters
#     ----------
#     domain : zipline.pipe.domain.Domain
#         The domain of execution for which we need to build a plan.
#     terms : dict
#         A dict mapping names to final output terms.
#     start_date : pd.Timestamp
#         The first date for which output is requested for ``terms``.
#     end_date : pd.Timestamp
#         The last date for which output is requested for ``terms``.
#
#     Attributes
#     ----------
#     domain
#     extra_rows
#     outputs
#     offset
#     """
#     def __init__(self,
#                  domain,
#                  terms,
#                  start_date,
#                  end_date,
#                  min_extra_rows=0):
#         super(ExecutionPlan, self).__init__(terms)
#
#         # Specialize all the LoadableTerms in the graph to our domain, so that
#         # when the engine requests an execution order, we emit the specialized
#         # versions of loadable terms.
#         #
#         # NOTE: We're explicitly avoiding using self.loadable_terms here.
#         #
#         # At this point the graph still contains un-specialized loadable terms,
#         # and this is where we're actually going through and specializing all
#         # of them. We don't want use self.loadable_terms because it's a
#         # lazyval, and we don't want its result to be cached until after we've
#         # specialized.
#         specializations = {
#             t: t.specialize(domain)
#             for t in self.graph if isinstance(t, LoadableTerm)
#         }
#         self.graph = nx.relabel_nodes(self.graph, specializations)
#
#         self.domain = domain
#
#         sessions = domain.all_sessions()
#         for term in terms.values():
#             self.set_extra_rows(
#                 term,
#                 sessions,
#                 start_date,
#                 end_date,
#                 min_extra_rows=min_extra_rows,
#             )
#
#         self._assert_all_loadable_terms_specialized_to(domain)
#
#     def set_extra_rows(self,
#                        term,
#                        all_dates,
#                        start_date,
#                        end_date,
#                        min_extra_rows):
#         # Specialize any loadable terms before adding extra rows.
#         term = maybe_specialize(term, self.domain)
#
#         # A term can require that additional extra rows beyond the minimum be
#         # computed.  This is most often used with downsampled terms, which need
#         # to ensure that the first date is a computation date.
#         extra_rows_for_term = term.compute_extra_rows(
#             all_dates,
#             start_date,
#             end_date,
#             min_extra_rows,
#         )
#         if extra_rows_for_term < min_extra_rows:
#             raise ValueError(
#                 "term %s requested fewer rows than the minimum of %d" % (
#                     term, min_extra_rows,
#                 )
#             )
#
#         self._ensure_extra_rows(term, extra_rows_for_term)
#
#         for dependency, additional_extra_rows in term.dependencies.items():
#             self.set_extra_rows(
#                 dependency,
#                 all_dates,
#                 start_date,
#                 end_date,
#                 min_extra_rows=extra_rows_for_term + additional_extra_rows,
#             )
#
#     @lazyval
#     def offset(self):
#         """
#         For all pairs (term, input) such that `input` is an input to `term`,
#         compute a mapping::
#
#             (term, input) -> offset(term, input)
#
#         where ``offset(term, input)`` is the number of rows that ``term``
#         should truncate off the raw array produced for ``input`` before using
#         it. We compute this value as follows::
#
#             offset(term, input) = (extra_rows_computed(input)
#                                    - extra_rows_computed(term)
#                                    - requested_extra_rows(term, input))
#         Examples
#         --------
#
#         Case 1
#         ~~~~~~
#
#         Factor A needs 5 extra rows of USEquityPricing.close, and Factor B
#         needs 3 extra rows of the same.  Factor A also requires 5 extra rows of
#         USEquityPricing.high, which no other Factor uses.  We don't require any
#         extra rows of Factor A or Factor B
#
#         We load 5 extra rows of both `price` and `high` to ensure we can
#         service Factor A, and the following offsets get computed::
#
#             offset[Factor A, USEquityPricing.close] == (5 - 0) - 5 == 0
#             offset[Factor A, USEquityPricing.high]  == (5 - 0) - 5 == 0
#             offset[Factor B, USEquityPricing.close] == (5 - 0) - 3 == 2
#             offset[Factor B, USEquityPricing.high] raises KeyError.
#
#         Case 2
#         ~~~~~~
#
#         Factor A needs 5 extra rows of USEquityPricing.close, and Factor B
#         needs 3 extra rows of Factor A, and Factor B needs 2 extra rows of
#         USEquityPricing.close.
#
#         We load 8 extra rows of USEquityPricing.close (enough to load 5 extra
#         rows of Factor A), and the following offsets get computed::
#
#             offset[Factor A, USEquityPricing.close] == (8 - 3) - 5 == 0
#             offset[Factor B, USEquityPricing.close] == (8 - 0) - 2 == 6
#             offset[Factor B, Factor A]              == (3 - 0) - 3 == 0
#
#         Notes
#         -----
#         `offset(term, input) >= 0` for all valid pairs, since `input` must be
#         an input to `term` if the pair appears in the mapping.
#
#         This value is useful because we load enough rows of each input to serve
#         all possible dependencies.  However, for any given dependency, we only
#         want to compute using the actual number of required extra rows for that
#         dependency.  We can do so by truncating off the first `offset` rows of
#         the loaded data for `input`.
#
#         See Also
#         --------
#         :meth:`zipline.pipe.graph.ExecutionPlan.offset`
#         :meth:`zipline.pipe.engine.ExecutionPlan.mask_and_dates_for_term`
#         :meth:`zipline.pipe.engine.SimplePipelineEngine._inputs_for_term`
#         """
#         extra = self.extra_rows
#
#         out = {}
#         for term in self.graph:
#             for dep, requested_extra_rows in term.dependencies.items():
#                 specialized_dep = maybe_specialize(dep, self.domain)
#
#                 # How much bigger is the result for dep compared to term?
#                 size_difference = extra[specialized_dep] - extra[term]
#
#                 # Subtract the portion of that difference that was required by
#                 # term's lookback window.
#                 offset = size_difference - requested_extra_rows
#                 out[term, specialized_dep] = offset
#
#         return out
#
#     @lazyval
#     def extra_rows(self):
#         """
#         A dict mapping `term` -> `# of extra rows to load/compute of `term`.
#
#         Notes
#         ----
#         This value depends on the other terms in the graph that require `term`
#         **as an input**.  This is not to be confused with `term.dependencies`,
#         which describes how many additional rows of `term`'s inputs we need to
#         load, and which is determined entirely by `Term` itself.
#
#         Examples
#         --------
#         Our graph contains the following terms:
#
#             A = SimpleMovingAverage([USEquityPricing.high], window_length=5)
#             B = SimpleMovingAverage([USEquityPricing.high], window_length=10)
#             C = SimpleMovingAverage([USEquityPricing.low], window_length=8)
#
#         To compute N rows of A, we need N + 4 extra rows of `high`.
#         To compute N rows of B, we need N + 9 extra rows of `high`.
#         To compute N rows of C, we need N + 7 extra rows of `low`.
#
#         We store the following extra_row requirements:
#
#         self.extra_rows[high] = 9  # Ensures that we can service B.
#         self.extra_rows[low] = 7
#
#         See Also
#         --------
#         :meth:`zipline.pipe.graph.ExecutionPlan.offset`
#         :meth:`zipline.pipe.Term.dependencies`
#         """
#         return {
#             term: attrs['extra_rows']
#             for term, attrs in iteritems(self.graph.node)
#         }
#
#     def _ensure_extra_rows(self, term, N):
#         """
#         Ensure that we're going to compute at least N extra rows of `term`.
#         """
#         attrs = self.graph.node[term]
#         attrs['extra_rows'] = max(N, attrs.get('extra_rows', 0))
#
#     def mask_and_dates_for_term(self,
#                                 term,
#                                 root_mask_term,
#                                 workspace,
#                                 all_dates):
#         """
#         Load mask and mask row labels for term.
#
#         Parameters
#         ----------
#         term : Term
#             The term to load the mask and labels for.
#         root_mask_term : Term
#             The term that represents the root asset exists mask.
#         workspace : dict[Term, any]
#             The values that have been computed for each term.
#         all_dates : pd.DatetimeIndex
#             All of the dates that are being computed for in the pipe.
#
#         Returns
#         -------
#         mask : np.ndarray
#             The correct mask for this term.
#         dates : np.ndarray
#             The slice of dates for this term.
#         """
#         mask = term.mask
#         mask_offset = self.extra_rows[mask] - self.extra_rows[term]
#
#         # This offset is computed against root_mask_term because that is what
#         # determines the shape of the top-level dates array.
#         dates_offset = (
#             self.extra_rows[root_mask_term] - self.extra_rows[term]
#         )
#
#         return workspace[mask][mask_offset:], all_dates[dates_offset:]
#
#     def _assert_all_loadable_terms_specialized_to(self, domain):
#         """Make sure that we've specialized all loadable terms in the graph.
#         """
#         for term in self.graph.node:
#             if isinstance(term, LoadableTerm):
#                 assert term.domain is domain

# import glob
#
# namespace = {}
#
# files = glob.glob('strategy/*.py')
# print(files)
#
# with open(files[2], 'r') as f:
#     exec(f.read(), namespace)
#
# print(namespace.keys())
#
# ins = namespace['PairWise']()
# print(ins)
#
# class AlgorithmSimulation(object):
#
#     EMISSION_TO_PERF_KEY_MAP = {
#         'minute': 'minute_perf',
#         'daily': 'daily_perf'
#     }
#
#     # def __init__(self,algo,sim_params,data_portal,benchmark_source):
#     #
#     #     self.algo = algo
#     #     self.sim_params = sim_params
#     #     self.data_portal = data_portal
#     #     self.benchmark = benchmark_source
#
#     def __init__(self, algo, sim_params, data_portal, clock, benchmark_source,
#                  restrictions, universe_func):
#
#         # ==============
#         # ArkQuant
#         # Param Setup
#         # ==============
#         self.sim_params = sim_params
#         self.data_portal = data_portal
#         self.restrictions = restrictions
#
#         # ==============
#         # Algo Setup
#         # ==============
#         self.algo = algo
#
#         # ==============
#         # Snapshot Setup
#         # ==============
#
#         # This object is the way that user algorithms interact with OHLCV data,
#         # fetcher data, and some API methods like `data.can_trade`.
#         self.current_data = self._create_bar_data(universe_func)
#
#         # We don't have a datetime for the current snapshot until we
#         # receive a message.
#         self.simulation_dt = None
#
#         self.clock = clock
#
#         self.benchmark_source = benchmark_source
#
#         # =============
#         # Logging Setup
#         # =============
#
#         # Processor function for injecting the algo_dt into
#         # user prints/logs.
#         def inject_algo_dt(record):
#             if 'algo_dt' not in record.extra:
#                 record.extra['algo_dt'] = self.simulation_dt
#
#     def get_simulation_dt(self):
#         return self.simulation_dt
#
#     #获取交易日数据，封装为一个API(fetch process flush other api)
#     def _create_bar_data(self, universe_func):
#         return BarData(
#             data_portal=self.data_portal,
#             simulation_dt_func=self.get_simulation_dt,
#             data_frequency=self.sim_params.data_frequency,
#             trading_calendar=self.algo.trading_calendar,
#             restrictions=self.restrictions,
#             universe_func=get_splits_divdend
#         )
#
#     def transfrom(self,dt):
#         """
#         Main generator work loop.
#         """
#         algo = self.algo
#         metrics_tracker = algo.metrics_tracker
#         emission_rate = metrics_tracker.emission_rate
#         engine = algo.engine
#         handle_data = algo.event_manager.handle_data
#
#         metrics_tracker.handle_market_open(dt, algo.data_portal)
#
#         def process_txn_commission(transactions,commissions):
#             for txn in transactions:
#                 metrics_tracker.process_transaction(txn)
#
#             for commission in commissions:
#                 metrics_tracker.process_commission(commission)
#
#         @contextlib.contextmanager
#         def once_a_day(dt):
#             payout = engine.get_payout(dt,metrics_tracker)
#             try:
#                 yield payout
#             finally:
#                 layout = engine.get_layout(dt,metrics_tracker)
#                 process_txn_commission(*layout)
#
#         def on_exit():
#             # Remove references to algo, data portal, et al to break cycles
#             # and ensure deterministic cleanup of these objects when the
#             # ArkQuant finishes.
#             self.algo = None
#             self.benchmark_source = self.data_portal = None
#
#         with ExitStack() as stack:
#             """
#             由于已注册的回调是按注册的相反顺序调用的，因此最终行为就好像with 已将多个嵌套语句与已注册的一组回调一起使用。
#             这甚至扩展到异常处理-如果内部回调抑制或替换异常，则外部回调将基于该更新状态传递自变量。
#             enter_context  输入一个新的上下文管理器，并将其__exit__()方法添加到回调堆栈中。返回值是上下文管理器自己的__enter__()方法的结果。
#             callback（回调，* args，** kwds ）接受任意的回调函数和参数，并将其添加到回调堆栈中。
#             """
#             stack.callback(on_exit())
#             stack.enter_context(ZiplineAPI(self.algo))
#
#             for dt in algo.trading_calendar:
#
#                 algo.on_dt_changed(dt)
#                 algo.before_trading_start(self.current_data(dt))
#                 with once_a_day(dt) as  action:
#                     process_txn_commission(*action)
#                 yield self._get_daily_message(dt, algo, metrics_tracker)
#
#             risk_message = metrics_tracker.handle_simulation_end(
#                 self.data_portal,
#             )
#             yield risk_message
#
#     def _get_daily_message(self, dt, algo, metrics_tracker):
#         """
#         Get a perf message for the given datetime.
#         """
#         perf_message = metrics_tracker.handle_market_close(
#             dt,
#             self.data_portal,
#         )
#         perf_message['daily_perf']['recorded_vars'] = algo.recorded_vars
#         return perf_message
#
#
# def run_pipeline(self, pipe, start_session, chunksize):
#     """
#     Compute `Pipeline`, providing values for at least `start_date`.
#
#     Produces a DataFrame containing data for days between `start_date` and
#     `end_date`, where `end_date` is defined by:
#
#         `end_date = min(start_date + chunksize trading days,
#                         simulation_end)`
#
#     Returns
#     -------
#     (data, valid_until) : tuple (pd.DataFrame, pd.Timestamp)
#
#     See Also
#     --------
#     PipelineEngine.run_pipeline
#     """
#     sessions = self.trading_calendar.all_sessions
#
#     # Load data starting from the previous trading day...
#     start_date_loc = sessions.get_loc(start_session)
#
#     # ...continuing until either the day before the ArkQuant end, or
#     # until chunksize days of data have been loaded.
#     sim_end_session = self.sim_params.end_session
#
#     end_loc = min(
#         start_date_loc + chunksize,
#         sessions.get_loc(sim_end_session)
#     )
#
#     end_session = sessions[end_loc]
#
#     return \
#         self.engine.run_pipeline(pipe, start_session, end_session), \
#         end_session
#
# @staticmethod
# def default_pipeline_domain(_calendar):
#     """
#     Get a default Pipeline domain for algorithms running on ``_calendar``.
#
#     This will be used to infer a domain for pipelines that only use generic
#     datasets when running in the context of a TradingAlgorithm.
#     """
#     return _DEFAULT_DOMAINS.get(_calendar.name, domain.GENERIC)
#
# def compute_eager_pipelines(self):
#     """
#     Compute any pipelines attached with eager=True.
#     """
#     for name, pipe in self._pipelines.items():
#         if pipe.eager:
#             self.pipeline_output(name)
#
#
#     def compute_chunk(self,
#                       graph,
#                       dates,
#                       sids,
#                       workspace,
#                       refcounts,
#                       execution_order,
#                       hooks):
#         """
#         Compute the Pipeline terms in the graph for the requested start and end
#         dates.
#
#         This is where we do the actual work of running a pipe.
#
#         Parameters
#         ----------
#         graph : zipline.pipe.graph.ExecutionPlan
#             Dependency graph of the terms to be executed.
#         dates : pd.DatetimeIndex
#             Row labels for our root mask.
#         sids : pd.Int64Index
#             Column labels for our root mask.
#         workspace : dict
#             Map from term -> output.
#             Must contain at least entry for `self._root_mask_term` whose shape
#             is `(len(dates), len(asset))`, but may contain additional
#             pre-computed terms for testing or optimization purposes.
#         refcounts : dict[Term, int]
#             Dictionary mapping terms to number of dependent terms. When a
#             term's refcount hits 0, it can be safely discarded from
#             ``workspace``. See TermGraph.decref_dependencies for more info.
#         execution_order : list[Term]
#             Order in which to execute terms.
#         hooks : implements(PipelineHooks)
#             Hooks to instrument pipe execution.
#
#         Returns
#         -------
#         results : dict
#             Dictionary mapping requested results to outputs.
#         """
#         for term in execution_order:
#             if isinstance(term, LoadableTerm):
#                 loader = get_loader(term)
#                 to_load = sorted(
#                     loader_groups[loader_group_key(term)],
#                     key=lambda t: t.dataset
#                 )
#                 with hooks.loading_terms(to_load):
#                     loaded = loader.load_adjusted_array(
#                         domain, to_load, mask_dates, sids, mask,
#                     )
#
#             else:
#                 with hooks.computing_term(term):
#                     workspace[term] = term._compute(
#                         self._inputs_for_term(
#                             term,
#                             workspace,
#                             graph,
#                             domain,
#                             refcounts,
#                         ),
#                         mask_dates,
#                         sids,
#                         mask,
#                     )
#
# import warnings
# from functools import wraps
#
#
# def deprecated(msg=None, stacklevel=2):
#     """
#     Used to mark a function as deprecated.
#     Parameters
#     ----------
#     msg : str
#         The message to display in the deprecation warning.
#     stacklevel : int
#         How far up the stack the warning needs to go, before
#         showing the relevant calling lines.
#     Usage
#     -----
#     @deprecated(msg='function_a is deprecated! Use function_b instead.')
#     def function_a(*args, **kwargs):
#     """
#     def deprecated_dec(fn):
#         @wraps(fn)
#         def wrapper(*args, **kwargs):
#             warnings.warn(
#                 msg or "Function %s is deprecated." % fn.__name__,
#                 category=DeprecationWarning,
#                 stacklevel=stacklevel
#             )
#             return fn(*args, **kwargs)
#         return wrapper
#     return deprecated_dec
#
# def copy(self):
#     """Copy an adjusted array, deep-copying the ``data`` array.
#     """
#     if self._unvalidated:
#         raise ValueError('cannot copy unvalidated AdjustedArray')
#
#     return type(self)()
#
#
# SQLITE_SPLITS_DIVIDEND_COLUMN_DTYPES = {
#     'declared_date':Timestamp,
#     'record_date':Timestamp,
#     'ex_date':Timestamp,
#     'pay_date':Timestamp,
#     'effective_date':Timestamp,
#     'sid_bonus':int,
#     'sid_transfer':int,
#     'bonus':float,
#     'progress':str,
#
# }
#
# SQLITE_STOCK_RIGHTS_COLUMN_DTYPES = {
#     'declared_date':Timestamp,
#     'record_date':Timestamp,
#     'ex_date':Timestamp,
#     'pay_date':Timestamp,
#     'effective_date':Timestamp,
#     'sid_bonus':int,
#     'right_price':float,
# }
#
# @classmethod
# def from_dict(cls, dict_):
#     """
#     Build an Asset instance from a dict.
#     """
#     return cls(**{k: v for k, v in dict_.items() if k in cls._kwargnames})
#
#
# class AssetExists(object):
#     """
#     Pseudo-filter describing whether or not an asset existed on a given day.
#     This is the default mask for all terms that haven't been passed a mask
#     explicitly.
#
#     This is morally a Filter, in the sense that it produces a boolean value for
#     every asset on every date.  We don't subclass Filter, however, because
#     `AssetExists` is computed directly by the PipelineEngine.
#
#     This term is guaranteed to be available as an input for any term computed
#     by SimplePipelineEngine.run_pipeline().
#
#     See Also
#     --------
#     zipline.asset.AssetFinder.lifetimes
#     """
#     dtype = bool_dtype
#     dataset = None
#     inputs = ()
#     dependencies = {}
#     mask = None
#     windowed = False
#
#     def __repr__(self):
#         return "AssetExists()"
#
#     graph_repr = __repr__
#
#     def _compute(self, today, asset, out):
#         raise NotImplementedError(
#             "AssetExists cannot be computed directly."
#             " Check your PipelineEngine configuration."
#         )


# @property
# def exchange(self):
#     raise NotImplemented
#
# @property
# def asset_name(self):
#     raise NotImplemented
#
# @property
# def first_traded(self):
#     raise NotImplemented

# @property
# def exchange(self):
#     exchange = 'SH' if self.sid.startswith('6') else 'SZ'
#     return exchange
#
# @property
# def asset_name(self):
#     self.equity_basics_info[1]
#
# @property
# def broker(self):
#     return self.equity_basics_info[2]
#
# @property
# def registered_eara(self):
#     return self.equity_basics_info[-1]
#
# @property
# def first_traded_dt(self):
#     return self.equity_basics_info[0]

# @property
# def pledge(self):
#     """股票质押率"""
#     pledge = ts.to_ts_pledge(self.sid)
#     return pledge

# def extra_description_about_asset(self):
#     """
#         extra information about asset --- equity structure
#         股票的总股本、流通股本，公告日期,变动日期结构
#         Warning: (1366, "Incorrect DECIMAL value: '0' for column '' at row -1")
#         Warning: (1292, "Truncated incorrect DECIMAL value: '--'")
#         --- 将 -- 变为0
#     """
#     fix_equity_cols = ['declared_date', 'effective_day', 'general', 'float', 'strict']
#     table = metadata.tables['symbol_equity_basics']
#     ins = sa.select([table.c.declared_date, table.c.effective_day,
#                      sa.cast(table.c.general_share, sa.Numeric(20, 3)),
#                      sa.cast(table.c.float_aShare, sa.Numeric(20, 3)),
#                      sa.cast(table.c.strict_aShare, sa.Numeric(20, 3))]).where(
#         table.c.sid == self.sid)
#     rp = self.engine.execute(ins)
#     raw = rp.fetchall()
#     equity = dict(zip(fix_equity_cols, raw))
#     return equity

# @property
# def asset_name(self):
#     self.equity_basics_info[0]
#
# @property
# def first_traded(self):
#     """ convert_dt """
#     return self.bond_basics_info[1]
#
# @property
# def last_traded(self):
#     return self.bond_basics_info[2]
#
# @property
# def convert_price(self):
#     return self.bond_basics_info[3]
#
#
# def extra_description_about_asset(self):
#     return self.bond_basics_info[-1]

# @property
# def first_traded(self):
#     """ convert_dt """
#     tbl = metadata.tables['fund_price']
#     ins = sa.select([func.min(tbl.c.trade_dt)]).\
#         where(tbl.c.sid == self.sid)
#     rp = self.engine.execute(ins)
#     init_date = rp.fetchall()[0]
#     return init_date

# asset_db_table_names = frozenset({
#     'symbol_naive_price',
#     'dual_symbol_price'
#     'bond_price',
#     'index_price',
#     'fund_price',
#     'symbol_equity_basics',
#     'bond_basics',
#     'symbol_splits',
#     'symbol_issue',
#     'symbol_mcap',
#     'symbol_massive',
#     'market_margin',
# })
#
#
# def _all_tables_present(self, txn):
#     """
#     Checks if any tables are present in the current asset database.
#
#     Parameters
#     ----------
#     txn : Transaction
#         The open transaction to check in.
#
#     Returns
#     -------
#     has_tables : bool
#         True if any tables are present, otherwise False.
#     """
#     conn = txn.connect()
#     for table_name in asset_db_table_names:
#         if txn.dialect.has_table(conn, table_name):
#             return True
#     return False
#
#
# def init_db(self, txn=None):
#     """Connect to database and create tables.
#
#     Parameters
#     ----------
#     txn : sa.engine.Connection, optional
#         The transaction to execute in. If this is not provided, a new
#         transaction will be started with the engine provided.
#
#     Returns
#     -------
#     metadata : sa.MetaData
#         The metadata that describes the new asset db.
#     """
#     with ExitStack() as stack:
#         if txn is None:
#             txn = stack.enter_context(self.engine.begin())
#
#         tables_already_exist = self._all_tables_present(txn)
#
#         # Create the SQL tables if they do not already exist.
#         metadata.create_all(txn, checkfirst=True)
#
#         if tables_already_exist:
#             check_version_info(txn, version_info, ASSET_DB_VERSION)
#         else:
#             write_version_info(txn, version_info, ASSET_DB_VERSION)
#
#
# class SimplePipelineEngine(PipelineEngine):
#     """
#     PipelineEngine class that computes each term independently.
#
#     Parameters
#     ----------
#     get_loader : callable
#         A function that is given a loadable term and returns a PipelineLoader
#         to use to retrieve raw data for that term.
#     asset_finder : zipline.asset.AssetFinder
#         An AssetFinder instance.  We depend on the AssetFinder to determine
#         which asset are in the top-level universe at any point in time.
#     populate_initial_workspace : callable, optional
#         A function which will be used to populate the initial workspace when
#         computing a pipe. See
#         :func:`zipline.pipe.engine.default_populate_initial_workspace`
#         for more info.
#     default_hooks : list, optional
#         List of hooks that should be used to instrument all pipelines executed
#         by this engine.
#
#     See Also
#     --------
#     :func:`zipline.pipe.engine.default_populate_initial_workspace`
#     """
#     __slots__ = (
#         '_get_loader',
#         '_finder',
#         '_root_mask_term',
#         '_root_mask_dates_term',
#         '_populate_initial_workspace',
#     )
#
#     def __init__(self,
#                  get_loader,
#                  asset_finder,
#                  default_domain=GENERIC,
#                  populate_initial_workspace=None,
#                  default_hooks=None):
#
#         self._get_loader = get_loader
#         self._finder = asset_finder
#
#         self._root_mask_term = AssetExists()
#         self._root_mask_dates_term = InputDates()
#
#         self._populate_initial_workspace = (
#             populate_initial_workspace or default_populate_initial_workspace
#         )
#         self._default_domain = default_domain
#
#         if default_hooks is None:
#             self._default_hooks = []
#         else:
#             self._default_hooks = list(default_hooks)
#
#     def _resolve_hooks(self, hooks):
#         if hooks is None:
#             hooks = []
#         return DelegatingHooks(self._default_hooks + hooks)
#
#     def run_pipeline(self, pipe, start_date, end_date, hooks=None):
#         """
#         Compute values for ``pipe`` from ``start_date`` to ``end_date``.
#
#         Parameters
#         ----------
#         pipe : zipline.pipe.Pipeline
#             The pipe to run.
#         start_date : pd.Timestamp
#             Start date of the computed matrix.
#         end_date : pd.Timestamp
#             End date of the computed matrix.
#         hooks : list[implements(PipelineHooks)], optional
#             Hooks for instrumenting Pipeline execution.
#
#         Returns
#         -------
#         result : pd.DataFrame
#             A frame of computed results.
#
#             The ``result`` columns correspond to the entries of
#             `pipe.columns`, which should be a dictionary mapping strings to
#             instances of :class:`zipline.pipe.Term`.
#
#             For each date between ``start_date`` and ``end_date``, ``result``
#             will contain a row for each asset that passed `pipe.screen`.
#             A screen of ``None`` indicates that a row should be returned for
#             each asset that existed each day.
#         """
#         hooks = self._resolve_hooks(hooks)
#         with hooks.running_pipeline(pipe, start_date, end_date):
#             return self._run_pipeline_impl(
#                 pipe,
#                 start_date,
#                 end_date,
#                 hooks,
#             )
#
#     def _run_pipeline_impl(self, pipe, start_date, end_date, hooks):
#         """Shared core for ``run_pipeline`` and ``run_chunked_pipeline``.
#         """
#         with hooks.computing_chunk(execution_order,
#                                    start_date,
#                                    end_date):
#
#             results = self.compute_chunk(
#                 graph=plan,
#                 dates=dates,
#                 sids=sids,
#                 workspace=workspace,
#                 refcounts=refcounts,
#                 execution_order=execution_order,
#                 hooks=hooks,
#             )
#
#         return self._to_narrow(
#             plan.outputs,
#             results,
#             results.pop(plan.screen_name),
#             dates[extra_rows:],
#             sids,
#         )
#
#
#     def run_chunked_pipeline(self,
#                              pipe,
#                              start_date,
#                              end_date,
#                              chunksize,
#                              hooks=None):
#         """
#         Compute values for ``pipe`` from ``start_date`` to ``end_date``, in
#         date chunks of size ``chunksize``.
#
#         Chunked execution reduces memory consumption, and may reduce
#         computation time depending on the contents of your pipe.
#
#         Parameters
#         ----------
#         pipe : Pipeline
#             The pipe to run.
#         start_date : pd.Timestamp
#             The start date to run the pipe for.
#         end_date : pd.Timestamp
#             The end date to run the pipe for.
#         chunksize : int
#             The number of days to execute at a time.
#         hooks : list[implements(PipelineHooks)], optional
#             Hooks for instrumenting Pipeline execution.
#
#         Returns
#         -------
#         result : pd.DataFrame
#             A frame of computed results.
#
#             The ``result`` columns correspond to the entries of
#             `pipe.columns`, which should be a dictionary mapping strings to
#             instances of :class:`zipline.pipe.Term`.
#
#             For each date between ``start_date`` and ``end_date``, ``result``
#             will contain a row for each asset that passed `pipe.screen`.
#             A screen of ``None`` indicates that a row should be returned for
#             each asset that existed each day.
#
#         See Also
#         --------
#         :meth:`zipline.pipe.engine.PipelineEngine.run_pipeline`
#         """
#
#         hooks = self._resolve_hooks(hooks)
#
#         run_pipeline = partial(self._run_pipeline_impl, pipe, hooks=hooks)
#         with hooks.running_pipeline(pipe, start_date, end_date):
#             chunks = [run_pipeline(s, e) for s, e in ranges]
#
#
# class EODCancel(CancelPolicy):
#     """
#         eod means the day which asset of order withdraw from market or suspend
#     """
#     @classmethod
#     def should_cancel(cls,asset,dt):
#         eod = asset.is_alive(dt)
#         return eod

import pandas as pd
from itertools import chain
# 按照固定时间去执行
# interval = 4 * 60 / size
# a = pd.date_range(start='09:30', end='11:30', freq='50min')
# b = pd.date_range(start='13:00', end='14:57', freq='50min')
# print(a)
# print(b)
# c = list(zip(a,b))
# print(c)
# d = list(chain(*zip(a,b)))
# last = pd.Timestamp('2020-06-17 14:57:00',freq = '50min')
# d.append(last)
# print(d)

# def _make_sids(tblattr):
#     def _(self):
#         return tuple(map(
#             itemgetter('sid'),
#             sa.select((
#                 getattr(self, tblattr).c.sid,
#             )).execute().fetchall(),
#         ))
#
#     return _

# equities_sids = property(
#     _make_sids('equities'),
#     doc='All of the sids for equities in the asset finder.',
# )

from numbers import Integral
from operator import itemgetter, attrgetter

from pandas import isnull
from six import with_metaclass, string_types, viewkeys, iteritems
from toolz import (
    compose,
    concat,
    # vertical itertools.chain
    concatv,
    curry,
    groupby,
    merge,
    partition_all,
    sliding_window,
    #valmap --- apply function to the values of dictionary
    valmap,
)

import array,binascii,struct , numpy as np,sqlalchemy as sa

# _cache = WeakValueDictionary()
#
#
# def __new__(cls,
#             engine,
#             metadata,
#             _trading_calendar):
#     identity = (engine, metadata, _trading_calendar)
#     try:
#         instance = cls._cache[identity]
#     except KeyError:
#         instance = cls._cache[identity] = super(AssetFinder, cls).__new__(cls)._initialize(*identity)
#     return instance

# if __name__ == '__main__':
#
#     engine = create_engine('mysql+pymysql://root:macpython@localhost:3306/spider',
#                             pool_size=50,
#                             max_overflow=100,
#                             pool_timeout=-1,
#                             pool_pre_ping=True,
#                             isolation_level="READ UNCOMMITTED")
#     con = engine.connect().execution_options(isolation_level = "READ UNCOMMITTED")
#     print(con.get_execution_options())
# hstock= sa.select(self.dual_symbol.hk).\
#           where(self.dual_symbol.c.sid == sid).\
#           execute().scalar()

# #A股交易日
# trading_calendar = sa.Table(
#     'trading_calendar',
#     metadata,
#     sa.Column(
#         'trading_day',
#         sa.Text,
#         unique=True,
#         nullable=False,
#         primary_key=True,
#         index = True,
#     ),
# )
# dividend_info.append({
#     "declared_date": dividend_tuple[1],
#     "ex_date": pd.Timestamp(dividend_tuple[2], unit="s"),
#     "pay_date": pd.Timestamp(dividend_tuple[3], unit="s"),
#     "payment_sid": dividend_tuple[4],
#     "ratio": dividend_tuple[5],
#     "record_date": pd.Timestamp(dividend_tuple[6], unit="s"),
#     "sid": dividend_tuple[7]
# })

# def shift_dates(dates, start_date, end_date, shift):
#     """
#     Shift dates of a pipe query back by `shift` days.
#
#     load_adjusted_array is called with dates on which the user's algo
#     will be shown data, which means we need to return the data that would
#     be known at the start of each date.  This is often labeled with a
#     previous date in the underlying data (e.g. at the start of today, we
#     have the data as of yesterday). In this case, we can shift the query
#     dates back to query the appropriate values.
#
#     Parameters
#     ----------
#     dates : DatetimeIndex
#         All known dates.
#     start_date : pd.Timestamp
#         Start date of the pipe query.
#     end_date : pd.Timestamp
#         End date of the pipe query.
#     shift : int
#         The number of days to shift back the query dates.
#     """
#     try:
#         start = dates.get_loc(start_date)
#     except KeyError:
#         if start_date < dates[0]:
#             raise NoFurtherDataError(
#                 msg=(
#                     "Pipeline Query requested data starting on {query_start}, "
#                     "but first known date is {calendar_start}"
#                 ).format(
#                     query_start=str(start_date),
#                     calendar_start=str(dates[0]),
#                 )
#             )
#         else:
#             raise ValueError("Query start %s not in _calendar" % start_date)
#
#     # Make sure that shifting doesn't push us out of the _calendar.
#     if start < shift:
#         raise NoFurtherDataError(
#             msg=(
#                 "Pipeline Query requested data from {shift}"
#                 " days before {query_start}, but first known date is only "
#                 "{start} days earlier."
#             ).format(shift=shift, query_start=start_date, start=start),
#         )
#
#     try:
#         end = dates.get_loc(end_date)
#     except KeyError:
#         if end_date > dates[-1]:
#             raise NoFurtherDataError(
#                 msg=(
#                     "Pipeline Query requesting data up to {query_end}, "
#                     "but last known date is {calendar_end}"
#                 ).format(
#                     query_end=end_date,
#                     calendar_end=dates[-1],
#                 )
#             )
#         else:
#             raise ValueError("Query end %s not in _calendar" % end_date)
#     return dates[start - shift], dates[end - shift]

# class HsymboleSessionReader(BarReader):
#
#     h_path = 'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=%s,day,%s,%s,10,qfq'
#
#     def __init__(self,
#                  trading_calenar,
#                  url = None):
#         self._trading_calenar = trading_calenar
#         self._url = url if url else self.h_path
#
#     def get_value(self,asset,edate,columns,window):
#         """
#             获取港股Kline , 针对于同时在A股上市的 , AH
#             load_daily_hSymbol('00168', '2011-01-01', '2012-01-06')
#             'us' + '.' + code
#         """
#         columns.apppend('trade_dt')
#         sdate = self._window_size_to_dt(edate,window)
#         request_sid = 'uk' + asset.sid
#         request_url = self._url%(request_sid,sdate,edate)
#         raw = _parse_url(request_url, bs=False, encoding=None)
#         raw = json.loads(raw)
#         data = raw['data']
#         if data and len(data):
#             daily = [item[:6] for item in data[request_sid]['day']]
#             # df = pd.DataFrame(daily,columns=['trade_dt','open','close','high','low','volume'])
#             df = pd.DataFrame(daily,columns=columns)
#             df.loc[:,'Hcode'] = asset.sid
#         return df
#
#     def load_raw_arrays(self, date, window, columns,asset):
#         raw_arrays = {}
#         _request_array = partial(self.get_value,edate = date,columns = columns,window = window)
#         #获取数据
#         for asset in asset:
#             raw_arrays[asset] = _request_array(asset = asset)
#         return raw_arrays

# 暂停上市股票
# symbol_lifetime = sa.Table(
#     'symbol_lifetime',
#     metadata,
#     sa.Column(
#         'sid',
#         sa.String(10),
#         sa.ForeignKey(equity_price.c.sid),
#         unique=True,
#         nullable=False,
#         primary_key=True,
#     ),
#     sa.Column('delist_date', sa.String(10)),
# )

# supspend_asset_url = 'http://datainterface.eastmoney.com/EM_DataCenter/JS.aspx?type=FD&sty=SRB&st=0&sr=-1&p=1&ps=500&' \
#                      'js={"pages":(pc),"data":[(x)]}&mkt=1&fd=%s'


import pandas as pd ,requests,json
from bs4 import BeautifulSoup
from collections import defaultdict

# Mapping from index symbol to appropriate bond data

ONE_HOUR = pd.Timedelta(hours=1)


def _parse_url(url, encoding='gbk', bs=True):
    Header = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36(KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'}
    req = requests.get(url, headers=Header, timeout=1)
    # if encoding:
    req.encoding = encoding
    if bs:
        raw = BeautifulSoup(req.text, features='lxml')
    else:
        raw = req.text
    return raw

# url = 'http://94.push2his.eastmoney.com/api/qt/stock/kline/get?secid=116.08231&fields1=f1' \
#       '&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57&klt=101&fqt=1&end=20500101&lmt=2'
# raw = _parse_url(url,bs = False)
# print(raw)
# data = json.loads(raw)['data']['klines']
# columns = ['trade_dt','open','close','high','low','volume','amount']
# print(data)
#
# delimeter = [d.split(',') for d in data]
#
# df = pd.DataFrame(delimeter,columns = columns)
# print(df)

# bond_asset_url = 'https://www.jisilu.cn/data/cbnew/cb_list/?'
# text = _parse_url(bond_asset_url, bs=False, encoding=None)
# text = json.loads(text)
# sids = text['rows']
# print(sids[100])

# supspend_url = 'http://datainterface.eastmoney.com/EM_DataCenter/JS.aspx?type=FD&sty=SRB&st=0&sr=-1&p=2&ps=500&' \
#                'js={"pages":(pc),"data":[(x)]}&mkt=1&fd=%s'

# supspend_url = 'http://datainterface.eastmoney.com/EM_DataCenter/JS.aspx?type=FD&sty=SRB&st=0&sr=-1&p=1&ps=50&' \
#                'js={"pages":(pc),"data":[(x)]}&mkt=1&fd=2020-05-13'
#
# text = _parse_url(supspend_url, bs=False, encoding=None)
# print(text)
# text = json.loads(text)
# print('supspend',text['data'])

#
# bond_asset_url = 'http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?type=KZZ_LB2.0' \
#                  '&token=70f12f2f4f091e459a279469fe49eca5&cmd=&sr=-1&p=1&ps=50&js={"pages":(tp),"data":(x)}'
# text = _parse_url(bond_asset_url, encoding= 'utf-8',bs=False)
# text = json.loads(text)
# sids = text['data']
# print(sids)
#
# # @preprocess(engine=coerce_string_to_eng(require_exists=False))
#
#公司基本情况
basics_url = 'https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/%s.phtml'
code = '002570'
url = basics_url % code
obj = _parse_url(url)
table = obj.find('table', {'id': 'comInfo1'})
tag = [item.findAll('td') for item in table.findAll('tr')]
tag_chain = list(chain(*tag))
raw = [item.get_text() for item in tag_chain]
# 去除格式
raw = [i.replace('：', '') for i in raw]
raw = [i.strip() for i in raw]
info = list(zip(raw[::2], raw[1::2]))
info_dict = {item[0]: item[1] for item in info}
# info_dict.update({'代码': code})
print(info_dict)
# #
# fund
import re
# fund_url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/jsonp.php/" \
#            "/Market_Center.getHQNodeDataSimple?page=%d&num=80&sort=symbol&asc=0&node=etf_hq_fund"
# page = 4
# url = fund_url % page
# obj = _parse_url(url,encoding='utf-8')
# text = obj.find('p').get_text()
# print('text',text)
# mid = re.findall('s[z|h][0-9]{6}', text)
# print(mid)

# 东方财富
fund_url = 'http://fund.eastmoney.com/cnjy_jzzzl.html'
obj = _parse_url(fund_url)
# print(obj.prettify())
from toolz import partition_all
raw = [data.find_all('td') for data in obj.find_all(id = 'tableDiv')]
text = [t.get_text() for t in raw[0]]
print(text)
df = pd.DataFrame(partition_all(14,text[18:]),columns = text[2:16])
df['基金简称'] = df['基金简称'].apply(lambda x: x[:-5])
print(df.head())
print(df['基金代码'].values)

equity_url = 'http://70.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&' \
             'fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12'
# 获取存量股票包括退市
raw = json.loads(_parse_url(equity_url, bs=False))
equities = [item['f12'] for item in raw['data']['diff']]
print('equities', equities)


asset = '002570'

cols = {'变动日期': 'ex_date', '公告日期': 'declared_date', '总股本': 'general', '流通A股': 'float', '限售A股': 'strict',
        '流通B股': 'b_float', '限售B股': 'b_strict', '流通H股': 'h_float'}

_url = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vCI_StockStructure/stockid/%s.phtml' % asset

frame = pd.DataFrame()
content = _parse_url(_url)
# resource = content['equity']
tbody = content.findAll('tbody')
if len(tbody) == 0:
    print('due to sina error ,it raise cannot set a frame with no defined index and a scalar when tbody is null')
for th in tbody:
    formatted = parse_content_from_header(th)
    frame = frame.append(formatted)
# 调整
frame.loc[:, 'sid'] = asset
frame.index = range(len(frame))
# deadline = self.deadline['equity_structure'][asset]
# equity = frame[frame['公告日期'] > deadline] if deadline else frame
# 需要rename cols
print(frame.columns)
print(frame.iloc[0,:])
frame.rename(columns=cols, inplace=True)
print('new', frame)

# def _generate_output_dataframe(self,data, default_cols):
#     """
#     Generates an output dataframe from the given subset of user-provided
#     data, the given column names, and the given default values.
#
#     Parameters
#     ----------
#     data : dict
#         A DataFrame, usually from an AssetData object,
#         that contains the user's input metadata for the asset type being
#         processed
#     default_cols : dict
#         A dict where the keys are the names of the columns of the desired
#         output DataFrame and the values are a function from dataframe and
#         column name to the default values to insert in the DataFrame if no user
#         data is provided
#
#     Returns
#     -------
#     DataFrame
#         A DataFrame containing all user-provided metadata, and default values
#         wherever user-provided metadata was missing
#     """
#     def _reformat_data(data):
#         _rename_cols = keyfilter(lambda x: x in equity_columns, default_cols)
#         insert_values = valmap(lambda x: data[x], _rename_cols)
#         return insert_values
#     #
#     data_subset = [_reformat_data(d) for d in data]
#     return data_subset

def _dt_to_epoch_ns(dt_series):
    """Convert a timeseries into an Int64Index of nanoseconds since the epoch.

    Parameters
    ----------
    dt_series : pd.Series
        The timeseries to convert.

    Returns
    -------
    idx : pd.Int64Index
        The index converted to nanoseconds since the epoch.
    """
    index = pd.to_datetime(dt_series.values)
    if index.tzinfo is None:
        index = index.tz_localize('UTC')
    else:
        index = index.tz_convert('UTC')
    return index.view(np.int64)

#asset_writer
# def split_delimited_symbol(symbol):
#     """
#     Takes in a symbol that may be delimited and splits it in to a company
#     symbol and share class symbol. Also returns the fuzzy symbol, which is the
#     symbol without any fuzzy characters at all.
#
#     Parameters
#     ----------
#     symbol : str
#         The possibly-delimited symbol to be split
#
#     Returns
#     -------
#     company_symbol : str
#         The company part of the symbol.
#     share_class_symbol : str
#         The share class part of a symbol.
#     """
#     # return blank strings for any bad fuzzy symbols, like NaN or None
#     if symbol in _delimited_symbol_default_triggers:
#         return '', ''
#
#     symbol = symbol.upper()
#
#     split_list = re.split(
#         pattern=_delimited_symbol_delimiters_regex,
#         string=symbol,
#         maxsplit=1,
#     )
#
#     # Break the list up in to its two extension, the company symbol and the
#     # share class symbol
#     company_symbol = split_list[0]
#     if len(split_list) > 1:
#         share_class_symbol = split_list[1]
#     else:
#         share_class_symbol = ''
#
#     return company_symbol, share_class_symbol
#
#
# def _check_asset_group(group):
#     row = group.sort_values('end_date').iloc[-1]
#     row.start_date = group.start_date.min()
#     row.end_date = group.end_date.max()
#     row.drop(list(symbol_columns), inplace=True)
#     return row
#
#
# def _format_range(r):
#     return (
#         str(pd.Timestamp(r.start, unit='ns')),
#         str(pd.Timestamp(r.stop, unit='ns')),
#     )
#
#
# def _check_symbol_mappings(df, exchanges, asset_exchange):
#     """Check that there are no cases where multiple symbols resolve to the same
#     asset at the same time in the same country.
#
#     Parameters
#     ----------
#     df : pd.DataFrame
#         The equity symbol mappings table.
#     exchanges : pd.DataFrame
#         The exchanges table.
#     asset_exchange : pd.Series
#         A series that maps sids to the exchange the asset is in.
#
#     Raises
#     ------
#     ValueError
#         Raised when there are ambiguous symbol mappings.
#     """
#     mappings = df.set_index('sid')[list(mapping_columns)].copy()
#     mappings['country_code'] = exchanges['country_code'][
#         asset_exchange.loc[df['sid']]
#     ].values
#     ambigious = {}
#
#     def check_intersections(persymbol):
#         intersections = list(intersecting_ranges(map(
#             from_tuple,
#             zip(persymbol.start_date, persymbol.end_date),
#         )))
#         if intersections:
#             data = persymbol[
#                 ['start_date', 'end_date']
#             ].astype('datetime64[ns]')
#             # indent the dataframe string, also compute this early because
#             # ``persymbol`` is a view and ``astype`` doesn't copy the index
#             # correctly in pandas 0.22
#             msg_component = '\n  '.join(str(data).splitlines())
#             ambigious[persymbol.name] = intersections, msg_component
#
#     mappings.groupby(['symbol', 'country_code']).apply(check_intersections)
#
#     if ambigious:
#         raise ValueError(
#             'Ambiguous ownership for %d symbol%s, multiple asset held the'
#             ' following symbols:\n%s' % (
#                 len(ambigious),
#                 '' if len(ambigious) == 1 else 's',
#                 '\n'.join(
#                     '%s (%s):\n  intersections: %s\n  %s' % (
#                         symbol,
#                         country_code,
#                         tuple(map(_format_range, intersections)),
#                         cs,
#                     )
#                     for (symbol, country_code), (intersections, cs) in sorted(
#                         ambigious.items(),
#                         key=first,
#                     ),
#                 ),
#             )
#         )
#
#
# def _split_symbol_mappings(df, exchanges):
#     """Split out the symbol: sid mappings from the raw data.
#
#     Parameters
#     ----------
#     df : pd.DataFrame
#         The dataframe with multiple rows for each symbol: sid pair.
#     exchanges : pd.DataFrame
#         The exchanges table.
#
#     Returns
#     -------
#     asset_info : pd.DataFrame
#         The asset info with one row per asset.
#     symbol_mappings : pd.DataFrame
#         The dataframe of just symbol: sid mappings. The index will be
#         the sid, then there will be three columns: symbol, start_date, and
#         end_date.
#     """
#     mappings = df[list(mapping_columns)]
#     with pd.option_context('mode.chained_assignment', None):
#         mappings['sid'] = mappings.index
#     mappings.reset_index(drop=True, inplace=True)
#
#     # take the most recent sid->exchange mapping based on end date
#     asset_exchange = df[
#         ['exchange', 'end_date']
#     ].sort_values('end_date').groupby(level=0)['exchange'].nth(-1)
#
#     _check_symbol_mappings(mappings, exchanges, asset_exchange)
#     return (
#         df.groupby(level=0).apply(_check_asset_group),
#         mappings,
#     )
#
#
# def _dt_to_epoch_ns(dt_series):
#     """Convert a timeseries into an Int64Index of nanoseconds since the epoch.
#
#     Parameters
#     ----------
#     dt_series : pd.Series
#         The timeseries to convert.
#
#     Returns
#     -------
#     idx : pd.Int64Index
#         The index converted to nanoseconds since the epoch.
#     """
#     index = pd.to_datetime(dt_series.values)
#     if index.tzinfo is None:
#         index = index.tz_localize('UTC')
#     else:
#         index = index.tz_convert('UTC')
#     return index.view(np.int64)
#
#
# def check_version_info(conn, version_table, expected_version):
#     """
#     Checks for a version value in the version table.
#
#     Parameters
#     ----------
#     conn : sa.Connection
#         The connection to use to perform the check.
#     version_table : sa.Table
#         The version table of the asset database
#     expected_version : int
#         The expected version of the asset database
#
#     Raises
#     ------
#     AssetDBVersionError
#         If the version is in the table and not equal to ASSET_DB_VERSION.
#     """
#
#     # Read the version out of the table
#     version_from_table = conn.execute(
#         sa.select((version_table.c.version,)),
#     ).scalar()
#
#     # A db without a version is considered v0
#     if version_from_table is None:
#         version_from_table = 0
#
#     # Raise an error if the versions do not match
#     if (version_from_table != expected_version):
#         raise AssetDBVersionError(db_version=version_from_table,
#                                   expected_version=expected_version)
#
#
# def write_version_info(conn, version_table, version_value):
#     """
#     Inserts the version value in to the version table.
#
#     Parameters
#     ----------
#     conn : sa.Connection
#         The connection to use to execute the insert.
#     version_table : sa.Table
#         The version table of the asset database
#     version_value : int
#         The version to write in to the database
#
#     """
#     conn.execute(sa.insert(version_table, values={'version': version_value}))
#
# Fuzzy symbol delimiters that may break up a company symbol and share class
# _delimited_symbol_delimiters_regex = re.compile(r'[./\-_]')
# _delimited_symbol_default_triggers = frozenset({np.nan, None, ''})
#
# def _default_none(df, column):
#     return None
#
# def _no_default(df, column):
#     if not df.empty:
#         raise ValueError('no default value for column %r' % column)
#
#
# # Default values for the equities DataFrame
# _equities_defaults = {
#     'symbol': _default_none,
#     'asset_name': _default_none,
#     'start_date': lambda df, col: 0,
#     # Machine limits for integer types.
#     'end_date': lambda df, col: np.iinfo(np.int64).max,
#     'first_traded': _default_none,
#     'auto_close_date': _default_none,
#     # the full exchange name
#     'exchange': _no_default,
# }


# Default values for the root_symbols DataFrame
# _root_symbols_defaults = {
#     'sector': _default_none,
#     'description': _default_none,
#     'exchange': _default_none,
# }
#
# # Default values for the equity_supplementary_mappings DataFrame
# _equity_supplementary_mappings_defaults = {
#     'value': _default_none,
#     'field': _default_none,
#     'start_date': lambda df, col: 0,
#     'end_date': lambda df, col: np.iinfo(np.int64).max,
# }
#
# # Default values for the equity_symbol_mappings DataFrame
# _equity_symbol_mappings_defaults = {
#     'sid': _no_default,
#     'company_symbol': _default_none,
#     'share_class_symbol': _default_none,
#     'symbol': _default_none,
#     'start_date': lambda df, col: 0,
#     'end_date': lambda df, col: np.iinfo(np.int64).max,
# }


from six.moves.urllib_error import HTTPError

#
# # 获取沪港通和深港通股票数据 , is_new = 1 表示沪港通的标的， is_new = 0 表示已经被踢出的沪港通的股票,exchange : SH SZ
# con_exchange = sa.Table(
#     'con_exchange',
#     metadata,
#     sa.Column('sid',
#               sa.String(6),
#               sa.ForeignKey(asset_router.c.sid),
#               nullable=False,
#               primary_key=True,
#               ),
#     sa.Column(
#         'exchange',
#         sa.String(6),
#         nullable=False,
#     ),
#     sa.Column(
#         'in_date',
#         sa.String(8),
#         nullable = False,
#     ),
#     sa.Column(
#         'out_date',
#         sa.String(6),
#         default = 'null'
#     ),
#     sa.Column('status', sa.Integer, nullable=False),
# )


# Default values for the exchanges DataFrame
# _exchanges_defaults = {
#     'canonical_name': lambda df, col: df.index,
#     'country_code': lambda df, col: '??',
# }

# MarketType

# The columns provided.
# 在ar1中但不在ar2中的已排序的唯一值
# missing_sids = np.setdiff1d(asset, self.sids)
# def compute_asset_lifetimes(frames):
#     """
#     Parameters
#     ----------
#     frames : dict[str, pd.DataFrame]
#         A dict mapping each OHLCV field to a dataframe with a row for
#         each date and a column for each sid, as passed to write().
#
#     Returns
#     -------
#     start_date_ixs : np.array[int64]
#         The index of the first date with non-nan values, for each sid.
#     end_date_ixs : np.array[int64]
#         The index of the last date with non-nan values, for each sid.
#     """
#     # Build a 2D array (dates x sids), where an entry is True if all
#     # fields are nan for the given day and sid.
#     is_null_matrix = np.logical_and.reduce(
#         [frames[field].isnull().values for field in FIELDS],
#     )
#     if not is_null_matrix.size:
#         empty = np.array([], dtype='int64')
#         return empty, empty.copy()
#
#     # Offset of the first null from the start of the input.
#     start_date_ixs = is_null_matrix.argmin(axis=0)
#     # Offset of the last null from the **end** of the input.
#     end_offsets = is_null_matrix[::-1].argmin(axis=0)
#     # Offset of the last null from the start of the input
#     end_date_ixs = is_null_matrix.shape[0] - end_offsets - 1
#     return start_date_ixs, end_date_ixs
#
# def _make_sids():
#     asset = np.array(asset)
#     sid_selector = self.sids.searchsorted(asset)
#     #查找相同的列，invert = True
#     unknown = np.in1d(asset, self.sids, invert=True)
#     sid_selector[unknown] = -1
#     return sid_selector
#
#
# def contextmanager(f):
#     """
#     Wrapper for contextlib.contextmanager that tracks which methods of
#     PipelineHooks are contextmanagers in CONTEXT_MANAGER_METHODS.
#     """
#     PIPELINE_HOOKS_CONTEXT_MANAGERS.add(f.__name__)
#     return contextmanager(f)

# def _load_cached_data(filename, first_date, last_date, now, resource_name,
#                       environ=None):
#     if resource_name == 'benchmark':
#         def from_csv(path):
#             return pd.read_csv(
#                 path,
#                 parse_dates=[0],
#                 index_col=0,
#                 header=None,
#                 # Pass squeeze=True so that we get a series instead of a frame.
#                 squeeze=True,
#             ).tz_localize('UTC')
#     else:
#         def from_csv(path):
#             return pd.read_csv(
#                 path,
#                 parse_dates=[0],
#                 index_col=0,
#             ).tz_localize('UTC')
#
#def mk(dt):
#  if not os.path.exists(dr):
#      os.makedirs(dr)
#  os.path.join(dr, name)

# import warnings
#
# class SidView:
#
#     """
#     This class exists to temporarily support the deprecated data[sid(N)] API.
#     """
#     def __init__(self, asset, data_portal, simulation_dt_func, data_frequency):
#         """
#         Parameters
#         ---------
#         asset : Asset
#             The asset for which the instance retrieves data.
#
#         data_portal : DataPortal
#             Provider for bar pricing data.
#
#         simulation_dt_func: function
#             Function which returns the current ArkQuant time.
#             This is usually bound to a method of TradingSimulation.
#
#         data_frequency: string
#             The frequency of the bar data; i.e. whether the data is
#             'daily' or 'minute' bars
#         """
#         self.asset = asset
#         self.data_portal = data_portal
#         self.simulation_dt_func = simulation_dt_func
#         self.data_frequency = data_frequency
#
#     def __getattr__(self, column):
#         # backwards compatibility code for Q1 API
#         if column == "close_price":
#             column = "close"
#         elif column == "open_price":
#             column = "open"
#         elif column == "dt":
#             return self.dt
#         elif column == "datetime":
#             return self.datetime
#         elif column == "sid":
#             return self.sid
#
#         return self.data_portal.get_spot_value(
#             self.asset,
#             column,
#             self.simulation_dt_func(),
#             self.data_frequency
#         )
#
#     def __contains__(self, column):
#         return self.data_portal.contains(self.asset, column)
#
#     def __getitem__(self, column):
#         return self.__getattr__(column)
#
#     @property
#     def sid(self):
#         return self.asset
#
#     @property
#     def dt(self):
#         return self.datetime
#
#     @property
#     def datetime(self):
#         return self.data_portal.get_last_traded_dt(
#                 self.asset,
#                 self.simulation_dt_func(),
#                 self.data_frequency)
#
#     @property
#     def current_dt(self):
#         return self.simulation_dt_func()
#
#     def mavg(self, num_minutes):
#         self._warn_deprecated("The `mavg` method is deprecated.")
#         return self.data_portal.get_simple_transform(
#             self.asset, "mavg", self.simulation_dt_func(),
#             self.data_frequency, bars=num_minutes
#         )
#
#     def stddev(self, num_minutes):
#         self._warn_deprecated("The `stddev` method is deprecated.")
#         return self.data_portal.get_simple_transform(
#             self.asset, "stddev", self.simulation_dt_func(),
#             self.data_frequency, bars=num_minutes
#         )
#
#     def vwap(self, num_minutes):
#         self._warn_deprecated("The `vwap` method is deprecated.")
#         return self.data_portal.get_simple_transform(
#             self.asset, "vwap", self.simulation_dt_func(),
#             self.data_frequency, bars=num_minutes
#         )
#
#     def returns(self):
#         self._warn_deprecated("The `returns` method is deprecated.")
#         return self.data_portal.get_simple_transform(
#             self.asset, "returns", self.simulation_dt_func(),
#             self.data_frequency
#         )
#
#     def _warn_deprecated(self, msg):
#         warnings.warn(
#             msg,
#             category=ZiplineDeprecationWarning,
#             stacklevel=1
#         )

# def _create_clock(self):
#     """
#     If the clock property is not set, then create one based on frequency.
#     """
#     trading_o_and_c = self.trading_calendar.schedule.ix[
#         self.sim_params.sessions]
#     market_closes = trading_o_and_c['market_close']
#     minutely_emission = False
#
#     if self.sim_params.data_frequency == 'minute':
#         market_opens = trading_o_and_c['market_open']
#         minutely_emission = self.sim_params.emission_rate == "minute"
#
#         # The _calendar's execution times are the minutes over which we
#         # actually want to run the clock. Typically the execution times
#         # simply adhere to the market open and close times. In the case of
#         # the futures _calendar, for example, we only want to simulate over
#         # a subset of the full 24 hour _calendar, so the execution times
#         # dictate a market open time of 6:31am US/Eastern and a close of
#         # 5:00pm US/Eastern.
#         execution_opens = \
#             self.trading_calendar.execution_time_from_open(market_opens)
#         execution_closes = \
#             self.trading_calendar.execution_time_from_close(market_closes)
#     else:
#         # in daily mode, we want to have one bar per session, timestamped
#         # as the last minute of the session.
#         execution_closes = \
#             self.trading_calendar.execution_time_from_close(market_closes)
#         execution_opens = execution_closes
#
#     # FIXME generalize these values
#     before_trading_start_minutes = days_at_time(
#         self.sim_params.sessions,
#         time(8, 45),
#         "US/Eastern"
#     )
#
#     return MinuteSimulationClock(
#         self.sim_params.sessions,
#         execution_opens,
#         execution_closes,
#         before_trading_start_minutes,
#         minute_emission=minutely_emission,
#     )

#
# Copyright 2018 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# import pandas as pd
#
#
# class BenchmarkSource(object):
#     def __init__(self,
#                  benchmark_asset,
#                  trading_calendar,
#                  sessions,
#                  data_portal,
#                  emission_rate="daily"):
#         self.benchmark_asset = benchmark_asset
#         self.sessions = sessions
#         self.emission_rate = emission_rate
#         self.data_portal = data_portal
#
#         if len(sessions) == 0:
#             self._precalculated_series = pd.Series()
#         elif benchmark_asset is not None:
#             self._validate_benchmark(benchmark_asset)
#             (self._precalculated_series,
#              self._daily_returns) = self._initialize_precalculated_series(
#                  benchmark_asset,
#                  trading_calendar,
#                  sessions,
#                  data_portal
#               )
#         elif benchmark_returns is not None:
#             self._daily_returns = daily_series = benchmark_returns.reindex(
#                 sessions,
#             ).fillna(0)
#
#             if self.emission_rate == "minute":
#                 # we need to take the env's benchmark returns, which are daily,
#                 # and resample them to minute
#                 minutes = trading_calendar.minutes_for_sessions_in_range(
#                     sessions[0],
#                     sessions[-1]
#                 )
#
#                 minute_series = daily_series.reindex(
#                     index=minutes,
#                     method="ffill"
#                 )
#
#                 self._precalculated_series = minute_series
#             else:
#                 self._precalculated_series = daily_series
#         else:
#             raise Exception("Must provide either benchmark_asset or "
#                             "benchmark_returns.")
#
#     def get_value(self, dt):
#         """Look up the returns for a given dt.
#
#         Parameters
#         ----------
#         dt : datetime
#             The label to look up.
#
#         Returns
#         -------
#         returns : float
#             The returns at the given dt or session.
#
#         See Also
#         --------
#         :class:`zipline.sources.benchmark_source.BenchmarkSource.daily_returns`
#
#         .. warning::
#
#            This method expects minute inputs if ``emission_rate == 'minute'``
#            and session labels when ``emission_rate == 'daily``.
#         """
#         return self._precalculated_series.loc[dt]
#
#     def get_range(self, start_dt, end_dt):
#         """Look up the returns for a given period.
#
#         Parameters
#         ----------
#         start_dt : datetime
#             The inclusive start label.
#         end_dt : datetime
#             The inclusive end label.
#
#         Returns
#         -------
#         returns : pd.Series
#             The series of returns.
#
#         See Also
#         --------
#         :class:`zipline.sources.benchmark_source.BenchmarkSource.daily_returns`
#
#         .. warning::
#
#            This method expects minute inputs if ``emission_rate == 'minute'``
#            and session labels when ``emission_rate == 'daily``.
#         """
#         return self._precalculated_series.loc[start_dt:end_dt]
#
#     def daily_returns(self, start, end=None):
#         """Returns the daily returns for the given period.
#
#         Parameters
#         ----------
#         start : datetime
#             The inclusive starting session label.
#         end : datetime, optional
#             The inclusive ending session label. If not provided, treat
#             ``start`` as a scalar key.
#
#         Returns
#         -------
#         returns : pd.Series or float
#             The returns in the given period. The index will be the trading
#             _calendar in the range [start, end]. If just ``start`` is provided,
#             return the scalar value on that day.
#         """
#         if end is None:
#             return self._daily_returns[start]
#
#         return self._daily_returns[start:end]
#
#     def _validate_benchmark(self, benchmark_asset):
#         # check if this security has a stock dividend.  if so, raise an
#         # error suggesting that the user pick a different asset to use
#         # as benchmark.
#         stock_dividends = \
#             self.data_portal.get_stock_dividends(self.benchmark_asset,
#                                                  self.sessions)
#
#         if len(stock_dividends) > 0:
#             raise InvalidBenchmarkAsset(
#                 sid=str(self.benchmark_asset),
#                 dt=stock_dividends[0]["ex_date"]
#             )
#
#         if benchmark_asset.start_date > self.sessions[0]:
#             # the asset started trading after the first ArkQuant day
#             raise BenchmarkAssetNotAvailableTooEarly(
#                 sid=str(self.benchmark_asset),
#                 dt=self.sessions[0],
#                 start_dt=benchmark_asset.start_date
#             )
#
#         if benchmark_asset.end_date < self.sessions[-1]:
#             # the asset stopped trading before the last ArkQuant day
#             raise BenchmarkAssetNotAvailableTooLate(
#                 sid=str(self.benchmark_asset),
#                 dt=self.sessions[-1],
#                 end_dt=benchmark_asset.end_date
#             )
#
#     @staticmethod
#     def _compute_daily_returns(g):
#         return (g[-1] - g[0]) / g[0]
#
#     @classmethod
#     def downsample_minute_return_series(cls,
#                                         trading_calendar,
#                                         minutely_returns):
#         sessions = trading_calendar.minute_index_to_session_labels(
#             minutely_returns.index,
#         )
#         closes = trading_calendar.session_closes_in_range(
#             sessions[0],
#             sessions[-1],
#         )
#         daily_returns = minutely_returns[closes].pct_change()
#         daily_returns.index = closes.index
#         return daily_returns.iloc[1:]
#
#     def _initialize_precalculated_series(self,
#                                          asset,
#                                          trading_calendar,
#                                          trading_days,
#                                          data_portal):
#         """
#         Internal method that pre-calculates the benchmark return series for
#         use in the ArkQuant.
#
#         Parameters
#         ----------
#         asset:  Asset to use
#
#         trading_calendar: TradingCalendar
#
#         trading_days: pd.DateTimeIndex
#
#         data_portal: DataPortal
#
#         Notes
#         -----
#         If the benchmark asset started trading after the ArkQuant start,
#         or finished trading before the ArkQuant end, exceptions are raised.
#
#         If the benchmark asset started trading the same day as the ArkQuant
#         start, the first available minute price on that day is used instead
#         of the previous close.
#
#         We use history to get an adjusted price history for each day's close,
#         as of the look-back date (the last day of the ArkQuant).  Prices are
#         fully adjusted for dividends, splits, and mergers.
#
#         Returns
#         -------
#         returns : pd.Series
#             indexed by trading day, whose values represent the %
#             change from close to close.
#         daily_returns : pd.Series
#             the partial daily returns for each minute
#         """
#         if self.emission_rate == "minute":
#             minutes = trading_calendar.minutes_for_sessions_in_range(
#                 self.sessions[0], self.sessions[-1]
#             )
#             benchmark_series = data_portal.get_history_window(
#                 [asset],
#                 minutes[-1],
#                 bar_count=len(minutes) + 1,
#                 frequency="1m",
#                 field="price",
#                 data_frequency=self.emission_rate,
#                 ffill=True
#             )[asset]
#
#             return (
#                 benchmark_series.pct_change()[1:],
#                 self.downsample_minute_return_series(
#                     trading_calendar,
#                     benchmark_series,
#                 ),
#             )
#
#         start_date = asset.start_date
#         if start_date < trading_days[0]:
#             # get the window of close prices for benchmark_asset from the
#             # last trading day of the ArkQuant, going up to one day
#             # before the ArkQuant start day (so that we can get the %
#             # change on day 1)
#             benchmark_series = data_portal.get_history_window(
#                 [asset],
#                 trading_days[-1],
#                 bar_count=len(trading_days) + 1,
#                 frequency="1d",
#                 field="price",
#                 data_frequency=self.emission_rate,
#                 ffill=True
#             )[asset]
#
#             returns = benchmark_series.pct_change()[1:]
#             return returns, returns
#         elif start_date == trading_days[0]:
#             # Attempt to handle case where stock data starts on first
#             # day, in this case use the open to close return.
#             benchmark_series = data_portal.get_history_window(
#                 [asset],
#                 trading_days[-1],
#                 bar_count=len(trading_days),
#                 frequency="1d",
#                 field="price",
#                 data_frequency=self.emission_rate,
#                 ffill=True
#             )[asset]
#
#             # get a minute history window of the first day
#             first_open = data_portal.get_spot_value(
#                 asset,
#                 'open',
#                 trading_days[0],
#                 'daily',
#             )
#             first_close = data_portal.get_spot_value(
#                 asset,
#                 'close',
#                 trading_days[0],
#                 'daily',
#             )
#
#             first_day_return = (first_close - first_open) / first_open
#
#             returns = benchmark_series.pct_change()[:]
#             returns[0] = first_day_return
#             return returns, returns
#         else:
#             raise ValueError(
#                 'cannot set benchmark to asset that does not exist during'
#                 ' the ArkQuant period (asset start date=%r)' % start_date
#             )
# from numpy import iinfo, uint32
# UINT32_MAX = iinfo(uint32).max
#
# class Engine(ABC):
#     """
#         1 存在价格笼子
#         2 无跌停限制但是存在竞价机制（10%基准价格），以及临时停盘制度
#         有存在竞价限制，科创板2% ，或者可转债10%
#         第十八条 债券现券竞价交易不实行价格涨跌幅限制。
# 　　             第十九条 债券上市首日开盘集合竞价的有效竞价范围为发行价的上下 30%，连续竞价、收盘集合竞价的有效竞价范围为最近成交价的上下 10%；
#         非上市首日开盘集合竞价的有效竞价范围为前收盘价的上下 10%，连续竞价、收盘集合竞价的有效竞价范围为最近成交价的上下 10%。
#          一、可转换公司债券竞价交易出现下列情形的，本所可以对其实施盘中临时停牌措施：
#     　　（一）盘中成交价较前收盘价首次上涨或下跌达到或超过20%的；
#     　　（二）盘中成交价较前收盘价首次上涨或下跌达到或超过30%的。
#     """
#     def reset(self):
#         self.engine_transactions = []
#
#     @abstractmethod
#     def _create_orders(self,asset,raw,**kwargs):
#         """
#             按照价格在10% 至 -10%范围内基于特定的统计分布模拟价格 --- 方向为开盘的涨跌幅 ，不适用于科创板（竞价机制要求）---买入价格不能超过基准价格（卖一的
#             102%，卖出价格不得低于买入价格98%，申报最小200股，递增可以以1股为单位 ；设立市价委托必须设立最高价以及最低价 ；
#             A股主板，中小板首日涨幅最大为44%而后10%波动，而科创板前5个交易日不设立涨跌停而后20%波动但是30%，60%临时停盘10分钟，如果超过2.57(复盘)；
#             科创板盘后固定价格交易 --- 以后15:00收盘价格进行交易 --- 15:00 -- 15:30(按照时间优先原则，逐步撮合成交）
#         """
#         raise NotImplementedError
#
#     @abstractmethod
#     def simulate_dist(self,alpha,size):
#         """
#         simulate price distribution to place on transactions
#         :param size: number of transactions
#         :param raw:  data for compute
#         :param multiplier: slippage base on vol_pct ,e.g. slippage = np.exp(vol_pct)
#         :return: array of simualtion price
#         """
#         raise NotImplementedError
#
#     def market_orders(self, capital, asset):
#         """按照市价竞价按照size --- 时间分割 TickerOrder"""
#         min_base_cost = self.commission.min_base_cost
#         size = capital / min_base_cost
#         tick_interval = self.simulate_dist(size)
#         for dts in tick_interval:
#             # 根据设立时间去定义订单
#             order = TickerOrder(asset,dts,min_base_cost)
#             self.broker(order, eager=True)
#
#     def call(self, capital, asset,raw,min_base_cost):
#         """执行前固化的订单买入计划"""
#         if not asset.bid_rule:
#             """按照价格区间竞价,适用于没有没有竞价机制要求，不会产生无效订单 PriceOrder"""
#             under_orders = self._create_orders(asset,
#                                                raw,
#                                                capital=capital,
#                                                min_base_cost = min_base_cost)
#         else:
#             under_orders = self.market_orders(capital, asset)
#         #执行买入订单
#         self.internal_oms(under_orders)
#
#     def _infer_order(self,capital_dct):
#         """基于时点执行买入订单,时间为进行入OMS系统的时间 --- 为了衔接卖出与买入"""
#         orders = [RealtimeOrder(asset,capital) for asset,capital in capital_dct]
#         self.broker(orders)
#
#     def _put_impl(self,position,raw,min_base_cost):
#         """按照市价竞价"""
#         amount = position.inner_position.amount
#         asset = position.inner_position.asset
#         last_sync_price = position.inner_position.last_sync_price
#         if not asset.bid_rule:
#             """按照价格区间竞价,适用于没有没有竞价机制要求，不会产生无效订单"""
#             tiny_put_orders = self._create_orders(asset,
#                                                   raw,
#                                                   amount = amount,
#                                                   min_base_cost = min_base_cost)
#         else:
#             min_base_cost = self.commission.min_base_cost
#             per_amount = np.ceil(self.multiplier['put'] * min_base_cost / (last_sync_price * 100))
#             size = amount / per_amount
#             #按照size --- 时间分割
#             intervals = self.simulate_tick(size)
#             for dts in intervals:
#                 tiny_put_orders = TickerOrder(per_amount * 100,asset,dts)
#         return tiny_put_orders
#
#     @staticmethod
#     def simulate_tick(size,final = True):
#         interval = 4 * 60 / size
#         # 按照固定时间去执行
#         day_m = pd.date_range(start='09:30', end='11:30', freq='%dmin'%interval)
#         day_a = pd.date_range(start='13:00', end='14:57', freq='%dmin'%interval)
#         day_ticker = list(chain(*zip(day_m, day_a)))
#         if final:
#             last = pd.Timestamp('2020-06-17 14:57:00',freq='%dmin'%interval)
#             day_ticker.append(last)
#         return day_ticker
#
#     def put(self,puts,raw,min_base_cost):
#         put_impl = partial(self._put_impl,
#                            raw = raw,
#                            min_base_cost = min_base_cost)
#         with Pool(processes=len(puts))as pool:
#             results = [pool.apply_async(put_impl,position)
#                        for position in puts.values]
#             put_orders = chain(*results)
#             # 执行卖出订单 --- 返回标识
#         for txn in self.internal_oms(put_orders, dual=True):
#                 #一旦有订单成交 基于队虽然有延迟，但是不影响
#                 txn_capital = txn.amount * txn.price
#                 yield txn_capital
#
#     @abstractmethod
#     def internal_oms(self,orders,eager = True):
#         """
#             principle --- 只要发出卖出信号的最大限度的卖出，如果没有完全卖出直接转入下一个交易日继续卖出
#             订单 --- priceOrder TickerOrder Intime
#             engine --- xtp or simulate(slippage_factor = self.slippage.calculate_slippage_factor)
#             dual -- True 双方向
#                   -- False 单方向（提交订单）
#             eager --- True 最后接近收盘时候集中将为成交的订单成交撮合成交保持最大持仓
#                   --- False 将为成交的订单追加之前由于restrict_rule里面的为成交订单里面
#             具体逻辑：
#                 当产生执行卖出订单时一旦成交接着执行买入算法，要求卖出订单的应该是买入Per买入标的的times，
#                 保证一次卖出成交金额可以覆盖买入标的
#             优势：提前基于一定的算法将订单根据时间或者价格提前设定好，在一定程度避免了被监测的程度。
#             成交的订单放入队列里面，不断的get
#             针对于put orders 生成的买入ticker_orders （逻辑 --- 滞后的订单是优先提交，主要由于订单生成到提交存在一定延迟)
#             订单优先级 --- Intime (first) > TickerOrder > priceOrder
#             基于asset计算订单成交比例
#             获取当天实时的ticer实点的数据，并且增加一些滑加，+ /-0.01
#         """
#         raise NotImplementedError

# class BackEngine(Engine):
#     """
#         基于ticker --- 进行回测,在执行具体的买入标的基于ticker数据真实模拟
#     """
#     def __init__(self,
#                 slippageModel,
#                 multiplier = {'call':1.5,'put':2},
#                 ):
#
#         # multipiler --- 针对基于保持最低交易成的capital的倍数进行交易
#         self.multiplier = multiplier
#         self.slippage = slippageModel()
#         self.engine_transactions = []
#
#     def _create_orders(self,asset,raw,**kwargs):
#         """
#             按照价格在10% 至 -10%范围内基于特定的统计分布模拟价格 --- 方向为开盘的涨跌幅 ，不适用于科创板（竞价机制要求）---买入价格不能超过基准价格（卖一的
#             102%，卖出价格不得低于买入价格98%，申报最小200股，递增可以以1股为单位 ；设立市价委托必须设立最高价以及最低价 ；
#             A股主板，中小板首日涨幅最大为44%而后10%波动，而科创板前5个交易日不设立涨跌停而后20%波动但是30%，60%临时停盘10分钟，如果超过2.57(复盘)；
#             科创板盘后固定价格交易 --- 以后15:00收盘价格进行交易 --- 15:00 -- 15:30(按照时间优先原则，逐步撮合成交）
#         """
#         multiplier = self.multiplier['call']
#         min_base_cost = kwargs['min_base_cost']
#         preclose = raw['preclose'][asset]
#         open_pct = raw['open_pct'][asset]
#         volume_restriction = self.volume_limit[asset]
#         try:
#             capital = kwargs['capital']
#             #ensuer_amount --- 手
#             bottom_amount = np.floor(capital / (preclose * 110))
#             if bottom_amount == 0:
#                 raise ValueError('satisfied at least 100 stocks')
#             #是否超过限制
#             ensure_amount = bottom_amount if bottom_amount <= volume_restriction else volume_restriction
#         except KeyError:
#             amount = kwargs['amount']
#             ensure_amount = amount if amount <= volume_restriction else volume_restriction
#         # 计算拆分订单的个数，以及单个订单金额
#         min_per_value = 90 * preclose / (open_pct + 1)
#         ensure_per_amount = np.ceil(multiplier * min_base_cost / min_per_value)
#         # 模拟价格分布的参数 --- 个数 数据 滑价系数
#         size = ensure_amount // ensure_per_amount
#         # volume = raw['volume'][asset]
#         alpha = 1 if open_pct == 0.00 else 100 * open_pct
#         sim_pct = self.simulate_dist(abs(alpha),size)
#         # 限价原则 --- 确定交易执行价格 针对于非科创板，创业板股票
#         # limit = self.style.get_limit_price() if self.style.get_limit_price() else asset.price_limit(dts)
#         # stop = self.style.get_stop_price() if self.style.get_stop_price() else asset.price_limit(dts)
#         limit = self.style.get_limit_price() if self.style.get_limit_price() else 0.1
#         stop = self.style.get_stop_price() if self.style.get_stop_price() else 0.1
#         clip_price = np.clip(sim_pct,-stop,limit) * preclose
#         # 将多余的手分散
#         sim_amount = np.tile([ensure_per_amount], size) if size > 0 else [ensure_amount]
#         random_idx = np.random.randint(0, size, ensure_amount % ensure_per_amount)
#         for idx in random_idx:
#             sim_amount[idx] += 1
#         #形成订单
#         tiny_orders =  [PriceOrder(asset,args[0],args[1])
#                      for args in zip(sim_amount,clip_price)]
#         return tiny_orders
#
#     def simulate_dist(self,alpha,size):
#         """
#         simulate price distribution to place on transactions
#         :param size: number of transactions
#         :param raw:  data for compute
#         :param multiplier: slippage base on vol_pct ,e.g. slippage = np.exp(vol_pct)
#         :return: array of simualtion price
#         """
#         # 涉及slippage --- 基于ensure_amount --- multiplier
#         if size > 0:
#             #模拟价格分布
#             dist = 1 + np.copysign(alpha,np.random.beta(alpha,100,size))
#         else:
#             dist = [1 + alpha  / 100]
#         return dist
#
#     def internal_oms(self,orders,eager = True):
#         """
#             principle --- 只要发出卖出信号的最大限度的卖出，如果没有完全卖出直接转入下一个交易日继续卖出
#             订单 --- priceOrder TickerOrder Intime
#             engine --- xtp or simulate(slippage_factor = self.slippage.calculate_slippage_factor)
#             dual -- True 双方向
#                   -- False 单方向（提交订单）
#             eager --- True 最后接近收盘时候集中将为成交的订单成交撮合成交保持最大持仓
#                   --- False 将为成交的订单追加之前由于restrict_rule里面的为成交订单里面
#             具体逻辑：
#                 当产生执行卖出订单时一旦成交接着执行买入算法，要求卖出订单的应该是买入Per买入标的的times，
#                 保证一次卖出成交金额可以覆盖买入标的
#             优势：提前基于一定的算法将订单根据时间或者价格提前设定好，在一定程度避免了被监测的程度。
#             成交的订单放入队列里面，不断的get
#             针对于put orders 生成的买入ticker_orders （逻辑 --- 滞后的订单是优先提交，主要由于订单生成到提交存在一定延迟)
#             订单优先级 --- Intime (first) > TickerOrder > priceOrder
#             基于asset计算订单成交比例
#             获取当天实时的ticer实点的数据，并且增加一些滑加，+ /-0.01
#         """
#         raise NotImplementedError()

# def load_data(self,dt,asset):
#     raw = self.adjust_array.load_array_for_sids(dt,0,['open','close','volume','amount','pct'],asset)
#     volume = { k : v['volume'] for k,v in raw.items()}
#     if raw:
#         """说明历史回测 --- 存在数据"""
#         preclose = { k: v['close'] / (v['pct'] +1 ) for k,v in raw.items()}
#         open_pct = { k: v['open'] / preclose[k] for k,v in raw.items()}
#     else:
#         """实时回测 , 9.25之后"""
#         raw = self.adjust_array.load_pricing_adjusted_array(dt,2,['open','close','pct'],asset)
#         minutes = self.adjust_array.load_minutes_for_sid(asset)
#         if not minutes:
#             raise ValueError('时间问题或者网路问题')
#         preclose = { k : v['close'][-1]  for k,v in raw.items() }
#         open_pct = { k : v.iloc[0,0] / preclose[k] for k,v in minutes.items()}
#     dct = {'preclose':preclose,'open_pct':open_pct,'volume':volume}
#     return dct

# class MatchUp(object):
#     """ 撮合成交
#         如果open_pct 达到10% --- 是否买入
#         分为不同的模块 创业板，科创板，ETF
#         包含 --- sell orders buy orders 同时存在，但是buy_orders --- 基于sell orders 和 ledger
#         通过限制买入capital的pct实现分布买入
#         但是卖出订单 --- 通过追加未成交订单来实现
#         如何连接卖出与买入模块
#
#         由capital --- calculate orders 应该属于在统一模块 ，最后将订单 --- 引擎生成交易 --- 执行计划式的，非手动操作类型的
#         剔除ReachCancel --- 10%
#         剔除SwatCancel --- 黑天鹅
#     """
#     def __init__(self,
#                 multiplier = 5,
#                 cancel_policy,
#                 execution_style,
#                 slippageMode):
#
#         #确定订单类型默认为市价单
#         self.style = execution_style
#         self.commission = AssetCommission(multiplier)
#         self.cancel_policy = ComposedCancel(cancel_policy)
#         self.engine = Engine()
#         self.adjust_array = AdjustArray()
#         self.record_transactions = OrderedDict()
#         self.record_efficiency = OrderedDict()
#         self.prune_closed_assets = OrderedDict()
#
#     @property
#     def _fraction(self):
#         """设立成交量限制，默认为前一个交易日的百分之一"""
#         return 0.05
#
#     @_fraction.setter
#     def _fraction(self,val):
#         return val
#
#     def execute_cancel_policy(self,target):
#         """买入 --- 如果以涨停价格开盘过滤，主要针对买入标的"""
#         _target =[self.cancel_policy.should_cancel(item) for item in target]
#         result = _target[0] if _target else None
#         return result
#
#     def _restrict_buy_rule(self,dct):
#         """
#             主要针对于买入标的的
#             对于卖出的，遵循最大程度卖出
#         """
#         self.capital_limit = valmap(lambda x : x * self._fraction,dct)
#
#     def attach_pruned_holdings(self,puts,holdings):
#         closed_holdings = valfilter(lambda x: x.inner_position.asset in self.prune_closed_assets, holdings)
#         puts.update(closed_holdings)
#         return puts
#
#     def carry_out(self,engine,ledger):
#         """建立执行计划"""
#         #engine --- 获得可以交易的标的
#         puts, calls,holdings,capital,dts = engine.execute_engine(ledger)
#         #将未完成的卖出的标的放入puts
#         puts = self.attach_pruned_holdings(puts,holdings)
#         self.commission._init_base_cost(dts)
#         #获取计算订单所需数据
#         asset = set([position.inner_position.asset for position in holdings]) | set(chain(*calls.values()))
#         raw = self.load_data(dts,asset)
#         #过滤针对标的
#         calls = valmap(lambda x:self.execute_cancel_policy(x),calls)
#         calls = valfilter(lambda x : x is not None,calls)
#         call_assets = list(calls.values())
#         #已有持仓标的
#         holding_assets = [holding.inner_position.asset for holding in holdings]
#         #卖出持仓标的
#         put_assets = [ put.inner_position.asset for put in puts]
#         # 限制 --- buys_amount,sell --- volume
#         self._restrict_rule(raw['amount'])
#         #固化参数
#         match_impl = partial(self.positive_match(holdings = holding_assets,capital = capital,raw = raw,dts = dts))
#         _match_impl = partial(self.dual_match(holdings = holding_assets,capital = capital,raw = raw,dts = dts))
#         _match_impl(put_assets,call_assets) if puts else match_impl(call_assets)
#         #获取存量的transactions
#         final_txns = self._init_engine(dts)
#         #计算关于总的订单拆分引擎撮合成交的的效率
#         self.evaluate_efficiency(capital,puts,dts)
#         #将未完成需要卖出的标的继续卖出
#         self.to_be_pruned(puts)
#         return final_txns
#
#     def _init_engine(self,dts):
#         txns = self.engine.engine_transactions
#         self.record_transactions[dts] = txns
#         self.engine.reset()
#         return txns
#
#     def evaluate_efficiency(self,capital,puts,dts):
#         """
#             根据标的追踪 --- 具体卖入订单根据volume计算成交率，买入订单根据成交额来计算资金利用率 --- 评估撮合引擎撮合的的效率
#         """
#         txns = self.record_transactions[dts]
#         call_efficiency = sum([ txn.amount * txn.price for txn in txns if txn.amount > 0 ]) / capital
#         put_efficiency = sum([txn.amount for txn in txns if txn.amount < 0]) / \
#                          sum([position.inner_position.amount for position in puts.values()]) if puts else 0
#         self.record_efficiency[dts] = {'call':call_efficiency,'put':put_efficiency}
#
#     def to_be_pruned(self,dts,puts):
#         #将未完全卖出的position存储继续卖出
#         txns = self.record_transactions[dts]
#         txn_put_amount = {txn.asset:txn.amount for txn in txns if txn.amount < 0}
#         position_amount = {position.inner_position.asset : position.inner_position.amount for position in puts}
#         pct = txn_put_amount / position_amount
#         uncompleted = keyfilter(lambda x : x < 1,pct)
#         self.prune_closed_assets[dts] = uncompleted.keys()
#
#     def positive_match(self,calls,holdings,capital,raw,dts):
#         """buys or sells parallel"""
#         if calls:
#             capital_dct = self.policy.calculate(calls,capital,dts)
#         else:
#             capital_dct = self.policy.calculate(holdings, capital,dts)
#         #买入金额限制
#         restrict_capital = {asset : self.capital_limit[asset] if capital >= self.capital_limit[asset]
#                                     else capital  for asset ,capital in capital_dct.items()}
#
#         call_impl = partial(self.engine.call,raw = raw,min_base_cost = self.commission.min_base_cost)
#         with Pool(processes=len(restrict_capital))as pool:
#             results = [pool.apply_async(call_impl,asset,capital)
#                        for asset,capital in restrict_capital.items()]
#             txns = chain(*results)
#         return txns
#
#     def dual_match(self,puts,calls,holdings,capital,dts,raw):
#         #双向匹配
#         """基于capital生成priceOrder"""
#         txns = dict()
#         if calls:
#             capital_dct = self.policy.calculate(calls,capital,dts)
#         else:
#             left_holdings = set(holdings) - set(puts)
#             capital_dct = self.policy.calculate(left_holdings,capital,dts)
#         #call orders
#         txns['call'] = self.positive_match(calls,holdings,capital_dct,raw,dts)
#         #put orders
#         # --- 直接以open_price卖出;如果卖出的话 --- 将未成交的卖出订单orders持续化
#         for txn_capital in self.engine.put(puts,calls,raw,self.commission.min_base_cost):
#             agg = sum(capital_dct.values())
#             trading_capital = valmap(lambda x : x * txn_capital / agg,capital_dct )
#             self.engine._infer_order(trading_capital)

# @abstractmethod
# def get_resampled(self, *args):
#     """
#      List of DatetimeIndex representing the minutes to exclude because
#      of early closes.
#     """
#     raise NotImplementedError()
#
# def get_resampled(self, sessions, dts, sids, field=default):
#     """
#         select specific dts minutes ,e,g --- 9:30,10:30
#     """
#     resample_tickers = {}
#     arrays = self.load_raw_arrays(sessions[0], sessions[1], sids, field)
#     for sid, raw in arrays.items():
#         ticker_seconds = dts.split(':')[0] * 60 * 60 + dts.split(':')[0] * 60
#         data = raw.fetchwhere("(timestamp - {0}) % {1} == 0".format(ticker_seconds, self._seconds_per_day))
#         resample_tickers[sid] = data
#     return resample_tickers
#
# def get_resampled(self,dts,window,frequency,sids,field = default):
#     """
#         select specific dts  Year Month D Min Second
#     """
#     resampled = {}
#     arrays = self.load_raw_arrays(dts,window,sids,field)
#     sdate = self._window_dt(dts,window)
#     pds = [dt.strftime('%Y%m%d') for dt in pd.date_range(sdate,dts,freq = frequency)]
#     for sid,raw in arrays.items():
#         resampled[sid] = raw.reindex(pds)
#     return resampled

# def _resovle_conflicts(self, outs, ins, holdings):
#     """
#         防止策略冲突 当pipeline的结果与ump的结果出现重叠 --- 说明存在问题，正常情况退出策略与买入策略应该不存交集
#
#         1. engine共用一个ump ---- 解决了不同策略产生的相同标的可以同一时间退出
#         2. engine --- 不同的pipeline对应不同的ump,产生1中的问题，相同的标的不会在同一时间退出是否合理（冲突）
#
#         退出策略 --- 针对标的，与标的是如何产生的不存在直接关系;只能根据资产类别的有关 --- 1
#         如果产生冲突 --- 当天卖出标的与买入标的产生重叠 说明策略是有问题的ump --- pipelines 对立的
#         symbol ,etf 的退出策略可以相同，但是bond不行属于T+0
#         return ---- name : [position , [pipeline_output]]
#
#         两个部分 pipelines - ledger
#                 positions -
#
#         建仓逻辑 --- 逐步建仓 1/2 原则 --- 1 优先发生信号先建仓 ，后发信号仓位变为剩下的1/2（为了提高资金利用效率）
#                                         2 如果没新的信号 --- 在已经持仓的基础加仓（不管资金是否足够或者设定一个底层资金池）
#         ---- 变相限定了单次单个标的最大持仓为1/2
#         position + pipe - ledger ---  (当ledger为空 --- position也为空)
#
#         关于ump --- 只要当天不是一直在跌停价格，以全部出货为原则，涉及一个滑价问题（position的成交额 与前一周的成交额占比
#         评估滑价），如果当天没有买入，可以适当放宽（开盘的时候卖出大部分，剩下的等等） ；
#         如果存在买入标的的行为则直接按照全部出货原则以open价格最大比例卖出 ，一般来讲集合竞价的代表主力卖入意愿强度）
#         ---- 侧面解决了卖出转为买入的断层问题 transfer1
#     """
#     intersection = set([item.inner_position.asset for item in outs]) & set(chain(*ins.values()))
#     if intersection:
#         raise ValueError('ump should not have intersection with pipelines')
#     out_dict = {position.inner_position.asset.origin: position
#                 for position in outs}
#     waited = set(ins) - (set(holdings) - out_dict)
#     result = keyfilter(lambda x: x in waited, ins)
#     return out_dict, result
# def intern_tunnel(self, p, c, dts):
#     """
#         holding , asset ,dts
#         基于触发器构建 通道 基于策略 卖出 --- 买入
#         principle --- 只要发出卖出信号的最大限度的卖出，如果没有完全卖出直接转入下一个交易日继续卖出
#         订单 --- priceOrder TickerOrder Intime
#         engine --- xtp or simulate(slippage_factor = self.slippage.calculate_slippage_factor)
#         dual -- True 双方向
#               -- False 单方向（提交订单）
#         eager --- True 最后接近收盘时候集中将为成交的订单成交撮合成交保持最大持仓
#               --- False 将为成交的订单追加之前由于restrict_rule里面的为成交订单里面
#         具体逻辑：
#             当产生执行卖出订单时一旦成交接着执行买入算法，要求卖出订单的应该是买入Per买入标的的times，
#             保证一次卖出成交金额可以覆盖买入标的
#         优势：提前基于一定的算法将订单根据时间或者价格提前设定好，在一定程度避免了被监测的程度。
#         成交的订单放入队列里面，不断的get
#         针对于put orders 生成的买入ticker_orders （逻辑 --- 滞后的订单是优先提交，主要由于订单生成到提交存在一定延迟)
#         订单优先级 --- Intime (first) > TickerOrder > priceOrder
#         基于asset计算订单成交比例
#         获取当天实时的ticer实点的数据，并且增加一些滑加，+ /-0.01
#         卖出标的 --- 对应买入标的 ，闲于的资金
#     """
#     # 计算卖出持仓
#     asset = p.inner_position.asset
#     # 获取数据
#     p_minutes = self._create_BarData(dts, asset)
#     # amount
#     q = p.inner_position.amount
#     # 每次卖出size
#     p_size = np.ceil(self.per_capital / (asset.tick_size * asset.preclose))
#     # 构建卖出组合
#     size_array = np.tile([p_size], int(q / p_size))
#     idx = np.random(int(q / p_size))
#     size_array[idx] += q % p_size
#     # 生成卖出订单
#     p_orders = self.create_order(asset, size_array)
#     # 根据 p_orders ---- 生成对应成交的ticker
#     p_transactions = [simulate_transaction(p_order, p_minutes, self.commission)
#                       for p_order in p_orders]
#     p_transaction_price = np.array([t.price for t in p_transactions])
#     # 执行对应的买入算法
#     # 获取买入标的的数据 c
#     c_minutes = c.minutes
#     # 增加ticker shift
#     c_tickers = [pd.Timedelta(minutes=self.delay) + t.ticker for t in p_transactions]
#     c_ticker_price = np.array([c_minutes[ticker] for ticker in c_tickers])
#     # 计算买入数据基于价格比值
#     times = p_transaction_price / c_ticker_price
#     c_size = [np.floor(size * times) for size in size_array]
#     # 构建对应买入订单
#     c_transactions = [create_transaction(c, c_size, c_price, c_ticker)
#                       for c_price, c_ticker in
#                       zip(c_ticker_price, c_tickers)]
#     return p_transactions, c_transactions

# def _cleanup_expired_assets(self, dt, position_assets):
#     """
#     Clear out any asset that have expired before starting a new sim day.
#
#     Performs two functions:
#
#     1. Finds all asset for which we have open orders and clears any
#        orders whose asset are on or after their auto_close_date.
#
#     2. Finds all asset for which we have positions and generates
#        close_position events for any asset that have reached their
#        auto_close_date.
#     """
#     algo = self.algo
#
#     def past_auto_close_date(asset):
#         acd = asset.auto_close_date
#         return acd is not None and acd <= dt
#
#     # Remove positions in any sids that have reached their auto_close date.
#     assets_to_clear = \
#         [asset for asset in position_assets if past_auto_close_date(asset)]
#     metrics_tracker = algo.metrics_tracker
#     data_portal = self.data_portal
#     for asset in assets_to_clear:
#         metrics_tracker.process_close_position(asset, dt, data_portal)
#
#     # Remove open orders for any sids that have reached their auto close
#     # date. These orders get processed immediately because otherwise they
#     # would not be processed until the first bar of the next day.
#     broker = algo.broker
#     assets_to_cancel = [
#         asset for asset in broker.open_orders
#         if past_auto_close_date(asset)
#     ]
#     for asset in assets_to_cancel:
#         broker.cancel_all_orders_for_asset(asset)
#
#     # Make a copy here so that we are not modifying the list that is being
#     # iterated over.
#     for order in copy(broker.new_orders):
#         if order.status == ORDER_STATUS.CANCELLED:
#             metrics_tracker.process_order(order)
#             broker.new_orders.remove(order)

# cash_utilization = 1 - (cash_blance /capital_blance).mean()

# def calculate(self, transaction):
#     self._init_base_cost(transaction.dt)
#     transaction_cost = transaction.amount * transaction.price
#     # 印花税 --- 1‰(卖的时候才收取，此为国家税收，全国统一)
#     stamp_cost = 0 if transaction.amount > 0 else transaction_cost * 1e-3
#     # 过户费：深圳交易所无此项费用，上海交易所收费标准(按成交金额的0.02)
#     transfer_cost = 2 * transaction_cost * 1e-5 if transaction.asset.startswith('6') else 0
#     # 交易佣金：最高收费为3‰，最低收费5元。各家劵商收费不一，开户前可咨询清楚。 2015年之后万/3
#     commission_cost = transaction_cost * self.commission_rate \
#         if transaction_cost > self.min_base_cost else self.min_cost
#     txn_cost = stamp_cost + transfer_cost + commission_cost
#     return txn_cost

# def _adjust_cost_basis_for_commission(self, txn_cost):
#     prev_cost = self.amount * self.cost_basis
#     new_cost = prev_cost + txn_cost
#     self.cost_basis = new_cost / self.amount
#
# def get_dividends(self, sids, trading_days):
#     """
#     splits --- divdends
#
#     Returns all the stock dividends for a specific sid that occur
#     in the given trading range.
#
#     Parameters
#     ----------
#     sid: int
#         The asset whose stock dividends should be returned.
#
#     trading_days: pd.DatetimeIndex
#         The trading range.
#
#     Returns
#     -------
#     list: A list of objects with all relevant attributes populated.
#     All timestamp fields are converted to pd.Timestamps.
#     """
#     extra = set(sids) - set(self._divdends_cache)
#     if extra:
#         for sid in extra:
#             divdends = self.adjustment_reader.load_splits_for_sid(sid)
#             self._divdends_cache[sid] = divdends
#     cache = keyfilter(lambda x: x in sids, self._splits_cache)
#     out = valmap(lambda x: x[x['pay_date'].isin(trading_days)] if x else x, cache)
#     return out
#
#
# def get_rights(self, sids, trading_days):
#     """
#     Returns all the stock dividends for a specific sid that occur
#     in the given trading range.
#
#     Parameters
#     ----------
#     sid: int
#         The asset whose stock dividends should be returned.
#
#     trading_days: pd.DatetimeIndex
#         The trading range.
#
#     Returns
#     -------
#     list: A list of objects with all relevant attributes populated.
#     All timestamp fields are converted to pd.Timestamps.
#     """
#     extra = set(sids) - set(self._rights_cache)
#     if extra:
#         for sid in extra:
#             rights = self.adjustment_reader.load_splits_for_sid(sid)
#             self._rights_cache[sid] = rights
#     #
#     cache = keyfilter(lambda x: x in sids, self._rights_cache)
#     out = valmap(lambda x: x[x['pay_date'].isin(trading_days)] if x else x, cache)
#     return out
#
# class OrderExecutionBit(object):
#     '''
#     Intended to hold information about order execution. A "bit" does not
#     determine if the order has been fully/partially executed, it just holds
#     information.
#
#     Member Attributes:
#
#       - dt: datetime (float) execution time
#       - size: how much was executed
#       - price: execution price
#       - closed: how much of the execution closed an existing postion
#       - opened: how much of the execution opened a new position
#       - openedvalue: market value of the "opened" part
#       - closedvalue: market value of the "closed" part
#       - closedcomm: commission for the "closed" part
#       - openedcomm: commission for the "opened" part
#
#       - value: market value for the entire bit size
#       - comm: commission for the entire bit execution
#       - pnl: pnl generated by this bit (if something was closed)
#
#       - psize: current open position size
#       - pprice: current open position price
#
#     '''
#
#     def __init__(self,
#                  dt=None, size=0, price=0.0,
#                  closed=0, closedvalue=0.0, closedcomm=0.0,
#                  opened=0, openedvalue=0.0, openedcomm=0.0,
#                  pnl=0.0,
#                  psize=0, pprice=0.0):
#
#         self.dt = dt
#         self.size = size
#         self.price = price
#
#         self.closed = closed
#         self.opened = opened
#         self.closedvalue = closedvalue
#         self.openedvalue = openedvalue
#         self.closedcomm = closedcomm
#         self.openedcomm = openedcomm
#
#         self.value = closedvalue + openedvalue
#         self.comm = closedcomm + openedcomm
#         self.pnl = pnl
#
#         self.psize = psize
#         self.pprice = pprice
#
#
# class OrderData(object):
#     '''
#     Holds actual order data for Creation and Execution.
#
#     In the case of Creation the request made and in the case of Execution the
#     actual outcome.
#
#     Member Attributes:
#
#       - exbits : iterable of OrderExecutionBits for this OrderData
#
#       - dt: datetime (float) creation/execution time
#       - size: requested/executed size
#       - price: execution price
#         Note: if no price is given and no pricelimite is given, the closing
#         price at the time or order creation will be used as reference
#       - pricelimit: holds pricelimit for StopLimit (which has trigger first)
#       - trailamount: absolute price distance in trailing stops
#       - trailpercent: percentage price distance in trailing stops
#
#       - value: market value for the entire bit size
#       - comm: commission for the entire bit execution
#       - pnl: pnl generated by this bit (if something was closed)
#       - margin: margin incurred by the Order (if any)
#
#       - psize: current open position size
#       - pprice: current open position price
#
#     '''
#     # According to the docs, collections.deque is thread-safe with appends at
#     # both ends, there will be no pop (nowhere) and therefore to know which the
#     # new exbits are two indices are needed. At time of cloning (__copy__) the
#     # indices can be updated to match the previous end, and the new end
#     # (len(exbits)
#     # Example: start 0, 0 -> islice(exbits, 0, 0) -> []
#     # One added -> copy -> updated 0, 1 -> islice(exbits, 0, 1) -> [1 elem]
#     # Other added -> copy -> updated 1, 2 -> islice(exbits, 1, 2) -> [1 elem]
#     # "add" and "__copy__" happen always in the same thread (with all current
#     # implementations) and therefore no append will happen during a copy and
#     # the len of the exbits can be queried with no concerns about another
#     # thread making an append and with no need for a lock
#
#     def __init__(self, dt=None, size=0, price=0.0, pricelimit=0.0, remsize=0,
#                  pclose=0.0, trailamount=0.0, trailpercent=0.0):
#
#         self.pclose = pclose
#         self.exbits = collections.deque()  # for historical purposes
#         self.p1, self.p2 = 0, 0  # indices to pending notifications
#
#         self.dt = dt
#         self.size = size
#         self.remsize = remsize
#         self.price = price
#         self.pricelimit = pricelimit
#         self.trailamount = trailamount
#         self.trailpercent = trailpercent
#
#         if not pricelimit:
#             # if no pricelimit is given, use the given price
#             self.pricelimit = self.price
#
#         if pricelimit and not price:
#             # price must always be set if pricelimit is set ...
#             self.price = pricelimit
#
#         self.plimit = pricelimit
#
#         self.value = 0.0
#         self.comm = 0.0
#         self.margin = None
#         self.pnl = 0.0
#
#         self.psize = 0
#         self.pprice = 0
#
#     def _getplimit(self):
#         return self._plimit
#
#     def _setplimit(self, val):
#         self._plimit = val
#
#     plimit = property(_getplimit, _setplimit)
#
#     def __len__(self):
#         return len(self.exbits)
#
#     def __getitem__(self, key):
#         return self.exbits[key]
#
#     def add(self, dt, size, price,
#             closed=0, closedvalue=0.0, closedcomm=0.0,
#             opened=0, openedvalue=0.0, openedcomm=0.0,
#             pnl=0.0,
#             psize=0, pprice=0.0):
#
#         self.addbit(
#             OrderExecutionBit(dt, size, price,
#                               closed, closedvalue, closedcomm,
#                               opened, openedvalue, openedcomm, pnl,
#                               psize, pprice))
#
#     def addbit(self, exbit):
#         # Stores an ExecutionBit and recalculates own values from ExBit
#         self.exbits.append(exbit)
#
#         self.remsize -= exbit.size
#
#         self.dt = exbit.dt
#         oldvalue = self.size * self.price
#         newvalue = exbit.size * exbit.price
#         self.size += exbit.size
#         self.price = (oldvalue + newvalue) / self.size
#         self.value += exbit.value
#         self.comm += exbit.comm
#         self.pnl += exbit.pnl
#         self.psize = exbit.psize
#         self.pprice = exbit.pprice
#
#     def getpending(self):
#         return list(self.iterpending())
#
#     def iterpending(self):
#         return itertools.islice(self.exbits, self.p1, self.p2)
#
#     def markpending(self):
#         # rebuild the indices to mark which exbits are pending in clone
#         self.p1, self.p2 = self.p2, len(self.exbits)
#
#     def clone(self):
#         obj = copy(self)
#         obj.markpending()
#         return obj
#
#
# class OrderBase(with_metaclass(MetaParams, object)):
#     params = (
#         ('owner', None), ('data', None),
#         ('size', None), ('price', None), ('pricelimit', None),
#         ('exectype', None), ('valid', None), ('tradeid', 0), ('oco', None),
#         ('trailamount', None), ('trailpercent', None),
#         ('parent', None), ('transmit', True),
#         ('simulated', False),
#         # To support historical order evaluation
#         ('histnotify', False),
#     )
#
#     DAY = datetime.timedelta()  # constant for DAY order identification
#
#     # Time Restrictions for orders
#     T_Close, T_Day, T_Date, T_None = range(4)
#
#     # Volume Restrictions for orders
#     V_None = range(1)
#
#     (Market, Close, Limit, Stop, StopLimit, StopTrail, StopTrailLimit,
#      Historical) = range(8)
#     ExecTypes = ['Market', 'Close', 'Limit', 'Stop', 'StopLimit', 'StopTrail',
#                  'StopTrailLimit', 'Historical']
#
#     OrdTypes = ['Buy', 'Sell']
#     Buy, Sell = range(2)
#
#     Created, Submitted, Accepted, Partial, Completed, \
#         Canceled, Expired, Margin, Rejected = range(9)
#
#     Cancelled = Canceled  # alias
#
#     Status = [
#         'Created', 'Submitted', 'Accepted', 'Partial', 'Completed',
#         'Canceled', 'Expired', 'Margin', 'Rejected',
#     ]
#
#     refbasis = itertools.count(1)  # for a unique identifier per order
#
#     def _getplimit(self):
#         return self._plimit
#
#     def _setplimit(self, val):
#         self._plimit = val
#
#     plimit = property(_getplimit, _setplimit)
#
#     def __getattr__(self, name):
#         # Return attr from params if not found in order
#         return getattr(self.params, name)
#
#     def __setattribute__(self, name, value):
#         if hasattr(self.params, name):
#             setattr(self.params, name, value)
#         else:
#             super(Order, self).__setattribute__(name, value)
#
#     def __str__(self):
#         tojoin = list()
#         tojoin.append('Ref: {}'.format(self.ref))
#         tojoin.append('OrdType: {}'.format(self.ordtype))
#         tojoin.append('OrdType: {}'.format(self.ordtypename()))
#         tojoin.append('Status: {}'.format(self.status))
#         tojoin.append('Status: {}'.format(self.getstatusname()))
#         tojoin.append('Size: {}'.format(self.size))
#         tojoin.append('Price: {}'.format(self.price))
#         tojoin.append('Price Limit: {}'.format(self.pricelimit))
#         tojoin.append('TrailAmount: {}'.format(self.trailamount))
#         tojoin.append('TrailPercent: {}'.format(self.trailpercent))
#         tojoin.append('ExecType: {}'.format(self.exectype))
#         tojoin.append('ExecType: {}'.format(self.getordername()))
#         tojoin.append('CommInfo: {}'.format(self.comminfo))
#         tojoin.append('End of Session: {}'.format(self.dteos))
#         tojoin.append('Info: {}'.format(self.info))
#         tojoin.append('Broker: {}'.format(self.broker))
#         tojoin.append('Alive: {}'.format(self.alive()))
#
#         return '\n'.join(tojoin)
#
#     def __init__(self):
#         self.ref = next(self.refbasis)
#         self.broker = None
#         self.info = AutoOrderedDict()
#         self.comminfo = None
#         self.triggered = False
#
#         self._active = self.parent is None
#         self.status = Order.Created
#
#         self.plimit = self.p.pricelimit  # alias via property
#
#         if self.exectype is None:
#             self.exectype = Order.Market
#
#         if not self.isbuy():
#             self.size = -self.size
#
#         # Set a reference price if price is not set using
#         # the close price
#         pclose = self.data.close[0] if not self.simulated else self.price
#         if not self.price and not self.pricelimit:
#             price = pclose
#         else:
#             price = self.price
#
#         dcreated = self.data.datetime[0] if not self.p.simulated else 0.0
#         self.created = OrderData(dt=dcreated,
#                                  size=self.size,
#                                  price=price,
#                                  pricelimit=self.pricelimit,
#                                  pclose=pclose,
#                                  trailamount=self.trailamount,
#                                  trailpercent=self.trailpercent)
#
#         # Adjust price in case a trailing limit is wished
#         if self.exectype in [Order.StopTrail, Order.StopTrailLimit]:
#             self._limitoffset = self.created.price - self.created.pricelimit
#             price = self.created.price
#             self.created.price = float('inf' * self.isbuy() or '-inf')
#             self.trailadjust(price)
#         else:
#             self._limitoffset = 0.0
#
#         self.executed = OrderData(remsize=self.size)
#         self.position = 0
#
#         if isinstance(self.valid, datetime.date):
#             # comparison will later be done against the raw datetime[0] value
#             self.valid = self.data.date2num(self.valid)
#         elif isinstance(self.valid, datetime.timedelta):
#             # offset with regards to now ... get utcnow + offset
#             # when reading with date2num ... it will be automatically localized
#             if self.valid == self.DAY:
#                 valid = datetime.datetime.combine(
#                     self.data.datetime.date(), datetime.time(23, 59, 59, 9999))
#             else:
#                 valid = self.data.datetime.datetime() + self.valid
#
#             self.valid = self.data.date2num(valid)
#
#         elif self.valid is not None:
#             if not self.valid:  # avoid comparing None and 0
#                 valid = datetime.datetime.combine(
#                     self.data.datetime.date(), datetime.time(23, 59, 59, 9999))
#             else:  # assume float
#                 valid = self.data.datetime[0] + self.valid
#
#         if not self.p.simulated:
#             # provisional end-of-session
#             # get next session end
#             dtime = self.data.datetime.datetime(0)
#             session = self.data.p.sessionend
#             dteos = dtime.replace(hour=session.hour, minute=session.minute,
#                                   second=session.second,
#                                   microsecond=session.microsecond)
#
#             if dteos < dtime:
#                 # eos before current time ... no ... must be at least next day
#                 dteos += datetime.timedelta(days=1)
#
#             self.dteos = self.data.date2num(dteos)
#         else:
#             self.dteos = 0.0
#
#     def clone(self):
#         # status, triggered and executed are the only moving parts in order
#         # status and triggered are covered by copy
#         # executed has to be replaced with an intelligent clone of itself
#         obj = copy(self)
#         obj.executed = self.executed.clone()
#         return obj  # status could change in next to completed
#
#     def getstatusname(self, status=None):
#         '''Returns the name for a given status or the one of the order'''
#         return self.Status[self.status if status is None else status]
#
#     def getordername(self, exectype=None):
#         '''Returns the name for a given exectype or the one of the order'''
#         return self.ExecTypes[self.exectype if exectype is None else exectype]
#
#     @classmethod
#     def ExecType(cls, exectype):
#         return getattr(cls, exectype)
#
#     def ordtypename(self, ordtype=None):
#         '''Returns the name for a given ordtype or the one of the order'''
#         return self.OrdTypes[self.ordtype if ordtype is None else ordtype]
#
#     def active(self):
#         return self._active
#
#     def activate(self):
#         self._active = True
#
#     def alive(self):
#         '''Returns True if the order is in a status in which it can still be
#         executed
#         '''
#         return self.status in [Order.Created, Order.Submitted,
#                                Order.Partial, Order.Accepted]
#
#     def addcomminfo(self, comminfo):
#         '''Stores a CommInfo scheme associated with the asset'''
#         self.comminfo = comminfo
#
#     def addinfo(self, **kwargs):
#         '''Add the keys, values of kwargs to the internal info dictionary to
#         hold custom information in the order
#         '''
#         for key, val in iteritems(kwargs):
#             self.info[key] = val
#
#     def __eq__(self, other):
#         return other is not None and self.ref == other.ref
#
#     def __ne__(self, other):
#         return self.ref != other.ref
#
#     def isbuy(self):
#         '''Returns True if the order is a Buy order'''
#         return self.ordtype == self.Buy
#
#     def issell(self):
#         '''Returns True if the order is a Sell order'''
#         return self.ordtype == self.Sell
#
#     def setposition(self, position):
#         '''Receives the current position for the asset and stotres it'''
#         self.position = position
#
#     def submit(self, broker=None):
#         '''Marks an order as submitted and stores the broker to which it was
#         submitted'''
#         self.status = Order.Submitted
#         self.broker = broker
#         self.plen = len(self.data)
#
#     def accept(self, broker=None):
#         '''Marks an order as accepted'''
#         self.status = Order.Accepted
#         self.broker = broker
#
#     def brokerstatus(self):
#         '''Tries to retrieve the status from the broker in which the order is.
#
#         Defaults to last known status if no broker is associated'''
#         if self.broker:
#             return self.broker.orderstatus(self)
#
#         return self.status
#
#     def reject(self, broker=None):
#         '''Marks an order as rejected'''
#         if self.status == Order.Rejected:
#             return False
#
#         self.status = Order.Rejected
#         self.executed.dt = self.data.datetime[0]
#         self.broker = broker
#         return True
#
#     def cancel(self):
#         '''Marks an order as cancelled'''
#         self.status = Order.Canceled
#         self.executed.dt = self.data.datetime[0]
#
#     def margin(self):
#         '''Marks an order as having met a margin call'''
#         self.status = Order.Margin
#         self.executed.dt = self.data.datetime[0]
#
#     def completed(self):
#         '''Marks an order as completely filled'''
#         self.status = self.Completed
#
#     def partial(self):
#         '''Marks an order as partially filled'''
#         self.status = self.Partial
#
#     def execute(self, dt, size, price,
#                 closed, closedvalue, closedcomm,
#                 opened, openedvalue, openedcomm,
#                 margin, pnl,
#                 psize, pprice):
#
#         '''Receives data execution input and stores it'''
#         if not size:
#             return
#
#         self.executed.add(dt, size, price,
#                           closed, closedvalue, closedcomm,
#                           opened, openedvalue, openedcomm,
#                           pnl, psize, pprice)
#
#         self.executed.margin = margin
#
#     def expire(self):
#         '''Marks an order as expired. Returns True if it worked'''
#         self.status = self.Expired
#         return True
#
#     def trailadjust(self, price):
#         pass  # generic interface
#
#
# class Order(OrderBase):
#     '''
#     Class which holds creation/execution data and type of oder.
#
#     The order may have the following status:
#
#       - Submitted: sent to the broker and awaiting confirmation
#       - Accepted: accepted by the broker
#       - Partial: partially executed
#       - Completed: fully exexcuted
#       - Canceled/Cancelled: canceled by the user
#       - Expired: expired
#       - Margin: not enough cash to execute the order.
#       - Rejected: Rejected by the broker
#
#         This can happen during order submission (and therefore the order will
#         not reach the Accepted status) or before execution with each new bar
#         price because cash has been drawn by other sources (future-like
#         instruments may have reduced the cash or orders orders may have been
#         executed)
#
#     Member Attributes:
#
#       - ref: unique order identifier
#       - created: OrderData holding creation data
#       - executed: OrderData holding execution data
#
#       - info: custom information passed over method :func:`addinfo`. It is kept
#         in the form of an OrderedDict which has been subclassed, so that keys
#         can also be specified using '.' notation
#
#     User Methods:
#
#       - isbuy(): returns bool indicating if the order buys
#       - issell(): returns bool indicating if the order sells
#       - alive(): returns bool if order is in status Partial or Accepted
#     '''
#
#     def execute(self, dt, size, price,
#                 closed, closedvalue, closedcomm,
#                 opened, openedvalue, openedcomm,
#                 margin, pnl,
#                 psize, pprice):
#
#         super(Order, self).execute(dt, size, price,
#                                    closed, closedvalue, closedcomm,
#                                    opened, openedvalue, openedcomm,
#                                    margin, pnl, psize, pprice)
#
#         if self.executed.remsize:
#             self.status = Order.Partial
#         else:
#             self.status = Order.Completed
#
#         # self.comminfo = None
#
#     def expire(self):
#         if self.exectype == Order.Market:
#             return False  # will be executed yes or yes
#
#         if self.valid and self.data.datetime[0] > self.valid:
#             self.status = Order.Expired
#             self.executed.dt = self.data.datetime[0]
#             return True
#
#         return False
#
#     def trailadjust(self, price):
#         if self.trailamount:
#             pamount = self.trailamount
#         elif self.trailpercent:
#             pamount = price * self.trailpercent
#         else:
#             pamount = 0.0
#
#         # Stop sell is below (-), stop buy is above, move only if needed
#         if self.isbuy():
#             price += pamount
#             if price < self.created.price:
#                 self.created.price = price
#                 if self.exectype == Order.StopTrailLimit:
#                     self.created.pricelimit = price - self._limitoffset
#         else:
#             price -= pamount
#             if price > self.created.price:
#                 self.created.price = price
#                 if self.exectype == Order.StopTrailLimit:
#                     # limitoffset is negative when pricelimit was greater
#                     # the - allows increasing the price limit if stop increases
#                     self.created.pricelimit = price - self._limitoffset
#
#
# class BuyOrder(Order):
#     ordtype = Order.Buy
#
#
# class StopBuyOrder(BuyOrder):
#     pass
#
#
# class StopLimitBuyOrder(BuyOrder):
#     pass
#
#
# class SellOrder(Order):
#     ordtype = Order.Sell
#
#
# class StopSellOrder(SellOrder):
#     pass
#
#
# class StopLimitSellOrder(SellOrder):
#     pass

# class TickerOrder(Order):
#     # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
#     # Order objects and we keep them all in memory, so it's worthwhile trying
#     # to cut down on the memory footprint of this object.
#     """
#         Parameters
#         ----------
#         asset : AssetEvent
#             The asset that this order is for.
#         amount : int
#             The amount of shares to order. If ``amount`` is positive, this is
#             the number of shares to buy or cover. If ``amount`` is negative,
#             this is the number of shares to sell or short.
#         dt : str, optional
#             The date created order.
#
#         市价单 --- 针对与卖出 --- 被动算法 ，基于时刻去卖出，这样避免被检测到 --- 将大订单拆分多个小订单然后基于时点去按照市价卖出
#
#     """
#     __slot__ = ['asset','_created_dt','capital']
#
#     def __init__(self,asset,ticker,capital):
#         self.asset = asset
#         self._created_dt = ticker
#         self.order_capital = capital
#         self.direction = math.copysign(1,capital)
#         self.filled = 0.0
#         self.broker_order_id = self.make_id()
#         self.order_type = StyleType.BOC
#
#     def check_trigger(self,dts):
#         if dts >= self._created_dt:
#             return True
#         return False
# def fulfill(self, data, iterator):
#     # 设定价格限制 , iterator里面的对象为第一个为price
#     bottom = data.pre['close'] * (1 - self._style.get_stop_price)
#     upper = data.pre['close'] * (1 + self._style.get_limit_price)
#     # 过滤
#     _iter = [item for item in iterator if bottom < item[0] < upper]
#     return _iter

# class MarketImpact(SlippageModel):
#     """
#         基于成交量进行对市场的影响进行测算
#     """
#     def __init__(self,func = np.exp):
#         self.adjust_func = func
#
#     def calculate_slippage_factor(self,target,volume):
#         psi = target / volume.mean()
#         factor = self.adjust_func(psi)
#         return factor
# from types import MappingProxyType as mappingproxy
# 返回一个动态映射视图

# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
# def load_portfolio_risk_factors(filepath_prefix=None, start=None, end=None):
#     """
#     Load risk factors Mkt-Rf, SMB, HML, Rf, and UMD.
#     Data is stored in HDF5 file. If the data is more than 2
#     days old, redownload from Dartmouth.
#     Returns
#     -------
#     five_factors : pd.DataFrame
#         Risk factors timeseries.
#     """
#
#     if start is None:
#         start = '1/1/1970'
#     if end is None:
#         end = _1_bday_ago()
#
#     start = get_utc_timestamp(start)
#     end = get_utc_timestamp(end)
#
#     if filepath_prefix is None:
#         filepath = data_path('factors.csv')
#     else:
#         filepath = filepath_prefix
#
#     five_factors = get_returns_cached(filepath, get_fama_french, end)
#
#     return five_factors.loc[start:end]
#
#
# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
# def get_treasury_yield(start=None, end=None, period='3MO'):
#     """
#     Load treasury yields from FRED.
#
#     Parameters
#     ----------
#     start : date, optional
#         Earliest date to fetch data for.
#         Defaults to earliest date available.
#     end : date, optional
#         Latest date to fetch data for.
#         Defaults to latest date available.
#     period : {'1MO', '3MO', '6MO', 1', '5', '10'}, optional
#         Which maturity to use.
#     Returns
#     -------
#     pd.Series
#         Annual treasury yield for every day.
#     """
#
#     if start is None:
#         start = '1/1/1970'
#     if end is None:
#         end = _1_bday_ago()
#
#     treasury = web.DataReader("DGS3{}".format(period), "fred",
#                               start, end)
#
#     treasury = treasury.ffill()
#
#     return treasury
#
#
# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
# def get_symbol_returns_from_yahoo(symbol, start=None, end=None):
#     """
#     Wrapper for pandas.io.data.get_data_yahoo().
#     Retrieves prices for symbol from yahoo and computes returns
#     based on adjusted closing prices.
#
#     Parameters
#     ----------
#     symbol : str
#         Symbol name to load, e.g. 'SPY'
#     start : pandas.Timestamp compatible, optional
#         Start date of time period to retrieve
#     end : pandas.Timestamp compatible, optional
#         End date of time period to retrieve
#
#     Returns
#     -------
#     pandas.DataFrame
#         Returns of symbol in requested period.
#     """
#
#     try:
#         px = web.get_data_yahoo(symbol, start=start, end=end)
#         px['date'] = pd.to_datetime(px['date'])
#         px.set_index('date', drop=False, inplace=True)
#         rets = px[['adjclose']].pct_change().dropna()
#     except Exception as e:
#         warnings.warn(
#             'Yahoo Finance read failed: {}, falling back to Google'.format(e),
#             UserWarning)
#         px = web.get_data_google(symbol, start=start, end=end)
#         rets = px[['Close']].pct_change().dropna()
#
#     rets.index = rets.index.tz_localize("UTC")
#     rets.columns = [symbol]
#     return rets
#
#
# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
# def default_returns_func(symbol, start=None, end=None):
#     """
#     Gets returns for a symbol.
#     Queries Yahoo Finance. Attempts to cache SPY.
#
#     Parameters
#     ----------
#     symbol : str
#         Ticker symbol, e.g. APPL.
#     start : date, optional
#         Earliest date to fetch data for.
#         Defaults to earliest date available.
#     end : date, optional
#         Latest date to fetch data for.
#         Defaults to latest date available.
#
#     Returns
#     -------
#     pd.Series
#         Daily returns for the symbol.
#          - See full explanation in tears.create_full_tear_sheet (returns).
#     """
#
#     if start is None:
#         start = '1/1/1970'
#     if end is None:
#         end = _1_bday_ago()
#
#     start = get_utc_timestamp(start)
#     end = get_utc_timestamp(end)
#
#     if symbol == 'SPY':
#         filepath = data_path('spy.csv')
#         rets = get_returns_cached(filepath,
#                                   get_symbol_returns_from_yahoo,
#                                   end,
#                                   symbol='SPY',
#                                   start='1/1/1970',
#                                   end=datetime.now())
#         rets = rets[start:end]
#     else:
#         rets = get_symbol_returns_from_yahoo(symbol, start=start, end=end)
#
#     return rets[symbol]
# @deprecated(msg=DATAREADER_DEPRECATION_WARNING)
# def get_fama_french():
#     """
#     Retrieve Fama-French factors via pandas-datareader
#     Returns
#     -------
#     pandas.DataFrame
#         Percent change of Fama-French factors
#     """
#
#     start = '1/1/1970'
#     research_factors = web.DataReader('F-F_Research_Data_Factors_daily',
#                                       'famafrench', start=start)[0]
#     momentum_factor = web.DataReader('F-F_Momentum_Factor_daily',
#                                      'famafrench', start=start)[0]
#     five_factors = research_factors.join(momentum_factor).dropna()
#     five_factors /= 100.
#     five_factors.index = five_factors.index.tz_localize('utc')
#
#     five_factors.columns = five_factors.columns.str.strip()
#
#     return five_factors
# try:
#     # fast versions
#     import bottleneck as bn
#
#     def _wrap_function(f):
#         @wraps(f)
#         def wrapped(*args, **kwargs):
#             out = kwargs.pop('out', None)
#             data = f(*args, **kwargs)
#             if out is None:
#                 out = data
#             else:
#                 out[()] = data
#
#             return out
#
#         return wrapped
#
#     nanmean = _wrap_function(bn.nanmean)
#     nanstd = _wrap_function(bn.nanstd)
#     nansum = _wrap_function(bn.nansum)
#     nanmax = _wrap_function(bn.nanmax)
#     nanmin = _wrap_function(bn.nanmin)
#     nanargmax = _wrap_function(bn.nanargmax)
#     nanargmin = _wrap_function(bn.nanargmin)
# except ImportError:
#     # slower numpy
#     nanmean = np.nanmean
#     nanstd = np.nanstd
#     nansum = np.nansum
#     nanmax = np.nanmax
#     nanmin = np.nanmin
#     nanargmax = np.nanargmax
#     nanargmin = np.nanargmin
#
#
# try:
#     from pandas_datareader import data as web
# except ImportError:
#     msg = ("Unable to import pandas_datareader. Suppressing import error and "
#            "continuing. All data reading functionality will raise errors; but "
#            "has been deprecated and will be removed in a later version.")
#     warnings.warn(msg)
# from .deprecate import deprecated
#
# DATAREADER_DEPRECATION_WARNING = \
#         ("Yahoo and Google Finance have suffered large API breaks with no "
#          "stable replacement. As a result, any data reading functionality "
#          "in empyrical has been deprecated and will be removed in a future "
#          "version. See README.md for more details: "
#          "\n\n"
#          "\thttps://github.com/quantopian/pyfolio/blob/master/README.md")

# from __future__ import division
# from multipledispatch import dispatch
# from .compat import PY2
# import numpy as np
#
# if PY2:
#     int_t = (int, long, np.int64)
# else:
#     int_t = (int, np.int64)
# from __future__ import division

# @property
# def default(self):
#     return self._default()
#
# def _default(self,dt):
#     """
#         a. 剔除停盘
#         b. 剔除上市不足一个月的 --- 次新股波动性太大
#         c. 剔除进入退市整理期的30个交易日
#     """
#     active_assets = self.asset_finder.was_active(dt)
#     sdate = self.trading_calendar._roll_forward(dt,StableEPeriod)
#     edate = self.trading_calendar._roll_forward(dt, -EnsurePeriod)
#     stable_alive = self.asset_finder.lifetime([sdate,edate])
#     default_assets = set(active_assets) & set(stable_alive)
#     return default_assets

# @staticmethod
# def _execution_open_and_close(_calendar, session):
#     open_, close = _calendar.open_and_close_for_session(session)
#     execution_open = _calendar.execution_time_from_open(open_)
#     execution_close = _calendar.execution_time_from_close(close)
    # cal = self._trading_calendar
    # self._market_open, self._market_close = self._execution_open_and_close(
    #     cal,
    #     session_label,
    # )
    # if self.emission_rate == 'daily':
    #     # this method is called for both minutely and daily emissions, but
    #     # this chunk of code here only applies for daily emissions. (since
    #     # it's done every minute, elsewhere, for minutely emission).
    #     self.sync_last_sale_prices(dt, data_portal)
    # session_ix = self._session_count
    # # increment the day counter before we move markers forward.
    # self._session_count += 1
    # self._total_session_count = len(sessions)
    # self._session_count = 0
    # self.emission_rate = emission_rate
    # emission_rate = 'daily',
    # import logging
    # logging.info(
    #     'Simulated {} trading days\n'
    #     'first open: {}\n'
    #     'last close: {}',
    #     self._session_count,
    #     self._trading_calendar.session_open(self._first_session),
    #     self._trading_calendar.session_close(self._last_session),
    # )

# def simple_returns(prices):
#     """
#     Compute simple returns from a timeseries of prices.
#
#     Parameters
#     ----------
#     prices : pd.Series, pd.DataFrame or np.ndarray
#         Prices of asset in wide-format, with asset as columns,
#         and indexed by datetimes.
#
#     Returns
#     -------
#     returns : array-like
#         Returns of asset in wide-format, with asset as columns,
#         and index coerced to be tz-aware.
#     """
#     if isinstance(prices, (pd.DataFrame, pd.Series)):
#         out = prices.pct_change().iloc[1:]
#     else:
#         # Assume np.ndarray np.diff ( back -before)
#         out = np.diff(prices, axis=0)
#         np.divide(out, prices[:-1], out=out)
#     return out

# Copied from Position and renamed.  This is used to handle cases where a user
# does something like `context.portfolio.positions[100]` instead of
# `context.portfolio.positions[sid(100)]`.
# class _DeprecatedSidLookupPosition(object):
#     def __init__(self, sid):
#         self.sid = sid
#         self.amount = 0
#         self.cost_basis = 0.0  # per share
#         self.last_sale_price = 0.0
#         self.last_sale_date = None
#
#     def __repr__(self):
#         return "_DeprecatedSidLookupPosition({0})".format(self.__dict__)
#
#     # If you are adding new attributes, don't update this set. This method
#     # is deprecated to normal attribute access so we don't want to encourage
#     # new usages.
#     __getitem__ = _deprecated_getitem_method(
#         'position', {
#             'sid',
#             'amount',
#             'cost_basis',
#             'last_sale_price',
#             'last_sale_date',
#         },
#     )
#
#
# class Positions(dict):
#     """A dict-like object containing the algorithm's current positions.
#     """
#
#     def __missing__(self, key):
#         if isinstance(key, Asset):
#             return Position(InnerPosition(key))
#         elif isinstance(key, int):
#             warnings.warn("Referencing positions by integer is deprecated."
#                  " Use an asset instead.")
#         else:
#             warnings.warn("Position lookup expected a value of type Asset but got {0}"
#                  " instead.".format(type(key).__name__))
#
#         return _DeprecatedSidLookupPosition(key)

# class DailyFieldLedger(object):
#
#     def __init__(self,ledger_field,packet_field = None):
#         self._get_ledger_field = op.attrgetter(ledger_field)
#         if packet_field is None:
#             self._packet_field = ledger_field.rsplit('.',1)[-1]
#         else:
#             self._packet_field = packet_field
#
#     def end_of_session(self,
#                        packet,
#                        ledger,
#                        session_ix):
#         field = self._packet_field
#         packet['daily_perf'][field] = \
#             self._get_ledger_field(ledger)
#
# class StartOfPeriodLedgerField(object):
#     """Keep track of the value of a ledger field at the start of the period.
#
#     Parameters
#     ----------
#     ledger_field : str
#         The ledger field to read.
#     packet_field : str, optional
#         The name of the field to populate in the packet. If not provided,
#         ``ledger_field`` will be used.
#     """
#     def __init__(self, ledger_field, packet_field=None):
#         self._get_ledger_field = op.attrgetter(ledger_field)
#         if packet_field is None:
#             self._packet_field = ledger_field.rsplit('.', 1)[-1]
#         else:
#             self._packet_field = packet_field
#
#     def start_of_simulation(self,
#                             ledger,
#                             benchmark,
#                             sessions):
#         self._start_of_simulation = self._get_ledger_field(ledger)
#
#     def start_of_session(self, ledger):
#         self._previous_day = self._get_ledger_field(ledger)
#
#     def _end_of_period(self, sub_field, packet,ledger):
#         packet_field = self._packet_field
#         # start_of_simulation 不变的
#         packet['cumulative_perf'][packet_field] = self._start_of_simulation
#         packet[sub_field][packet_field] = self._previous_day
#
#     def end_of_session(self,
#                        packet,
#                        ledger,
#                        session_ix):
#         self._end_of_period('daily_perf', packet,ledger)

# We don't have a datetime for the current snapshot until we
# receive a message.
# This object is the way that user algorithms interact with OHLCV data,
# fetcher data, and some API methods like `data.can_trade`.
# self.current_data = self._create_bar_data()

# #获取日数据，封装为一个API(fetch process flush other api)
# def _create_bar_data(self):
#     return BarData(
#         data_portal=self.data_portal,
#         data_frequency=self.sim_params.data_frequency,
#         trading_calendar=self.algo.trading_calendar,
#         restrictions=self.restrictions,
#     )
# if isinstance(asset, Asset):
#     return False
# return pd.Series(index=pd.Index(asset), data=False)
# if isinstance(asset, Asset):
#     return asset in self._restricted_set
# return pd.Series(
#     index=pd.Index(asset),
#     # list 内置的__contains__ 方法
#     # data=vectorized_is_element(asset, self._restricted_set)
#     data = np.vectorize(self._restricted_set.__contains__,otypes = [bool])(asset)
# )
# def is_restricted(self, asset, dt):
#     if isinstance(asset, Asset):
#         return any(
#             r.is_restricted(asset, dt) for r in self.sub_restrictions
#         )
#
#     return reduce(
#         operator.or_,
#         (r.is_restricted(asset, dt) for r in self.sub_restrictions)
#     )
# class LongOnly(TradingControl):
#
#     def __init__(self,on_error):
#         super(LongOnly,self).__init__(on_error)
#
#     def validate(self,
#                  txn,
#                  portfolio,
#                  algo_datetime,
#                  algo_current_data):
#
#         asset = txn.asset
#         amount = txn.amount
#         if portfolio.positons[asset].amount + amount  < 0 :
#             self.handle_violation(asset,amount,algo_datetime)
#
#
# class RestrictedListOrder(TradingControl):
#     """ represents a restricted list of asset that canont be ordered by the algorithm"""
#     def __init__(self,on_error,restrictions):
#         super(RestrictedListOrder,self).__init__(on_error)
#         self.restrictions = restrictions
#
#     def validate(self,
#                  txn,
#                  portfolio,
#                  algo_datetime,
#                  algo_current_data):
#
#         asset = txn.asset
#         amount = txn.amount
#         if self.restrictions.is_restricted(asset,algo_datetime):
#             self.handle_violation(asset,amount,algo_datetime)
#

# namespace = dict()
# with open('/Users/python/Library/Mobile Documents/com~apple~CloudDocs/ArkQuant/test/test_driver.py','r') as f:
#     exec(f.read(),namespace)
#
# print(namespace.keys())
# test = namespace['UnionEngine']
# print(test)
# # ins = test()
# # print(ins)
# # print(namespace['__builtins__'])
# # print(namespace['signature'])
#
# import glob
# res = glob.glob('/Users/python/Library/Mobile Documents/com~apple~CloudDocs/ArkQuant/pipe/strategy/*.py')
# print(list(res))
#
# print(__file__)
#
#
# # exec eval compile将字符串转化为可执行代码 , exec compile source into code or AST object ,if filename is None ,'<string>' is used
# # code = compile(self.algoscript, algo_filename, 'exec')
# # exec_(code, self.namespace)
# #
# # # dict get参数可以为方法或者默认参数
# # self._initialize = self.namespace.get('initialize', noop)
# # self._handle_data = self.namespace.get('handle_data', noop)
# # self._before_trading_start = self.namespace.get(
# #     'before_trading_start',
# # )

# class BarData:
#     """
#     Provides methods for accessing minutely and daily price/volume data from
#     Algorithm API functions.
#
#     Also provides utility methods to determine if an asset is alive, and if it
#     has recent trade data.
#
#     An instance of this object is passed as ``data`` to
#     :func:`~zipline.api.handle_data` and
#     :func:`~zipline.api.before_trading_start`.
#
#     Parameters
#     ----------
#     data_portal : DataPortal
#         Provider for bar pricing data.
#     data_frequency : {'minute', 'daily'}
#         The frequency of the bar data; i.e. whether the data is
#         daily or minute bars
#     restrictions : zipline.finance.asset_restrictions.Restrictions
#         Object that combines and returns restricted list information from
#         multiple sources
#     """
#
#     def __init__(self, data_portal, data_frequency,
#                  trading_calendar, restrictions):
#         self.data_portal = data_portal
#         self.data_frequency = data_frequency
#         self._trading_calendar = trading_calendar
#         self._is_restricted = restrictions.is_restricted
#
#     def get_current_ticker(self,asset,fields):
#         """
#         Returns the "current" value of the given fields for the given asset
#         at the current ArkQuant time.
#         :param asset: asset_type
#         :param fields: OHLCTV
#         :return: dict asset -> ticker
#         intended to return current ticker
#         """
#         cur = {}
#         for asset in asset:
#             ticker = self.data_portal.get_current(asset)
#             cur[asset] = ticker.loc[:,fields]
#         return cur
#
#     def history(self, asset, end_dt,bar_count, fields,frequency):
#         """
#         Returns a trailing window of length ``bar_count`` containing data for
#         the given asset, fields, and frequency.
#
#         Returned data is adjusted for splits, dividends, and mergers as of the
#         current ArkQuant time.
#
#         The semantics for missing data are identical to the ones described in
#         the notes for :meth:`current`.
#
#         Parameters
#         ----------
#         asset: zipline.asset.Asset or iterable of zipline.asset.Asset
#             The asset(s) for which data is requested.
#         fields: string or iterable of string.
#             Requested data field(s). Valid field names are: "price",
#             "last_traded", "open", "high", "low", "close", and "volume".
#         bar_count: int
#             Number of data observations requested.
#         frequency: str
#             String indicating whether to load daily or minutely data
#             observations. Pass '1m' for minutely data, '1d' for daily data.
#
#         Returns
#         -------
#         history : pd.Series or pd.DataFrame or pd.Panel
#             See notes below.
#
#         Notes
#         ------
#         returned panel has:
#         items: fields
#         major axis: dt
#         minor axis: asset
#         return pd.Panel(df_dict)
#         """
#         sliding_window = self.data_portal.get_history_window(asset,
#                                                              end_dt,
#                                                              bar_count,
#                                                              fields,
#                                                              frequency)
#         return sliding_window
#
#     def window_data(self,asset,end_dt,bar_count,fields,frequency):
#         window_array = self.data_portal.get_window_data(asset,
#                                                         end_dt,
#                                                         bar_count,
#                                                         fields,
#                                                         frequency)
#         return window_array
# ALLOWED_READ_CSV_KWARGS = {
#     'sep',
#     'dialect',
#     'doublequote',
#     'escapechar',
#     'quotechar',
#     'quoting',
#     'skipinitialspace',
#     'lineterminator',
#     'header',
#     'index_col',
#     'names',
#     'prefix',
#     'skiprows',
#     'skipfooter',
#     'skip_footer',
#     'na_values',
#     'true_values',
#     'false_values',
#     'delimiter',
#     'converters',
#     'dtype',
#     'delim_whitespace',
#     'as_recarray',
#     'na_filter',
#     'compact_ints',
#     'use_unsigned',
#     'buffer_lines',
#     'warn_bad_lines',
#     'error_bad_lines',
#     'keep_default_na',
#     'thousands',
#     'comment',
#     'decimal',
#     'keep_date_col',
#     'nrows',
#     'chunksize',
#     'encoding',
#     'usecols'
# }
# class MassiveSessionReader(BarReader):
#
#     def __init__(self,
#                  metadata,
#                  engine,
#                  trading_calenar):
#         self.metadata = metadata
#         self.engine = engine
#         self._trading_calenar = trading_calenar
#
#     def get_value(self, asset, dt):
#         table = self.metadata['massive']
#         sql = select([cast(table.c.bid_price, Numeric(10,2)),
#                       cast(table.c.discount, Numeric(10,5)),
#                       cast(table.c.bid_volume, Integer),
#                       table.c.buyer,
#                       table.c.seller,
#                       table.c.cleltszb]).where(and_(table.c.trade_dt == dt,table.c.sid == asset.sid))
#         raw = self.engine.execute(sql).fetchall()
#         share_massive = pd.DataFrame(raw,columns = ['bid_price','discount','bid_volume','buyer','seller','cleltszb'])
#         return share_massive
#
#     def load_raw_arrays(self, edate, window,asset):
#         sdate = self._window_size_to_dt(edate,window)
#         sids = [asset.sid for asset in asset]
#         #获取数据
#         table = self.metadata['massive']
#         sql = select([table.c.trade_dt,
#                       table.c.sid,
#                       cast(table.c.bid_price, Numeric(10,2)),
#                       cast(table.c.discount, Numeric(10,5)),
#                       cast(table.c.bid_volume, Integer),
#                       table.c.buyer,
#                       table.c.seller,
#                       table.c.cleltszb]).where(table.c.trade_dt.between(sdate,edate))
#         raw = self.engine.execute(sql).fetchall()
#         df = pd.DataFrame(raw,columns = ['trade_dt','code','bid_price','discount','bid_volume','buyer','seller','cleltszb'])
#         df.set_index('code',inplace= True)
#         massive = df.loc[sids]
#         return massive
#
#
# class ReleaseSessionReader(BarReader):
#
#     def __init__(self,
#                  metadata,
#                  engine,
#                  trading_calendar):
#         self.metadata = metadata
#         self.engine = engine
#         self._trading_calendar = trading_calendar
#
#     def get_value(self, asset,dt):
#         table = self.metadata['release']
#         sql = select([cast(table.c.release_type, Numeric(10, 2)),
#                       cast(table.c.cjeltszb, Numeric(10, 5)), ]).\
#             where(and_(table.c.release_date == dt,table.c.sid == asset.sid))
#         raw = self.engine.execute(sql).fetchall()
#         release = pd.DataFrame(raw, columns=['release_type', 'cjeltszb'])
#         return release
#
#     def load_raw_arrays(self, edate, window,asset):
#         sdate = self._window_size_to_dt(edate,window)
#         sids = [asset.sid for asset in asset]
#         table = self.metadata['release']
#         sql = select([table.c.sid,
#                       table.c.release_date,
#                       cast(table.c.release_type, Numeric(10, 2)),
#                       cast(table.c.cjeltszb, Numeric(10, 5)), ]).where \
#             (table.c.release_date.between(sdate, edate))
#         raw = self.engine.execute(sql).fetchall()
#         df = pd.DataFrame(raw, columns=['code', 'release_date', 'release_type', 'cjeltszb'])
#         df.set_index('code',inplace= True)
#         releases = df.loc[sids]
#         return releases
#
#
# class ShareholderSessionReader(BarReader):
#
#     def __init__(self,
#                  metadata,
#                  engine,
#                  trading_calendar):
#         self.metadata = metadata
#         self.engine = engine
#         self._trading_calendar = trading_calendar
#
#     def get_value(self, asset,dt):
#         """股东持仓变动"""
#         table = self.metadata['shareholder']
#         sql = select([table.c.股东,
#                       table.c.方式,
#                       cast(table.c.变动股本, Numeric(10,2)),
#                       cast(table.c.总持仓, Integer),
#                       cast(table.c.占总股本比例, Numeric(10, 5)),
#                       cast(table.c.总流通股, Integer),
#                       cast(table.c.占总流通比例, Numeric(10, 5))]).where(and_(table.c.公告日 == dt,table.c.sid == asset.sid))
#         raw = self.engine.execute(sql).fetchall()
#         share_tracker = pd.DataFrame(raw,columns = ['股东','方式','变动股本','总持仓','占总股本比例','总流通股','占总流通比例'])
#         return share_tracker
#
#     def load_raw_arrays(self, edate, window,asset):
#         sdate = self._window_size_to_dt(edate,window)
#         sids = [asset.sid for asset in asset]
#         """股东持仓变动"""
#         table = self.metadata['shareholder']
#         sql = select([table.c.sid,
#                       table.c.公告日,
#                       table.c.股东,
#                       table.c.方式,
#                       cast(table.c.变动股本, Numeric(10,2)),
#                       cast(table.c.总持仓, Integer),
#                       cast(table.c.占总股本比例, Numeric(10, 5)),
#                       cast(table.c.总流通股, Integer),
#                       cast(table.c.占总流通比例, Numeric(10, 5))]).where(
#                     table.c.公告日.between(sdate,edate))
#         raw = self.engine.execute(sql).fetchall()
#         df = pd.DataFrame(raw,columns = ['code','公告日','股东','方式','变动股本','总持仓','占总股本比例','总流通股','占总流通比例'])
#         df.set_index('code',inplace= True)
#         trackers = df.loc[sids]
#         return trackers
#
#
# class GrossSessionReader(BarReader):
#
#     GdpPath = 'http://data.eastmoney.com/cjsj/grossdomesticproduct.aspx?p=%d'
#
#     def __init__(self,
#                  trading_calendar,
#                  url = None):
#         self._trading_calendar = trading_calendar
#         self._url = url if url else self.GdpPath
#
#     def get_value(self, asset, dt, field):
#         print('get_values is deprescated by gpd ,use load_raw_arrays method')
#
#     def load_raw_arrays(self,edate,window):
#         sdate = self._window_size_to_dt(edate,window)
#         """获取GDP数据"""
#         page = 1
#         gross_value = pd.DataFrame()
#         while True:
#             req_url = self._url%page
#             obj = _parse_url(req_url)
#             raw = obj.findAll('div', {'class': 'Content'})
#             text = [t.get_text() for t in raw[1].findAll('td')]
#             text = [item.strip() for item in text]
#             data = zip(text[::9], text[1::9])
#             data = pd.DataFrame(data, columns=['季度', '总值'])
#             gross_value = gross_value.append(data)
#             if len(gross_value) != len(gross_value.drop_duplicates(ignore_index=True)):
#                 gross_value.drop_duplicates(inplace=True, ignore_index=True)
#                 return gross_value
#             page = page + 1
#         #截取
#         start_idx = gross_value.index(sdate)
#         end_idx = gross_value.index(edate)
#         return gross_value.iloc[start_idx:end_idx +1,:]
#
#
# class MarginSessionReader(BarReader):
#
#     MarginPath = 'http://api.dataide.eastmoney.com/data/get_rzrq_lshj?' \
#                  'orderby=dim_date&order=desc&pageindex=%d&pagesize=50'
#
#     def __init__(self,
#                  trading_calendar,
#                  _url):
#         self._trading_calendar = trading_calendar
#         self._url = _url if _url else self.MarginPath
#
#     def get_value(self, asset, dt, field):
#         raise NotImplementedError('get_values is deprescated ,use load_raw_arrays method')
#
#     def load_raw_arrays(self, edate, window):
#         sdate = self._window_size_to_dt(edate,window)
#         """获取市场全量融资融券"""
#         page = 1
#         margin = pd.DataFrame()
#         while True:
#             req_url = self._url% page
#             raw = _parse_url(req_url, bs=False)
#             raw = json.loads(raw)
#             raw = [
#                 [item['dim_date'], item['rzye'], item['rqye'], item['rzrqye'], item['rzrqyecz'], item['new'],
#                  item['zdf']]
#                 for item in raw['data']]
#             data = pd.DataFrame(raw, columns=['trade_dt', 'rzye', 'rqye', 'rzrqze', 'rzrqce', 'hs300', 'pct'])
#             data.loc[:, 'trade_dt'] = [datetime.datetime.fromtimestamp(dt / 1000) for dt in data['trade_dt']]
#             data.loc[:, 'trade_dt'] = [datetime.datetime.strftime(t, '%Y-%m-%d') for t in data['trade_dt']]
#             if len(data) == 0:
#                 break
#             margin = margin.append(data)
#             page = page + 1
#         margin.set_index('trade_dt', inplace=True)
#         #
#         start_idx = margin.index(sdate)
#         end_idx = margin.index(edate)
#         return margin.iloc[start_idx:end_idx +1,:]
#

# class ExecutionStyle(ABC):
#     """
#         base class for order execution style
#     """
#     @abstractmethod
#     def get_limit_price(self, is_buy):
#         raise NotImplementedError
#
#     @abstractmethod
#     def get_stop_price(self, is_buy):
#         raise NotImplementedError
#
#
# class MarketOrder(ExecutionStyle):
#
#     def __init__(self, exchange=None):
#         self._exchange = exchange
#
#     def get_limit_price(self, _is_buy):
#         return None
#
#     def get_stop_price(self, _is_buy):
#         return None
#
#
# class LimitOrder(ExecutionStyle):
#     """
#         limit price --- maximum price for buys or minimum price for sells
#     """
#
#     def __init__(self, limit_price, asset=None, exchange=None):
#         check_stoplimit_prices(limit_price, 'limit')
#
#         self.limit_price = limit_price
#         self._exchange = exchange
#         self.asset = asset
#
#     def get_limit_price(self, is_buy):
#         return asymmetric_round_price(self.limit_price, is_buy,
#                                       tick_size=(0.01 if self.asset is None else self.asset.tick_size))
#
#     def get_stop_price(self, _is_buy):
#         return None
#
#
# class StopOrder(ExecutionStyle):
#     """
#         stop_price ---- for sells the order will be placed if market price falls below this value .
#         for buys ,the order will be placed if market price rise above this value.
#     """
#
#     def __init__(self, stop_price, asset=None, exchange=None):
#         check_stoplimit_prices(stop_price, 'stop')
#
#         self.stop_price = stop_price
#         self._exchange = exchange
#         self.asset = asset
#
#     def get_limit_price(self, is_buy):
#         return None
#
#     def get_stop_price(self, is_buy):
#         return asymmetric_round_price(
#             self.stop_price,
#             not is_buy,
#             tick_size=(0.01 if self.asset is None else self.asset.tick_size)
#         )
#
# class StopLimitOrder(ExecutionStyle):
#     """
#         price reach a threahold
#     """
#
#     def __init__(self, limit_price, stop_price, asset=None, exchange=None):
#         check_stoplimit_prices(limit_price, 'limit')
#         check_stoplimit_prices(stop_price, 'stop')
#
#         self.limit_price = limit_price
#         self.stop_price = stop_price
#         self._exchange = exchange
#         self.asset = asset
#
#     def get_limit_price(self, is_buy):
#         return asymmetric_round_price(
#             self.limit_price,
#             is_buy,
#             tick_size=(0.01 if self.asset is None else self.asset.tick_size)
#         )
#
#     def get_stop_price(self, is_buy):
#         return asymmetric_round_price(
#             self.stop_price,
#             not is_buy,
#             tick_size=(0.01 if self.asset is None else self.asset.tick_size)
#         )
#
#
#
# def asymmetric_round_price(price, prefer_round_down, tick_size, diff=0.95):
#     """
#         for limit_price ,this means preferring to round down on buys and preferring to round up on sells.
#         for stop_price ,reverse
#     ---- narrow the sacle of limits and stop
#     :param price:
#     :param prefer_round_down:
#     :param tick_size:
#     :param diff:
#     :return:
#     """
#     # return 小数位数
#     precision = zp_math.number_of_decimal_places(tick_size)
#     multiplier = int(tick_size * (10 ** precision))
#     diff -= 0.5  # shift the difference down
#     diff *= (10 ** -precision)
#     # 保留tick_size
#     diff *= multiplier
#     # 保留系统精度
#     epsilon = sys.float_info * 10
#     diff = diff - epsilon
#
#     rounded = tick_size * consistent_round(
#         (price - (diff if prefer_round_down else -diff)) / tick_size
#     )
#     if zp_math.tolerant_equals(rounded, 0.0):
#         return 0.0
#     return rounded


# def order(self, asset, amount, style =None, order_id=None):
#     """Place an order.
#
#     Parameters
#     ----------
#     asset : zipline.asset.Asset
#         The asset that this order is for.
#     amount : int
#         The amount of shares to order. If ``amount`` is positive, this is
#         the number of shares to buy or cover. If ``amount`` is negative,
#         this is the number of shares to sell or short.
#     style : zipline.finance.execution.ExecutionStyle
#         The execution style for the order.
#     order_id : str, optional
#         The unique identifier for this order.
#
#     Returns
#     -------
#     order_id : str or None
#         The unique identifier for this order, or None if no order was
#         placed.
#
#     Notes
#     -----
#     amount > 0 :: Buy/Cover
#     amount < 0 :: Sell/Short
#     Market order:    order(asset, amount)
#     Limit order:     order(asset, amount, style=LimitOrder(limit_price))
#     Stop order:      order(asset, amount, style=StopOrder(stop_price))
#     StopLimit order: order(asset, amount, style=StopLimitOrder(limit_price,
#                            stop_price))
#     """
#     # something could be done with amount to further divide
#     # between buy by share count OR buy shares up to a dollar amount
#     # numeric == share count  AND  "$dollar.cents" == cost amount
#
#     if amount == 0:
#         # Don't bother placing orders for 0 shares.
#         return None
#     elif amount > self.max_shares:
#         # Arbitrary limit of 100 billion (US) shares will never be
#         # exceeded except by a buggy algorithm.
#         raise OverflowError("Can't order more than %d shares" %
#                             self.max_shares)
#
#     is_buy = (amount > 0)
#     order = Order(
#         dt=self.current_dt,
#         asset=asset,
#         amount=amount,
#         stop=style.get_stop_price(is_buy),
#         limit=style.get_limit_price(is_buy),
#         id=order_id
#     )
#
#     self.open_orders[order.asset].append(order)
#     self.orders[order.id] = order


# full_share_count = self.amount * float(ratio)
# new_cost_basics = round(self.cost_basis / float(ratio), 2)
# left_cash = (full_share_count - np.floor(full_share_count)) * new_cost_basics
# self.cost_basis = np.floor(new_cost_basics)
# self.amount = full_share_count
# return left_cash

# def update(self,txn):
#     if self.asset != txn.asset:
#         raise Exception('transaction must be the same with position asset')
#
#     if self.last_sale_dt is None or txn.dt > self.last_sale_dt:
#         self.last_sale_dt = txn.dt
#         self.last_sale_price = txn.price
#
#     total_shares = txn.amount + self.amount
#     if total_shares == 0:
#         # 用于统计transaction是否盈利
#         # self.cost_basis = 0.0
#         position_return = (txn.price - self.cost_basis)/self.cost_basis
#         self.cost_basis = position_return
#     elif total_shares < 0:
#         raise Exception('for present put action is not allowed')
#     else:
#         total_cost = txn.amout * txn.price + self.amount * self.cost_basis
#         new_cost_basis = total_cost / total_shares
#         self.cost_basis = new_cost_basis
#
#     self.amount = total_shares

# def update_position(self,
#                     asset,
#                     amount = None,
#                     last_sale_price = None,
#                     last_sale_date = None,
#                     cost_basis = None):
#     self._dirty_stats = True
#
#     try:
#         position = self.positions[asset]
#     except KeyError:
#         position = Position(asset)
#
#     if amount is not None:
#         position.amount = amount
#     if last_sale_price is not None :
#         position.last_sale_price = last_sale_price
#     if last_sale_date is not None :
#         position.last_sale_date = last_sale_date
#     if cost_basis is not None :
#         position.cost_basis = cost_basis
#
# # 执行
# def execute_transaction(self,txn):
#     self._dirty_stats = True
#
#     asset = txn.asset
#
#     # 新的股票仓位
#     if asset not in self.positions:
#         position = Position(asset)
#     else:
#         position = self.positions[asset]
#
#     position.update(txn)
#
#     if position.amount ==0 :
#         #统计策略的对应的收益率
#         dt = txn.dt
#         algorithm_ret = position.cost_basis
#         asset_origin = position.asset.reason
#         self.record_vars[asset_origin] = {str(dt):algorithm_ret}
#
#         del self.positions[asset]

# def handle_spilts(self,splits):
#     total_leftover_cash = 0
#
#     for asset,ratio in splits.items():
#         if asset in self.positions:
#             position = self.positions[asset]
#             leftover_cash = position.handle_split(asset,ratio)
#             total_leftover_cash += leftover_cash
#     return total_leftover_cash

#将分红或者配股的数据分类存储
# def earn_divdends(self,cash_divdends,stock_divdends):
#     """
#         given a list of divdends where ex_date all the next_trading
#         including divdend and stock_divdend
#     """
#     for cash_divdend in cash_divdends:
#         div_owned = self.positions[cash_divdend['paymen_asset']].earn_divdend(cash_divdend)
#         self._unpaid_divdend[cash_divdend.pay_date].apppend(div_owned)
#
#     for stock_divdend in stock_divdends:
#         div_owned_ = self.positions[stock_divdend['payment_asset']].earn_stock_divdend(stock_divdend)
#         self._unpaid_stock_divdends[stock_divdend.pay_date].append(div_owned_)

# 根据时间执行分红或者配股
# def pay_divdends(self,next_trading_day):
#     """
#         股权登记日，股权除息日（为股权登记日下一个交易日）
#         但是红股的到账时间不一致（制度是固定的）
#         根据上海证券交易规则，对投资者享受的红股和股息实行自动划拨到账。股权（息）登记日为R日，除权（息）基准日为R+1日，
#         投资者的红股在R+1日自动到账，并可进行交易，股息在R+2日自动到帐，
#         其中对于分红的时间存在差异
#
#         根据深圳证券交易所交易规则，投资者的红股在R+3日自动到账，并可进行交易，股息在R+5日自动到账，
#
#         持股超过1年：税负5%;持股1个月至1年：税负10%;持股1个月以内：税负20%新政实施后，上市公司会先按照5%的最低税率代缴红利税
#     """
#     net_cash_payment = 0.0
#
#     # cash divdend
#     try:
#         payments = self._unpaid_divdend[next_trading_day]
#         del self._unpaid_divdend[next_trading_day]
#     except KeyError:
#         payments = []
#
#     for payment in payments:
#         net_cash_payment += payment['cash_amount']
#
#     #stock divdend
#     try:
#         stock_payments = self._unpaid_stock_divdends[next_trading_day]
#     except KeyError:
#         stock_payments = []
#
#     for stock_payment in stock_payments:
#         payment_asset = stock_payment['payment_asset']
#         share_amount = stock_payment['share_amount']
#         if payment_asset in self.positions:
#             position = self.positions[payment_asset]
#         else:
#             position = self.positions[payment_asset] = Position(payment_asset)
#         position.amount  += share_amount
#     return net_cash_payment

# def calculate_position_tracker_stats(positions,stats):
#     """
#         stats ---- PositionStats
#     """
#     longs_count = 0
#     long_exposure = 0
#     shorts_count = 0
#     short_exposure = 0
#
#     for outer_position in positions.values():
#         position = outer_position.inner_position
#         #daily更新价格
#         exposure = position.amount * position.last_sale_price
#         if exposure > 0:
#             longs_count += 1
#             long_exposure += exposure
#         elif exposure < 0:
#             shorts_count +=1
#             short_exposure += exposure
#     #
#     net_exposure = long_exposure + short_exposure
#     gross_exposure = long_exposure - short_exposure
#
#     stats.gross_exposure = gross_exposure
#     stats.long_exposure = long_exposure
#     stats.longs_count = longs_count
#     stats.net_exposure = net_exposure
#     stats.short_exposure = short_exposure
#     stats.shorts_count = shorts_count

# def process_transaction(self,transaction):
#     position = self.position_tracker.positions[asset]
#     amount = position.amount
#     left_amount = amount + transaction.amount
#     if left_amount == 0:
#         self._cash_flow( - self.commission.calculate(transaction))
#         del self._payout_last_sale_price[asset]
#     elif left_amount < 0:
#         raise Exception('禁止融券卖出')
#     # calculate cash
#     self._cash_flow( - transaction.amount * transaction.price)
#     #execute transaction
#     self.position_tracker.execute_transaction(transaction)
#     transaction_dict = transaction.to_dict()
#     self._processed_transaction[transaction.dt].append(transaction_dict)

# def process_commission(self,commission):
#     asset = commission['asset']
#     cost = commission['cost']
#
#     self.position_tracker.handle_commission(asset,cost)
#     self._cash_flow(-cost)

# def process_split(self,splits):
#     """
#         splits --- (asset,ratio)
#     :param splits:
#     :return:
#     """
#     leftover_cash = self.position_tracker.handle_spilts(splits)
#     if leftover_cash > 0 :
#         self._cash_flow(leftover_cash)
#
# def process_divdends(self,next_session,adjustment_reader):
#     """
#         基于时间、仓位获取对应的现金分红、股票分红
#     """
#     position_tracker = self.position_tracker
#     #针对字典 --- set return keys
#     held_sids = set(position_tracker.positions)
#     if held_sids:
#         cash_divdend = adjustment_reader.get_dividends_with_ex_date(
#             held_sids,
#             next_session,
#         )
#         stock_dividends = (
#             adjustment_reader.get_stock_dividends_with_ex_date(
#                 held_sids,
#                 next_session,
#             )
#         )
#     #添加
#     position_tracker.earn_divdends(
#         cash_divdend,stock_dividends
#     )
#     #基于session --- pay_date 处理
#     self._cash_flow(
#         position_tracker.pay_divdends(next_session)
#     )
# self.record_vars
# def update_portfolio(self):
#     """
#         force a computation of the current portfolio
#         portofolio 保留最新
#     """
#     if not self._dirty_portfolio:
#         return
#
#     portfolio = self._portfolio
#     pt = self.position_tracker
#
#     portfolio.positions = pt.get_positions()
#     #计算positioin stats --- sync_last_sale_price
#     position_stats = pt.stats
#
#     portfolio.positions_value = position_value = (
#         position_stats.net_value
#     )
#
#     portfolio.positions_exposure = position_stats.net_exposure
#     self._cash_flow(self._get_payout_total(pt.positions))
#
#     # portfolio_value 初始化capital_value
#     start_value = portfolio.portfolio_value
#     portfolio.portfolio_value = end_value = portfolio.cash + position_value
#
#     # daily 每天账户净值波动
#     pnl = end_value - start_value
#     if start_value !=0 :
#         returns = pnl/start_value
#     else:
#         returns = 0.0
#
#     #pnl --- 投资收益
#     portfolio.pnl += pnl
#     # 每天的区间收益率 --- 递归方式
#     portfolio.returns = (
#         (1+portfolio.returns) *
#         (1+returns) - 1
#     )
#     self._dirty_portfolio = False
# for asset, old_price in payout_last_sale_prices.items():
#     position = positions[asset]
#     payout_last_sale_prices[asset] = price = position.last_sale_price
#     amount = position.amount
#     total += calculate_payout(
#         amount,
#         old_price,
#         price,
#         asset.price_multiplier,
#     )
# return total
import numpy as np
from scipy.optimize import fsolve
#
# a = [15.3,14.7,14.9,14.01,15.2,16.7,16.9]
# print(np.std(a))
# b = np.std(a)
#
#
# def func(paramlist):
#
#     a,b =paramlist[0],paramlist[1]
#     return [a / (a+b) - 0.0476,
#             (a*b) /((a+b+1) * (a+b) ** 2) - 0.0021]
# c1,c2=fsolve(func,[0,0])
# print(c1,c2)
# e = c1 / (c1+c2)


# a = 10
# b = 200
# e = a/(a + b)
# s = (a*b) /((a+b+1) * (a+b) ** 2)
# print(e,s)
# pct = [0.02,-0.03,0.04,0.05,0.08,-0.06,0.07]
# print(np.std(pct))
# def override_account_fields(self,
#                             settled_cash=not_overridden,
#                             total_positions_values=not_overridden,
#                             total_position_exposure=not_overridden,
#                             cushion=not_overridden,
#                             gross_leverage=not_overridden,
#                             net_leverage=not_overridden,
#                             ):
#     # locals ---函数内部的参数
#     self._account_overrides = kwargs = {k: v for k, v in locals().items() if v is not not_overridden}
#     del kwargs['self']
# for k, v in self._account_overrides:
#     setattr(account, k, v)
from itertools import product

# p = {'a':[1,2,3],'b':[4,5,6],'c':[7,8,9]}
# items = p.items()
#
# keys, values = zip(*items)
# print(keys)
# print(values)
# #product --- 每个列表里面取一个元素
# for v in product(*values):
#     params = dict(zip(keys, v))
#     print(params)
#
# from toolz import concatv
#
# list(concatv([], ["a"], ["b", "c"]))
# #['a', 'b', 'c']
#
# #代码高亮
# try:
#     from pygments import highlight
#     from pygments.lexers import PythonLexer
#     from pygments.formatters import TerminalFormatter
#     PYGMENTS = True
# except ImportError:
#     PYGMENTS = False
#
# """
#     将不同的算法通过串行或者并行方式形成算法工厂 ，筛选过滤最终得出目标目标标的组合
#     串行：
#         1、串行特征工厂借鉴zipline或者scikit_learn Pipeline
#         2、串行理论基础：现行的策略大多数基于串行，比如多头排列、空头排列、行业龙头战法、统计指标筛选
#         3、缺点：确定存在主观去判断特征顺序，前提对于市场有一套自己的认识以及分析方法
#     并行：
#         1、并行的理论基础借鉴交集理论
#         2、基于结果反向分类strategy
#     难点：
#         不同算法的权重分配
#     input : stategies ,output : list or tuple of filtered asset
#
#             pipe of strategy to fit targeted asset
#     Parameters
#     -----------
#     steps :list
#         List of strategy
#         wgts: List,str or list , default : 'average'
#     wgts: List
#         List of (name,weight) tuples that allocate the weight of steps means the
#         importance, average wgts avoids the unbalance of steps
#     memory : joblib.Memory interface used to cache the fitted transformers of
#         the Pipeline. By default,no caching is performed. If a string is given,
#         it is the path to the caching directory. Enabling caching triggers a clone
#         of the transformers before fitting.Caching the transformers is advantageous
#         when fitting is time consuming.
#
#     This estimator applies a list of transformer objects in parallel to the
#     input data, then concatenates the results. This is useful to combine
#     several feature extraction mechanisms into a single transformer.
#
#     Parameters
#     ----------
#     transformer_list : List of transformer objects to be applied to the data
#     n_jobs : int --- Number of jobs to run in parallel,
#             -1 means using all processors.`
#     allocation: str(default=average) ,dict , callable
# """
#
# class Ump(object):
#     """
#         裁决模块 基于有效的特征集，针对特定的asset进行投票抉择
#         关于仲裁逻辑：
#             普通选股：针对备选池进行选股，迭代初始选股序列，在迭代中再迭代选股因子，选股因子决定是否对
#             symbol投出反对票，一旦一个因子投出反对票，即筛出序列
#     """
#
#     def __init__(self, poll_workers, thres=0.8):
#         super()._validate_steps(poll_workers)
#         self.voters = poll_workers
#         self._poll_picker = dict()
#         self.threshold = thres
#
#     def _set_params(self, **params):
#         for pname, pval in params.items():
#             self._poll_picker[pname] = pval
#
#     def poll_pick(self, res, v):
#         """
#            vote for feature and quantity the vote action
#            simple poll_pick --- calculate rank pct
#            return bool
#         """
#         formatting = pd.Series(range(1, len(res) + 1), index=res)
#         pct_rank = formatting.rank(pct=True)
#         polling = True if pct_rank[v] > self.thres else False
#         return polling
#
#     def _fit(self, worker, target):
#         '''因子对象针对每一个交易目标的投票结果'''
#         picker = super()._load_from_name(worker)
#         fit_result = picker(self._poll_picker[worker]).fit()
#         poll = self.poll_pick(fit_result, target)
#         return poll
#
#     def decision_function(self, asset):
#         vote_poll = dict()
#         for picker in self.voters:
#             vote_poll.update({picker: self._fit(picker, asset)})
#         decision = np.sum(list(vote_poll.values)) / len(vote_poll)
#         return decision
#
# class MIFeature(ABC):
#     """
#         strategy composed of features which are logically arranged
#         input : feature_list
#         return : asset_list
#         param : _n_field --- all needed field ,_max_window --- upper window along the window args
#         core_part : _domain --- logically combine all features
#     """
#     _n_fields  = []
#     _max_window = []
#     _feature_params = {}
#
#     def _load_features(self,name):
#         try:
#             feature_class = importlib.__import__(name, 'algorithm.features')
#         except:
#             raise ValueError('%s feature not implemented'%name)
#         return feature_class
#
#     def _verify_params(self,params):
#         if isinstance(params,dict):
#             for name,p in params:
#                 feature = self._load_features(name)
#                 if hasattr(feature,'_n_fields') and feature._n_fields != p['fields']:
#                     raise ValueError('fields must be same with feature : %s'%name)
#                 if feature.windowed and p['window'] is None:
#                     raise ValueError('window of feature  is not None : %s'%name)
#                 if feature._pairwise and not isinstance(p['window'],(tuple,list)):
#                     raise ValueError('when pairwise is True ,the length of window must be two')
#                 if hasattr(feature,'_triple') and not isinstance(p['window'],dict):
#                     raise ValueError('triple means three window , it specify macd --- fast,slow,period')
#         else:
#             raise TypeError('params must be dict type')
#
#     def _set_params(self,params):
#         self._verify_params(params)
#         return params
#
#
#     def _eval_feature(self,raw,name,p:dict):
#         """
#             特征分为主体、部分，其中部分特征只是作为主体特征的部分逻辑
#         """
#         feature_class = self._load_features(name)
#         if 'field' in p.keys():
#             print('filed exists spceify this feature should be initialized')
#             if 'window' in p.key():
#                 result = feature_class.calc_feature(raw[p['field']],p['window'])
#             else:
#                 result = feature_class.calc_feature(raw['field'])
#         else:
#             print('field not exists spceify this feature is just  a middle process used by outer faeture function')
#             result = None
#         return result
#
#
#     def _fit_main_features(self,raw):
#         """
#             计算每个标的的所有特征
#         """
#         filter_nan = {}
#         for name in self._n_features:
#             res = self._eval_feature(raw,name,self._feature_params[name])
#             if res:
#                 filter_nan.update({name:res})
#         return filter_nan
#
#
#     def _execute_main(self, trade_date,stock_list):
#         feature_res = {}
#         for code in stock_list:
#             event = Event(trade_date,code)
#             req = GateReq(event, field=self._n_fields, window=self._max_window)
#             raw = feed.addBars(req)
#             res = self._fit_main_features(raw)
#             feature_res.update({code:res})
#         return feature_res
#
#     @abstractmethod
#     def _domain(self,input):
#         """
#             MIFeature（构建有特征组成的接口类），特征按照一定逻辑组合处理为策略
#             实现： 逻辑组合抽象为不同的特征的逻辑运算，具体还是基于不同的特征的运行结果
#         """
#         NotImplemented
#
#
#     def run(self,trade_dt,stock_list:list) -> list:
#         exec_info= self._execute_main(trade_dt,stock_list)
#         filter_order = self._domain(exec_info)
#         return filter_order
#
#
# class MyStrategy(MIFeature):
#     """
#         以MyStrategy为例进行实现
#     """
#
#     _n_features = ['DMA','Reg']
#
#     def __init__(self,params):
#         self._feature_params = super()._set_params(params)
#         self._n_fields = [ v['field'] for k,v in params.items() if 'field' in v.keys()]
#         self._max_window = [ v['window'] for k,v in params.items() if 'window' in v.keys()].max()
#
#     def __enter__(self):
#         return self
#
#     def _domain(self,input):
#         """
#             策略核心逻辑： DMA --- 短期MA大于的长期MA概率超过80%以及收盘价处于最高价与最低价的形成夹角1/2位以上，则asset有效
#             return ranked_list
#         """
#         df = pd.DataFrame.from_dict(input)
#         result = df.T
#         hit_rate = result['DMA'].applymap(lambda x : len(x[x>0])/len(x) > 0.75)
#         reg = result['Reg'].map(lambda x : x > 0.6)
#         # union = set(reg.index) & set(hit_rate.index)
#         input = (pd.DataFrame([hit_rate,reg])).T
#         union = BaseScorer().calc_feature(input)
#         return union
#
#     def __exit__(self,exc_type,exc_val,exc_tb):
#         """
#             exc_type,exc_value,exc_tb(traceback), 当with 后面语句执行错误输出
#         """
#         if exc_val :
#             print('strategy fails to complete')
#         else:
#             print('successfully process')
#
#
# class UnionEngine(object):
#     """
#         组合不同算法---策略
#         返回 --- Order对象
#         initialize
#         handle_data
#         before_trading_start
#         1.判断已经持仓是否卖出
#         2.基于持仓限制确定是否执行买入操作
#     """
#     def __init__(self,algo_mappings,data_portal,broker,assign_policy):
#         self.data_portal = data_portal
#         self.postion_allocation = assign_policy
#         self.broker = broker
#         self.loaders = [self.get_loader_class(key,args) for key,args in algo_mappings.items()]
#
#     @staticmethod
#     def get_loader_class(key,args):
#         """
#         :param key: algo_name or algo_path
#         :param args: algo_params
#         :return: dict -- __name__ : instance
#         """
#
#     # @lru_cache(maxsize=32)
#     def compute_withdraw(self,dt):
#         def run(ins):
#             result = ins.before_trading_start(dt)
#             return result
#
#         with Pool(processes = len(self.loaders)) as pool:
#             exit_assets = [pool.apply_async(run,instance)
#                             for instance in self.loaders.values]
#         return exit_assets
#
#     # @lru_cache(maxsize=32)
#     def compute_algorithm(self,dt,metrics_tracker):
#         unprocessed_loaders = self.tracker_algorithm(metrics_tracker)
#         def run(algo):
#             ins = self.loaders[algo]
#             result = ins.initialize(dt)
#             return result
#
#         with Pool(processes=len(self.loaders)) as pool:
#             exit_assets = [pool.apply_async(run, algo)
#                            for algo in unprocessed_loaders]
#         return exit_assets
#
#     def tracker_algorithm(self,metrics_tracker):
#         unprocessed_algo = set(self.algorithm_mappings.keys()) - \
#                            set(map(lambda x : x.reason ,metrics_tracker.positions.asset))
#         return unprocessed_algo
#
#     def position_allocation(self):
#         return self.assign_policy.map_allocation(self.tracker_algorithm)
#
#     def _calculate_order_amount(self,asset,dt,total_value):
#         """
#             calculate how many shares to order based on the position managment
#             and price where is assigned to 10% limit in order to carry out order max amount
#         """
#         preclose = self.data_portal.get_preclose(asset,dt)
#         porportion = self.postion_allocation.compute_pos_placing(asset)
#         amount = np.floor(porportion * total_value / (preclose * 1.1))
#         return amount
#
#     def get_payout(self, dt,metrics_tracker):
#         """
#         :param metrics_tracker: to get the position
#         :return: sell_orders
#         """
#         assets_of_exit = self.compute_withdraw(dt)
#         positions = metrics_tracker.positions
#         if assets_of_exit:
#             [self.broker.order(asset,
#                                 positions[asset].amount)
#                                 for asset in assets_of_exit]
#             cleanup_transactions,additional_commissions = self.broker.get_transaction(self.data_portal)
#             return cleanup_transactions,additional_commissions
#
#     def get_layout(self,dt,metrics_tracker):
#         asset = self.compute_algorithm(dt,metrics_tracker)
#         avaiable_cash = metrics_tracker.portfolio.cash
#         [self.broker.order(asset,
#                             self._calculate_order_amount(asset,dt,avaiable_cash))
#                             for asset in asset]
#         transactions,new_commissions = self.broker.get_transaction(self.data_portal)
#         return transactions,new_commissions
#
#     def _pop_params(cls, kwargs):
#         """
#         Pop entries from the `kwargs` passed to cls.__new__ based on the values
#         in `cls.params`.
#
#         Parameters
#         ----------
#         kwargs : dict
#             The kwargs passed to cls.__new__.
#
#         Returns
#         -------
#         params : list[(str, object)]
#             A list of string, value pairs containing the entries in cls.params.
#
#         Raises
#         ------
#         TypeError
#             Raised if any parameter values are not passed or not hashable.
#         """
#         params = cls.params
#         if not isinstance(params, Mapping):
#             params = {k: NotSpecified for k in params}
#         param_values = []
#         for key, default_value in params.items():
#             try:
#                 value = kwargs.pop(key, default_value)
#                 if value is NotSpecified:
#                     raise KeyError(key)
#
#                 # Check here that the value is hashable so that we fail here
#                 # instead of trying to hash the param values tuple later.
#                 hash(value)
#             except KeyError:
#                 raise TypeError(
#                     "{typename} expected a keyword parameter {name!r}.".format(
#                         typename=cls.__name__,
#                         name=key
#                     )
#                 )
#             except TypeError:
#                 # Value wasn't hashable.
#                 raise TypeError(
#                     "{typename} expected a hashable value for parameter "
#                     "{name!r}, but got {value!r} instead.".format(
#                         typename=cls.__name__,
#                         name=key,
#                         value=value,
#                     )
#                 )
#
#             param_values.append((key, value))
#         return tuple(param_values)
#
#
# class NoHooks(PipelineHooks):
#     """A PipelineHooks that defines no-op methods for all available hooks.
#     """
#     @contextmanager
#     def running_pipeline(self, pipe, start_date, end_date):
#         yield
#
#     @contextmanager
#     def computing_chunk(self, terms, start_date, end_date):
#         yield
#
#     @contextmanager
#     def loading_terms(self, terms):
#         yield
#
#     @contextmanager
#     def computing_term(self, term):
#         yield

# @contextmanager
# @abstractmethod
# def loading_terms(self, terms):
#     """Contextmanager entered when loading a batch of LoadableTerms.
#
#     Parameters
#     ----------
#     terms : list[zipline.pipe.LoadableTerm]
#         Terms being loaded.
#     """
#
# @contextmanager
# @abstractmethod
# def computing_term(self, term):
#     """Contextmanager entered when computing a ComputableTerm.
#
#     Parameters
#     ----------
#     terms : zipline.pipe.ComputableTerm
#         Terms being computed.
#     """
#
# def delegating_hooks_method(method_name):
#     """Factory function for making DelegatingHooks methods.
#     """
#     if method_name in PIPELINE_HOOKS_CONTEXT_MANAGERS:
#         # Generate a contextmanager that enters the context of all child hooks.
#         # wraps --- callable
#         @wraps(getattr(PipelineHooks, method_name))
#         @contextmanager
#         def ctx(self, *args, **kwargs):
#             with ExitStack() as stack:
#                 for hook in self._hooks:
#                     sub_ctx = getattr(hook, method_name)(*args, **kwargs)
#                     stack.enter_context(sub_ctx)
#                 yield stack
#         return ctx
#     else:
#         # Generate a method that calls methods of all child hooks.
#         @wraps(getattr(PipelineHooks, method_name))
#         def method(self, *args, **kwargs):
#             for hook in self._hooks:
#                 sub_method = getattr(hook, method_name)
#                 sub_method(*args, **kwargs)
#
#         return method
#
#
# class DelegatingHooks(PipelineHooks):
#     """A PipelineHooks that delegates to one or more other hooks.
#
#     Parameters
#     ----------
#     hooks : list[implements(PipelineHooks)]
#         Sequence of hooks to delegate to.
#     """
#     def __new__(cls, hooks):
#         if len(hooks) == 0:
#             # OPTIMIZATION: Short-circuit to a NoHooks if we don't have any
#             # sub-hooks.
#             return NoHooks()
#         else:
#             self = super(DelegatingHooks, cls).__new__(cls)
#             self._hooks = hooks
#             return self
#
#     # Implement all interface methods by delegating to corresponding methods on
#     # input hooks. locals --- __dict__ 覆盖原来的方法
#     locals().update({
#         name: delegating_hooks_method(name)
#         # TODO: Expose this publicly on interface.
#         for name in PipelineHooks._signatures
#     })
#
#
# del delegating_hooks_method
#
# class AlgorithmSimulator(object):
#
#     EMISSION_TO_PERF_KEY_MAP = {
#         'minute': 'minute_perf',
#         'daily': 'daily_perf'
#     }
#
#     def __init__(self, algo, sim_params, data_portal, clock, benchmark_source,
#                  restrictions, universe_func):
#
#         # ==============
#         # ArkQuant
#         # Param Setup
#         # ==============
#         self.sim_params = sim_params
#         self.data_portal = data_portal
#         self.restrictions = restrictions
#
#         # ==============
#         # Algo Setup
#         # ==============
#         self.algo = algo
#
#         # ==============
#         # Snapshot Setup
#         # ==============
#
#         # We don't have a datetime for the current snapshot until we
#         # receive a message.
#         self.simulation_dt = None
#
#         self.clock = clock
#
#         self.benchmark_source = benchmark_source
#
#         # =============
#         # Logging Setup
#         # =============
#
#         # Processor function for injecting the algo_dt into
#         # user prints/logs.
#         # def inject_algo_dt(record):
#         #     if 'algo_dt' not in record.extra:
#         #         record.extra['algo_dt'] = self.simulation_dt
#         # self.processor = Processor(inject_algo_dt)
#
#         # This object is the way that user algorithms interact with OHLCV data,
#         # fetcher data, and some API methods like `data.can_trade`.
#         self.current_data = self._create_bar_data(universe_func)
#
#     def get_simulation_dt(self):
#         return self.simulation_dt
#
#     #获取日数据，封装为一个API(fetch process flush other api)
#     def _create_bar_data(self, universe_func):
#         return BarData(
#             data_portal=self.data_portal,
#             simulation_dt_func=self.get_simulation_dt,
#             data_frequency=self.sim_params.data_frequency,
#             trading_calendar=self.algo.trading_calendar,
#             restrictions=self.restrictions,
#             universe_func=universe_func
#         )
#
#     def transform(self):
#         """
#         Main generator work loop.
#         """
#         algo = self.algo
#         metrics_tracker = algo.metrics_tracker
#         emission_rate = metrics_tracker.emission_rate
#
#         #生成器yield方法 ，返回yield 生成的数据，next 执行yield 之后的方法
#         def every_bar(dt_to_use, current_data=self.current_data,
#                       handle_data=algo.event_manager.handle_data):
#             for capital_change in calculate_minute_capital_changes(dt_to_use):
#                 yield capital_change
#
#             self.simulation_dt = dt_to_use
#             # called every tick (minute or day).
#             algo.on_dt_changed(dt_to_use)
#
#             broker = algo.broker
#
#             # handle any transactions and commissions coming out new orders
#             # placed in the last bar
#             new_transactions, new_commissions, closed_orders = \
#                 broker.get_transactions(current_data)
#
#             broker.prune_orders(closed_orders)
#
#             for transaction in new_transactions:
#                 metrics_tracker.process_transaction(transaction)
#
#                 # since this order was modified, record it
#                 order = broker.orders[transaction.order_id]
#                 metrics_tracker.process_order(order)
#
#             for commission in new_commissions:
#                 metrics_tracker.process_commission(commission)
#
#             handle_data(algo, current_data, dt_to_use)
#
#             # grab any new orders from the broker, then clear the list.
#             # this includes cancelled orders.
#             new_orders = broker.new_orders
#             broker.new_orders = []
#
#             # if we have any new orders, record them so that we know
#             # in what perf period they were placed.
#             for new_order in new_orders:
#                 metrics_tracker.process_order(new_order)
#
#         def once_a_day(midnight_dt, current_data=self.current_data,
#                        data_portal=self.data_portal):
#             # process any capital changes that came overnight
#             for capital_change in algo.calculate_capital_changes(
#                     midnight_dt, emission_rate=emission_rate,
#                     is_interday=True):
#                 yield capital_change
#
#             # set all the timestamps
#             self.simulation_dt = midnight_dt
#             algo.on_dt_changed(midnight_dt)
#
#             metrics_tracker.handle_market_open(
#                 midnight_dt,
#                 algo.data_portal,
#             )
#
#             # handle any splits that impact any positions or any open orders.
#             assets_we_care_about = (
#                 viewkeys(metrics_tracker.positions) |
#                 viewkeys(algo.broker.open_orders)
#             )
#
#             if assets_we_care_about:
#                 splits = data_portal.get_splits(assets_we_care_about,
#                                                 midnight_dt)
#                 if splits:
#                     algo.broker.process_splits(splits)
#                     metrics_tracker.handle_splits(splits)
#
#         def on_exit():
#             # Remove references to algo, data portal, et al to break cycles
#             # and ensure deterministic cleanup of these objects when the
#             # ArkQuant finishes.
#             self.algo = None
#             self.benchmark_source = self.current_data = self.data_portal = None
#
#         with ExitStack() as stack:
#             """
#             由于已注册的回调是按注册的相反顺序调用的，因此最终行为就好像with 已将多个嵌套语句与已注册的一组回调一起使用。
#             这甚至扩展到异常处理-如果内部回调抑制或替换异常，则外部回调将基于该更新状态传递自变量。
#             enter_context  输入一个新的上下文管理器，并将其__exit__()方法添加到回调堆栈中。返回值是上下文管理器自己的__enter__()方法的结果。
#             callback（回调，* args，** kwds ）接受任意的回调函数和参数，并将其添加到回调堆栈中。
#             """
#             stack.callback(on_exit)
#             stack.enter_context(self.processor)
#             stack.enter_context(ZiplineAPI(self.algo))
#
#             if algo.data_frequency == 'minute':
#                 def execute_order_cancellation_policy():
#                     algo.broker.execute_cancel_policy(SESSION_END)
#
#                 def calculate_minute_capital_changes(dt):
#                     # process any capital changes that came between the last
#                     # and current minutes
#                     return algo.calculate_capital_changes(
#                         dt, emission_rate=emission_rate, is_interday=False)
#             else:
#                 def execute_order_cancellation_policy():
#                     pass
#
#                 def calculate_minute_capital_changes(dt):
#                     return []
#
#             for dt, action in self.clock:
#                 if action == BAR:
#                     for capital_change_packet in every_bar(dt):
#                         yield capital_change_packet
#                 elif action == SESSION_START:
#                     for capital_change_packet in once_a_day(dt):
#                         yield capital_change_packet
#                 elif action == SESSION_END:
#                     # End of the session.
#                     positions = metrics_tracker.positions
#                     position_assets = algo.asset_finder.retrieve_all(positions)
#                     self._cleanup_expired_assets(dt, position_assets)
#
#                     execute_order_cancellation_policy()
#                     algo.validate_account_controls()
#
#                     yield self._get_daily_message(dt, algo, metrics_tracker)
#                 elif action == BEFORE_TRADING_START_BAR:
#                     self.simulation_dt = dt
#                     algo.on_dt_changed(dt)
#                     algo.before_trading_start(self.current_data)
#                 elif action == MINUTE_END:
#                     minute_msg = self._get_minute_message(
#                         dt,
#                         algo,
#                         metrics_tracker,
#                     )
#
#                     yield minute_msg
#
#             risk_message = metrics_tracker.handle_simulation_end(
#                 self.data_portal,
#             )
#             yield risk_message
#
#     def _cleanup_expired_assets(self, dt, position_assets):
#         """
#         Clear out any asset that have expired before starting a new sim day.
#
#         Performs two functions:
#
#         1. Finds all asset for which we have open orders and clears any
#            orders whose asset are on or after their auto_close_date.
#
#         2. Finds all asset for which we have positions and generates
#            close_position events for any asset that have reached their
#            auto_close_date.
#         """
#         algo = self.algo
#
#         def past_auto_close_date(asset):
#             acd = asset.auto_close_date
#             return acd is not None and acd <= dt
#
#         # Remove positions in any sids that have reached their auto_close date.
#         assets_to_clear = \
#             [asset for asset in position_assets if past_auto_close_date(asset)]
#         metrics_tracker = algo.metrics_tracker
#         data_portal = self.data_portal
#         for asset in assets_to_clear:
#             metrics_tracker.process_close_position(asset, dt, data_portal)
#
#         # Remove open orders for any sids that have reached their auto close
#         # date. These orders get processed immediately because otherwise they
#         # would not be processed until the first bar of the next day.
#         broker = algo.broker
#         assets_to_cancel = [
#             asset for asset in broker.open_orders
#             if past_auto_close_date(asset)
#         ]
#         for asset in assets_to_cancel:
#             broker.cancel_all_orders_for_asset(asset)
#
#         # Make a copy here so that we are not modifying the list that is being
#         # iterated over.
#         for order in copy(broker.new_orders):
#             if order.status == ORDER_STATUS.CANCELLED:
#                 metrics_tracker.process_order(order)
#                 broker.new_orders.remove(order)
#
#     def _get_daily_message(self, dt, algo, metrics_tracker):
#         """
#         Get a perf message for the given datetime.
#         """
#         perf_message = metrics_tracker.handle_market_close(
#             dt,
#             self.data_portal,
#         )
#         perf_message['daily_perf']['recorded_vars'] = algo.recorded_vars
#         return perf_message
#
#     def _get_minute_message(self, dt, algo, metrics_tracker):
#         """
#         Get a perf message for the given datetime.
#         """
#         rvars = algo.recorded_vars
#
#         minute_message = metrics_tracker.handle_minute_close(
#             dt,
#             self.data_portal,
#         )
#
#         minute_message['minute_perf']['recorded_vars'] = rvars
#         return minute_message
#
#         # =============
#         # Logging Setup
#         # =============
#
#         # Processor function for injecting the algo_dt into
#         # user prints/logs.
#         # def inject_algo_dt(record):
#         #     if 'algo_dt' not in record.extra:
#         #         record.extra['algo_dt'] = self.simulation_dt
#         # self.processor = Processor(inject_algo_dt)
#
#
# class PeriodLabel(object):
#     """Backwards compat, please kill me.
#     """
#     def start_of_session(self, ledger, session, data_portal):
#         self._label = session.strftime('%Y-%m')
#
#     def end_of_bar(self, packet, *args):
#         packet['cumulative_risk_metrics']['period_label'] = self._label
#
#     end_of_session = end_of_bar
#
#
# class _ConstantCumulativeRiskMetric(object):
#     """A metric which does not change, ever.
#
#     Notes
#     -----
#     This exists to maintain the existing structure of the perf packets. We
#     should kill this as soon as possible.
#     """
#     def __init__(self, field, value):
#         self._field = field
#         self._value = value
#
#     def start_of_session(self, packet,*args):
#         packet['cumulative_risk_metrics'][self._field] = self._value
#
#     def end_of_session(self, packet, *args):
#         packet['cumulative_risk_metrics'][self._field] = self._value


# If you are adding new attributes, don't update this set. This method
# is deprecated to normal attribute access so we don't want to encourage
# new usages.
# __getitem__ = _deprecated_getitem_method(
#     'portfolio', {
#         'capital_used',
#         'starting_cash',
#         'portfolio_value',
#         'pnl',
#         'returns',
#         'cash',
#         'positions',
#         'start_date',
#         'positions_value',
#     },
# )

#toolz.itertoolz.groupby(key, seq)
from dateutil.relativedelta import relativedelta
# import datetime , pandas as pd
#
# start_session = datetime.datetime.strptime('2010-01-31','%Y-%m-%d')
# end_session = datetime.datetime.strptime('2012-01-31','%Y-%m-%d')
#
# print(start_session,end_session)
#
# # end = end_session.replace(day=1) + relativedelta(months=1)
# end = end_session
# print(end)
#
# months = pd.date_range(
#     start=start_session,
#     # Ensure we have at least one month
#     end=end,
#     freq='M',
#     tz='utc',
#     closed = 'left'
# )
# print('months',months.size)
# print(type(months),months)
# months.iloc[-1] = 'test'
# period = months[0].to_period(freq='%dM' % 3)
# print(months[::3])
# print('period',period.end_date)


# for period_timestamp in months:
#     period = period_timestamp.to_period(freq='%dM' % months_per)

# # 下个月第一天
# end = end_session.replace(day=1) + relativedelta(months=1)
# months = pd.date_range(
#     start=start_session,
#     # Ensure we have at least one month
#     end=end - datetime.timedelta(days=1),
#     freq='M',
#     tz='utc',
# )
# 分析指标:
# 策略共执行{}个交易日 策略资金利用率比例  策略买入成交比例 平均获利期望 平均亏损期望
# 策略持股天数平均值,策略持股天数中位数,策略期望收益,策略期望亏损,前后两两生效交易时间相减,
# 计算平均生效间隔时间,计算cost各种统计度量值,计算资金对应的成交比例
# from sys import float_info
#
# def asymmetric_round_price(price, prefer_round_down, tick_size, diff=0.95):
#     """
#     Asymmetric rounding function for adjusting prices to the specified number
#     of places in a way that "improves" the price. For limit prices, this means
#     preferring to round down on buys and preferring to round up on sells.
#     For stop prices, it means the reverse.
#
#     If prefer_round_down == True:
#         When .05 below to .95 above a specified decimal place, use it.
#     If prefer_round_down == False:
#         When .95 below to .05 above a specified decimal place, use it.
#
#     In math-speak:
#     If prefer_round_down: [<X-1>.0095, X.0195) -> round to X.01.
#     If not prefer_round_down: (<X-1>.0005, X.0105] -> round to X.01.
#     """
#     # 返回位数
#     precision = zp_math.number_of_decimal_places(tick_size)
#     multiplier = int(tick_size * (10 ** precision))
#     diff -= 0.5  # shift the difference down
#     diff *= (10 ** -precision)  # adjust diff to precision of tick size
#     diff *= multiplier  # adjust diff to value of tick_size
#
#     # Subtracting an epsilon from diff to enforce the open-ness of the upper
#     # bound on buys and the lower bound on sells.  Using the actual system
#     # epsilon doesn't quite get there, so use a slightly less epsilon-ey value.
#     epsilon = float_info.epsilon * 10
#     diff = diff - epsilon
#
#     # relies on rounding half away from zero, unlike numpy's bankers' rounding
#     rounded = tick_size * consistent_round(
#         (price - (diff if prefer_round_down else -diff)) / tick_size
#     )
#     if zp_math.tolerant_equals(rounded, 0.0):
#         return 0.0
#     return rounded
#
#
# # 生成器yield方法 ，返回yield 生成的数据，next 执行yield 之后的方法
# def every_bar(dt_to_use, current_data=self.current_data,
#               handle_data=algo.event_manager.handle_data):
#     for capital_change in calculate_minute_capital_changes(dt_to_use):
#         yield capital_change
#
#     self.simulation_dt = dt_to_use
#     # called every tick (minute or day).
#     algo.on_dt_changed(dt_to_use)
#
#     broker = algo.broker
#
#     # handle any transactions and commissions coming out new orders
#     # placed in the last bar
#     new_transactions, new_commissions, closed_orders = \
#         broker.get_transactions(current_data)
#
#     broker.prune_orders(closed_orders)
#
#     for transaction in new_transactions:
#         metrics_tracker.process_transaction(transaction)
#
#         # since this order was modified, record it
#         order = broker.orders[transaction.order_id]
#         metrics_tracker.process_order(order)
#
#     for commission in new_commissions:
#         metrics_tracker.process_commission(commission)
#
#     handle_data(algo, current_data, dt_to_use)
#
#     # grab any new orders from the broker, then clear the list.
#     # this includes cancelled orders.
#     new_orders = broker.new_orders
#     broker.new_orders = []
#
#     # if we have any new orders, record them so that we know
#     # in what perf period they were placed.
#     for new_order in new_orders:
#         metrics_tracker.process_order(new_order)
#
# def once_a_day(midnight_dt, current_data=self.current_data,
#                data_portal=self.data_portal):
#     # process any capital changes that came overnight
#     for capital_change in algo.calculate_capital_changes(
#             midnight_dt, emission_rate=emission_rate,
#             is_interday=True):
#         yield capital_change
#
#     # set all the timestamps
#     self.simulation_dt = midnight_dt
#     algo.on_dt_changed(midnight_dt)
#
#     metrics_tracker.handle_market_open(
#         midnight_dt,
#         algo.data_portal,
#     )
#
#     # handle any splits that impact any positions or any open orders.
#     assets_we_care_about = (
#         viewkeys(metrics_tracker.positions) |
#         viewkeys(algo.broker.open_orders)
#     )
#
#     if assets_we_care_about:
#         splits = data_portal.get_splits(assets_we_care_about,
#                                         midnight_dt)
#         if splits:
#             algo.broker.process_splits(splits)
#             metrics_tracker.handle_splits(splits)
#

# -*- coding:utf-8 -*-

# import unittest
#
# class NamesTestCase(unittest.TestCase):
#     """
#         所有以test_开头的方法都会自动运行
#         assertEqual,assertNotEqual,assertTrue,assertFalse,assertIn,assertNotIn
#         setUp -- called before test method ; setUpClass --A  class method called before tests in an individual class are run
#     """
#     @classmethod
#     def setUpClass(cls) -> None:
#         pass
#
#     def test_first_last_name(self):
#         pass
#
#     @classmethod
#     def tearDownClass(cls) -> None:
#         pass


# @property
# def birth(self):
#     return self._birth
#
# @birth.setter
# def birth(self, value):
#     self._birth = value
#
# @birth.getter
# def birth(self):
#     return self._birth
#
# getter  ---- property ;  setter --- @func.setter
#
# __delete__(instance), __get__(instance,owner) , __set__(instance,value) 描述器 , 实例为类的类属性
# __getattribute__ --- __getattr__ (显式访问不存在饿属性,除非显示调用或引发AttributeError异常） ）
#
#
# __delete__(self,instance) ,__del__(self)
#
# math.copysign(x, y)
# Return x with the sign of y. On a platform that supports signed zeros, copysign(1.0, -0.0) returns -1.0.

# def get_pctchange(self, asset,dt):
#     tbl = self.metadata['equity_price']
#     orm = sa.select([sa.cast(tbl.c.pct, sa.Numeric(10, 2)).label('pct')])\
#         .where(sa.and_(tbl.c.trade_dt == dt,tbl.c.sid == asset.sid))
#     rp = self.engine.execute(orm)
#     data = rp.scalar()
#     return data[0]

# stamp = dt
# final = datetime.datetime(stamp.year,stamp.month,stamp.day,hour = 15,minute = 0,second=0) if last else \
#     datetime.datetime(stamp.year,stamp.month,stamp.day,hour = 9,minute = 30,second=0)

# @preprocess(conn=coerce_string_to_conn(require_exists=True))

# def _init_raw_array(self, asset, edate, window):
#     if self.reader.data_frequency == 'daily':
#         bars = self.reader.load_raw_arrays(edate, window, asset, ['close'])
#     elif self.reader.data_frequency == 'minute':
#         bars = self.reader.get_resampled(edate, window, '15:00', asset, ['close'])
#     return bars
# def window_arrays(self, edate, window, asset, field):
#     """基于固定的fields才需要adjust"""
#     #获取时间区间
#     sessions = self.trading_calendar.sessions_in_range(edate, window, include=True)
#     # # 获取原始数据
#     # raw_arrays = self.array(sessions,asset, field)
#     #需要调整的
#     adjusted_fields = set(field) & self.FIELDS
#     if adjusted_fields:
#         #调整系数
#         adjustments, raw_arrays = self._adjustment.calculate_adjustments_in_sessions(edate,window,asset)
#         #计算调整数据
#         adjust_arrays = {}
#         for asset in asset:
#             sid = asset.sid
#             qfq = adjustments[sid]
#             raw = raw_arrays[sid]
#             try:
#                 qfq = qfq.reindex(sessions)
#                 qfq.fillna(method = 'bfill',inplace = True)
#                 qfq.fillna(1.0,inplace=True)
#                 raw[adjusted_fields] = raw.loc[:, adjusted_fields].multiply(qfq, axis=0)
#             except Exception as e:
#                 print(e,asset)
#             adjust_arrays[sid] = raw
#     else:
#         adjust_arrays = raw_arrays
#     return adjust_arrays

# def history(self, asset, field, dts, window):
#     """
#     A window of pricing data with adjustments applied assuming that the
#     end of the window is the day before the current ArkQuant time.
#     default fields --- OHLCV
#
#     Parameters
#     ----------
#     asset : iterable of Assets
#         The asset in the window.
#     dts : iterable of datetime64-like
#         The datetimes for which to fetch data.
#         Makes an assumption that all dts are present and contiguous,
#         in the _calendar.
#     field : str or list
#         The OHLCV field for which to retrieve data.
#     window : int
#         The length of window
#     Returns
#     -------
#     out : np.ndarray with shape(len(days between start, end), len(asset))
#     """
#     if window != 1:
#         block_arrays = self._ensure_sliding_windows(
#                                         dts,
#                                         window,
#                                         asset,
#                                         field
#                                         )
#     else:
#         # 获取昨天的数据
#         pre_date = self.trading_calendar.dt_window_size(dts, 1)
#         block_arrays = self.adjust_window.array(pre_date, asset, field)
#     return block_arrays

# def load_pricing_adjustments(self, date, window,
#                              should_include_dividends=True,
#                              should_include_rights=True,
#                              ):
#     sessions = self.trading_calendar.sessions_in_range(date, window, include=True)
#     pricing_adjustments = self._load_adjustments_from_sqlite(
#                             sessions,
#                             should_include_dividends,
#                             should_include_rights)
#     return pricing_adjustments
# sessions = self.trading_calendar.sessions_in_range(date, window, include=True)

# def window_arrays(self, date, window, asset, field):
#     """
#     :param date: str
#     :param window: int
#     :param asset: Assets list
#     :param field: str or list
#     :return: arrays which is adjusted by divdends and rights
#     """
#     adjustments, raw_arrays, sessions = self._adjustment.calculate_adjustments_in_sessions(date, window, asset)
#     adjusted_fields = set(field) & self.FIELDS
#     if adjusted_fields:
#         #计算调整数据
#         adjust_arrays = {}
#         for asset in asset:
#             sid = asset.sid
#             qfq = adjustments[sid]
#             raw = raw_arrays[sid]
#             try:
#                 qfq = qfq.reindex(sessions)
#                 qfq.fillna(method='bfill', inplace=True)
#                 qfq.fillna(1.0, inplace=True)
#                 raw[adjusted_fields] = raw.loc[:, adjusted_fields].multiply(qfq, axis=0)
#             except Exception as e:
#                 print(e, asset)
#             adjust_arrays[sid] = raw
#     else:
#         adjust_arrays = raw_arrays
#     return adjust_arrays


# def calculate_adjustments_in_sessions(self, date, window, asset):
#     #     """
#     #     Returns
#     #     -------
#     #     adjustments : list[dict[int -> Adjustment]]
#     #         A list, where each element corresponds to the `columns`, of
#     #         mappings from index to adjustment objects to apply at that index.
#     #     """
#     #     adjs = {}
#     #     #获取全部的分红除权配股数据
#     #     adjustments = self._adjustments_reader.load_pricing_adjustments(date, window)
#     #     # 基于data_frequency --- 调整adjustments
#     #     adapted_adjustments = self.adapt_to_frequency(adjustments)
#     #     #获取对应的收盘价数据
#     #     history, sessions = self._load_raw_array(date, window, asset)
#     #     close = valmap(lambda x: x['close'], history)
#     #     #计算前复权系数
#     #     _calculate = partial(self._calculate_adjustments_for_sid, adjustments=adapted_adjustments, close=close)
#     #     for asset in asset:
#     #         adjs[asset] = _calculate(sid=asset.sid)
#     #     return adjs, history, sessions

# Get the first trading minute
# self._first_trading_minute, _ = (
#     _calendar.open_and_close_for_session(
#         [self._first_trading_day]
#     )
#     if self._first_trading_day is not None else (None, None)
# )
#
# # Store the locs of the first day and first minute
# self._first_trading_day_loc = (
#     _calendar.all_sessions.get_loc(self._first_trading_day)
#     if self._first_trading_day is not None else None
# )
# -*- coding : utf-8 -*-
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# index=True,
# index_label=first(tbl.primary_key.columns).name,

# def _update_cache_for_asset(self):
#     self._request_cache = {}
#     equities = self._fetch_equities_from_dfcf()
#     self._request_cache['equity'] = set(equities) - set(self._assets_cache['equity'])
#     self._assets_cache['equity'] = equities
#
#     convertibles = self._fetch_convertibles_from_dfcf()
#     self._request_cache['convertible'] = set(convertibles) - set(self._assets_cache['convertible'])
#     self._assets_cache['convertible'] = convertibles
#
#     funds = self._fetch_funds_from_dfcf()
#     self._request_cache['fund'] = set(funds['基金代码'].values) - set(self._assets_cache['fund'])
#     self._assets_cache['fund'] = funds
#
#     duals = self._fetch_duals_from_dfcf()
#     self._request_cache['dual'] = set(duals) - set(self._assets_cache['dual'])
#     self._assets_cache['dual'] = duals

# if self.symbol:
#     return '%s(%d [%s])' % (type(self).__name__, self.sid, self.symbol)
# else:
#     return '%s(%d)' % (type(self).__name__, self.sid)

# def _compute_date_range_slice(self, start_date, end_date):
#     # Get the index of the start of dates for ``start_date``.
#     start_ix = self.dates.searchsorted(start_date)
#
#     # Get the index of the start of the first date **after** end_date.
#     end_ix = self.dates.searchsorted(end_date, side='right')
#
#     return slice(start_ix, end_ix)
# ins = ins.groupby(table.c.sid)

# class AssetDateBounds(TradingControl):
#     """
#     TradingControl representing a prohibition against ordering an asset before
#     its start_date, or after its end_date.
#     """
#
#     def __init__(self, on_error):
#         super(AssetDateBounds, self).__init__(on_error)
#
#     def validate(self,
#                  asset,
#                  amount,
#                  portfolio,
#                  algo_datetime):
#         """
#         Fail if the algo has passed this Asset's end_date, or before the
#         Asset's start date.
#         """
#         # If the order is for 0 shares, then silently pass through.
#         if amount == 0:
#             return
#
#         normalized_algo_dt = pd.Timestamp(algo_datetime).normalize()
#
#         # Fail if the algo is before this Asset's start_date
#         if asset.start_date:
#             normalized_start = pd.Timestamp(asset.start_date).normalize()
#             if normalized_algo_dt < normalized_start:
#                 metadata = {
#                     'asset_start_date': normalized_start
#                 }
#                 self.handle_violation(
#                     asset, amount, algo_datetime, metadata=metadata)
#         # Fail if the algo has passed this Asset's end_date
#         if asset.end_date:
#             normalized_end = pd.Timestamp(asset.end_date).normalize()
#             if normalized_algo_dt > normalized_end:
#                 metadata = {
#                     'asset_end_date': normalized_end
#                 }
#                 self.handle_violation(
#                     asset, amount, algo_datetime, metadata=metadata)
# class RestrictedListOrder(TradingControl):
#     """TradingControl representing a restricted list of asset that
#     cannot be ordered by the algorithm.
#
#     Parameters
#     ----------
#     restrictions : zipline.finance.asset_restrictions.Restrictions
#         Object representing restrictions of a group of asset.
#     """
#
#     def __init__(self, on_error, restrictions):
#         super(RestrictedListOrder, self).__init__(on_error)
#         self.restrictions = restrictions
#
#     def validate(self,
#                  asset,
#                  amount,
#                  portfolio,
#                  algo_datetime):
#         """
#         Fail if the asset is in the restricted_list.
#         """
#         if self.restrictions.is_restricted(asset, algo_datetime):
#             self.handle_violation(asset, amount, algo_datetime)
#
#
# class MaxOrderCount(TradingControl):
#     """
#     TradingControl representing a limit on the number of orders that can be
#     placed in a given trading day.
#     """
#
#     def __init__(self, on_error, max_count):
#         super(MaxOrderCount, self).__init__(on_error, max_count=max_count)
#         self.orders_placed = 0
#         self.max_count = max_count
#         self.current_date = None
#
#     def validate(self,
#                  asset,
#                  amount,
#                  portfolio,
#                  algo_datetime):
#         """
#         Fail if we've already placed self.max_count orders today.
#         """
#         algo_date = algo_datetime.date()
#
#         # Reset order count if it's a new day.
#         if self.current_date and self.current_date != algo_date:
#             self.orders_placed = 0
#         self.current_date = algo_date
#
#         if self.orders_placed >= self.max_count:
#             self.handle_violation(asset, amount, algo_datetime)
#         self.orders_placed += 1
#
#
# class AccountControl(ABC):
#     """
#     Abstract base class representing a fail-safe control on the behavior of any
#     algorithm.
#     """
#
#     def __init__(self, **kwargs):
#         """
#         Track any arguments that should be printed in the error message
#         generated by self.fail.
#         """
#         self.__fail_args = kwargs
#
#     @abstractmethod
#     def validate(self,
#                  _portfolio,
#                  _account,
#                  _algo_datetime,
#                  _algo_current_data):
#         """
#         On each call to handle data by TradingAlgorithm, this method should be
#         called *exactly once* on each registered AccountControl object.
#
#         If the check does not violate this AccountControl's restraint given
#         the information in `portfolio` and `account`, this method should
#         return None and have no externally-visible side-effects.
#
#         If the desired order violates this AccountControl's contraint, this
#         method should call self.fail().
#         """
#         raise NotImplementedError
#
#     def fail(self):
#         """
#         Raise an AccountControlViolation with information about the failure.
#         """
#         raise AccountControlViolation(constraint=repr(self))
#
#     def __repr__(self):
#         return "{name}({attrs})".format(name=self.__class__.__name__,
#                                         attrs=self.__fail_args)
#
#
# class MaxLeverage(AccountControl):
#     """
#     AccountControl representing a limit on the maximum leverage allowed
#     by the algorithm.
#     """
#
#     def __init__(self, max_leverage):
#         """
#         max_leverage is the gross leverage in decimal form. For example,
#         2, limits an algorithm to trading at most double the account value.
#         """
#         super(MaxLeverage, self).__init__(max_leverage=max_leverage)
#         self.max_leverage = max_leverage
#
#         if max_leverage is None:
#             raise ValueError(
#                 "Must supply max_leverage"
#             )
#
#         if max_leverage < 0:
#             raise ValueError(
#                 "max_leverage must be positive"
#             )
#
#     def validate(self,
#                  _portfolio,
#                  _account,
#                  _algo_datetime,
#                  _algo_current_data):
#         """
#         Fail if the leverage is greater than the allowed leverage.
#         """
#         if _account.leverage > self.max_leverage:
#             self.fail()
#
#
# class MinLeverage(AccountControl):
#     """AccountControl representing a limit on the minimum leverage allowed
#     by the algorithm after a threshold period of time.
#
#     Parameters
#     ----------
#     min_leverage : float
#         The gross leverage in decimal form.
#     deadline : datetime
#         The date the min leverage must be achieved by.
#
#     For example, min_leverage=2 limits an algorithm to trading at minimum
#     double the account value by the deadline date.
#     """
#
#     @expect_types(
#         __funcname='MinLeverage',
#         min_leverage=(int, float),
#         deadline=datetime
#     )
#     @expect_bounded(__funcname='MinLeverage', min_leverage=(0, None))
#     def __init__(self, min_leverage, deadline):
#         super(MinLeverage, self).__init__(min_leverage=min_leverage,
#                                           deadline=deadline)
#         self.min_leverage = min_leverage
#         self.deadline = deadline
#
#     def validate(self,
#                  _portfolio,
#                  account,
#                  algo_datetime,
#                  _algo_current_data):
#         """
#         Make validation checks if we are after the deadline.
#         Fail if the leverage is less than the min leverage.
#         """
#         if (algo_datetime > self.deadline and
#                 account.leverage < self.min_leverage):
#             self.fail()
#
# @api_method
# def fetch_csv(self,
#               url,
#               pre_func=None,
#               post_func=None,
#               date_column='date',
#               date_format=None,
#               timezone=pytz.utc.zone,
#               symbol=None,
#               mask=True,
#               symbol_column=None,
#               special_params_checker=None,
#               country_code=None,
#               **kwargs):
#     """Fetch a csv from a remote url and register the data so that it is
#     queryable from the ``data`` object.
#
#     Parameters
#     ----------
#     url : str
#         The url of the csv file to load.
#     pre_func : callable[pd.DataFrame -> pd.DataFrame], optional
#         A callback to allow preprocessing the raw data returned from
#         fetch_csv before dates are paresed or symbols are mapped.
#     post_func : callable[pd.DataFrame -> pd.DataFrame], optional
#         A callback to allow postprocessing of the data after dates and
#         symbols have been mapped.
#     date_column : str, optional
#         The name of the column in the preprocessed dataframe containing
#         datetime information to map the data.
#     date_format : str, optional
#         The format of the dates in the ``date_column``. If not provided
#         ``fetch_csv`` will attempt to infer the format. For information
#         about the format of this string, see :func:`pandas.read_csv`.
#     timezone : tzinfo or str, optional
#         The timezone for the datetime in the ``date_column``.
#     symbol : str, optional
#         If the data is about a new asset or index then this string will
#         be the name used to identify the values in ``data``. For example,
#         one may use ``fetch_csv`` to load data for VIX, then this field
#         could be the string ``'VIX'``.
#     mask : bool, optional
#         Drop any rows which cannot be symbol mapped.
#     symbol_column : str
#         If the data is attaching some new attribute to each asset then this
#         argument is the name of the column in the preprocessed dataframe
#         containing the symbols. This will be used along with the date
#         information to map the sids in the asset finder.
#     country_code : str, optional
#         Country code to use to disambiguate symbol lookups.
#     **kwargs
#         Forwarded to :func:`pandas.read_csv`.
#
#     Returns
#     -------
#     csv_data_source : zipline.sources.requests_csv.PandasRequestsCSV
#         A requests source that will pull data from the url specified.
#     """
#     if country_code is None:
#         country_code = self.default_fetch_csv_country_code(
#             self.trading_calendar,
#         )
#
#     # Show all the logs every time fetcher is used.
#     csv_data_source = PandasRequestsCSV(
#         url,
#         pre_func,
#         post_func,
#         self.asset_finder,
#         self.trading_calendar.day,
#         self.sim_params.start_session,
#         self.sim_params.end_session,
#         date_column,
#         date_format,
#         timezone,
#         symbol,
#         mask,
#         symbol_column,
#         data_frequency=self.data_frequency,
#         country_code=country_code,
#         special_params_checker=special_params_checker,
#         **kwargs
#     )
#
#     # ingest this into dataportal
#     self.data_portal.handle_extra_source(csv_data_source.df,
#                                          self.sim_params)
#
#     return csv_data_source

# import glob
# res = glob.glob('/Users/python/Library/Mobile Documents/com~apple~CloudDocs/ArkQuant/pipe/strategy/*.py')
# print(list(res))
#
# print(__file__)

# ####################
# # Account Controls #
# ####################
#
# def register_account_control(self, control):
#     """
#     Register a new AccountControl to be checked on each bar.
#     """
#     if self.initialized:
#         raise RegisterAccountControlPostInit()
#     self.account_controls.append(control)
#
# def validate_account_controls(self):
#     for control in self.account_controls:
#         control.validate(self.portfolio,
#                          self.account,
#                          self.get_datetime(),
#                          self.trading_client.current_data)
#
# @api_method
# def set_max_leverage(self, max_leverage):
#     """Set a limit on the maximum leverage of the algorithm.
#
#     Parameters
#     ----------
#     max_leverage : float
#         The maximum leverage for the algorithm. If not provided there will
#         be no maximum.
#     """
#     control = MaxLeverage(max_leverage)
#     self.register_account_control(control)
#
# @api_method
# def set_min_leverage(self, min_leverage, grace_period):
#     """Set a limit on the minimum leverage of the algorithm.
#
#     Parameters
#     ----------
#     min_leverage : float
#         The minimum leverage for the algorithm.
#     grace_period : pd.Timedelta
#         The offset from the start date used to enforce a minimum leverage.
#     """
#     deadline = self.sim_params.start_session + grace_period
#     control = MinLeverage(min_leverage, deadline)
#     self.register_account_control(control)

# def evaluate(self, positions, cache):
#     _impl = partial(self._evaluate_for_sid, metadata=cache)
#     # 执行退出算法
#     with Pool(processes=len(positions))as pool:
#         picker_votes = [pool.apply_async(_impl, position)
#                         for position in positions]
#         # selector --- position or False
#         selector = [vote for vote in picker_votes if vote]
#     return selector

"""
    trading_calendar: zipline.util._calendar.exchange_calendar.TradingCalendar
        The _calendar instance used to provide minute->session information.
    first_trading_day : pd.Timestamp
        The first trading day for the ArkQuant.
    equity_daily_reader : BcolzDailyBarReader, optional
        The daily bar reader for equities. This will be used to service
        daily data backtests or daily history calls in a minute backetest.
        If a daily bar reader is not provided but a minute bar reader is,
        the minutes will be rolled up to serve the daily requests.
    equity_minute_reader : BcolzMinuteBarReader, optional
        The minute bar reader for equities. This will be used to service
        minute data backtests or minute history calls. This can be used
        to serve daily calls if no daily bar reader is provided.
    adjustment_reader : SQLiteAdjustmentWriter, optional
        The adjustment reader. This is used to apply splits, dividends, and
        other adjustment data to the raw data from the readers.
"""

# def _handle_transaction(self, transaction):
#     asset = transaction.asset
#     try:
#         position = self.positions[asset]
#     except KeyError:
#         position = self.positions[asset] = Position(asset)
#     cash_flow = position.update(transaction)
#     if position.closed:
#         dts = transaction.created_dt.strftime('%Y-%m-%d')
#         self.record_closed_position[dts].append(position)
#         del self.positions[asset]
#     return cash_flow

# return callable fetch attr from its operant
# operator.attrgetter(return a callable that fetches attr from operant)
# operate.itemgetter (return a callable that uses method __getitem__())
# rsplit 从右往左 参数 sep(默认为所有空字符) count=count（sep）

# def attach_pipeline(self, pipe, name, chunks=None, eager=True):
#     """Register a Pipeline to be computed at the start of each day.
#
#     Parameters
#     ----------
#     pipe : Pipeline
#         The Pipeline to have computed.
#     name : str
#         The name of the Pipeline.
#     chunks : int or iterator, optional
#         The number of days to compute Pipeline results for. Increasing
#         this number will make it longer to get the first results but
#         may improve the total runtime of the ArkQuant. If an iterator
#         is passed, we will run in chunks based on values of the iterator.
#         Default is True.
#     eager : bool, optional
#         Whether or not to compute this Pipeline prior to
#         before_trading_start.
#
#     Returns
#     -------
#     Pipeline : Pipeline
#         Returns the Pipeline that was attached unchanged.
#
#     See Also
#     --------
#     :func:`zipline.api.pipeline_output`
#     """
#     if chunks is None:
#         chunks = chain([5], repeat(126))
#     elif isinstance(chunks, int):
#         chunks = repeat(chunks)
#
#     if name in self._pipelines:
#         raise DuplicatePipelineName(name=name)
#     return pipe

# def _sync_last_sale_prices(self, dt=None):
#     """Sync the last sale prices on the metric tracker to a given
#     datetime.
#
#     Parameters
#     ----------
#     dt : datetime
#         The time to sync the prices to.
#
#     Notes
#     -----
#     This call is cached by the datetime. Repeated calls in the same bar
#     are cheap.
#     """
#     if dt is None:
#         dt = self.datetime
#
#     if dt != self._last_sync_time:
#         self.metrics_tracker.sync_last_sale_prices(
#             dt,
#             self.data_portal,
#         )
#         self._last_sync_time = dt

# @property
# def portfolio(self):
#     self._sync_last_sale_prices()
#     return self.metrics_tracker.portfolio
#
# @property
# def account(self):
#     self._sync_last_sale_prices()
#     return self.metrics_tracker.account

# # 根据dt获取change,动态计算，更新数据
# def calculate_capital_changes(self, dt, emission_rate, is_interday,
#                               portfolio_value_adjustment=0.0):
#     """
#     If there is a capital change for a given dt, this means the the change
#     occurs before `handle_data` on the given dt. In the case of the
#     change being a target value, the change will be computed on the
#     portfolio value according to prices at the given dt
#
#     `portfolio_value_adjustment`, if specified, will be removed from the
#     portfolio_value of the cumulative performance when calculating deltas
#     from target capital changes.
#     """
#     try:
#         capital_change = self.capital_changes[dt]
#     except KeyError:
#         return
#
#     self._sync_last_sale_prices()
#     if capital_change['type'] == 'target':
#         target = capital_change['value']
#         capital_change_amount = (
#             target -
#             (
#                 self.portfolio.portfolio_value -
#                 portfolio_value_adjustment
#             )
#         )
#
#         logging.log.info('Processing capital change to target %s at %s. Capital '
#                  'change delta is %s' % (target, dt,
#                                          capital_change_amount))
#     elif capital_change['type'] == 'delta':
#         target = None
#         capital_change_amount = capital_change['value']
#         logging.log.info('Processing capital change of delta %s at %s'
#                  % (capital_change_amount, dt))
#     else:
#         logging.log.error("Capital change %s does not indicate a valid type "
#                   "('target' or 'delta')" % capital_change)
#         return
#
#     self.capital_change_deltas.update({dt: capital_change_amount})
#     self.metrics_tracker.capital_change(capital_change_amount)
#
#     yield {
#         'capital_change':
#             {'date': dt,
#              'type': 'cash',
#              'target': target,
#              'delta': capital_change_amount}
#     }
from copy import copy
from datetime import tzinfo
import logging

# @api_method
# @preprocess(tz=coerce_string(pytz.timezone))
# @expect_types(tz=optional(tzinfo))
# def get_datetime(self, tz=None):
#     """
#     Returns the current ArkQuant datetime.
#
#     Parameters
#     ----------
#     tz : tzinfo or str, optional
#         The timezone to return the datetime in. This defaults to utc.
#
#     Returns
#     -------
#     dt : datetime
#         The current ArkQuant datetime converted to ``tz``.
#     """
#     dt = self.datetime
#     assert dt.tzinfo == pytz.utc, "algorithm should have a utc datetime"
#     if tz is not None:
#         dt = dt.astimezone(tz)
#     return dt

# control = RestrictedListOrder(on_error, restrictions)
# self.register_trading_control(control)
# self.restrictions |= restrictions

# sa.ForeignKey(equity_basics.c.sid),

# grouped_by_sid = source_df.groupby(["sid"])
# group_names = grouped_by_sid.groups.keys()
# group_dict = {}
# for group_name in group_names:
#     group_dict[group_name] = grouped_by_sid.get_group(group_name)
# for col_name in df.columns.difference(['sid']):

"""
Construction of sentinel objects.

Sentinel objects are used when you only care to check for object identity.
"""
import sys
from textwrap import dedent


class _Sentinel(object):
    """Base class for Sentinel objects.
    """
    __slots__ = ('__weakref__',)


def is_sentinel(obj):
    return isinstance(obj, _Sentinel)


# 返回 目标的具体信息文件名、行号基于_getframe
def sentinel(name, doc=None):
    try:
        value = sentinel._cache[name]  # memoized
    except KeyError:
        pass
    else:
        if doc == value.__doc__:
            return value

        raise ValueError(dedent(
            """\
            New sentinel value %r conflicts with an existing sentinel of the
            same name.
            Old sentinel docstring: %r
            New sentinel docstring: %r

            The old sentinel was created at: %s

            Resolve this conflict by changing the name of one of the sentinels.
            """,
        ) % (name, value.__doc__, doc, value._created_at))

    try:
        frame = sys._getframe(1)
    except ValueError:
        frame = None

    if frame is None:
        created_at = '<unknown>'
    else:
        created_at = '%s:%s' % (frame.f_code.co_filename, frame.f_lineno)

    @object.__new__   # bind a single instance to the name 'Sentinel'
    class Sentinel(_Sentinel):
        __doc__ = doc
        __name__ = name

        # store created_at so that we can report this in case of a duplicate
        # name violation
        _created_at = created_at

        def __new__(cls):
            raise TypeError('cannot create %r instances' % name)

        def __repr__(self):
            return 'sentinel(%r)' % name

        def __reduce__(self):
            return sentinel, (name, doc)

        def __deepcopy__(self, _memo):
            return self

        def __copy__(self):
            return self

    cls = type(Sentinel)
    try:
        cls.__module__ = frame.f_globals['__name__']
    except (AttributeError, KeyError):
        # Couldn't get the name from the calling scope, just use None.
        # AttributeError is when frame is None, KeyError is when f_globals
        # doesn't hold '__name__'
        cls.__module__ = None

    sentinel._cache[name] = Sentinel  # cache result
    return Sentinel


sentinel._cache = {}

# 字典键值对转换
def _invert(d):
    return dict(zip(d.values(), d.keys()))

handler = StreamHandler(sys.stdout, format_string=" | {record.message}")
logger = Logger(__name__)
logger.handlers.append(handler)

if not csvdir:
    csvdir = environ.get('CSVDIR')
    if not csvdir:
        raise ValueError("CSVDIR environment variable is not set")

if not os.path.isdir(csvdir):
    raise ValueError("%s is not a directory" % csvdir)


# def maybe_create_close_position_transaction(self, asset):
#     """强制平仓机制 --- 持仓特定标的的仓位"""
#     raise NotImplementedError('automatic operation')

# def manual_withdraw_operation(self, assets):
#     """
#         self.position_tracker.maybe_create_close_position_transaction
#         self.process_transaction(txn)
#     """
#     warnings.warn('avoid interupt automatic process')
#     self.position_tracker.maybe_create_close_position_transaction(assets)

# def copy_process_env(self):
#     """为子进程拷贝主进程中的设置执行，在add_process_env_sig装饰器中调用，外部不应主动使用"""
#     for module in self.register_module():
#         # 迭代注册了的需要拷贝内存设置的模块, 筛选模块中以g_或者_g_开头的, 且不能callable，即不是方法
#         sig_env = list(filter(
#             lambda sig: not callable(sig) and (sig.startswith('g_') or sig.startswith('_g_')), dir(module)))
#         module_name = module.__name__
#         for _sig in sig_env:
#             # 格式化类变量中对应模块属性的key
#             name = '{}_{}'.format(module_name, _sig)
#             # 根据应模块属性的key（name）getattr获取属性值
#             val = getattr(self, name)
#             # 为子模块内存变量进行值拷贝
#             module.__dict__[_sig] = val

# def add_process_env_sig(func):
#     """
#     初始化装饰器时给被装饰函数添加env关键字参数，在wrapper中将env对象进行子进程copy
#     由于要改方法签名，多个装饰器的情况要放在最下面
#     :param func:
#     :return:
#     """
#     # 获取原始函数参数签名，给并行方法添加env参数
#     sig = signature(func)
#
#     if 'env' not in list(sig.parameters.keys()):
#         parameters = list(sig.parameters.values())
#         # 通过强制关键字参数，给方法加上env
#         parameters.append(Parameter('env', Parameter.KEYWORD_ONLY, default=None))
#         # wrapper的__signature__进行替换
#         wrapper.__signature__ = sig.replace(parameters=parameters)
#
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         # env = kwargs.pop('env', None)
#         if 'env' in kwargs:
#             """
#                 实际上linux, mac os上并不需要进行进程间模块内存拷贝，
#                 子进程fork后携带了父进程的内存信息，win上是需要的，
#                 暂时不做区分，都进行进程间的内存拷贝，如特别在乎效率的
#                 情况下基于linux系统，mac os可以不需要拷贝，如下：
#                 if kwargs['env'] is not None and not ABuEnv.g_is_mac_os:
#                     # 只有windows进行内存设置拷贝
#                     env.copy_process_env()
#             """
#             # if kwargs['env'] is not None and not ABuEnv.g_is_mac_os:
#             env = kwargs.pop('env', None)
#             if env is not None:
#                 # 将主进程中的env拷贝到子进程中
#                 env.copy_process_env()
#         return func(*args, **kwargs)
#
#     return wrapper

# def compute(self, today, assets, out, data):
#     drawdowns = fmax.accumulate(data, axis=0) - data
#     drawdowns[isnan(drawdowns)] = NINF
#     drawdown_ends = nanargmax(drawdowns, axis=0)
#
#     # TODO: Accelerate this loop in Cython or Numba.
#     for i, end in enumerate(drawdown_ends):
#         peak = nanmax(data[:end + 1, i])
#         out[i] = (peak - data[end, i]) / data[end, i]

# def function_application(func):
#     """
#     Factory function for producing function application methods for Factor
#     subclasses.
#     """
#     if func not in NUMEXPR_MATH_FUNCS:
#         raise ValueError("Unsupported mathematical function '%s'" % func)
#
#     docstring = dedent(
#         """\
#         Construct a Factor that computes ``{}()`` on each output of ``self``.
#
#         Returns
#         -------
#         factor : zipline.pipe.Factor
#         """.format(func)
#     )
#
#     @with_doc(docstring)
#     @with_name(func)
#     def mathfunc(self):
#         if isinstance(self, NumericalExpression):
#             return NumExprFactor(
#                 "{func}({expr})".format(func=func, expr=self._expr),
#                 self.inputs,
#                 dtype=float64_dtype,
#             )
#         else:
#             return NumExprFactor(
#                 "{func}(x_0)".format(func=func),
#                 (self,),
#                 dtype=float64_dtype,
#             )
#     return mathfunc

# with_missing = pd.Series(
#     data=pd.Categorical(
#         result.values,
#         result.values.categories.union([self.missing_value]),
#     ),
#     index=result.index,
# )


def _compute(self, arrays, dates, assets, mask):
    data = arrays[0]
    bins = self.params['bins']
    to_bin = where(mask, data, nan)
    result = quantiles(to_bin, bins)
    # Write self.missing_value into nan locations, whether they were
    # generated by our input mask or not.
    result[isnan(result)] = self.missing_value
    return result.astype(int64_dtype)


class PeerCount(SingleInputMixin, CustomFactor):
    """
    Peer Count of distinct categories in a given classifier.  This factor
    is returned by the classifier instance method peer_count()

    **Default Inputs:** None

    **Default Window Length:** 1
    """
    window_length = 1

    def _validate(self):
        super(PeerCount, self)._validate()
        if self.window_length != 1:
            raise ValueError(
                "'PeerCount' expected a window length of 1, but was given"
                "{window_length}.".format(window_length=self.window_length)
            )

    def compute(self, today, assets, out, classifier_values):
        # Convert classifier array to group label int array
        group_labels, null_label = self.inputs[0]._to_integral(
            classifier_values[0]
        )
        _, inverse, counts = unique(  # Get counts, idx of unique groups
            group_labels,
            return_counts=True,
            return_inverse=True,
        )
        # Copies values from one array to another, broadcasting as necessary.
        copyto(out, counts[inverse], where=(group_labels != null_label))

# gdp统计数据水分太多
# class Temperature(BaseFeature):
#     """
#         市值于GDP比率（判断市场是否过热）
#     """
#     @classmethod
#     def calc_feature(cls, feed, kwargs):
#         gdp = feed.copy()
#         market_value = gdp['mkv'].sum(axis = 1)
#         index = {'第一季度':'0330','第二季度':'0630','第三季度':'0930','第四季度':'1230'}
#         gdp = raw['gdp']
#         gdp.index = [i.replace(index) for i in gdp.index]
#         ratio_windowed = market_value.rolling(window).mean() /raw['gdp']
#         return ratio_windowed
# from numbers import Number
# from math import ceil
# from textwrap import dedent
# from scipy.stats import rankdata

# class SMMA(BaseFeature):
#     """
#         平滑移动平均线(SMMA)
#         SMMA1 = SUM(CLOSE(i), N) / N
#         经过运算转换公式可以简化为：SMMA (i) = (SMMA (i - 1) * (N - 1) + CLOSE (i)) / N
#     """
#     @classmethod
#     def _calc_feature(cls, feed, kwargs):
#         raw = feed.copy()
#         init_smma = raw.rolling(window = window).mean()
#         wgt = (window - 1)/window
#         smma = raw * (1-wgt) + init_smma * wgt
#         return smma
# uint8_dtype = dtype('uint8')

# uint32_dtype = dtype('uint32')
# uint64_dtype = dtype('uint64')
# int64_dtype = dtype('int64')
#
# float32_dtype = dtype('float32')
# float64_dtype = dtype('float64')
#
# complex128_dtype = dtype('complex128')
#
# datetime64D_dtype = dtype('datetime64[D]')
# datetime64ns_dtype = dtype('datetime64[ns]')
#
# object_dtype = dtype('O')
# # We use object arrays for strings.
# categorical_dtype = object_dtype
#
# make_datetime64ns = flip(datetime64, 'ns')
# make_datetime64D = flip(datetime64, 'D')

# CLASSIFIER_DTYPES = frozenset({object_dtype, int64_dtype})
# FACTOR_DTYPES = frozenset({datetime64ns_dtype, float64_dtype, int64_dtype})


# class FastStochasticOscillator(CustomFactor):
#     """
#     Fast Stochastic Oscillator Indicator [%K, Momentum Indicator]
#     https://wiki.timetotrade.eu/Stochastic
#
#     This stochastic is considered volatile, and varies a lot when used in
#     market analysis. It is recommended to use the slow stochastic oscillator
#     or a moving average of the %K [%D].
#
#     **Default Inputs:** :data: `zipline.pipe.data.EquityPricing.close`
#                         :data: `zipline.pipe.data.EquityPricing.low`
#                         :data: `zipline.pipe.data.EquityPricing.high`
#
#     **Default Window Length:** 14
#
#     Returns
#     -------
#     out: %K oscillator
#     """
#     inputs = (EquityPricing.close, EquityPricing.low, EquityPricing.high)
#     window_safe = True
#     window_length = 14
#
#     def compute(self, today, assets, out, closes, lows, highs):
#
#         highest_highs = nanmax(highs, axis=0)
#         lowest_lows = nanmin(lows, axis=0)
#         today_closes = closes[-1]
#
#         evaluate(
#             '((tc - ll) / (hh - ll)) * 100',
#             local_dict={
#                 'tc': today_closes,
#                 'll': lowest_lows,
#                 'hh': highest_highs,
#             },
#             global_dict={},
#             out=out,
#         )
#
# @singleton
# class Ignore(object):
#     def __str__(self):
#         return 'Argument.ignore'
#     __repr__ = __str__
#
#
# class Expired(Exception):
#     """Marks that a :class:`CachedObject` has expired.
#     """
# >>> from scipy.stats import rankdata
# >>> rankdata([0, 2, 3, 2])
# array([ 1. ,  2.5,  4. ,  2.5])
# >>> rankdata([0, 2, 3, 2], method='min')
# array([ 1,  2,  4,  2])
# >>> rankdata([0, 2, 3, 2], method='max')
# array([ 1,  3,  4,  3])
# >>> rankdata([0, 2, 3, 2], method='dense')
# array([ 1,  2,  3,  2])
# >>> rankdata([0, 2, 3, 2], method='ordinal')
# array([ 1,  2,  4,  3])

# if g_is_ipython and not g_is_py3:
#     """ipython在python2的一些版本需要reload logging模块，否则不显示log信息"""
#     # noinspection PyUnresolvedReferences, PyCompatibility
#     reload(logging)
#     # pass
from distutils.version import StrictVersion


pandas_version = StrictVersion(pd.__version__)
new_pandas = pandas_version >= StrictVersion('0.19')
if pandas_version >= StrictVersion('0.20'):
    def normalize_date(dt):
        """
        Normalize datetime.datetime value to midnight. Returns datetime.date as
        a datetime.datetime at midnight

        Returns
        -------
        normalized : datetime.datetime or Timestamp
        """
        return dt.normalize()
else:
    from pandas.tseries.tools import normalize_date

from numpy.lib.stride_tricks import as_strided
# def repeat_first_axis(array, count):
#     """
#     Restride `array` to repeat `count` times along the first axis.
#
#     Parameters
#     ----------
#     array : np.array
#         The array to restride.
#     count : int
#         Number of times to repeat `array`.
#
#     Returns
#     -------
#     result : array
#         Array of shape (count,) + array.shape, composed of `array` repeated
#         `count` times along the first axis.
#
#     Example
#     -------
#     >>> from numpy import arange
#     >>> a = arange(3); a
#     array([0, 1, 2])
#     >>> repeat_first_axis(a, 2)
#     array([[0, 1, 2],
#            [0, 1, 2]])
#     >>> repeat_first_axis(a, 4)
#     array([[0, 1, 2],
#            [0, 1, 2],
#            [0, 1, 2],
#            [0, 1, 2]])
#
#     Notes
#     ----
#     The resulting array will share memory with `array`.  If you need to assign
#     to the input or output, you should probably make a copy first.
#
#     See Also
#     --------
#     repeat_last_axis
#     """
#     return as_strided(array, (count,) + array.shape, (0,) + array.strides)
#
#
# def repeat_last_axis(array, count):
#     """
#     Restride `array` to repeat `count` times along the last axis.
#
#     Parameters
#     ----------
#     array : np.array
#         The array to restride.
#     count : int
#         Number of times to repeat `array`.
#
#     Returns
#     -------
#     result : array
#         Array of shape array.shape + (count,) composed of `array` repeated
#         `count` times along the last axis.
#
#     Example
#     -------
#     >>> from numpy import arange
#     >>> a = arange(3); a
#     array([0, 1, 2])
#     >>> repeat_last_axis(a, 2)
#     array([[0, 0],
#            [1, 1],
#            [2, 2]])
#     >>> repeat_last_axis(a, 4)
#     array([[0, 0, 0, 0],
#            [1, 1, 1, 1],
#            [2, 2, 2, 2]])
#
#     Notes
#     ----
#     The resulting array will share memory with `array`.  If you need to assign
#     to the input or output, you should probably make a copy first.
#
#     See Also
#     --------
#     repeat_last_axis
#     """
#     return as_strided(array, array.shape + (count,), array.strides + (0,))
#
#
# def rolling_window(array, length):
#     """
#     Restride an array of shape
#
#         (X_0, ... X_N)
#
#     into an array of shape
#
#         (length, X_0 - length + 1, ... X_N)
#
#     where each slice at index i along the first axis is equivalent to
#
#         result[i] = array[length * i:length * (i + 1)]
#
#     Parameters
#     ----------
#     array : np.ndarray
#         The base array.
#     length : int
#         Length of the synthetic first axis to generate.
#
#     Returns
#     -------
#     out : np.ndarray
#
#     Example
#     -------
#     >>> from numpy import arange
#     >>> a = arange(25).reshape(5, 5)
#     >>> a
#     array([[ 0,  1,  2,  3,  4],
#            [ 5,  6,  7,  8,  9],
#            [10, 11, 12, 13, 14],
#            [15, 16, 17, 18, 19],
#            [20, 21, 22, 23, 24]])
#
#     >>> rolling_window(a, 2)
#     array([[[ 0,  1,  2,  3,  4],
#             [ 5,  6,  7,  8,  9]],
#     <BLANKLINE>
#            [[ 5,  6,  7,  8,  9],
#             [10, 11, 12, 13, 14]],
#     <BLANKLINE>
#            [[10, 11, 12, 13, 14],
#             [15, 16, 17, 18, 19]],
#     <BLANKLINE>
#            [[15, 16, 17, 18, 19],
#             [20, 21, 22, 23, 24]]])
#     """
#     orig_shape = array.shape
#     if not orig_shape:
#         raise IndexError("Can't restride a scalar.")
#     elif orig_shape[0] <= length:
#         raise IndexError(
#             "Can't restride array of shape {shape} with"
#             " a window length of {len}".format(
#                 shape=orig_shape,
#                 len=length,
#             )
#         )
#
#     num_windows = (orig_shape[0] - length + 1)
#     new_shape = (num_windows, length) + orig_shape[1:]
#
#     new_strides = (array.strides[0],) + array.strides
#
#     return as_strided(array, new_shape, new_strides)
# Sentinel value that isn't NaT.
# _notNaT = make_datetime64D(0)

# out = np.full((len(all_dates), len(all_sids)), -1, dtype=np.int64)
#
# sid_ixs = all_sids.searchsorted(event_sids)
# # side='right' here ensures that we include the event date itself
# # if it's in all_dates.
# dt_ixs = all_dates.searchsorted(event_dates, side='right')
# ts_ixs = data_query_cutoff.searchsorted(event_timestamps, side='right')

# class Event(object):
#
#     def __init__(self, initial_values=None):
#         if initial_values:
#             self.__dict__.update(initial_values)
#
#     def keys(self):
#         return self.__dict__.keys()
#
#     def __eq__(self, other):
#         return hasattr(other, '__dict__') and self.__dict__ == other.__dict__
#
#     def __contains__(self, name):
#         return name in self.__dict__
#
#     def __repr__(self):
#         return "Event({0})".format(self.__dict__)
#
#     def to_series(self, index=None):
#         return pd.Series(self.__dict__, index=index)

# # shape: (N, M)
# ind_residual = independent - nanmean(independent, axis=0)
#
# # shape: (M,)
# covariances = nanmean(ind_residual * dependents, axis=0)
#
# # We end up with different variances in each column here because each
# # column may have a different subset of the data dropped due to missing
# # data in the corresponding dependent column.
# # shape: (M,)
# independent_variances = nanmean(ind_residual ** 2, axis=0)
#
# # shape: (M,)
# np.divide(covariances, independent_variances, out=out)
#
# # Write nans back to locations where we have more then allowed number of
# # missing entries.
# nanlocs = isnan(independent).sum(axis=0) > allowed_missing
# out[nanlocs] = nan

# # class ExponentialWeightedMovingStdDev(_ExponentialWeightedFactor):
# def compute(self, today, assets, out, data, decay_rate):
#     weights = exponential_weights(len(data), decay_rate)
#
#     mean = average(data, axis=0, weights=weights)
#     variance = average((data - mean) ** 2, axis=0, weights=weights)
#
#     squared_weight_sum = (np_sum(weights) ** 2)
#     bias_correction = (
#         squared_weight_sum / (squared_weight_sum - np_sum(weights ** 2))
#     )
#     out[:] = sqrt(variance * bias_correction)
#
#
# # class LinearWeightedMovingAverage(SingleInputMixin, CustomFactor):
# def compute(self, today, assets, out, data):
#     ndays = data.shape[0]
#
#     # Initialize weights array
#     weights = arange(1, ndays + 1, dtype=float64_dtype).reshape(ndays, 1)
#
#     # Compute normalizer
#     normalizer = (ndays * (ndays + 1)) / 2
#
#     # Weight the data
#     weighted_data = data * weights
#
#     # Compute weighted averages
#     out[:] = nansum(weighted_data, axis=0) / normalizer
#
#
# # class AnnualizedVolatility(CustomFactor):
# def compute(self, today, assets, out, returns, annualization_factor):
#     out[:] = nanstd(returns, axis=0) * (annualization_factor ** .5)

# def __hash__(self):
#     return id(self)
#
#
# def __contains__(self, column):
#     return column in self._table_expressions
#
#
# def __getitem__(self, column):
#     return self._table_expressions[column]
#
#
# def __iter__(self):
#     return iter(self._table_expressions)
#
#
# def __len__(self):
#     return len(self._table_expressions)
#
#
# def __call__(self, column):
#     if column in self:
#         return self
#     raise KeyError(column)

# getmtime --- 生成文件的时间

# def get_returns_cached(filepath, update_func, latest_dt, **kwargs):
#     """
#     Get returns from a cached file if the cache is recent enough,
#     otherwise, try to retrieve via a provided update function and
#     update the cache file.
#     Parameters
#     ----------
#     filepath : str
#         Path to cached csv file
#     update_func : function
#         Function to call in case cache is not up-to-date.
#     latest_dt : pd.Timestamp (tz=UTC)
#         Latest datetime required in csv file.
#     **kwargs : Keyword arguments
#         Optional keyword arguments will be passed to update_func()
#     Returns
#     -------
#     pandas.DataFrame
#         DataFrame containing returns
#     """
#
#     update_cache = False
#
#     try:
#         mtime = getmtime(filepath)
#     except OSError as e:
#         if e.errno != errno.ENOENT:
#             raise
#         update_cache = True
#     else:
#
#         file_dt = pd.Timestamp(mtime, unit='s')
#
#         if latest_dt.tzinfo:
#             file_dt = file_dt.tz_localize('utc')
#
#         if file_dt < latest_dt:
#             update_cache = True
#         else:
#             returns = pd.read_csv(filepath, index_col=0, parse_dates=True)
#             returns.index = returns.index.tz_localize("UTC")
#
#     if update_cache:
#         returns = update_func(**kwargs)
#         try:
#             ensure_directory(cache_dir())
#         except OSError as e:
#             warnings.warn(
#                 'could not update cache: {}. {}: {}'.format(
#                     filepath, type(e).__name__, e,
#                 ),
#                 UserWarning,
#             )
#
#         try:
#             returns.to_csv(filepath)
#         except OSError as e:
#             warnings.warn(
#                 'could not update cache {}. {}: {}'.format(
#                     filepath, type(e).__name__, e,
#                 ),
#                 UserWarning,
#             )
#     return returns
# mtime = getmtime(filepath)

# def last_modified_time(path):
#     """
#     Get the last modified time of path as a Timestamp.
#     """
#     return pd.Timestamp(os.path.getmtime(path), unit='s', tz='UTC')


# def load_prices_from_csv(filepath, identifier_col, tz='UTC'):
#     data = pd.read_csv(filepath, index_col=identifier_col)
#     data.index = pd.DatetimeIndex(data.index, tz=tz)
#     data.sort_index(inplace=True)
#     return data
#
#
# def load_prices_from_csv_folder(folder, identifier_col, tz='UTC'):
#     data = None
#     for file in os.listdir(folder):
#         if '.csv' not in file:
#             continue
#         raw = load_prices_from_csv(os.path.join(folder, file),
#                                    identifier_col, tz)
#         if data is None:
#             data = raw
#         else:
#             data = pd.concat([data, raw], axis=1)
#     return data


# def has_data_for_dates(series_or_df, first_date, last_date):
#     """
#     Does `series_or_df` have data on or before first_date and on or after
#     last_date?
#     """
#     dts = series_or_df.index
#     if not isinstance(dts, pd.DatetimeIndex):
#         raise TypeError("Expected a DatetimeIndex, but got %s." % type(dts))
#     first, last = dts[[0, -1]]
#     return (first <= first_date) and (last >= last_date)

# class Crawler(ABC):
#
#     @property
#     def metadata(self):
#         return MetaData(bind=engine)
#
#     @staticmethod
#     @lru_cache(maxsize=128)
#     def spider_proxies():
#         raw = _parse_url(ProxyIp, encoding='utf-8')
#         table = raw.find('table')
#         # ip
#         ip_item = [item.find('td', {'data-title': 'IP'}) for item in table.findAll('tr')]
#         ip = [ele.get_text() for ele in ip_item if ele]
#         # port
#         port_item = [item.find('td', {'data-title': 'PORT'}) for item in table.findAll('tr')]
#         port = [ele.get_text() for ele in port_item if ele]
#         # category
#         category_item = [item.find('td', {'data-title': '类型'}) for item in table.findAll('tr')]
#         category = [ele.get_text() for ele in category_item if ele]
#         # construct proxy
#         proxies = []
#         for item in zip(category, ip, port):
#             proxy = item[0].lower() + '://' + item[1] + ':' + item[-1]
#             proxies.append(proxy)
#         return proxies
#
#     @property
#     def proxy(self):
#         # proxies = self.spider_proxies()
#         proxies = None
#         proxy_func = partial(_parse_url, proxy=proxies)
#         return proxy_func
#
#     @abstractmethod
#     def writer(self, *args):
#         """
#             intend to spider data from online
#         :return:
#         """
#         raise NotImplementedError()
# ins = test()
# print(ins)
# print(namespace['__builtins__'])
# print(namespace['signature'])

# exec eval compile将字符串转化为可执行代码 , exec compile source into code or AST object ,if filename is None ,'<string>' is used
# code = compile(self.algoscript, algo_filename, 'exec')
# exec_(code, self.namespace)
#
# def noop(*args, **kwargs):
#     pass
#
#
# if algo_filename is None:
#     algo_filename = '<string>'
#     # exec eval compile将字符串转化为可执行代码 , exec compile source into code or AST object ,
#     # if filename is None ,'<string>' is used
#     code = compile(self.algoscript, algo_filename, 'exec')
#     exec(code, self.namespace)
#
#     # dict get参数可以为方法或者默认参数
#     self._initialize = self.namespace.get('initialize', noop)
#     self._handle_data = self.namespace.get('handle_data', noop)
#     self._before_trading_start = self.namespace.get(
#         'before_trading_start',
#     )
#     # Optional analyze function, gets called after run
#     self._analyze = self.namespace.get('analyze')
# else:
#     self._initialize = initialize or (lambda self: None)
#     self._handle_data = handle_data
#     self._before_trading_start = before_trading_start
#     self._analyze = analyze
namespace = dict()
with open('/Users/python/Library/Mobile Documents/com~apple~CloudDocs/ArkQuant/test/test_driver.py','r') as f:
    exec(f.read(), namespace)

print(namespace.keys())
test = namespace['UnionEngine']
print(test)

# from graphviz import Digraph,Graph
#
# h = Graph('hello', format='svg')
#
# h.edge('Hello', 'World')
#
# print(h.pipe().decode('utf-8'))


# import pandas as pd, numpy as np
# from gateway.driver.client import tsclient
#
# status = tsclient.to_ts_stats()
# print('status', status['list_date'].dtype)
#
# path = '/Users/python/Library/Mobile Documents/com~apple~CloudDocs/ArkQuant/gateway/spider/equity_basics.csv'
#
# frame = pd.read_csv(path, dtype={'代码': np.str, '发行价格': np.float, 'list_date': np.str})
# frame['发行价格'].fillna('0.00', inplace=True)
# frame['发行价格'].astype(np.float, copy=False)
# print('frame 32', frame.iloc[31, :])
# frame.set_index('代码', drop=False, inplace=True)
# frame['list_date'].fillna('', inplace=True)
# print(frame['list_date'])
# print(frame[frame['代码']=='000003']['list_date'])

# locals() 只读, globals() 可读可写


# if calendar is None:
#     cal = self.trading_calendar
# elif calendar is calendars.US_EQUITIES:
#     cal = get_calendar('XNYS')
# elif calendar is calendars.US_FUTURES:
#     cal = get_calendar('us_futures')

# asset_db_table_names = frozenset(['asset_router',  'equity_status', 'equity_basics', 'convertible_basics',
#                                   'equity_price', 'convertible_price', 'fund_price', 'equity_splits', 'equity_rights',
#                                   'ownership', 'holder', 'unfreeze', 'massive', 'm_cap', 'version_info'])

# metadata.reflect(bind=engine)

# current mappings   data: a nested dictionary: knowledge_date -> lookup_date ->
# {add: [symbol list], 'delete': []}, delete: [symbol list]}
#     def update_current(self, effective_date, symbols, change_func):

# assets = [asset for asset in chain(*self._asset_type_cache.values())
#           if asset.asset_type == category]

# def window_arrays(self, sessions, assets, field):
#     """
#     :param sessions: [a,b]
#     :param assets: Assets list
#     :param field: str or list
#     :return: arrays which is adjusted by divdends and rights
#     """
#     adjustments, raw_arrays, sessions = self._compatible_adjustment.calculate_adjustments_in_sessions(sessions, assets)
#     adjusted_fields = set(field) & AdjustFields
#     if adjusted_fields:
#         # 计算调整数据
#         adjust_arrays = {}
#         for asset in assets:
#             sid = asset.sid
#             qfq = adjustments[sid]
#             raw = raw_arrays[sid]
#             try:
#                 qfq = qfq.reindex(sessions)
#                 qfq.fillna(method='bfill', inplace=True)
#                 qfq.fillna(1.0, inplace=True)
#                 raw[adjusted_fields] = raw.loc[:, adjusted_fields].multiply(qfq, axis=0)
#             except Exception as e:
#                 print(e, asset)
#             adjust_arrays[sid] = raw
#     else:
#         adjust_arrays = raw_arrays
#     return adjust_arrays

# def get_fetcher_assets(self, sids):
#     """
#     Returns a list of asset for the current date, as defined by the
#     fetcher data.
#
#     Returns
#     -------
#     list: a list of Asset objects.
#     """
#     # return a list of asset for the current date, as defined by the
#     # fetcher source
#     found, missing = self.asset_finder.retrieve_asset(sids)
#     return found, missing
#
# def get_all_assets(self, asset_type=None):
#     all_assets = self.asset_finder.retrieve_all(asset_type)
#     return all_assets

# from functools import reduce
# int64_dtype = dtype('int64')
#
# bool_dtype = dtype('bool')
#
# bool_dtype = dtype('bool')
#
# FILTER_DTYPES = frozenset({bool_dtype})
#
#
# def make_kind_check(python_types, numpy_kind):
#     """
#     Make a function that checks whether a scalar or array is of a given kind
#     (e.g. float, int, datetime, timedelta).
#     """
#     def check(value):
#         if hasattr(value, 'dtype'):
#             return value.dtype.kind == numpy_kind
#         return isinstance(value, python_types)
#     return check
#
#
# is_float = make_kind_check(float, 'f')
# is_int = make_kind_check(int, 'i')
# is_datetime = make_kind_check(datetime, 'M')
# is_object = make_kind_check(object, 'O')
#
#
# def isnat(obj):
#     """
#     Check if a value is np.NaT.
#     """
#     if obj.dtype.kind not in ('m', 'M'):
#         raise ValueError("%s is not a numpy datetime or timedelta")
#     return obj.view(int64_dtype) == iNaT
#
# def is_missing(data, missing_value):
#     """
#     Generic is_missing function that handles NaN and NaT.
#     """
#     if is_float(data) and isnan(missing_value):
#         return isnan(data)
#     elif is_datetime(data) and isnat(missing_value):
#         return isnat(data)
#     return data == missing_value

# def _downsampled_type(self, *args, **kwargs):
#     """
#     The expression type to return from self.downsample().
#     """
#     raise NotImplementedError(
#         "downsampling is not yet implemented "
#         "for instances of %s." % type(self).__name__
#     )
#
# def downsample(self, frequency):
#     """
#     Make a term that computes from ``self`` at lower-than-daily frequency.
#
#     Parameters
#     ----------
#     {frequency}
#     """
#     return self._downsampled_type(term=self, frequency=frequency)

# import os, glob
#
# p = os.path.abspath('__file__')
# print('p', p)
# dir = os.path.dirname(p)
# print('dir', dir)
# base = os.path.basename(p)
# print('base', base)
# p_dir = os.getcwd()
# print('now directory', p_dir)
# test = os.path.split(os.getcwd())
# print('test', test)
# target = os.path.join(os.path.split(os.getcwd())[0], 'strat')
# print('target', target)
# files = os.path.join(target, '*')
# p = glob.glob(target + os.sep + 'cross.py')
# print('p', p)
# files = glob.glob('strat/*')
# print('files', files)
# LRU 缓存只在当你想要重用之前计算的结果时使用
from functools import lru_cache

# chop off any minutes or hours on the given start and end dates,
# as we only support session labels here (and we represent session
# labels as midnight UTC).
# self._start_session = normalize_date(start_session)
# self._end_session = normalize_date(end_session)
import pytz, numbers
from datetime import datetime
from zipline.protocol import DATASOURCE_TYPE


def assert_datasource_protocol(event):
    """Assert that an event meets the protocol for datasource outputs."""
    assert event.type in DATASOURCE_TYPE

    # Done packets have no dt.
    if not event.type == DATASOURCE_TYPE.DONE:
        assert isinstance(event.dt, datetime)
        assert event.dt.tzinfo == pytz.utc


def assert_trade_protocol(event):
    """Assert that an event meets the protocol for datasource TRADE outputs."""
    assert_datasource_protocol(event)

    assert event.type == DATASOURCE_TYPE.TRADE
    assert isinstance(event.price, numbers.Real)
    assert isinstance(event.volume, numbers.Integral)
    assert isinstance(event.dt, datetime)


def assert_datasource_unframe_protocol(event):
    """Assert that an event is valid output of zp.DATASOURCE_UNFRAME."""
    assert event.type in DATASOURCE_TYPE


from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def _fit_sklearn(x, y):
    reg = LinearRegression(fit_intercept=False).fit(x, y)
    # reg.intercept_
    coef = reg.coef_
    return coef


def _fit_statsmodel(x, y):
    # statsmodels.regression.linear_model  intercept = model.params[0]，rad = model.params[1]
    X = sm.add_constant(x)
    #const coef
    res = sm.OLS(y, X).fit()
    return res[-1]


# def load_divdends_for_sid(self, sid, date):
#     sql_dialect = sa.select([self.equity_splits.c.ex_date,
#                              sa.cast(self.equity_splits.c.sid_bonus, sa.Numeric(5, 2)),
#                              sa.cast(self.equity_splits.c.sid_transfer, sa.Numeric(5, 2)),
#                              sa.cast(self.equity_splits.c.bonus, sa.Numeric(5, 2))]).\
#                             where(sa.and_(self.equity_splits.c.sid == sid,
#                                   self.equity_splits.c.progress.like('实施'),
#                                   self.equity_splits.c.pay_date == date))
#     rp = self.engine.execute(sql_dialect)
#     dividends = pd.DataFrame(rp.fetchall(), columns=['ex_date', 'sid_bonus',
#                                                      'sid_transfer', 'bonus'])
#     adjust_divdends = self._adjust_frame_type(dividends)
#     return adjust_divdends
#
# def load_rights_for_sid(self, sid, date):
#     sql = sa.select([self.equity_rights.c.ex_date,
#                      sa.cast(self.equity_rights.c.rights_bonus, sa.Numeric(5, 2)),
#                      sa.cast(self.equity_rights.c.rights_price, sa.Numeric(5, 2))]).\
#                     where(sa.and_(self.equity_rights.c.sid == sid,
#                                   self.equity_rights.c.pay_date == date))
#     rp = self.engine.execute(sql)
#     rights = pd.DataFrame(rp.fetchall(), columns=['ex_date', 'right_bonus', 'right_price'])
#     adjust_rights = self._adjust_frame_type(rights)
#     return adjust_rights

# def handle_splits(self, dts):
#     total_left_cash = 0
#     dividends = portal.get_dividends(set(self.positions), dts)
#     for asset, position in self.positions.items():
#         # update last_sync_date
#         position.last_sync_date = dts
#         try:
#             amount_ratio, cash_ratio = self._calculate_adjust_ratio(dividends.loc[asset.sid, :])
#             left_cash = position.handle_split(amount_ratio, cash_ratio)
#             total_left_cash += left_cash
#         except KeyError:
#             pass
#     return total_left_cash

# @staticmethod
# def resolve_pipeline_final(outputs):
#     mappings = dict()
#     for item in outputs:
#         mappings.update(item)
#
#     # to find out the final asset of each pipe , notice ---  NamedPipe list
#     # group by pipe name
#     sample_by_pipe = groupby(lambda x: x.name, outputs)
#     # priority 0 --- high ,diminish by increase number
#     group_sorted = valmap(lambda x: x.sort(key=lambda k: k.priority), sample_by_pipe)
#     # namedPipe --- event priority
#     finals = valmap(lambda x: x[0].asset if x else None, group_sorted)
#     return finals

# 不同的pipeline 可以相同持仓，但是不一定同一时间卖出
# 其他的pipeline产生相同的持仓，持仓动态变动，所以只能做收盘的分析 或者每个minute产生一个account view

# trading controls --- List of account controls to be checked on each bar
# cancel_policy = cancel_policy or NeverCancel()
# List of trading controls to be used to validate orders.
# trading_controls = control or [MaxOrderSize, MaxPositionSize]
# List of account controls to be checked on each bar.
# self.account_controls = []

# broker --- combine pipe_engine and blotter ; when reality trading broker ---- xtp
# set manual risk management --- manual close positions
# risk_manual = risk_manual or PortfolioRisk
# self.manual_controls = Manual(risk_manual)
# metrics_set and initialize metric tracker


# @api_method
# def set_cancel_policy(self, cancel_policy):
#     """Sets the order cancellation policy for the ArkQuant.
#
#     Parameters
#     ----------
#     cancel_policy : CancelPolicy
#         The cancellation policy to use.
#
#     See Also
#     --------
#     :class:`zipline.api.EODCancel`
#     :class:`zipline.api.NeverCancel`
#     """
#     if not isinstance(cancel_policy, CancelPolicy):
#         raise UnsupportedCancelPolicy()
#
#     if self.initialized:
#         raise SetCancelPolicyPostInit()
#
#     self.blotter.cancel_policy = cancel_policy


# class LongOnly(TradingControl):
#     """
#     TradingControl representing a prohibition against holding short positions.
#     """
#     def __init__(self, on_error='fail'):
#         self.on_error = on_error
#         self.__fail_args = 'short action is not allowed'
#
#     def validate_call(self,
#                       asset,
#                       capital,
#                       portfolio,
#                       algo_datetime):
#         super().validate_call(asset,
#                               capital,
#                               portfolio,
#                               algo_datetime)
#
#     def validate_put(self,
#                      position,
#                      portfolio,
#                      algo_datetime):
#         """
#         Fail if we would hold negative shares of asset after completing this order.
#         """
#         super().validate_put(position,
#                              portfolio,
#                              algo_datetime)

# futures_metadata : dict or DataFrame or file-like object, optional
#     The same layout as ``equities_metadata`` except that it is used
#     for futures information.
# identifiers : list, optional
#     Any asset identifiers that are not provided in the
#     equities_metadata, but will be traded by this TradingAlgorithm.
# get_pipeline_loader : callable[BoundColumn -> pipe], optional
#     The function that maps Pipeline columns to their loaders.

"""A class that represents a trading strategy and parameters to execute
    the strategy.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to ``initialize`` unless listed below.
    initialize : callable[context -> None], optional
        Function that is called at the start of the ArkQuant to
        setup the initial context.
    handle_data : callable[(context, data) -> None], optional
        Function called on every bar. This is where most logic should be
        implemented.
    before_trading_start : callable[(context, data) -> None], optional
        Function that is called before any bars have been processed each
        day.
    analyze : callable[(context, DataFrame) -> None], optional
        Function that is called at the end of the backtest. This is passed
        the context and the performance results for the backtest.
    script : str, optional
        Algoscript that contains the definitions for the four algorithm
        lifecycle functions and any supporting code.
    namespace : dict, optional
        The namespace to execute the algoscript in. By default this is an
        empty namespace that will include only python built ins.
    algo_filename : str, optional
        The filename for the algoscript. This will be used in exception
        tracebacks. default: '<string>'.

    equities_metadata : dict or DataFrame or file-like object, optional
        If dict is provided, it must have the following structure:
        * keys are the identifiers
        * values are dicts containing the metadata, with the metadata
          field name as the key
        If pandas.DataFrame is provided, it must have the
        following structure:
        * column names must be the metadata fields
        * index must be the different asset identifiers
        * array contents should be the metadata value
        If an object with a ``read`` method is provided, ``read`` must
        return rows containing at least one of 'sid' or 'symbol' along
        with the other metadata fields.
    create_event_context : callable[BarData -> context manager], optional
        A function used to create a context mananger that wraps the
        execution of all events that are scheduled for a bar.
        This function will be passed the data for the bar and should
        return the actual context manager that will be entered.
    history_container_class : type, optional
        The type of history container to use. default: HistoryContainer
    platform : str, optional
        The platform the ArkQuant is running on. This can be queried for
        in the ArkQuant with ``get_environment``. This allows algorithms
        to conditionally execute code based on platform it is running on.
        default: 'zipline'

    量化交易系统:
        a.策略识别（搜索策略 ， 挖掘优势 ， 交易频率）
        b.回溯测试（获取数据 ， 分析策略性能 ，剔除偏差）
        c.交割系统（经纪商接口 ，交易自动化 ， 交易成本最小化）
        d.风险管理（最优资本配置 ， 最优赌注或者凯利准则 ， 海龟仓位管理）
"""

# base_dir = '/Users/python/Library/Mobile Documents/com~apple~CloudDocs/ArkQuant/strat'

# @staticmethod
# def resolve_pipeline_final(outputs):
#     group_sorted = dict()
#     for item in outputs:
#         group_sorted.update(item)
#     finals = valmap(lambda x: x[0], group_sorted)
#     return finals


# class CancelPolicy(ABC):
#     """
#         Abstract cancellation policy interface.
#         --- manual interface
#     """
#     @abstractmethod
#     def should_cancel(self, asset):
#         """Should open order be cancelled
#         Returns
#         -------
#         should_cancel : bool
#         """
#         raise NotImplementedError()
#
#
# class NoCancel(CancelPolicy):
#     """Orders are never automatically canceled.
#     """
#
#     def __init__(self):
#         self.warn_on_cancel = False
#
#     def should_cancel(self, asset):
#         return False
#
#
# class EODCancel(CancelPolicy):
#     """
#         This policy cancels open orders which created dt in session of last_traded and eod_window
#         --- 取消标的退市之前的一段时间的内订单
#     """
#     def __init__(self, window):
#         """
#         :param window: int
#         """
#         self.eod_window = window
#
#     def should_cancel(self, asset):
#         last_traded = asset.last_traded
#         previous = calendar.dt_window_size(last_traded, self.eod_window)
#         return previous <= last_traded.strftime('%Y-%m-%d')
#
#
# class ExtraCancel(CancelPolicy):
#     """
#         the policy cancel order which order asset is suffer negative affairs  --- black swat
#     """
#
#
# class ComposedCancel(CancelPolicy):
#     """
#      compose rules with some composing function
#     """
#     def __init__(self, policies):
#
#         if not np.all([isinstance(p, CancelPolicy) for p in policies]):
#             raise ValueError('only cancel policy can be composed')
#         self.sub_policies = policies
#
#     def should_cancel(self, asset):
#         return np.all([p.shoud_cancel(asset) for p in self.sub_policies])
#
# def protect(cls):
#     def handler(signum, frame):
#         print(signum, frame)
#         raise SystemExit
#
#     while True:
#         # signal.signal(signal.SIGINT, handler)
#         signal.signal(signal.SIGINT, signal.SIG_IGN)
#         # signal.signal(signal.SIGINT, signal.SIG_DFL)
#
# def protect(cls):
#     def handler(signum, frame):
#         print(signum, frame)
#         print('now time', time.time())
#         time.sleep(10)
#         # raise SystemExit
#
#     signal.signal(signal.SIGALRM, handler)
#     signal.alarm(1)
#     while True:
#         print('test')
#
# def protect(cls):
#     # Define signal handler function
#     def myHandler(signum, frame):
#         print('I received: ', signum)
#
#     # register signal.SIGTSTP's handler
#     signal.signal(signal.SIGINT, myHandler)
#     signal.pause()
#     print('End of Signal Demo')
"""
       name : str
            The name of the pipeline.
        chunks : int or iterator, optional
            The number of days to compute pipeline results for. Increasing
            this number will make it longer to get the first results but
            may improve the total runtime of the simulation. If an iterator
            is passed, we will run in chunks based on values of the iterator.
            Default is True.
        eager : bool, optional
            Whether or not to compute this pipeline prior to
            before_trading_start.
"""

# def _validate_benchmark(self, benchmark_asset):
#     # check if this security has a stock dividend.  if so, raise an
#     # error suggesting that the user pick a different asset to use
#     # as benchmark.
#     stock_dividends = \
#         self.data_portal.get_stock_dividends(self.benchmark_asset,
#                                              self.sessions)
#
#     if len(stock_dividends) > 0:
#         raise InvalidBenchmarkAsset(
#             sid=str(self.benchmark_asset),
#             dt=stock_dividends[0]["ex_date"]
#         )
#
#     if benchmark_asset.start_date > self.sessions[0]:
#         # the asset started trading after the first simulation day
#         raise BenchmarkAssetNotAvailableTooEarly(
#             sid=str(self.benchmark_asset),
#             dt=self.sessions[0],
#             start_dt=benchmark_asset.start_date
#         )
#
#     if benchmark_asset.end_date < self.sessions[-1]:
#         # the asset stopped trading before the last simulation day
#         raise BenchmarkAssetNotAvailableTooLate(
#             sid=str(self.benchmark_asset),
#             dt=self.sessions[-1],
#             end_dt=benchmark_asset.end_date
#         )
# ----------- asset deadlines
#
# class AssetDateBounds(TradingControl):
#     """
#     TradingControl representing a prohibition against ordering an asset before
#     its start_date, or after its end_date.
#     """
#
#     def __init__(self, on_error):
#         super(AssetDateBounds, self).__init__(on_error)
#
#     def validate(self,
#                  asset,
#                  amount,
#                  portfolio,
#                  algo_datetime,
#                  algo_current_data):
#         """
#         Fail if the algo has passed this Asset's end_date, or before the
#         Asset's start date.
#         """
#         # If the order is for 0 shares, then silently pass through.
#         if amount == 0:
#             return
#
#         normalized_algo_dt = pd.Timestamp(algo_datetime).normalize()
#
#         # Fail if the algo is before this Asset's start_date
#         if asset.start_date:
#             normalized_start = pd.Timestamp(asset.start_date).normalize()
#             if normalized_algo_dt < normalized_start:
#                 metadata = {
#                     'asset_start_date': normalized_start
#                 }
#                 self.handle_violation(
#                     asset, amount, algo_datetime, metadata=metadata)
#         # Fail if the algo has passed this Asset's end_date
#         if asset.end_date:
#             normalized_end = pd.Timestamp(asset.end_date).normalize()
#             if normalized_algo_dt > normalized_end:
#                 metadata = {
#                     'asset_end_date': normalized_end
#                 }
#                 self.handle_violation(
#                     asset, amount, algo_datetime, metadata=metadata)

# @expect_types(
#     __funcname='MinLeverage',
#     min_leverage=(int, float),
#     deadline=datetime
# )
# @expect_bounded(__funcname='MinLeverage', min_leverage=(0, None))

# class MaxLeverage(AccountControl):
#     """
#     AccountControl representing a limit on the maximum leverage allowed
#     by the algorithm.
#     """
#
#     def __init__(self, max_leverage):
#         """
#         max_leverage is the gross leverage in decimal form. For example,
#         2, limits an algorithm to trading at most double the account value.
#         """
#         super(MaxLeverage, self).__init__(max_leverage=max_leverage)
#         self.max_leverage = max_leverage
#
#         if max_leverage is None:
#             raise ValueError(
#                 "Must supply max_leverage"
#             )
#
#         if max_leverage < 0:
#             raise ValueError(
#                 "max_leverage must be positive"
#             )
#
#     def validate(self,
#                  portfolio,
#                  account,
#                  algo_datetime):
#         """
#         Fail if the leverage is greater than the allowed leverage.
#         """
#         if account.leverage > self.max_leverage:
#             self.fail()
#
#
# class MinLeverage(AccountControl):
#     """AccountControl representing a limit on the minimum leverage allowed
#     by the algorithm after a threshold period of time.
#
#     Parameters
#     ----------
#     min_leverage : float
#         The gross leverage in decimal form.
#     deadline : datetime
#         The date the min leverage must be achieved by.
#
#     For example, min_leverage=2 limits an algorithm to trading at minimum
#     double the account value by the deadline date.
#     """
#     def __init__(self, min_leverage, deadline):
#         super(MinLeverage, self).__init__(min_leverage=min_leverage,
#                                           deadline=deadline)
#         self.min_leverage = min_leverage
#         self.deadline = deadline
#
#     def validate(self,
#                  portfolio,
#                  account,
#                  algo_datetime):
#         """
#         Make validation checks if we are after the deadline.
#         Fail if the leverage is less than the min leverage.
#         """
#         if (algo_datetime > self.deadline and
#                 account.leverage < self.min_leverage):
#             self.fail()
# __func__ ---指向函数对象

# max_capital = portfolio.portfolio_value * (self.max_notional - weights)
#     agg_shares = current_share + amount
#     price = portfolio.positions[asset].last_sync_price
#     multiplier = np.floor(max_capital / (price * asset.tick_size))
#     amount = asset.tick_size * multiplier
#
# if weights[asset.sid] >= self.max_notional and amount > 0:
#     self.handle_violation(asset, amount, algo_datetime)
#     amount = 0
# else:
#     max_capital = portfolio.portfolio_value * (self.max_notional - weights)
#     price = portfolio.positions[asset].last_sync_price
#     multiplier = np.floor(max_capital / (price * asset.tick_size))
#     amount = asset.tick_size * multiplier


# class MaxPositionSize(TradingControl):
#     """
#     TradingControl representing a limit on the magnitude of any single order
#     placed with the given asset.  Can be specified by share or by dollar
#     value. 深圳ST股票买入卖出都不受限制，上海买入限制50万股，卖出没有限制
#     """
#
#     def __init__(self,
#                  max_notional,
#                  sliding_window=1,
#                  on_error='log',
#                  _fail_args='position amount exceed'):
#         super(MaxPositionSize, self).__init__(
#                                             on_error=on_error,
#                                             _fail_args=_fail_args)
#         self.max_notional = max_notional
#         self.window = sliding_window
#
#     def validate(self,
#                  asset,
#                  amount,
#                  portfolio,
#                  algo_datetime):
#         """
#         Fail if the magnitude of the given order exceeds either self.max_shares
#         or self.max_notional.
#         """
#         try:
#             p = portfolio.positions[asset]
#             current_share = p.amount
#             sync_price = p.last_sync_price
#         except KeyError:
#             current_share = 0
#             sync_price = 0.0
#         agg_shares = current_share + amount
#
#         volume_window = portal.get_window([asset], algo_datetime, - abs(self.window), ['volume'])
#         max_share = volume_window[asset.sid].mean() * self.max_notional
#         too_many_shares = agg_shares > max_share
#         if too_many_shares:
#             self.handle_violation(asset, amount, algo_datetime)
#             amount = max_share
#         return amount

# def set_asset_restrictions(self, restrictions, on_error='fail'):
#     """Set a restriction on which assets can be ordered.
#
#     Parameters
#     ----------
#     restricted_list : Restrictions
#         An object providing information about restricted assets.
#
#     See Also
#     --------
#     zipline.finance.asset_restrictions.Restrictions
#     """
#     control = RestrictedListOrder(on_error, restrictions)
#     self.register_trading_control(control)
#     self.restrictions |= restrictions

# @api_method
# def set_min_leverage(self, min_leverage, grace_period):
#     """Set a limit on the minimum leverage of the algorithm.
#
#     Parameters
#     ----------
#     min_leverage : float
#         The minimum leverage for the algorithm.
#     grace_period : pd.Timedelta
#         The offset from the start date used to enforce a minimum leverage.
#     """
#     deadline = self.sim_params.start_session + grace_period
#     control = MinLeverage(min_leverage, deadline)
#     self.register_account_control(control)

def validate_account_controls(self):
    for control in self.account_controls:
        control.validate(self.portfolio,
                         self.account,
                         self.get_datetime())

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod

# Consistent error to be thrown in various cases regarding overriding
# `final` attributes.
_type_error = TypeError('Cannot override final attribute')


# 多重继承 __mro__
def bases_mro(bases):
    """
    Yield classes in the order that methods should be looked up from the
    base classes of an object.
    """
    for base in bases:
        for class_ in base.__mro__:
            yield class_


def is_final(name, mro):
    """
    Checks if `name` is a `final` object in the given `mro`.
    We need to check the mro because we need to directly go into the __dict__
    of the classes. Because `final` objects are descriptor, we need to grab
    them _BEFORE_ the `__call__` is invoked.
    """
    return any(isinstance(getattr(c, '__dict__', {}).get(name), final)
               for c in bases_mro(mro))


class FinalMeta(type):
    """A metaclass template for classes the want to prevent subclassess from
    overriding some methods or attributes.
    """
    def __new__(mcls, name, bases, dict_):
        # for k, v in iteritems(dict_):
        for k, v in dict_.items():
            if is_final(k, bases):
                raise _type_error

        setattr_ = dict_.get('__setattr__')
        if setattr_ is None:
            # No `__setattr__` was explicitly defined, look up the super
            # class's. `bases[0]` will have a `__setattr__` because
            # `object` does so we don't need to worry about the mro.
            setattr_ = bases[0].__setattr__

        if not is_final('__setattr__', bases) \
           and not isinstance(setattr_, final):
            # implicitly make the `__setattr__` a `final` object so that
            # users cannot just avoid the descriptor protocol.
            dict_['__setattr__'] = final(setattr_)

        return super(FinalMeta, mcls).__new__(mcls, name, bases, dict_)

    def __setattr__(self, name, value):
        """This stops the `final` attributes from being reassigned on the
        class object.
        """
        if is_final(name, self.__mro__):
            raise _type_error

        super(FinalMeta, self).__setattr__(name, value)


class final(ABC):
    """
    An attribute that cannot be overridden.
    This is like the final modifier in Java.

    Example usage:
    >>> from six import with_metaclass
    >>> class C(with_metaclass(FinalMeta, object)):
    ...    @final
    ...    def f(self):
    ...        return 'value'
    ...

    This constructs a class with final method `f`. This cannot be overridden
    on the class object or on any instance. You cannot override this by
    subclassing `C`; attempting to do so will raise a `TypeError` at class
    construction time.
    """
    def __new__(cls, attr):
        # Decide if this is a method wrapper or an attribute wrapper.
        # We are going to cache the `callable` check by creating a
        # method or attribute wrapper.
        if hasattr(attr, '__get__'):
            return object.__new__(finaldescriptor)
        else:
            return object.__new__(finalvalue)

    def __init__(self, attr):
        self._attr = attr

    def __set__(self, instance, value):
        """
        `final` objects cannot be reassigned. This is the most import concept
        about `final`s.

        Unlike a `property` object, this will raise a `TypeError` when you
        attempt to reassign it.
        """
        raise _type_error

    def __get__(self, instance, owner):
        raise NotImplementedError('__get__')


class finalvalue(final):
    """
    A wrapper for a non-descriptor attribute.
    """
    def __get__(self, instance, owner):
        return self._attr


class finaldescriptor(final):
    """
    A final wrapper around a descriptor.
    """
    def __get__(self, instance, owner):
        return self._attr.__get__(instance, owner)


# def locate_pos(price, minutes, direction):
#     try:
#         loc = list(minutes.index[minutes == price])
#         if not loc:
#             # 当卖出价格大于bid价格才会成交，买入价格低于bid价格才会成交
#             loc = list(minutes.index[minutes <= price]) if direction == '1' else \
#                 list(minutes.index[minutes >= price])
#         return loc[0]
#     except IndexError:
#         print('price out of minutes')
#         return None



# class Position(object):
#     """
#     A protocol position which is not mutated ,but inner can be changed
#     """
#     # __slots__ = ['_underlying_position']
#
#     def __init__(self, inner):
#         self._underlying_position = MutableView(inner)
#         # self._underlying_position = inner
#
#     def __getattr__(self, attr):
#         # return self.__dict__[attr]
#         return self._underlying_position.__dict__[attr]
#         # return getattr(self._underlying_position, attr)
#
#     # def __setattr__(self, attr, value):
#     #     raise AttributeError('cannot mutate Position objects')
#
#     def __repr__(self):
#         return 'Position(%r)' % {
#             k: getattr(self, k)
#             for k in (
#                 'asset',
#                 'amount',
#                 'cost_basis',
#                 'last_sync_price',
#                 'last_sync_date',
#             )
#         }
#
#     # If you are adding new attributes, don't update this set. This method
#     # is deprecated to normal attribute access so we don't want to encourage
#     # new usages.
#     __getitem__ = _deprecated_getitem_method(
#         'position', {
#             'sid',
#             'amount',
#             'cost_basis',
#             'last_sale_price',
#             'last_sale_date',
#         },
#     )

# def _decref_recursive(self, metadata, mask):
#     """
#         internal method for decref_recursive
#         decrease by layer
#     """
#     print('decref graph', self._graph.nodes)
#     # return in_degree == 0 nodes
#     decref_nodes = self._graph.decref_dependencies()
#     print('decref nodes', decref_nodes)
#     if decref_nodes:
#         for node in decref_nodes:
#             node_mask = self._combine_term_dependence(node, mask)
#             print('node_mask', node_mask)
#             output = node.compute(metadata, list(node_mask))
#             self._workspace[node] = output
#             print('_workspace', self._workspace)
#         self._decref_recursive(metadata, mask)
#
# def _decref_dependence(self, metadata, mask):
#     """
#     Return a topologically-sorted list of the terms in ``self`` which
#     need to be computed.
#
#     Filters out any terms that are already present in ``workspace``, as
#     well as any terms with refcounts of 0.
#
#     Parameters
#     ----------
#     metadata : dict[Term, np.ndarray]
#         Initial state of workspace for a pipe execution. May contain
#         pre-computed values provided by ``populate_initial_workspace``.
#     mask : asset list
#         Reference counts for terms to be computed. Terms with reference
#         counts of 0 do not need to be computed.
#     """
#     self._decref_recursive(metadata, mask)

# @staticmethod
# def multi_process(ledger, iterable):
#     def proc(dct):
#         for k, v in dct.items():
#             ledger.process_transaction(v)
#     with Pool(processes=3) as pool:
#         [pool.apply_async(proc, item) for item in iterable]


# from toolz import keyfilter
#
# def resolve_conflicts(calls, puts, holdings):
#     # position name means pipeline_name ; asset tag name means pipeline_name
#     call_proxy = {r.tag: r for r in calls} if calls else {}
#     put_proxy = {r.name: r for r in puts} if puts else {}
#     hold_proxy = {p.name: p for p in holdings} if holdings else {}
#     # 判断买入标的的sid与卖出持仓的sid是否存在冲突
#     positive_sids = [r.sid for r in calls] if calls else []
#     negatives_sids = [p.asset.sid for p in puts] if puts else []
#     union_sids = set(positive_sids) & set(negatives_sids)
#     assert not union_sids, 'call assets should not be put at meantime'
#     # 基于capital执行直接买入标的的
#     extra = set(call_proxy) - set(hold_proxy)
#     if extra:
#         direct_positives = keyfilter(lambda x: x in extra, call_proxy)
#     else:
#         direct_positives = dict()
#     # 一个pipeline同时存在买入和卖出行为 --- 基于pipeline name
#     # common pipe name
#     common_pipe = set(call_proxy) & set(put_proxy)
#     negatives = set(put_proxy) - set(common_pipe)
#     direct_negatives = keyfilter(lambda x: x in negatives, put_proxy)
#     # 卖出持仓买入对应标的 --- (position, asset)
#     if common_pipe:
#         conflicts = [name for name in common_pipe if put_proxy[name].asset == call_proxy[name]]
#         assert not conflicts, ValueError('name : %r have conflicts between ump and pipe ' % conflicts)
#         dual = [(put_proxy[name], call_proxy[name]) for name in common_pipe]
#     else:
#         dual = set()
#     return direct_positives, direct_negatives, dual
#
#
# if __name__ == '__main__':
#
#     print(resolve_conflicts(None,(),()))

# def run(self):
#     """Run the algorithm.
#     """
#     # Create px_trade and loop through simulated_trading.
#     # Each iteration returns a perf dictionary
#     try:
#         perfs = []
#         for perf in self.yield_simulation():
#             perfs.append(perf)
#         # convert perf dict to pandas frame
#         daily_stats = self._create_daily_stats(perfs)
#         analysis = self.analyse(daily_stats)
#         return analysis
#     except Exception as e:
#         print('error:', e)

# # !/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Mar 12 15:37:47 2019
#
# @author: python
# """
# from abc import ABC, abstractmethod
# import numpy as np, pandas as pd
# from _calendar.trading_calendar import calendar
# from gateway.driver.adjustArray import (
#                         AdjustedDailyWindow,
#                         AdjustedMinuteWindow)
#
# DefaultFields = frozenset(['open', 'high', 'low', 'close', 'amount', 'volume'])
#
#
# # cacheObject --- bar_reader
# class Expired(Exception):
#     """
#         mark a cacheobject has expired
#     """
#
#
# class CachedObject(object):
#     """
#     A simple struct for maintaining a cached object with an expiration date.
#
#     Parameters
#     ----------
#     value : object
#         The object to cache.
#     expires : datetime-like []
#         Expiration date of `value`. The cache is considered invalid for dates
#         **strictly greater** than `expires`.
#     """
#     def __init__(self, value, expires):
#         self._value = value
#         self._expires = expires
#
#     def unwrap(self, dts):
#         """
#         Get the cached value.
#         dts: sessions
#         dts : [start_date, end_date]
#
#         Returns
#         -------
#         value : object
#             The cached value.
#
#         Raises
#         ------
#         Expired
#             Raised when `dt` is greater than self.expires.
#         """
#         expires = self._expires
#         if dts[0] < expires[0] or dts[-1] > expires[-1]:
#             raise Expired(expires)
#         return self._value
#
#     def _unsafe_get_value(self):
#         """You almost certainly shouldn't use this."""
#         return self._value
#
#
# class ExpiredCache(object):
#     """
#     A cache of multiple CachedObjects, which returns the wrapped the value
#     or raises and deletes the CachedObject if the value has expired.
#
#     Parameters
#     ----------
#     cache : dict-like, optional
#         An instance of a dict-like object which needs to support at least:
#         `__del__`, `__getitem__`, `__setitem__`
#         If `None`, than a dict is used as a default.
#
#     cleanup : callable, optional
#         A method that takes a single argument, a cached object, and is called
#         upon expiry of the cached object, prior to deleting the object. If not
#         provided, defaults to a no-op.
#
#     """
#     def __init__(self):
#         self._cache = {}
#         # cleanup = lambda value_to_clean: None
#
#     def get(self, key, dts):
#         """Get the value of a cached object.
#
#         Parameters
#         ----------
#         key : any
#             The key to lookup.
#         dts : datetime list e.g.[start, end]
#             The time of the lookup.
#
#         Returns
#         -------
#         result : any
#             The value for ``key``.
#
#         Raises
#         ------
#         KeyError
#             Raised if the key is not in the cache or the value for the key
#             has expired.
#         """
#         value = self._cache[key].unwrap(dts)
#         return value
#
#     def set(self, key, value, expiration_dt):
#         """Adds a new key value pair to the cache.
#
#         Parameters
#         ----------
#         key : sid
#             Asset object sid attribute
#         value : any
#             The value to store under the name ``key``.
#         expiration_dt : datetime
#             When should this mapping expire? The cache is considered invalid
#             for dates **strictly greater** than ``expiration_dt``.
#         """
#         self._cache[key] = CachedObject(value, expiration_dt)
#
#     def remove(self, key):
#         del self._cache[key]
#
#
# class HistoryLoader(ABC):
#
#     @property
#     def trading_calendar(self):
#         return calendar
#
#     @property
#     def frequency(self):
#         raise NotImplementedError()
#
#     def get_spot_value(self, dt, asset, fields):
#         spot = self.adjust_window.get_spot_value(dt, asset, fields)
#         return spot
#
#     def get_stack_value(self, tbl, dt, window):
#         sdate = self.trading_calendar.dt_window_size(dt, window)
#         stack = self.adjust_window.get_stack_value(tbl, [sdate, dt])
#         return stack
#
#     @abstractmethod
#     def _compute_slice_window(self, data, date, window):
#         raise NotImplementedError
#
#     def _ensure_sliding_windows(self, assets, fields, dts, window):
#         """
#         Ensure that there is a Float64Multiply window for each asset that can
#         provide data for the given parameters.
#         If the corresponding window for the (asset, len(dts), field) does not
#         exist, then create a new one.
#         If a corresponding window does exist for (asset, len(dts), field), but
#         can not provide data for the current dts range, then create a new
#         one and replace the expired window.
#
#         Parameters
#         ----------
#         assets : iterable of Assets
#             The asset in the window
#         dts : iterable of datetime64-like
#             The datetimes for which to fetch data.
#             Makes an assumption that all dts are present and contiguous,
#             in the _calendar.
#         fields : str or list
#             The OHLCV field for which to retrieve data.
#         Returns
#         -------
#         out : list of Float64Window with sufficient data so that each asset's
#         window can provide `get` for the index corresponding with the last
#         value in `dts`
#         """
#         asset_windows = {}
#         needed_assets = []
#         sdate = self.trading_calendar.dt_window_size(dts, window)
#         session = [sdate, dts]
#         # original window
#         orig_window = self.window(assets, fields, dts, window)
#         for asset_obj in assets:
#             # print('blocks', self._window_blocks)
#             # print('asset', asset_obj)
#             try:
#                 cache_window = self._window_blocks.get(
#                     asset_obj, session)
#                 # print('cache_window', cache_window)
#             except Expired:
#                 # del self._window_blocks[asset_obj]
#                 self._window_blocks.remove(asset_obj)
#             except KeyError:
#                 needed_assets.append(asset_obj)
#             else:
#                 slice_window = self._compute_slice_window(cache_window, session)
#                 slice_origin = orig_window[asset_obj.sid]
#                 slice_origin.sort_index(inplace=True)
#                 # series index
#                 ratio = slice_origin['close'].iloc[-1] / slice_window['close'].iloc[-1]
#                 # print('ratio', ratio)
#                 asset_windows[asset_obj.sid] = slice_window.loc[:, fields] * ratio
#
#         if needed_assets:
#             for i, target_asset in enumerate(needed_assets):
#                 sliding_window = self.adjust_window.window_arrays(
#                         session,
#                         [target_asset],
#                         list(DefaultFields)
#                             )[target_asset.sid]
#                 # ExpiredCache
#                 self._window_blocks.set(
#                     target_asset,
#                     sliding_window,
#                     session)
#                 asset_windows[target_asset.sid] = sliding_window.loc[:, fields] \
#                     if not sliding_window.empty else sliding_window
#         return asset_windows
#
#     def window(self, assets, field, dts, window):
#         if window == -1:
#             frame = dict()
#             date = self.trading_calendar.dt_window_size(dts, -1)
#             for asset in assets:
#                 frame[asset.sid] = self.get_spot_value(date, asset, field)
#         else:
#             sessions = self.trading_calendar.session_in_window(dts, window)
#             frame = self.adjust_window.array([min(sessions), max(sessions)], assets, field)
#         return frame
#
#     def history(self, assets, field, dts, window):
#         """
#         A window of pricing data with adjustments applied assuming that the
#         end of the window is the day before the current ArkQuant time.
#         default fields --- OHLCV
#
#         Parameters
#         ----------
#         assets : iterable of Assets
#             The asset in the window.
#         dts : iterable of datetime64-like
#             The datetimes for which to fetch data.
#             Makes an assumption that all dts are present and contiguous,
#             in the _calendar.
#         field : str or list
#             The OHLCV field for which to retrieve data.
#         window : int
#             The length of window
#         Returns
#         -------
#         out : np.ndarray with shape(len(days between start, end), len(asset))
#         """
#         # 不包括当天数据
#         assert window < 0, 'to avoid forward prospective error'
#         if window != -1:
#             block_arrays = self._ensure_sliding_windows(
#                                             assets,
#                                             field,
#                                             dts,
#                                             window
#                                             )
#         else:
#             block_arrays = self.window(assets, field, dts, window)
#         return block_arrays
#
#
# class HistoryDailyLoader(HistoryLoader):
#     """
#         生成调整后的序列
#         优化 --- 缓存
#     """
#
#     def __init__(self,
#                  _daily_reader,
#                  equity_adjustment_reader):
#         self.adjust_window = AdjustedDailyWindow(
#                                             _daily_reader,
#                                             equity_adjustment_reader)
#         self._window_blocks = ExpiredCache()
#
#     @property
#     def frequency(self):
#         return 'daily'
#
#     def get_mkv_value(self, session, assets, fields):
#         mkv = self.adjust_window.get_mkv_value(session, assets, fields)
#         return mkv
#
#     @staticmethod
#     def _compute_slice_window(_window, dts):
#         sessions = calendar.session_in_range(*dts)
#         slice_window = _window.reindex(sessions)
#         slice_window.sort_index(inplace=True)
#         return slice_window
#
#
# class HistoryMinuteLoader(HistoryLoader):
#
#     def __init__(self,
#                  _minute_reader,
#                  equity_adjustment_reader):
#         self.adjust_window = AdjustedMinuteWindow(
#                                             _minute_reader,
#                                             equity_adjustment_reader)
#         self._window_blocks = ExpiredCache()
#
#     @property
#     def frequency(self):
#         return 'minute'
#
#     @staticmethod
#     def _compute_slice_window(_window, dts):
#         # print('dts', dts)
#         s_timestamp = pd.Timestamp(dts[0]).timestamp()
#         e_timestamp = pd.Timestamp(dts[1]).timestamp()
#         _idn = np.array(_window.index)
#         ticker = _idn[(_idn >= int(s_timestamp)) & (_idn <= int(e_timestamp))]
#         _slice_window = _window.reindex(ticker)
#         _slice_window.sort_index(inplace=True)
#         return _slice_window

# def _cleanup_expired_assets(self, dt):
#     """
#     Clear out any assets that have expired before starting a new sim day.
#
#     Finds all assets for which we have positions and generates
#     close_position events for any assets that have reached their
#     close_date.
#     """
#     def past_close_date(asset):
#         acd = asset.last_traded
#         return acd is not None and acd == dt
#     # Remove positions in any sids that have reached their auto_close date.
#     positions_to_clear = \
#         [p for p in self.positions if past_close_date(p)]
#     return positions_to_clear

# def get_rights_positions(self, dts):
#     # 获取当天为配股登记日的仓位 --- 卖出 因为需要停盘产生机会成本
#     assets = set(self.positions)
#     rights = self.position_tracker.retrieve_equity_rights(assets, dts)
#     p_mapping = {p.sid: p for p in self.positions}
#     right_positions = None if rights.empty else [p_mapping[symbol].protocol for symbol in rights.index]
#     return right_positions

# def get_violate_risk_positions(self):
#     # 获取违反风控管理的仓位
#     violate_positions = [p.protocol for p in self.positions
#                          if self.risk_alert.should_trigger(p)]
#     return violate_positions

# def __getattr__(self, item):
#     return self.__dict__[item]
#
# # def __getattr__(self, item):
# #     return self.__slots__[item]

# # distribution of price
# alpha = 1 if open_pct == 0.00 else 100 * open_pct
# print('alpha', alpha, size)
# if size > 0:
#     # dist = 1 + np.copysign(alpha, np.random.beta(abs(alpha), 100, size))/10
#     dist = np.copysign(alpha, np.random.beta(abs(alpha), 100, size))/10
#     print('beta', dist)
#     dist = dist + 1
# else:
#     dist = [1 + alpha / 100]

#
# import pandas as pd
# from itertools import chain
#
#
# def _uncover_by_ticker(size, asset, dts):
#     # ticker arranged on sequence
#     dts = pd.Timestamp(dts) if isinstance(dts, str) else dts
#     interval = 4 * 60 / size
#     print('uncover_by_ticker', interval)
#     # 按照固定时间去执行
#     upper = pd.date_range(start=dts + pd.Timedelta(hours=9, minutes=30), end=dts + pd.Timedelta(hours=11, minutes=30),
#                           freq='%dmin' % interval)
#     print('uncover by ticker upper', upper)
#     bottom = pd.date_range(start=dts + pd.Timedelta(hours=13, minutes=30), end=dts + pd.Timedelta(hours=14, minutes=57),
#                            freq='%dmin' % interval)
#     print('uncover by ticker bottom', bottom)
#     # 确保首尾
#     intervals = list(chain(*zip(upper, bottom)))
#     intervals if len(intervals) == size else intervals.append(dts + pd.Timedelta(hours=14, minutes=57))
#     print('tick_intervals', len(intervals), intervals)
#     return intervals
#
#
# if __name__ == '__main__':
#
#     _uncover_by_ticker(7, None, '2019-09-02')

# kline.index = [datetime.datetime.utcfromtimestamp(i).strftime('%Y-%m-%d %H:%M') for i in data['600000'].index]

# @property
# def cash_flow(self):
#     return self._cash_flow
#
# @cash_flow.setter
# def cash_flow(self, val):
#     self._cash_flow = val


# if __name__ == '__main__':
#
#     import signal
#
#     def handler(signum, frame):
#         print(signum, frame)
#         print('unexpected action')
#
#     # signal.signal(signal.SIGALRM, handler)
#     # signal.alarm(1)
#     signal.signal(signal.SIGINT, handler)
#     signal.pause()
#     print('End of Signal Demo')

# class Profits(object):
#     """
#         track the daily profit of the algorithm
#         dict --- asset : profit
#     """
#     def end_of_session(self,
#                        packet,
#                        ledger,
#                        session_ix):
#         packet['daily_perf']['profit'] = ledger.daily_position_stats(session_ix)

# class Weights(object):
#
#     def end_of_session(self,
#                         packet,
#                         ledger,
#                         session_ix):
#         # print('ledger', ledger)
#         weights = ledger.portolio.current_portfolio_weights
#         packet['cumulative_risk_metrics']['portfolio_weights'] = weights

# class CashFlow(object):
#     """Tracks daily and cumulative cash flow.
#     Notes
#     -----
#     For historical reasons, this field is named 'capital_used' in the packets.
#     """
#
#     def __init__(self):
#         self._previous_cash_flow = 0.0
#
#     def start_of_simulation(self,
#                             ledger,
#                             benchmark_returns,
#                             sessions):
#         self._previous_cash_flow = 0.0
#
#     def start_of_session(self, ledger):
#         self._previous_cash_flow = ledger.portfolio.cash_flow
#
#     def end_of_session(self,
#                        packet,
#                        ledger,
#                        session_ix):
#         cash_flow = ledger.portfolio.cash_flow
#         packet['daily_perf']['capital_used'] = (
#                 cash_flow - self._previous_cash_flow
#         )
#         packet['cumulative_perf']['capital_used'] = cash_flow
#         self._previous_cash_flow = cash_flow

# class Proportion(object):
#     """
#         计算持仓按照资产类别计算比例
#     """
#     def end_of_session(self,
#                        packet,
#                        ledger,
#                        session_ix):
#         # 按照资产类别计算持仓
#         portfolio = ledger.portfolio
#         portfolio_position_values = portfolio.portfolio_value - portfolio.start_cash
#         # 持仓分类
#         protocols = ledger.get_positions()
#         mappings = groupby(lambda x: x.asset.asset_type, protocols)
#         # 计算大类权重
#         from toolz import valmap
#         mappings_value = valmap(lambda x: sum([p.amount * p.last_sync_price for p in x]), mappings)
#         ratio = valmap(lambda x : x / portfolio_position_values, mappings_value)
#         packet['cumulative_risk_metrics']['proportion'] = ratio

# @staticmethod
# def _create_daily_stats(perfs):
#     # create daily and cumulative stats frame
#     daily_perfs = []
#     for perf in perfs:
#         if 'daily_perf' in perf:
#             perf['daily_perf'].update(perf['cumulative_risk_metrics'])
#             daily_perfs.append(perf['daily_perf'])
#     daily_dts = pd.DatetimeIndex(
#         [p['period_close'] for p in daily_perfs], tz='UTC'
#     )
#     daily_stats = pd.DataFrame(daily_perfs, index=daily_dts)
#     daily_stats = pd.DataFrame(daily_perfs)
#     print('daily_stats', daily_stats)
#     return daily_stats


# def divided_by_capital(self, asset, capital, portfolio, dts):
#     """
#         split order into plenty of tiny orders
#         a. calculate amount to determine size
#         b. create ticker_array depend on size
#         c. simulate order according to ticker_price , ticker_size , ticker_price
#             --- 存在竞价机制的情况将订单分散在不同时刻，符合最大成交原则
#             --- 无竞价机制的情况下，模拟的价格分布，将异常的价格集中以收盘价价格进行成交
#         d. principle:
#             a. pipe 买入策略信号会滞后 ， dt对象与dt + 1对象可能相同的 --- 分段加仓
#             b. 针对于卖出标的 -- 遵循最大程度卖出（当天）
#             c. 执行买入算法的需要涉及比如最大持仓比例，持仓量等限制
#         order amount --- negative
#
#         针对于买入操作
#         a. 计算满足最低capital(基于手续费逻辑），同时计算size
#         b. 存在竞价机制 --- 基于size设立时点order
#         c. 不存在竞价机制 --- 模拟价格分布提前确定价格单，14:57集中撮合
#
#     """
#     open_pct, ensure_price, per_amount = self._calculate_division_data(asset, dts)
#     print('division data', open_pct, ensure_price, per_amount)
#     amount = asset.tick_size * np.floor(capital / (ensure_price * asset.tick_size)) \
#         if asset.increment else np.floor(capital / ensure_price)
#     print('ensure amount', amount)
#     assert amount >= asset.tick_size, 'amount must be at least tick_size'
#     control_amount = self.trade_controls.validate(asset, amount, portfolio, dts)
#     print('capital control_amount', control_amount)
#     zip_iterables = self.uncover_func.create_iterables(asset, control_amount, per_amount, dts)
#     capital_orders = self._simulate_iterator(asset, zip_iterables)
#     print('capital_orders', capital_orders)
#     return capital_orders

# def create_iterables(self, asset, amount, per_amount, dt):
#     amount_arrays, size = self._underneath_size(asset, amount, per_amount, dt)
#     if asset.bid_mechanism:
#         dist_arrays = self._uncover_by_ticker(size, asset, dt)
#     else:
#         dist_arrays = self._uncover_by_price(size, asset, dt)
#     iterables = zip(amount_arrays, dist_arrays)
#     return iterables


# def validate(self,
#              asset,
#              amount,
#              portfolio,
#              algo_datetime):
#     """
#     Fail if the given order would cause the magnitude of our position to be
#     greater in shares than self.max_shares or greater in dollar value than
#     self.max_notional.
#     """
#     # 基于sid 不是asset(由于不同的pipeline作为asset属性)
#     weights = portfolio.current_portfolio_weights
#
#     if amount < 0:
#         return amount
#     elif weights[asset.sid] >= self.max_notional:
#         self.handle_violation(asset, amount, algo_datetime)
#         amount = 0
#     else:
#         try:
#             p = portfolio.positions[asset]
#             current_share = p.amount
#             sync_price = p.last_sync_price
#         except KeyError:
#             current_share = 0
#             pctchange, pre_close = portal.get_open_pct(asset, algo_datetime)
#             sync_price = pre_close * (1 + asset.restricted(algo_datetime))
#         # calculate amount
#         max_capital = portfolio.portfolio_value * self.max_notional
#         max_amount = int(max_capital / sync_price)
#         amount = max_amount - current_share
#     return amount

# def synchronize(self):
#     """
#         a. sync last_sale_price of position (close price)
#         b. update position return series
#         c. update last_sync_date
#     """
#     sync_date = set([p.last_sync_date for p in self.positions.values()])
#     # print('sync_date', sync_date)
#     if sync_date:
#         assert len(sync_date) == 1, 'all positions must be sync on the same date'
#         get_price = partial(portal.get_spot_value,
#                             dts=list(sync_date)[0],
#                             frequency='daily',
#                             field='close')
#         for asset, p in self.positions.items():
#             p.inner_position.last_sync_price = get_price(asset=asset)
#             # update position_returns
#             p.calculate_returns()

# np.add(returns, 1, out=out)
# out.cumprod(axis=0, out=out)
# # cum_returns_s = np.exp(np.log(1 + returns_s).cumsum())
# np.subtract(out, 1, out=out)
#
# out = np.cumprod(returns.values())

# # np.multiply(out, starting_value, out=out)
# if returns.ndim == 1 and isinstance(returns, pd.Series):
#     out = pd.Series(out, index=returns.index)
# elif isinstance(returns, pd.DataFrame):
#     out = pd.DataFrame(
#         out, index=returns.index, columns=returns.columns,
#         )
# return out

#
# def cum_returns_final(
#                      returns,
#                      benchmark_returns,
#                      risk_free=0.0,
#                      required_return=0.0
#                      ):
#     """
#     Compute total returns from simple returns.
#
#     Parameters
#     ----------
#     returns : pd.DataFrame, pd.Series, or np.ndarray
#        Noncumulative simple returns of one or more timeseries.
#     starting_value : float, optional
#        The starting returns.
#
#     Returns
#     -------
#     total_returns : pd.Series, np.ndarray, or float
#         If input is 1-dimensional (a Series or 1D numpy array), the result is a
#         scalar.
#
#         If input is 2-dimensional (a DataFrame or 2D numpy array), the result
#         is a 1D array containing cumulative returns for each column of input.
#     """
#     if len(returns) == 0:
#         return np.nan
#     if isinstance(returns, pd.DataFrame):
#         result = (returns + 1).prod()
#     else:
#         result = np.nanprod(returns + 1, axis=0)
#     return result
#

# def _create_generator(self):
#     """
#     :param dist: distribution module (simulate_dist , simulate_ticker)
#                 to generate price timeseries or ticker timeseries
#     generator --- compute capital or position to transactions
#     """
#     # generator
#     generator_class = Generator(self.sim_params.delay,
#                                 self.blotter,
#                                 self.division_model)
#     return generator_class

# universe_func=self._calculate_universe

# initialize necessary module
# self.pipeline_engine = self._construct_pipeline_engine(
#                                             disallow_righted,
#                                             disallowed_violation)
# self.ledger = Ledger(self.sim_params, risk_models, risk_fuse)
# self.underneath_module = uncover_algorithm or UncoverAlgorithm()
# self.division_model = Division(self.underneath_module,
#                                self.trading_controls,
#                                self.sim_params.capital_base)
# self.blotter = SimulationBlotter(self.commission,
#                                  self.slippage,
#                                  self.execution_style)
#
# self.generator = self._create_generator()
# self.broker = self._initialize_broker()
