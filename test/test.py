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
# #                 smallestDateTime = utils.safe_min(smallestDateTime, subject.peekDateTime())
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
# #     :class:`~zipline.pipeline.data.DataSet` objects, each of which has the same
# #     columns, domain, and ndim.
# #
# #     :class:`DataSetFamily` objects are defined with by one or more
# #     :class:`~zipline.pipeline.data.Column` objects, plus one additional field:
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
# #     To work with a :class:`DataSetFamily` in a pipeline expression, one must
# #     choose a specific value for each of the extra dimensions using the
# #     :meth:`~zipline.pipeline.data.DataSetFamily.slice` method.
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
# #             A regular pipeline dataset indexed by asset and date.
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
#     def __init__(self,algo_mappings,data_portal,blotter,assign_policy):
#         self.data_portal = data_portal
#         self.postion_allocation = assign_policy
#         self.blotter = blotter
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
#                            set(map(lambda x : x.reason ,metrics_tracker.positions.assets))
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
#             [self.blotter.order(asset,
#                                 positions[asset].amount)
#                                 for asset in assets_of_exit]
#             cleanup_transactions,additional_commissions = self.blotter.get_transaction(self.data_portal)
#             return cleanup_transactions,additional_commissions
#
#     def get_layout(self,dt,metrics_tracker):
#         assets = self.compute_algorithm(dt,metrics_tracker)
#         avaiable_cash = metrics_tracker.portfolio.cash
#         [self.blotter.order(asset,
#                             self._calculate_order_amount(asset,dt,avaiable_cash))
#                             for asset in assets]
#         transactions,new_commissions = self.blotter.get_transaction(self.data_portal)
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
#     :meth:`~zipline.pipeline.graph.TermGraph.decref_dependencies`
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
#         Initial state of workspace for a pipeline execution. May contain
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
#     domain : zipline.pipeline.domain.Domain
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
#         :meth:`zipline.pipeline.graph.ExecutionPlan.offset`
#         :meth:`zipline.pipeline.engine.ExecutionPlan.mask_and_dates_for_term`
#         :meth:`zipline.pipeline.engine.SimplePipelineEngine._inputs_for_term`
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
#         :meth:`zipline.pipeline.graph.ExecutionPlan.offset`
#         :meth:`zipline.pipeline.Term.dependencies`
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
#             All of the dates that are being computed for in the pipeline.
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
#         # Simulation
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
#             # simulation finishes.
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
# def run_pipeline(self, pipeline, start_session, chunksize):
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
#     # ...continuing until either the day before the simulation end, or
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
#         self.engine.run_pipeline(pipeline, start_session, end_session), \
#         end_session
#
# @staticmethod
# def default_pipeline_domain(calendar):
#     """
#     Get a default Pipeline domain for algorithms running on ``calendar``.
#
#     This will be used to infer a domain for pipelines that only use generic
#     datasets when running in the context of a TradingAlgorithm.
#     """
#     return _DEFAULT_DOMAINS.get(calendar.name, domain.GENERIC)
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
#         This is where we do the actual work of running a pipeline.
#
#         Parameters
#         ----------
#         graph : zipline.pipeline.graph.ExecutionPlan
#             Dependency graph of the terms to be executed.
#         dates : pd.DatetimeIndex
#             Row labels for our root mask.
#         sids : pd.Int64Index
#             Column labels for our root mask.
#         workspace : dict
#             Map from term -> output.
#             Must contain at least entry for `self._root_mask_term` whose shape
#             is `(len(dates), len(assets))`, but may contain additional
#             pre-computed terms for testing or optimization purposes.
#         refcounts : dict[Term, int]
#             Dictionary mapping terms to number of dependent terms. When a
#             term's refcount hits 0, it can be safely discarded from
#             ``workspace``. See TermGraph.decref_dependencies for more info.
#         execution_order : list[Term]
#             Order in which to execute terms.
#         hooks : implements(PipelineHooks)
#             Hooks to instrument pipeline execution.
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
#     zipline.assets.AssetFinder.lifetimes
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
#     def _compute(self, today, assets, out):
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
#     Checks if any tables are present in the current assets database.
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
#         The metadata that describes the new assets db.
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
#     asset_finder : zipline.assets.AssetFinder
#         An AssetFinder instance.  We depend on the AssetFinder to determine
#         which assets are in the top-level universe at any point in time.
#     populate_initial_workspace : callable, optional
#         A function which will be used to populate the initial workspace when
#         computing a pipeline. See
#         :func:`zipline.pipeline.engine.default_populate_initial_workspace`
#         for more info.
#     default_hooks : list, optional
#         List of hooks that should be used to instrument all pipelines executed
#         by this engine.
#
#     See Also
#     --------
#     :func:`zipline.pipeline.engine.default_populate_initial_workspace`
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
#     def run_pipeline(self, pipeline, start_date, end_date, hooks=None):
#         """
#         Compute values for ``pipeline`` from ``start_date`` to ``end_date``.
#
#         Parameters
#         ----------
#         pipeline : zipline.pipeline.Pipeline
#             The pipeline to run.
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
#             `pipeline.columns`, which should be a dictionary mapping strings to
#             instances of :class:`zipline.pipeline.Term`.
#
#             For each date between ``start_date`` and ``end_date``, ``result``
#             will contain a row for each asset that passed `pipeline.screen`.
#             A screen of ``None`` indicates that a row should be returned for
#             each asset that existed each day.
#         """
#         hooks = self._resolve_hooks(hooks)
#         with hooks.running_pipeline(pipeline, start_date, end_date):
#             return self._run_pipeline_impl(
#                 pipeline,
#                 start_date,
#                 end_date,
#                 hooks,
#             )
#
#     def _run_pipeline_impl(self, pipeline, start_date, end_date, hooks):
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
#                              pipeline,
#                              start_date,
#                              end_date,
#                              chunksize,
#                              hooks=None):
#         """
#         Compute values for ``pipeline`` from ``start_date`` to ``end_date``, in
#         date chunks of size ``chunksize``.
#
#         Chunked execution reduces memory consumption, and may reduce
#         computation time depending on the contents of your pipeline.
#
#         Parameters
#         ----------
#         pipeline : Pipeline
#             The pipeline to run.
#         start_date : pd.Timestamp
#             The start date to run the pipeline for.
#         end_date : pd.Timestamp
#             The end date to run the pipeline for.
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
#             `pipeline.columns`, which should be a dictionary mapping strings to
#             instances of :class:`zipline.pipeline.Term`.
#
#             For each date between ``start_date`` and ``end_date``, ``result``
#             will contain a row for each asset that passed `pipeline.screen`.
#             A screen of ``None`` indicates that a row should be returned for
#             each asset that existed each day.
#
#         See Also
#         --------
#         :meth:`zipline.pipeline.engine.PipelineEngine.run_pipeline`
#         """
#
#         hooks = self._resolve_hooks(hooks)
#
#         run_pipeline = partial(self._run_pipeline_impl, pipeline, hooks=hooks)
#         with hooks.running_pipeline(pipeline, start_date, end_date):
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
#     Shift dates of a pipeline query back by `shift` days.
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
#         Start date of the pipeline query.
#     end_date : pd.Timestamp
#         End date of the pipeline query.
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
#             raise ValueError("Query start %s not in calendar" % start_date)
#
#     # Make sure that shifting doesn't push us out of the calendar.
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
#             raise ValueError("Query end %s not in calendar" % end_date)
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
#     def load_raw_arrays(self, date, window, columns,assets):
#         raw_arrays = {}
#         _request_array = partial(self.get_value,edate = date,columns = columns,window = window)
#         #获取数据
#         for asset in assets:
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

bond_asset_url = 'https://www.jisilu.cn/data/cbnew/cb_list/?'
text = _parse_url(bond_asset_url, bs=False, encoding=None)
text = json.loads(text)
sids = text['rows']
print(sids[100])
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
# #公司基本情况
# basics_url = 'https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/%s.phtml'
# code = '002570'
# url = basics_url % code
# obj = _parse_url(url)
# table = obj.find('table', {'id': 'comInfo1'})
# tag = [item.findAll('td') for item in table.findAll('tr')]
# tag_chain = list(chain(*tag))
# raw = [item.get_text() for item in tag_chain]
# # 去除格式
# raw = [i.replace('：', '') for i in raw]
# raw = [i.strip() for i in raw]
# info = list(zip(raw[::2], raw[1::2]))
# info_dict = {item[0]: item[1] for item in info}
# # info_dict.update({'代码': code})
# print(info_dict)
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
# fund_url = 'http://fund.eastmoney.com/cnjy_jzzzl.html'
# obj = _parse_url(fund_url)
# # print(obj.prettify())
# from toolz import partition_all
# raw = [data.find_all('td') for data in obj.find_all(id = 'tableDiv')]
# text = [t.get_text() for t in raw[0]]
# print(text)
# df = pd.DataFrame(partition_all(14,text[18:]),columns = text[2:16])
# df['基金简称'] = df['基金简称'].apply(lambda x: x[:-5])
# print(df.head())
# print(df['基金代码'].values)
# # equity_url = 'http://70.push2.eastmoney.com/api/qt/clist/get?pn=1&pz=10000&po=1&np=1&fltt=2&invt=2&' \
#              'fid=f3&fs=m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12'
# # 获取存量股票包括退市
# raw = json.loads(_parse_url(equity_url, bs=False))
# equities = [item['f12'] for item in raw['data']['diff']]
# print('equities',equities)

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
#     # Break the list up in to its two components, the company symbol and the
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
#             'Ambiguous ownership for %d symbol%s, multiple assets held the'
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



