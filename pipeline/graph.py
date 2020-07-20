# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import glob, uuid,networkx as nx,warnings,pandas as pd,numpy as np
from collections import OrderedDict
from toolz import valfilter, keyfilter,merge,merge_with,groupby
from contextlib import ExitStack ,contextmanager
from weakref import WeakValueDictionary
from functools import partial,reduce
from multiprocessing import Pool
from itertools import chain
from abc import ABC,abstractmethod

from gateWay.assets.asset_ext import  AssetFinder

class NotSpecific(Exception):

    def __str__(self):
        return ('object not specific')

    __repr__ = __str__()

@contextmanager
def ignore_pandas_nan_categorical_warning():
    with warnings.catch_warnings():
        # Pandas >= 0.18 doesn't like null-ish values in categories, but
        # avoiding that requires a broader change to how missing values are
        # handled in pipeline, so for now just silence the warning.
        warnings.filterwarnings(
            'ignore',
            category=FutureWarning,
        )
        yield

def contextmanager(f):
    """
    Wrapper for contextlib.contextmanager that tracks which methods of
    PipelineHooks are contextmanagers in CONTEXT_MANAGER_METHODS.
    """
    PIPELINE_HOOKS_CONTEXT_MANAGERS.add(f.__name__)
    return contextmanager(f)


class Term(object):
    """
        term.specialize(domain)
        执行算法 --- 拓扑结构
        退出算法 --- 裁决模块
        Dependency-Graph representation of Pipeline API terms.
        结构:
            1 节点 --- 算法，基于拓扑结构 --- 实现算法逻辑 表明算法的组合方式
            2 不同的节点已经应该继承相同的接口，不需要区分pipeline还是featureUnion
            3 同一层级的不同节点相互独立，一旦有连接不同层级
            4 同一层级节点计算的标的集合交集得出下一层级的输入，不同节点之间不考虑权重分配因为交集（存在所有节点）
            5 每一个节点产生有序的股票集合，一层所有的交集按照各自节点形成综合排序
            6 最终节点 --- 返回一个有序有限的集合
        节点:
            1 inputs --- asset list
            2 compute ---- algorithm list
            3 outputs --- algorithm list & asset list
    """
    default_input = NotSpecific
    final = False

    namespace = dict()
    _term_cache = WeakValueDictionary

    __slots__ = ['domain','script','params','dtype']

    def __new__(cls,
                domain,
                script,
                params,
                dtype = None
                ):

        dtype = dtype if dtype else list

        script_path= glob.glob('strategy/%s.py'%script)
        with open(script_path, 'r') as f:
            exec(f.read(), cls.namespace)
        logic_cls = cls.namespace[script]
        identity = cls._static_identity(domain,logic_cls,params,dtype)

        try:
            return cls._term_cache[identity]
        except KeyError:
            new_instance = cls._term_cache[identity] = \
                super(Term,cls).__new__(cls)._init(domain,logic_cls,params,dtype)
            return new_instance

    @classmethod
    def _static_identity(cls,domain,script_class,script_params,dtype):

        return (domain,script_class,script_params,dtype)

    def _init(self, domain,script,params,dtype):
        """
            __new__已经初始化后，不需要在__init__里面调用
            Noop constructor to play nicely with our caching __new__.  Subclasses
            should implement _init instead of this method.

            When a class' __new__ returns an instance of that class, Python will
            automatically call __init__ on the object, even if a new object wasn't
            actually constructed.  Because we memoize instances, we often return an
            object that was already initialized from __new__, in which case we
            don't want to call __init__ again.

            Subclasses that need to initialize new instances should override _init,
            which is guaranteed to be called only once.
            Parameters
            ----------
            domain : zipline.pipeline.domain.Domain
                The domain of this term.
            dtype : np.dtype
                Dtype of this term's output.
            params : tuple[(str, hashable)]
                Tuple of key/value pairs of additional parameters.
        """
        self.domain = domain
        self.dtype = dtype
        try:
            instance = script(params)
            self.term_logic = instance
            self._validate()
        except TypeError:
            raise Exception('cannot initialize strategy')
            self._subclass_called_validate = False

        assert self._subclass_called_super_validate, (
            "Term._validate() was not called.\n"
            "This probably means that logic cannot be initialized."
        )
        del self._subclass_called_super_validate

        return self

    def _validate(self):
        """
        Assert that this term is well-formed.  This should be called exactly
        once, at the end of Term._init().
        """
        # mark that we got here to enforce that subclasses overriding _validate
        # call super().
        self._subclass_called_super_validate = True

    def postprocess(self,data):
        """
            called with an result of self ,after any user-defined screens have been applied
            this is mostly useful for transforming  the dtype of an output

            the default implementation is to just return data unchange
        """
        if self.dtype == bool:
            if not isinstance(data,self.dtype):
                raise TypeError('style of data is not %s' % self.dtype)
            return data
        else:
            try:
                data = self.dtype(data)
            except:
                raise TypeError('cannot transform the style of data to %s'%self.dtype)
            return data

    @property
    def dependencies(self):
        """
        A dictionary mapping terms that must be computed before `self` to the
        number of extra rows needed for those terms.
        """
        return self.default_input

    @dependencies.setter
    def dependencies(self,terms):
        for item in terms:
            if not isinstance(item,self):
                raise TypeError('dependencies must be Term')
        return terms

    def _compute(self,inputs,data):
        """
            1. subclass should implement when _verify_asset_finder is True
            2. self.postprocess()
        """
        output = self.term_logic.compute(inputs,data)
        validate_output = self.postprocess(output)
        return validate_output

    def recursive_repr(self):
        """A short repr to use when recursively rendering terms with inputs.
        """
        # Default recursive_repr is just the name of the type.
        return type(self).__name__


class TermGraph(object):
    """
    An abstract representation of Pipeline Term dependencies.

    This class does not keep any additional metadata about any term relations
    other than dependency ordering.  As such it is only useful in contexts
    where you care exclusively about order properties (for example, when
    drawing visualizations of execution order).

    Graph represention of Pipeline Term dependencies that includes metadata
    about extra rows required to perform computations.

    Each node in the graph has an `extra_rows` attribute, indicating how many,
    if any, extra rows we should compute for the node.  Extra rows are most
    often needed when a term is an input to a rolling window computation.
    """
    def __init__(self, terms):

        self.graph = nx.DiGraph()

        self._frozen = False

        for term in terms:
            self._add_to_graph(term)

        self._outputs = terms

        self._frozen = True

    def _add_to_graph(self,term):
        """
            先增加节点 --- 增加edge
        """
        if self._frozen:
            raise ValueError(
                "Can't mutate %s after construction." % type(self).__name__
            )

        self.graph.add_node(term)

        for dependency in term.dependencies:
            self._add_to_graph(dependency)
            self.graph.add_edge(dependency,term)

    @property
    def outputs(self):
        """
        Dict mapping names to designated output terms.
        """
        return self._outputs

    @property
    def screen_name(self):
        """Name of the specially-designated ``screen`` term for the pipeline.
        """
        SCREEN_NAME = 'screen_' + uuid.uuid4().hex
        return SCREEN_NAME

    def __contains__(self, term):
        return term in self.graph

    def __len__(self):
        return len(self.graph)

    def ordered(self):
        return iter(nx.topological_sort(self.graph))

    def _decref_dependencies(self):
        """
        Decrement in-edges for ``term`` after computation.

        Parameters
        ----------
        term : zipline.pipeline.Term
            The term whose parents should be decref'ed.
        refcounts : dict[Term -> int]
            Dictionary of refcounts.

        Return
        ------
        terms which need to decref
        """
        refcounts = dict(self.graph.in_degree())
        nodes = valfilter(lambda x : x == 0 ,refcounts)
        for node in nodes:
            self.graph.remove_node(node)
        return nodes


class Pipeline(object):
    """
        拓扑执行逻辑
    """
    __slots__ = ['_term_store','default_type']

    def __init__(self,terms,default):
        self._terms_store = terms
        self.default_type = default
        self._init_graph()
        self.initialzed = False
        self.asset_finder = AssetFinder()

    @property
    def name(self):
        return uuid.uuid4().hex()

    def __add__(self,term):
        if not isinstance(term, Term):
            raise TypeError(
                "{term} is not a valid pipeline column. Did you mean to "
                "append '.latest'?".format(term=term)
            )
        if term in self.pipeline_graph.nodes:
            raise Exception('term object already exists in pipeline')
        self._terms_store.append(term)
        return self

    def __sub__(self,term):
        try:
            self._terms_store.remove(term)
        except Exception as e:
            raise TypeError(e)
        return self

    def __setattr__(self, key, value):

        raise NotImplementedError

    def _init_graph(self):
        self.pipeline_graph = self._to_simple_graph()

    def attach_default(self,dt,window = 30):
        """
            a. 剔除停盘
            b. 剔除进入退市整理期的30个交易日
            c. 剔除上市不足一个月的 --- 为了防止筛选出来的都是无法买入的次新股由于流通性问题 -- 次新股波动性太大缺少一定逻辑性,而且到时基于pipeline
               算法筛选出得都是无法买入的次新股的概率大大增加
        """
        suspend = self.asset_finder.fuzzy_symbols_ownership_by_suspend(dt)
        delist = self.asset_finder.fuzzy_symbol_ownership_by_delist(dt)
        unsufficient = self.asset_finder.fuzzy_symbol_ownership_by_ipo(dt,window)
        self.default = set(unsufficient) - (set(suspend) | set(delist))
        self.initialzed = True

    def _to_simple_graph(self):
        """
        Compile into a simple TermGraph with no extra row metadata.

        Parameters
        ----------
        default_screen : zipline.pipeline.Term
            Term to use as a screen if self.screen is None.

        Returns
        -------
        graph : zipline.pipeline.graph.TermGraph
            Graph encoding term dependencies.
        """
        graph = TermGraph(self._terms_store).graph
        return graph

    def _inputs_for_term(self, term):
        if term.dependencies != NotSpecific :
            slice_inputs = keyfilter(lambda x: x in term.dependencies,
                                     self.initial_workspace_cache)
            input_of_term = reduce(lambda x, y: set(x) & set(y),
                                   slice_inputs.values())
        else:
            input_of_term = self.default
        return input_of_term

    def _decref_recursive(self,metadata):
        """
        Return a topologically-sorted list of the terms in ``self`` which
        need to be computed.

        Filters out any terms that are already present in ``workspace``, as
        well as any terms with refcounts of 0.

        Parameters
        ----------
        workspace : dict[Term, np.ndarray]
            Initial state of workspace for a pipeline execution. May contain
            pre-computed values provided by ``populate_initial_workspace``.
        refcounts : dict[Term, int]
            Reference counts for terms to be computed. Terms with reference
            counts of 0 do not need to be computed.
        """
        decref_nodes = self.pipeline_graph._decref_dependencies()

        for node in decref_nodes:
            _input = self._inputs_for_term(node)
            output = node._compute(_input,metadata)
            self.initial_workspace_cache[node] = output
            self._decref_recursive(metadata)

    def tag(self,outs):
        """将pipeline.name --- outs"""
        for asset in outs:
            asset.origin = self.name

    def to_execution_plan(self,pipeline_metadata,alternative = 1):
        """
            source: accumulated data from all terms
        """
        assert self.initialzied , ValueError('attach_default first')
        self.initial_workspace_cache = OrderedDict()
        self._decref_recursive(pipeline_metadata)
        outputs = self.initial_workspace_cache.popitem(last=True)
        self.initialzed = False
        #tag
        pipes = self.tag(outputs)
        #保留一定个数的标的
        return {self.name : pipes[:alternative + 1]}


class UmpPickers(object):
    """
        包括 --- 止损策略
        Examples:
            FeatureUnion(_name_estimators(transformers),weight = weight)
        裁决模块 基于有效的特征集，针对特定的asset进行投票抉择
        关于仲裁逻辑：
        H0假设 --- 标的退出
            迭代选股序列因子 --- 一旦任何因子投出反对票无法通过HO假设
        基于一个因子判断是否退出股票有失偏颇
    """
    def __init__(self,pickers):
        self._validate_features(pickers)

    def __setattr__(self, key, value):
        raise NotImplementedError

    def _validate_features(self,features):
        for feature in features:
            assert isinstance(feature,Term),ValueError('term type')
            if feature.dtype != bool:
                raise Exception('bool term needed for ump')
        self._poll_pickers = features

    def evaluate(self,holdings,_cache):
        _implement = partial(self._pick_for_sid,metadata = _cache)
        #执行退出算法
        with Pool(processes=len(holdings))as pool:
            picker_votes = [pool.apply_async(_implement, position)
                      for position in holdings]
            selector = [vote for vote in picker_votes if vote]
        return selector

    def _pick_for_sid(self,position, metadata):
        votes = [term_picker._compute([position.asset],metadata)
                                for term_picker in self._poll_pickers]
        if np.all(votes):
            return position
        return False


class SimplePipelineEngine(object):
    """
    Computation engines for executing Pipelines.

    This module defines the core computation algorithms for executing Pipelines.

    The primary entrypoint of this file is SimplePipelineEngine.run_pipeline, which
    implements the following algorithm for executing pipelines:

    1、Determine the domain of the pipeline.The domain determines the top-level
        set of dates and field that serves as row and column ---- data needed
        to compute the pipeline

    2. Build a dependency graph of all terms in TernmGraph with information
     about tropological tree of terms.

    3. Combine the domains of all terms to produce a overall data source.
        Each entry nodes(term) calculate outputs based on it.

    4. Iterate over the terms in the order computed . For each term:

       a. Fetch the term's inputs from the workspace and set_assert_finder
          with inputs

       b. Call ``term._compute`` with source . Store the results into
          the workspace.

       c. Decrement terms on the tropological tree and recursive the
          process.
    5. a. 不同的pipeline --- 各自执行算法，不干涉 ，就算标的重合（但是不同时间的买入），但是会在同一时间退出
       b. 每个Pipeline 存在一个alternatives(确保最大限度可以成交）,默认为最大持仓个数 --- len(self.pipelines)
          如果alternatives 太大 --- 降低标的准备行影响收益 ，如果太小 --- 到时空仓的概率变大影响收益（由于国内涨跌停制度）
    """
    __slots__ = (
        '_get_loader',
        'pipelines',
        'ump_picker'
    )

    def __init__(self,pipelines,ump_pickers):

        self._pipeline_cache = {}
        self._init_engine(pipelines,ump_pickers)

    def __setattr__(self, key, value):
        raise NotImplementedError

    def _init_engine(self,pipelines,_pickers):
        _inner_terms = chain(pipeline._terms_store
                             for pipeline in pipelines)
        engine_terms = set(_picker._poll_pickers + _inner_terms)
        self._get_loader = PricingLoader(engine_terms)
        self.ump_pickers = _pickers
        self.pipelines = pipelines

    def _cache_metadata(self,dts):
        """
        Register a Pipeline default for pipeline on every day.
        :param dts: initialize attach pipeline and cache metadata for engine
        :return:
        """
        #init pipelines
        for pipeline in self.pipelines:
            pipeline.attach_default(dts)
        # _cache_metada
        pipeline_type = [pipeline.default_type for pipeline in self.pipelines]
        metadata = self._get_loader.load_pipeline_arrays(dts,pipeline_type)
        return metadata

    def execute_engine(self, ledger):
        """
            计算ump和所有pipelines --- 如果ump为0，但是pipelines得到与持仓一直的标的相当于变相加仓
            umps --- 根据资产类别话费不同退出策略 ，symbols , etf , bond
        """
        capital = ledger.porfolio.cash
        holdings = ledger.positions
        dts = set([holding.inner_position.last_sync_date
                   for holding in holdings.values()])
        assert len(dts) == 1,Exception('positions must sync at the same time')
        # _cache_metdata
        _metadata = self._cache_metadata(dts[0])
        #执行算法逻辑
        polls = self._run_ump_pickers(holdings,_metadata)
        pipes = self._run_pipelines(_metadata)
        # --- 如果selectors 与 outs 存在交集
        puts,calls = self._resovle_conflicts(polls,pipes,holdings)

        self._pipeline_cache[dts[0]] = (puts,calls,holdings)
        return puts,calls,holdings,capital,dts[0]

    def _run_ump_pickers(self,holdings,_ump_metadata):
        dct = groupby(lambda x : x.inner_position.asset.asset_type,holdings)
        ump_outputs = []
        for name , position in dct.items():
            result = self.ump_pickers[name].evalute(position,_ump_metadata)
            ump_outputs.extend(result)
        return ump_outputs

    def _run_pipelines(self,pipeline_metadata):
        """
        Compute values for  pipelines on a specific date.
        Parameters
        ----------
        pipeline_metadata : cache data for pipeline
        """
        workers = len(self.pipelines)
        _implement = partial(self._run_pipeline_impl,
                             metadata = pipeline_metadata,
                             alternatives = workers)

        with Pool(processes = workers)as pool:
            results = [pool.apply_async(_implement, pipeline)
                      for pipeline in self.pipelines]
            outputs = merge(results)
        return outputs

    def _run_pipeline_impl(self,pipeline,metadata,alternatives):
        """
        ----------
        pipeline : zipline.pipeline.Pipeline
            The pipeline to run.
        """
        yield pipeline.to_execution_plan(metadata,alternatives)

    def _resovle_conflicts(self,outs,ins,holdings):
        """
            防止策略冲突 当pipeline的结果与ump的结果出现重叠 --- 说明存在问题，正常情况退出策略与买入策略应该不存交集

            1. engine共用一个ump ---- 解决了不同策略产生的相同标的可以同一时间退出
            2. engine --- 不同的pipeline对应不同的ump,产生1中的问题，相同的标的不会在同一时间退出是否合理（冲突）

            退出策略 --- 针对标的，与标的是如何产生的不存在直接关系;只能根据资产类别的有关 --- 1
            如果产生冲突 --- 当天卖出标的与买入标的产生重叠 说明策略是有问题的ump --- pipelines 对立的
            symbol ,etf 的退出策略可以相同，但是bond不行属于T+0
            return ---- name : [position , [pipeline_output]]

            两个部分 pipelines - ledger
                    positions -

            建仓逻辑 --- 逐步建仓 1/2 原则 --- 1 优先发生信号先建仓 ，后发信号仓位变为剩下的1/2（为了提高资金利用效率）
                                            2 如果没新的信号 --- 在已经持仓的基础加仓（不管资金是否足够或者设定一个底层资金池）
            ---- 变相限定了单次单个标的最大持仓为1/2
            position + pipeline - ledger ---  (当ledger为空 --- position也为空)

            关于ump --- 只要当天不是一直在跌停价格，以全部出货为原则，涉及一个滑价问题（position的成交额 与前一周的成交额占比
            评估滑价），如果当天没有买入，可以适当放宽（开盘的时候卖出大部分，剩下的等等） ；
            如果存在买入标的的行为则直接按照全部出货原则以open价格最大比例卖出 ，一般来讲集合竞价的代表主力卖入意愿强度）
            ---- 侧面解决了卖出转为买入的断层问题 transfer1
        """
        intersection = set([item.inner_position.asset for item in outs]) & set(chain(*ins.values()))
        if intersection:
            raise ValueError('ump should not have intersection with pipelines')
        out_dict = {position.inner_position.asset.origin : position
               for position in outs}
        waited = set(ins) - (set(holdings) - out_dict)
        result = keyfilter(lambda x : x in waited,ins)
        return out_dict,result


class NoEngineRegistered(Exception):
    """
    Raised if a user tries to call pipeline_output in an algorithm that hasn't
    set up a pipeline engine.
    """


def init_engine(self, pipelines, ump_pickers):
    """
    Initialize Pipeline API data.
    self.init_engine(get_pipeline_loader)
    self._pipelines = {}
    Construct and store a PipelineEngine from loader.

    If get_loader is None, constructs an ExplodingPipelineEngine
    """
    try:
        self.engine = SimplePipelineEngine(
            pipelines,
            ump_pickers
        )
    except Exception as e:
        print(e)


#撮合模块
import math,uuid,numpy as np
from abc import ABC, abstractmethod
from functools import lru_cache
from collections import OrderedDict
from numpy import isfinite
from enum import Enum
from toolz import valmap


class ExecutionStyle(ABC):
    """Base class for order execution styles.
    """

    @abstractmethod
    def get_limit_price(self,_is_buy):
        """
        Get the limit price for this order.
        Returns either None or a numerical value >= 0.
        """
        raise NotImplementedError

    @abstractmethod
    def get_stop_price(self,_is_buy):
        """
        Get the stop price for this order.
        Returns either None or a numerical value >= 0.
        """
        raise NotImplementedError


class MarketOrder(ExecutionStyle):
    """
    Execution style for orders to be filled at current market price.

    This is the default for orders placed with :func:`~zipline.api.order`.
    """

    def __init__(self, exchange=None):
        self._exchange = exchange

    def get_limit_price(self,_is_buy):
        return None

    def get_stop_price(self,_is_buy):
        return None


class LimitOrder(ExecutionStyle):
    """
    Execution style for orders to be filled at a price equal to or better than
    a specified limit price.

    Parameters
    ----------
    limit_price : float
        Maximum price for buys, or minimum price for sells, at which the order
        should be filled.
    """
    def __init__(self, limit_price):
        check_stoplimit_prices(limit_price, 'limit')

        self.limit_price = limit_price

    def get_limit_price(self,_is_buy):
        return self.limit_price

    def get_stop_price(self,_is_buy):
        return None


class StopOrder(ExecutionStyle):
    """
    Execution style representing a market order to be placed if market price
    reaches a threshold.

    Parameters
    ----------
    stop_price : float
        Price threshold at which the order should be placed. For sells, the
        order will be placed if market price falls below this value. For buys,
        the order will be placed if market price rises above this value.
    """
    def __init__(self, stop_price):
        check_stoplimit_prices(stop_price, 'stop')

        self.stop_price = stop_price

    def get_limit_price(self,_is_buy):
        return None

    def get_stop_price(self, _is_buy):
        return self.get_stop_price()


class StopLimitOrder(ExecutionStyle):
    """
    Execution style representing a limit order to be placed if market price
    reaches a threshold.

    Parameters
    ----------
    limit_price : float
        Maximum price for buys, or minimum price for sells, at which the order
        should be filled, if placed.
    stop_price : float
        Price threshold at which the order should be placed. For sells, the
        order will be placed if market price falls below this value. For buys,
        the order will be placed if market price rises above this value.
    """
    def __init__(self, limit_price, stop_price):
        check_stoplimit_prices(limit_price, 'limit')
        check_stoplimit_prices(stop_price, 'stop')

        self.limit_price = limit_price
        self.stop_price = stop_price

    def get_limit_price(self, _is_buy):
        return self.limit_price,

    def get_stop_price(self, _is_buy):
        return self.stop_price,

def check_stoplimit_prices(price, label):
    """
    Check to make sure the stop/limit prices are reasonable and raise
    a BadOrderParameters exception if not.
    """
    try:
        if not isfinite(price):
            raise Exception(
                "Attempted to place an order with a {} price "
                    "of {}.".format(label, price)
            )
    # This catches arbitrary objects
    except TypeError:
        raise Exception(
            "Attempted to place an order with a {} price "
                "of {}.".format(label, type(price))
        )

    if price < 0:
        raise Exception(
            "Can't place a {} order with a negative price.".format(label)
        )


class StyleType(Enum):
    """
        Market Price (市价单）
    """

    LMT = 'lmt'
    BOC = 'boc'
    BOP = 'bop'
    ITC = 'itc'
    B5TC = 'b5tc'
    B5TL = 'b5tl'
    FOK =  'fok'
    FAK =  'fak'


class Order(ABC):

    def make_id(self):
        return  uuid.uuid4().hex()

    @property
    def open_amount(self):
        return self.amount - self.filled

    @property
    def sid(self):
        # For backwards compatibility because we pass this object to
        # custom slippage models.
        return self.asset.sid

    @property
    def status(self):
        self._status = OrderStatus.OPEN

    @status.setter
    def status(self,status):
        self._status = status

    def to_dict(self):
        dct = {name : getattr(self.name)
               for name in self.__slots__}
        return dct

    def __repr__(self):
        """
        String representation for this object.
        """
        return "Order(%s)" % self.to_dict().__repr__()

    def __getstate__(self):
        """ pickle -- __getstate__ , __setstate__"""
        return self.__dict__()

    @abstractmethod
    def check_trigger(self,price,dt):
        """
        Given an order and a trade event, return a tuple of
        (stop_reached, limit_reached).
        For market orders, will return (False, False).
        For stop orders, limit_reached will always be False.
        For limit orders, stop_reached will always be False.
        For stop limit orders a Boolean is returned to flag
        that the stop has been reached.

        Orders that have been triggered already (price targets reached),
        the order's current values are returned.
        """
        raise NotImplementedError


class TickerOrder(Order):
    # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    """
        Parameters
        ----------
        asset : AssetEvent
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        dt : str, optional
            The date created order.

        市价单 --- 针对与卖出 --- 被动算法 ，基于时刻去卖出，这样避免被检测到 --- 将大订单拆分多个小订单然后基于时点去按照市价卖出

    """
    __slot__ = ['asset','_created_dt','capital']

    def __init__(self,asset,ticker,capital):
        self.asset = asset
        self._created_dt = ticker
        self.order_capital = capital
        self.direction = math.copysign(1,capital)
        self.filled = 0.0
        self.broker_order_id = self.make_id()
        self.order_type = StyleType.BOC

    def check_trigger(self,dts):
        if dts >= self._created_dt:
            return True
        return False


class RealtimeOrder(Order):
    # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    """
        Parameters
        ----------
        asset : AssetEvent
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        dt : str, optional
            The date created order.

        市价单 --- 针对与卖出 --- 被动算法 ，基于时刻去卖出，这样避免被检测到 --- 将大订单拆分多个小订单然后基于时点去按照市价卖出
        实时订单
    """
    __slot__ = ['asset', 'capital']

    def __init__(self, asset, capital):
        self.asset = asset
        self.order_capital = capital
        self.direction = math.copysign(1, capital)
        self.filled = 0.0
        self.broker_order_id = self.make_id()
        self.order_type = StyleType.BOC

    def check_trigger(self, dts):
        return True


class PriceOrder(Order):
    # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    """
        Parameters
        ----------
        asset : AssetEvent
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        dt : str, optional
            The date created order.

        限价单 --- 执行买入算法， 如果确定标的可以买入，偏向于涨的概率，主动买入而不是被动买入

        买1 价格超过卖1，买方以卖1价成交
    """
    __slot__ = ['asset','amount','lmt']

    def __init__(self,asset,amount,price):
        self.asset = asset
        self.amount = amount
        self.lmt_price = price
        self._created_dt = dt
        self.direction = math.copysign(1,self.amount)
        self.filled = 0.0
        self.broker_order_id = self.make_id()
        self.order_type = StyleType.LMT

    def check_trigger(self,bid):
        if bid <= self.lmt_price:
            return True
        return False


#过滤模块
class CancelPolicy(ABC):

    def to_asset(self,obj):
        if isinstance(obj, Position):
            asset = Position.inner_position.asset
        elif isinstance(obj, Asset):
            asset = obj
        else:
            raise TypeError()
        return asset

    @abstractmethod
    def should_cancel(self,obj,dt):
        raise NotImplementedError


class RestrictCancel(CancelPolicy):

    def __init__(self):
        self.adjust_array = AdjustArray()

    def should_cancel(self,obj,dts):
        """
            计算当天的open_pct是否达到涨停
            针对买入asset = self.to_asset(obj)
        """


class SwatCancel(CancelPolicy):

    def __init__(self):
        self.black_swat = Bar()

    @classmethod
    def should_cancel(self,obj,dts):
        asset = self.to_asset(obj)
        black = self.black_swat.load_blackSwat_kline(dts)
        try:
            event = black[asset]
        except KeyError:
            event = False
        return event


class ComposedCancel(CancelPolicy):
    """
     compose two rule with some composing function
    """
    def __init__(self,first,second):
        if not np.all(isinstance(first,CancelPolicy) and isinstance(second,CancelPolicy)):
            raise ValueError('only StatelessRule can be composed')

        self.first = first
        self.second = second

    def should_trigger(self,order):

        return self.first.should_cancel(order) & self.second.should_cancel(order)


#交易成本
class CommissionModel(ABC):
    """
        交易成本 分为订单如果有交易成本，考虑是否增加额外成本 ； 如果没有交易成本则需要计算
    """

    @abstractmethod
    def calculate(self, transaction):
        raise NotImplementedError


class NoCommission(CommissionModel):

    @staticmethod
    def calculate(order, transaction):
        return 0.0


class AssetCommission(CommissionModel):
    """
        1、印花税：1‰(卖的时候才收取，此为国家税收，全国统一)。
        2、过户费：深圳交易所无此项费用，上海交易所收费标准(按成交金额的0.02‰人民币[3])。
        3、交易佣金：最高收费为3‰，最低收费5元。各家劵商收费不一，开户前可咨询清楚。 2015年之后万/3
    """

    def __init__(self, multiplier = 3):
        self.mulitplier = multiplier

    @property
    def min_cost(self):
        return 5

    @min_cost.setter
    def min_cost(self,val):
        return val

    @lru_cache(maxsize= 10)
    def _init_base_cost(self,dt):
        base_fee = 1e-4 if dt > '2015-06-09' else 1e-3
        self.commission_rate = base_fee * self.mulitplier
        self.min_base_cost = self.min_cost / self.commission_rate

    def calculate(self, transaction):
        self._init_base_cost(transaction.dt)
        transaction_cost = transaction.amount * transaction.price
        #stamp_cost 印花税
        stamp_cost = 0 if transaction.amount > 0  else transaction_cost * 1e-3
        transfer_cost = transaction_cost * 1e-5 if transaction.asset.startswith('6') else 0
        trade_cost = transaction_cost * self.commission_rate \
            if transaction_cost > self.min_base_cost else self.min_cost
        txn_cost = stamp_cost + transfer_cost + trade_cost
        return txn_cost


class Transaction(object):

    # @expect_types(asset=Asset)
    __slots__ = ['asset_type','amount','price','dt']

    def __init__(self, asset,amount,price,dts):
        self.asset = asset
        self.amount = amount
        self.price = price
        self._created_dt = dts

    def __getitem__(self, name):
        return self.__dict__[name]

    def __repr__(self):
        template = (
            "{cls}(asset={asset}, dt={dt},"
            " amount={amount}, price={price})"
        )

        return template.format(
            cls=type(self).__name__,
            asset=self.asset,
            dt=self.dt,
            amount=self.amount,
            price=self.price
        )

    def __getstate__(self):
        """
            pickle dumps
        """
        p_dict = {name : self.name for name in self.__slots__}
        return p_dict


def create_transaction(price,amount,asset,dt):

    # floor the amount to protect against non-whole number orders
    # TODO: Investigate whether we can add a robust check in blotter
    # and/or tradesimulation, as well.
    amount_magnitude = int(abs(amount))

    if amount_magnitude < 100:
        raise Exception("Transaction magnitude must be at least 100 share")

    transaction = Transaction(
        amount=int(amount),
        price=price,
        asset = asset,
        dt = dt
    )

    return transaction


class SlippageModel(ABC):

    @abstractmethod
    def calculate_slippage_factor(self, *args):
        raise NotImplementedError


class NoSlippage(SlippageModel):
    """
        ideal model
    """

    def calculate_slippage_factor(self):
        return 1.0


class FixedBasisPointSlippage(SlippageModel):
    """
        basics_points * 0.0001
    """

    def __init__(self, basis_points=1.0):
        super(FixedBasisPointSlippage, self).__init__()
        self.basis_points = basis_points

    def calculate_slippage_factor(self, *args):
        return self.basis_points


class MarketImpact(SlippageModel):
    """
        基于成交量进行对市场的影响进行测算
    """
    def __init__(self,func = np.exp):
        self.adjust_func = func

    def calculate_slippage_factor(self,target,volume):
        psi = target / volume.mean()
        factor = self.adjust_func(psi)
        return factor


#资金分配
class CapitalAssign(ABC):
    """
        distribution base class
    """
    def __init__(self):
        self.adjust_portal = AdjustedArray()

    @abstractmethod
    def compute(self,asset_types,cash):
        raise NotImplementedError


class Average(CapitalAssign):

    def compute(self,assets,cash):
        per_cash = cash / len(assets)
        return {event.sid : per_cash for event in assets}


class Turtle(CapitalAssign):
    """
        基于波动率测算持仓比例 --- 基于策略形成的净值的波动性分配比例
        --- 收益率的sharp_ratio
    """
    def __init__(self,window):

        self._window = window

    def handle_data(self,sids,dt):
        adjust_arrays = self.adjust_portal.load_pricing_adjusted_array(
                                                            dt,
                                                            self.window,
                                                            ['open','high','low','close','volume'],
                                                            sids)
        return adjust_arrays

    def compute(self,dt,assets,cash):
        """基于数据的波动性以及均值"""
        raise NotImplementedError


class Kelly(CapitalAssign):
    """
        基于策略的胜率反向推导出凯利仓位
        ---- 策略胜率
    """
    def __init__(self,hitrate):
        assert hitrate , 'strategy hirate must not be None'
        self.win_rate = hitrate

    def _calculate_kelly(self,sid):
        """
            标的基于什么策略生产的
        """
        rate = self.win_rate[sid.reason]
        return 2 * rate -1

    def compute(self,dt,assets,cash):
        kelly_weight = {
                        asset: cash * self._calculate_kelly(asset)
                        for asset in assets
                        }
        return kelly_weight


import pandas as pd,datetime as dt
from itertools import chain


class MatchUp(object):
    """ 撮合成交
        如果open_pct 达到10% --- 是否买入
        分为不同的模块 创业板，科创板，ETF
        包含 --- sell orders buy orders 同时存在，但是buy_orders --- 基于sell orders 和 ledger
        通过限制买入capital的pct实现分布买入
        但是卖出订单 --- 通过追加未成交订单来实现
        如何连接卖出与买入模块

        由capital --- calculate orders 应该属于在统一模块 ，最后将订单 --- 引擎生成交易 --- 执行计划式的，非手动操作类型的
        剔除ReachCancel --- 10%
        剔除SwatCancel --- 黑天鹅
    """
    def __init__(self,
                multiplier = 5,
                cancel_policy = [ReachCancel,SwatCancel],
                execution_style = MarketOrder,
                slippageModel = MarketImpact):

        #确定订单类型默认为市价单
        self.style = execution_style
        self.commission = AssetCommission(multiplier)
        self.cancel_policy = ComposedCancel(cancel_policy)
        self.engine = Engine()
        self.adjust_array = AdjustArray()
        self.record_transactions = OrderedDict()
        self.record_efficiency = OrderedDict()
        self.prune_closed_assets = OrderedDict()

    @property
    def _fraction(self):
        """设立成交量限制，默认为前一个交易日的百分之一"""
        return 0.05

    @_fraction.setter
    def _fraction(self,val):
        return val

    def load_data(self,dt,assets):
        raw = self.adjust_array.load_array_for_sids(dt,0,['open','close','volume','amount','pct'],assets)
        volume = { k : v['volume'] for k,v in raw.items()}
        if raw:
            """说明历史回测 --- 存在数据"""
            preclose = { k: v['close'] / (v['pct'] +1 ) for k,v in raw.items()}
            open_pct = { k: v['open'] / preclose[k] for k,v in raw.items()}
        else:
            """实时回测 , 9.25之后"""
            raw = self.adjust_array.load_pricing_adjusted_array(dt,2,['open','close','pct'],assets)
            minutes = self.adjust_array.load_minutes_for_sid(assets)
            if not minutes:
                raise ValueError('时间问题或者网路问题')
            preclose = { k : v['close'][-1]  for k,v in raw.items() }
            open_pct = { k : v.iloc[0,0] / preclose[k] for k,v in minutes.items()}
        dct = {'preclose':preclose,'open_pct':open_pct,'volume':volume}
        return dct

    #获取日数据，封装为一个API(fetch process flush other api)
    # def _create_bar_data(self, universe_func):
    #     return BarData(
    #         data_portal=self.data_portal,
    #         simulation_dt_func=self.get_simulation_dt,
    #         data_frequency=self.sim_params.data_frequency,
    #         trading_calendar=self.algo.trading_calendar,
    #         restrictions=self.restrictions,
    #         universe_func=universe_func
    #     )

    # def _load_tick_data(self,asset,ticker):
    #     """获取当天实时的ticer实点的数据，并且增加一些滑加，+/-0.01"""
    #     minutes = self.adjust_array.load_minutes_for_sid(asset,ticker)
    #     return minutes

    def execute_cancel_policy(self,target):
        """买入 --- 如果以涨停价格开盘过滤，主要针对买入标的"""
        _target =[self.cancel_policy.should_cancel(item) for item in target]
        result = _target[0] if _target else None
        return result

    def _restrict_buy_rule(self,dct):
        """
            主要针对于买入标的的
            对于卖出的，遵循最大程度卖出
        """
        self.capital_limit = valmap(lambda x : x * self._fraction,dct)

    def attach_pruned_holdings(self,puts,holdings):
        closed_holdings = valfilter(lambda x: x.inner_position.asset in self.prune_closed_assets, holdings)
        puts.update(closed_holdings)
        return puts

    def carry_out(self,engine,ledger):
        """建立执行计划"""
        #engine --- 获得可以交易的标的
        puts, calls,holdings,capital,dts = engine.execute_engine(ledger)
        #将未完成的卖出的标的放入puts
        puts = self.attach_pruned_holdings(puts,holdings)
        self.commission._init_base_cost(dts)
        #获取计算订单所需数据
        assets = set([position.inner_position.asset for position in holdings]) | set(chain(*calls.values()))
        raw = self.load_data(dts,assets)
        #过滤针对标的
        calls = valmap(lambda x:self.execute_cancel_policy(x),calls)
        calls = valfilter(lambda x : x is not None,calls)
        call_assets = list(calls.values())
        #已有持仓标的
        holding_assets = [holding.inner_position.asset for holding in holdings]
        #卖出持仓标的
        put_assets = [ put.inner_position.asset for put in puts]
        # 限制 --- buys_amount,sell --- volume
        self._restrict_rule(raw['amount'])
        #固化参数
        match_impl = partial(self.positive_match(holdings = holding_assets,capital = capital,raw = raw,dts = dts))
        _match_impl = partial(self.dual_match(holdings = holding_assets,capital = capital,raw = raw,dts = dts))
        _match_impl(put_assets,call_assets) if puts else match_impl(call_assets)
        #获取存量的transactions
        final_txns = self._init_engine(dts)
        #计算关于总的订单拆分引擎撮合成交的的效率
        self.evaluate_efficiency(capital,puts,dts)
        #将未完成需要卖出的标的继续卖出
        self.to_be_pruned(puts)
        return final_txns

    def _init_engine(self,dts):
        txns = self.engine.engine_transactions
        self.record_transactions[dts] = txns
        self.engine.reset()
        return txns

    def evaluate_efficiency(self,capital,puts,dts):
        """
            根据标的追踪 --- 具体卖入订单根据volume计算成交率，买入订单根据成交额来计算资金利用率 --- 评估撮合引擎撮合的的效率
        """
        txns = self.record_transactions[dts]
        call_efficiency = sum([ txn.amount * txn.price for txn in txns if txn.amount > 0 ]) / capital
        put_efficiency = sum([txn.amount for txn in txns if txn.amount < 0]) / \
                         sum([position.inner_position.amount for position in puts.values()]) if puts else 0
        self.record_efficiency[dts] = {'call':call_efficiency,'put':put_efficiency}

    def to_be_pruned(self,dts,puts):
        #将未完全卖出的position存储继续卖出
        txns = self.record_transactions[dts]
        txn_put_amount = {txn.asset:txn.amount for txn in txns if txn.amount < 0}
        position_amount = {position.inner_position.asset : position.inner_position.amount for position in puts}
        pct = txn_put_amount / position_amount
        uncompleted = keyfilter(lambda x : x < 1,pct)
        self.prune_closed_assets[dts] = uncompleted.keys()

    def positive_match(self,calls,holdings,capital,raw,dts):
        """buys or sells parallel"""
        if calls:
            capital_dct = self.policy.calculate(calls,capital,dts)
        else:
            capital_dct = self.policy.calculate(holdings, capital,dts)
        #买入金额限制
        restrict_capital = {asset : self.capital_limit[asset] if capital >= self.capital_limit[asset]
                                    else capital  for asset ,capital in capital_dct.items()}

        call_impl = partial(self.engine.call,raw = raw,min_base_cost = self.commission.min_base_cost)
        with Pool(processes=len(restrict_capital))as pool:
            results = [pool.apply_async(call_impl,asset,capital)
                       for asset,capital in restrict_capital.items()]
            txns = chain(*results)
        return txns

    def dual_match(self,puts,calls,holdings,capital,dts,raw):
        #双向匹配
        """基于capital生成priceOrder"""
        txns = dict()
        if calls:
            capital_dct = self.policy.calculate(calls,capital,dts)
        else:
            left_holdings = set(holdings) - set(puts)
            capital_dct = self.policy.calculate(left_holdings,capital,dts)
        #call orders
        txns['call'] = self.positive_match(calls,holdings,capital_dct,raw,dts)
        #put orders
        # --- 直接以open_price卖出;如果卖出的话 --- 将未成交的卖出订单orders持续化
        for txn_capital in self.engine.put(puts,calls,raw,self.commission.min_base_cost):
            agg = sum(capital_dct.values())
            trading_capital = valmap(lambda x : x * txn_capital / agg,capital_dct )
            self.engine._infer_order(trading_capital)


class Engine(ABC):
    """
        1 存在价格笼子
        2 无跌停限制但是存在竞价机制（10%基准价格），以及临时停盘制度
        有存在竞价限制，科创板2% ，或者可转债10%
        第十八条 债券现券竞价交易不实行价格涨跌幅限制。
　　             第十九条 债券上市首日开盘集合竞价的有效竞价范围为发行价的上下 30%，连续竞价、收盘集合竞价的有效竞价范围为最近成交价的上下 10%；
        非上市首日开盘集合竞价的有效竞价范围为前收盘价的上下 10%，连续竞价、收盘集合竞价的有效竞价范围为最近成交价的上下 10%。
         一、可转换公司债券竞价交易出现下列情形的，本所可以对其实施盘中临时停牌措施：
    　　（一）盘中成交价较前收盘价首次上涨或下跌达到或超过20%的；
    　　（二）盘中成交价较前收盘价首次上涨或下跌达到或超过30%的。
    """
    def reset(self):
        #当MatchUp运行结束之后
        self.engine_transactions = []

    @abstractmethod
    def _create_orders(self,asset,raw,**kwargs):
        """
            按照价格在10% 至 -10%范围内基于特定的统计分布模拟价格 --- 方向为开盘的涨跌幅 ，不适用于科创板（竞价机制要求）---买入价格不能超过基准价格（卖一的
            102%，卖出价格不得低于买入价格98%，申报最小200股，递增可以以1股为单位 ；设立市价委托必须设立最高价以及最低价 ；
            A股主板，中小板首日涨幅最大为44%而后10%波动，而科创板前5个交易日不设立涨跌停而后20%波动但是30%，60%临时停盘10分钟，如果超过2.57(复盘)；
            科创板盘后固定价格交易 --- 以后15:00收盘价格进行交易 --- 15:00 -- 15:30(按照时间优先原则，逐步撮合成交）
        """
        raise NotImplementedError

    @abstractmethod
    def simulate_dist(self,alpha,size):
        """
        simulate price distribution to place on transactions
        :param size: number of transactions
        :param raw:  data for compute
        :param multiplier: slippage base on vol_pct ,e.g. slippage = np.exp(vol_pct)
        :return: array of simualtion price
        """
        raise NotImplementedError

    def market_orders(self, capital, asset):
        """按照市价竞价按照size --- 时间分割 TickerOrder"""
        min_base_cost = self.commission.min_base_cost
        size = capital / min_base_cost
        tick_interval = self.simulate_dist(size)
        for dts in tick_interval:
            # 根据设立时间去定义订单
            order = TickerOrder(asset,dts,min_base_cost)
            self.oms(order, eager=True)

    def call(self, capital, asset,raw,min_base_cost):
        """执行前固化的订单买入计划"""
        if not asset.bid_rule:
            """按照价格区间竞价,适用于没有没有竞价机制要求，不会产生无效订单 PriceOrder"""
            under_orders = self._create_orders(asset,
                                               raw,
                                               capital=capital,
                                               min_base_cost = min_base_cost)
        else:
            under_orders = self.market_orders(capital, asset)
        #执行买入订单
        self.internal_oms(under_orders)

    def _infer_order(self,capital_dct):
        """基于时点执行买入订单,时间为进行入OMS系统的时间 --- 为了衔接卖出与买入"""
        orders = [RealtimeOrder(asset,capital) for asset,capital in capital_dct]
        self.oms(orders)

    def _put_impl(self,position,raw,min_base_cost):
        """按照市价竞价"""
        amount = position.inner_position.amount
        asset = position.inner_position.asset
        last_sync_price = position.inner_position.last_sync_price
        if not asset.bid_rule:
            """按照价格区间竞价,适用于没有没有竞价机制要求，不会产生无效订单"""
            tiny_put_orders = self._create_orders(asset,
                                                  raw,
                                                  amount = amount,
                                                  min_base_cost = min_base_cost)
        else:
            min_base_cost = self.commission.min_base_cost
            per_amount = np.ceil(self.multiplier['put'] * min_base_cost / (last_sync_price * 100))
            size = amount / per_amount
            #按照size --- 时间分割
            intervals = self.simulate_tick(size)
            for dts in intervals:
                tiny_put_orders = TickerOrder(per_amount * 100,asset,dts)
        return tiny_put_orders

    @staticmethod
    def simulate_tick(size,final = True):
        interval = 4 * 60 / size
        # 按照固定时间去执行
        day_m = pd.date_range(start='09:30', end='11:30', freq='%dmin'%interval)
        day_a = pd.date_range(start='13:00', end='14:57', freq='%dmin'%interval)
        day_ticker = list(chain(*zip(day_m, day_a)))
        if final:
            last = pd.Timestamp('2020-06-17 14:57:00',freq='%dmin'%interval)
            day_ticker.append(last)
        return day_ticker

    def put(self,puts,raw,min_base_cost):
        put_impl = partial(self._put_impl,
                           raw = raw,
                           min_base_cost = min_base_cost)
        with Pool(processes=len(puts))as pool:
            results = [pool.apply_async(put_impl,position)
                       for position in puts.values]
            put_orders = chain(*results)
            # 执行卖出订单 --- 返回标识
        for txn in self.internal_oms(put_orders, dual=True):
                #一旦有订单成交 基于队虽然有延迟，但是不影响
                txn_capital = txn.amount * txn.price
                yield txn_capital

    @abstractmethod
    def internal_oms(self,orders,eager = True):
        """
            principle --- 只要发出卖出信号的最大限度的卖出，如果没有完全卖出直接转入下一个交易日继续卖出
            订单 --- priceOrder TickerOrder Intime
            engine --- xtp or simulate(slippage_factor = self.slippage.calculate_slippage_factor)
            dual -- True 双方向
                  -- False 单方向（提交订单）
            eager --- True 最后接近收盘时候集中将为成交的订单成交撮合成交保持最大持仓
                  --- False 将为成交的订单追加之前由于restrict_rule里面的为成交订单里面
            具体逻辑：
                当产生执行卖出订单时一旦成交接着执行买入算法，要求卖出订单的应该是买入Per买入标的的times，
                保证一次卖出成交金额可以覆盖买入标的
            优势：提前基于一定的算法将订单根据时间或者价格提前设定好，在一定程度避免了被监测的程度。
            成交的订单放入队列里面，不断的get
            针对于put orders 生成的买入ticker_orders （逻辑 --- 滞后的订单是优先提交，主要由于订单生成到提交存在一定延迟)
            订单优先级 --- Intime (first) > TickerOrder > priceOrder
            基于asset计算订单成交比例
            获取当天实时的ticer实点的数据，并且增加一些滑加，+ /-0.01
        """
        raise NotImplementedError


class BackEngine(Engine):
    """
        基于ticker --- 进行回测,在执行具体的买入标的基于ticker数据真实模拟
    """
    def __init__(self,
                multiplier = {'call':1.5,'put':2},
                slippageModel = MarketImpact):

        # multipiler --- 针对基于保持最低交易成的capital的倍数进行交易
        self.multiplier = multiplier
        self.slippage = slippageModel()
        self.engine_transactions = []

    def _create_orders(self,asset,raw,**kwargs):
        """
            按照价格在10% 至 -10%范围内基于特定的统计分布模拟价格 --- 方向为开盘的涨跌幅 ，不适用于科创板（竞价机制要求）---买入价格不能超过基准价格（卖一的
            102%，卖出价格不得低于买入价格98%，申报最小200股，递增可以以1股为单位 ；设立市价委托必须设立最高价以及最低价 ；
            A股主板，中小板首日涨幅最大为44%而后10%波动，而科创板前5个交易日不设立涨跌停而后20%波动但是30%，60%临时停盘10分钟，如果超过2.57(复盘)；
            科创板盘后固定价格交易 --- 以后15:00收盘价格进行交易 --- 15:00 -- 15:30(按照时间优先原则，逐步撮合成交）
        """
        multiplier = self.multiplier['call']
        min_base_cost = kwargs['min_base_cost']
        preclose = raw['preclose'][asset]
        open_pct = raw['open_pct'][asset]
        volume_restriction = self.volume_limit[asset]
        try:
            capital = kwargs['capital']
            #ensuer_amount --- 手
            bottom_amount = np.floor(capital / (preclose * 110))
            if bottom_amount == 0:
                raise ValueError('satisfied at least 100 stocks')
            #是否超过限制
            ensure_amount = bottom_amount if bottom_amount <= volume_restriction else volume_restriction
        except KeyError:
            amount = kwargs['amount']
            ensure_amount = amount if amount <= volume_restriction else volume_restriction
        # 计算拆分订单的个数，以及单个订单金额
        min_per_value = 90 * preclose / (open_pct + 1)
        ensure_per_amount = np.ceil(multiplier * min_base_cost / min_per_value)
        # 模拟价格分布的参数 --- 个数 数据 滑价系数
        size = ensure_amount // ensure_per_amount
        # volume = raw['volume'][asset]
        alpha = 1 if open_pct == 0.00 else 100 * open_pct
        sim_pct = self.simulate_dist(abs(alpha),size)
        # 限价原则 --- 确定交易执行价格 针对于非科创板，创业板股票
        # limit = self.style.get_limit_price() if self.style.get_limit_price() else asset.price_limit(dts)
        # stop = self.style.get_stop_price() if self.style.get_stop_price() else asset.price_limit(dts)
        limit = self.style.get_limit_price() if self.style.get_limit_price() else 0.1
        stop = self.style.get_stop_price() if self.style.get_stop_price() else 0.1
        clip_price = np.clip(sim_pct,-stop,limit) * preclose
        # 将多余的手分散
        sim_amount = np.tile([ensure_per_amount], size) if size > 0 else [ensure_amount]
        random_idx = np.random.randint(0, size, ensure_amount % ensure_per_amount)
        for idx in random_idx:
            sim_amount[idx] += 1
        #形成订单
        tiny_orders =  [PriceOrder(asset,args[0],args[1])
                     for args in zip(sim_amount,clip_price)]
        return tiny_orders

    def simulate_dist(self,alpha,size):
        """
        simulate price distribution to place on transactions
        :param size: number of transactions
        :param raw:  data for compute
        :param multiplier: slippage base on vol_pct ,e.g. slippage = np.exp(vol_pct)
        :return: array of simualtion price
        """
        # 涉及slippage --- 基于ensure_amount --- multiplier
        if size > 0:
            #模拟价格分布
            dist = 1 + np.copysign(alpha,np.random.beta(alpha,100,size))
        else:
            dist = [1 + alpha  / 100]
        return dist

    def internal_oms(self,orders,eager = True):
        """
            principle --- 只要发出卖出信号的最大限度的卖出，如果没有完全卖出直接转入下一个交易日继续卖出
            订单 --- priceOrder TickerOrder Intime
            engine --- xtp or simulate(slippage_factor = self.slippage.calculate_slippage_factor)
            dual -- True 双方向
                  -- False 单方向（提交订单）
            eager --- True 最后接近收盘时候集中将为成交的订单成交撮合成交保持最大持仓
                  --- False 将为成交的订单追加之前由于restrict_rule里面的为成交订单里面
            具体逻辑：
                当产生执行卖出订单时一旦成交接着执行买入算法，要求卖出订单的应该是买入Per买入标的的times，
                保证一次卖出成交金额可以覆盖买入标的
            优势：提前基于一定的算法将订单根据时间或者价格提前设定好，在一定程度避免了被监测的程度。
            成交的订单放入队列里面，不断的get
            针对于put orders 生成的买入ticker_orders （逻辑 --- 滞后的订单是优先提交，主要由于订单生成到提交存在一定延迟)
            订单优先级 --- Intime (first) > TickerOrder > priceOrder
            基于asset计算订单成交比例
            获取当天实时的ticer实点的数据，并且增加一些滑加，+ /-0.01
        """


class AlgorithmSimulator(object):

    EMISSION_TO_PERF_KEY_MAP = {
        'minute': 'minute_perf',
        'daily': 'daily_perf'
    }

    def __init__(self, algo,sim_params, clock, benchmark,
                universe_func):

        # ==============
        # Simulation
        # Param Setup
        # ==============
        self.sim_params = sim_params

        # ==============
        # Algo Setup
        # ==============
        self.algo = algo

        self.benchmark = benchmark

        # We don't have a datetime for the current snapshot until we
        # receive a message.
        self.simulation_dt = None
        self.clock = clock
        # This object is the way that user algorithms interact with OHLCV data,
        # fetcher data, and some API methods like `data.can_trade`.
        self.current_data = self._create_bar_data(universe_func)

    def get_simulation_dt(self):
        return self.simulation_dt

    #获取日数据，封装为一个API(fetch process flush other api)
    def _create_bar_data(self, universe_func):
        return BarData(
            data_portal=self.data_portal,
            simulation_dt_func=self.get_simulation_dt,
            data_frequency=self.sim_params.data_frequency,
            trading_calendar=self.algo.trading_calendar,
            restrictions=self.restrictions,
            universe_func=universe_func
        )

    def _create_clock(self):
        """
        If the clock property is not set, then create one based on frequency.
        """
        trading_o_and_c = self.trading_calendar.schedule.ix[
            self.sim_params.sessions]
        market_closes = trading_o_and_c['market_close']
        minutely_emission = False

        if self.sim_params.data_frequency == 'minute':
            market_opens = trading_o_and_c['market_open']
            minutely_emission = self.sim_params.emission_rate == "minute"

            # The calendar's execution times are the minutes over which we
            # actually want to run the clock. Typically the execution times
            # simply adhere to the market open and close times. In the case of
            # the futures calendar, for example, we only want to simulate over
            # a subset of the full 24 hour calendar, so the execution times
            # dictate a market open time of 6:31am US/Eastern and a close of
            # 5:00pm US/Eastern.
            execution_opens = \
                self.trading_calendar.execution_time_from_open(market_opens)
            execution_closes = \
                self.trading_calendar.execution_time_from_close(market_closes)
        else:
            # in daily mode, we want to have one bar per session, timestamped
            # as the last minute of the session.
            execution_closes = \
                self.trading_calendar.execution_time_from_close(market_closes)
            execution_opens = execution_closes

        # FIXME generalize these values
        before_trading_start_minutes = days_at_time(
            self.sim_params.sessions,
            time(8, 45),
            "US/Eastern"
        )

        return MinuteSimulationClock(
            self.sim_params.sessions,
            execution_opens,
            execution_closes,
            before_trading_start_minutes,
            minute_emission=minutely_emission,
        )


    def tranform(self):
        """
        Main generator work loop.
        """
        algo = self.algo
        ledger = algo.ledger
        metrics_tracker = algo.metrics_tracker

        def once_a_day(dt):
            # daily metrics run
            metrics_tracker.handle_market_open(dt)
            #生成交易订单
            txns = MatchUp.carry_out(algo.engine,ledger)
            #处理交易订单
            ledger.process_transaction(txns)
            algo.calculate_capital_changes(midnight_dt,
                                           emission_rate=emission_rate,
                                           is_interday=True)
            algo.on_dt_changed(midnight_dt)

        def on_exit():
            # Remove references to algo, data portal, et al to break cycles
            # and ensure deterministic cleanup of these objects when the
            # simulation finishes.
            self.algo = None
            self.benchmark_source = self.current_data = self.data_portal = None

        with ExitStack() as stack:
            """
            由于已注册的回调是按注册的相反顺序调用的，因此最终行为就好像with 已将多个嵌套语句与已注册的一组回调一起使用。
            这甚至扩展到异常处理-如果内部回调抑制或替换异常，则外部回调将基于该更新状态传递自变量。
            enter_context  输入一个新的上下文管理器，并将其__exit__()方法添加到回调堆栈中。返回值是上下文管理器自己的__enter__()方法的结果。
            callback（回调，* args，** kwds ）接受任意的回调函数和参数，并将其添加到回调堆栈中。
            """

            stack.callback(on_exit)
            stack.enter_context(self.processor)
            stack.enter_context(ZiplineAPI(self.algo))

            metrics_tracker.handle_start_of_simulation(self.benchmark)

            for dt, action in self.clock:
                if action == BAR:
                    for capital_change_packet in every_bar(dt):
                        yield capital_change_packet
                elif action == SESSION_START:
                    for capital_change_packet in once_a_day(dt):
                        yield capital_change_packet
                elif action == SESSION_END:
                    # End of the session.
                    positions = metrics_tracker.positions
                    position_assets = algo.asset_finder.retrieve_all(positions)
                    self._cleanup_expired_assets(dt, position_assets)

                    execute_order_cancellation_policy()
                    algo.validate_account_controls()

                    yield self._get_daily_message(dt, algo, metrics_tracker)
                elif action == BEFORE_TRADING_START_BAR:
                    self.simulation_dt = dt
                    algo.on_dt_changed(dt)
                    algo.before_trading_start(self.current_data)
                elif action == MINUTE_END:
                    minute_msg = self._get_minute_message(
                        dt,
                        algo,
                        metrics_tracker,
                    )
                    yield minute_msg

            for dt in algo.trading_calendar:
                once_a_day(dt)

            # risk_message = metrics_tracker.handle_simulation_end(
            #     self.data_portal,
            # )
            risk_message = metrics_tracker.handle_simulation_end()

            yield risk_message

    def _get_daily_message(self, algo, metrics_tracker):
        """
        Get a perf message for the given datetime.
        """
        perf_message = metrics_tracker.handle_market_close()

        # perf_message = metrics_tracker.handle_market_close(
        #     dt,
        #     self.data_portal,
        # )
        perf_message['daily_perf']['recorded_vars'] = algo.recorded_vars
        return perf_message

    def _create_benchmark_source(self):
        if self.benchmark_sid is not None:
            benchmark_asset = self.asset_finder.retrieve_asset(
                self.benchmark_sid
            )
            benchmark_returns = None
        else:
            if self.benchmark_returns is None:
                raise ValueError("Must specify either benchmark_sid "
                                 "or benchmark_returns.")
            benchmark_asset = None
            # get benchmark info from trading environment, which defaults to
            # downloading data from IEX Trading.
            benchmark_returns = self.benchmark_returns
        return BenchmarkSource(
            benchmark_asset=benchmark_asset,
            benchmark_returns=benchmark_returns,
            trading_calendar=self.trading_calendar,
            sessions=self.sim_params.sessions,
            data_portal=self.data_portal,
            emission_rate=self.sim_params.emission_rate,
        )

    def _create_metrics_tracker(self):
        #'start_of_simulation','end_of_simulation','start_of_session'，'end_of_session','end_of_bar'
        return MetricsTracker(
            trading_calendar=self.trading_calendar,
            first_session=self.sim_params.start_session,
            last_session=self.sim_params.end_session,
            capital_base=self.sim_params.capital_base,
            emission_rate=self.sim_params.emission_rate,
            data_frequency=self.sim_params.data_frequency,
            asset_finder=self.asset_finder,
            metrics=self._metrics_set,
        )

    def _create_generator(self, sim_params):
        if sim_params is not None:
            self.sim_params = sim_params

        self.metrics_tracker = metrics_tracker = self._create_metrics_tracker()

        # Set the dt initially to the period start by forcing it to change.
        self.on_dt_changed(self.sim_params.start_session)

        if not self.initialized:
            self.initialize(**self.initialize_kwargs)
            self.initialized = True

        benchmark_source = self._create_benchmark_source()

        self.trading_client = AlgorithmSimulator(
            self,
            sim_params,
            self.data_portal,
            self._create_clock(),
            benchmark_source,
            self.restrictions,
            universe_func=self._calculate_universe
        )

        metrics_tracker.handle_start_of_simulation(benchmark_source)
        return self.trading_client.transform()

    def get_generator(self):
        """
        Override this method to add new logic to the construction
        of the generator. Overrides can use the _create_generator
        method to get a standard construction generator.
        """
        return self._create_generator(self.sim_params)

    def analyze(self, perf):
        # 分析stats
        if self._analyze is None:
            return

        with ZiplineAPI(self):
            self._analyze(self, perf)

    def run(self, data_portal=None):
        """Run the algorithm.
        """
        # HACK: I don't think we really want to support passing a data portal
        # this late in the long term, but this is needed for now for backwards
        # compat downstream.
        if data_portal is not None:
            self.data_portal = data_portal
            self.asset_finder = data_portal.asset_finder
        elif self.data_portal is None:
            raise RuntimeError(
                "No data portal in TradingAlgorithm.run().\n"
                "Either pass a DataPortal to TradingAlgorithm() or to run()."
            )
        else:
            assert self.asset_finder is not None, \
                "Have data portal without asset_finder."

        # Create zipline and loop through simulated_trading.
        # Each iteration returns a perf dictionary
        try:
            perfs = []
            for perf in self.get_generator():
                perfs.append(perf)

            # convert perf dict to pandas dataframe
            daily_stats = self._create_daily_stats(perfs)

            self.analyze(daily_stats)
        finally:
            self.data_portal = None
            self.metrics_tracker = None

        return daily_stats

    def _create_daily_stats(self, perfs):
        # create daily and cumulative stats dataframe
        daily_perfs = []
        # TODO: the loop here could overwrite expected properties
        # of daily_perf. Could potentially raise or log a
        # warning.
        for perf in perfs:
            if 'daily_perf' in perf:

                perf['daily_perf'].update(
                    perf['daily_perf'].pop('recorded_vars')
                )
                perf['daily_perf'].update(perf['cumulative_risk_metrics'])
                daily_perfs.append(perf['daily_perf'])
            else:
                self.risk_report = perf

        daily_dts = pd.DatetimeIndex(
            [p['period_close'] for p in daily_perfs], tz='UTC'
        )
        daily_stats = pd.DataFrame(daily_perfs, index=daily_dts)
        return daily_stats

    #根据dt获取change,动态计算，更新数据
    def calculate_capital_changes(self, dt, emission_rate, is_interday,
                                  portfolio_value_adjustment=0.0):
        """
        If there is a capital change for a given dt, this means the the change
        occurs before `handle_data` on the given dt. In the case of the
        change being a target value, the change will be computed on the
        portfolio value according to prices at the given dt

        `portfolio_value_adjustment`, if specified, will be removed from the
        portfolio_value of the cumulative performance when calculating deltas
        from target capital changes.
        """
        try:
            capital_change = self.capital_changes[dt]
        except KeyError:
            return

        self._sync_last_sale_prices()
        if capital_change['type'] == 'target':
            target = capital_change['value']
            capital_change_amount = (
                target -
                (
                    self.portfolio.portfolio_value -
                    portfolio_value_adjustment
                )
            )

            log.info('Processing capital change to target %s at %s. Capital '
                     'change delta is %s' % (target, dt,
                                             capital_change_amount))
        elif capital_change['type'] == 'delta':
            target = None
            capital_change_amount = capital_change['value']
            log.info('Processing capital change of delta %s at %s'
                     % (capital_change_amount, dt))
        else:
            log.error("Capital change %s does not indicate a valid type "
                      "('target' or 'delta')" % capital_change)
            return

        self.capital_change_deltas.update({dt: capital_change_amount})
        self.metrics_tracker.capital_change(capital_change_amount)

        yield {
            'capital_change':
                {'date': dt,
                 'type': 'cash',
                 'target': target,
                 'delta': capital_change_amount}
        }

    @api_method
    def get_environment(self, field='platform'):
        """Query the execution environment.

        Parameters
        ----------
        field : {'platform', 'arena', 'data_frequency',
                 'start', 'end', 'capital_base', 'platform', '*'}
            The field to query. The options have the following meanings:
              arena : str
                  The arena from the simulation parameters. This will normally
                  be ``'backtest'`` but some systems may use this distinguish
                  live trading from backtesting.
              data_frequency : {'daily', 'minute'}
                  data_frequency tells the algorithm if it is running with
                  daily data or minute data.
              start : datetime
                  The start date for the simulation.
              end : datetime
                  The end date for the simulation.
              capital_base : float
                  The starting capital for the simulation.
              platform : str
                  The platform that the code is running on. By default this
                  will be the string 'zipline'. This can allow algorithms to
                  know if they are running on the Quantopian platform instead.
              * : dict[str -> any]
                  Returns all of the fields in a dictionary.

        Returns
        -------
        val : any
            The value for the field queried. See above for more information.

        Raises
        ------
        ValueError
            Raised when ``field`` is not a valid option.
        """
        env = {
            'arena': self.sim_params.arena,
            'data_frequency': self.sim_params.data_frequency,
            'start': self.sim_params.first_open,
            'end': self.sim_params.last_close,
            'capital_base': self.sim_params.capital_base,
            'platform': self._platform
        }
        if field == '*':
            return env
        else:
            try:
                return env[field]
            except KeyError:
                raise ValueError(
                    '%r is not a valid field for get_environment' % field,
                )

    def add_event(self, rule, callback):
        """Adds an event to the algorithm's EventManager.

        Parameters
        ----------
        rule : EventRule
            The rule for when the callback should be triggered.
        callback : callable[(context, data) -> None]
            The function to execute when the rule is triggered.
        """
        self.event_manager.add_event(
            zipline.utils.events.Event(rule, callback),
        )

    @api_method
    def schedule_function(self,
                          func,
                          date_rule=None,
                          time_rule=None,
                          half_days=True,
                          calendar=None):
        """
        Schedule a function to be called repeatedly in the future.

        Parameters
        ----------
        func : callable
            The function to execute when the rule is triggered. ``func`` should
            have the same signature as ``handle_data``.
        date_rule : zipline.utils.events.EventRule, optional
            Rule for the dates on which to execute ``func``. If not
            passed, the function will run every trading day.
        time_rule : zipline.utils.events.EventRule, optional
            Rule for the time at which to execute ``func``. If not passed, the
            function will execute at the end of the first market minute of the
            day.
        half_days : bool, optional
            Should this rule fire on half days? Default is True.
        calendar : Sentinel, optional
            Calendar used to compute rules that depend on the trading calendar.

        See Also
        --------
        :class:`zipline.api.date_rules`
        :class:`zipline.api.time_rules`
        """

        # When the user calls schedule_function(func, <time_rule>), assume that
        # the user meant to specify a time rule but no date rule, instead of
        # a date rule and no time rule as the signature suggests
        if isinstance(date_rule, (AfterOpen, BeforeClose)) and not time_rule:
            warnings.warn('Got a time rule for the second positional argument '
                          'date_rule. You should use keyword argument '
                          'time_rule= when calling schedule_function without '
                          'specifying a date_rule', stacklevel=3)

        date_rule = date_rule or date_rules.every_day()
        time_rule = ((time_rule or time_rules.every_minute())
                     if self.sim_params.data_frequency == 'minute' else
                     # If we are in daily mode the time_rule is ignored.
                     time_rules.every_minute())

        # Check the type of the algorithm's schedule before pulling calendar
        # Note that the ExchangeTradingSchedule is currently the only
        # TradingSchedule class, so this is unlikely to be hit
        if calendar is None:
            cal = self.trading_calendar
        elif calendar is calendars.US_EQUITIES:
            cal = get_calendar('XNYS')
        elif calendar is calendars.US_FUTURES:
            cal = get_calendar('us_futures')
        else:
            raise ScheduleFunctionInvalidCalendar(
                given_calendar=calendar,
                allowed_calendars=(
                    '[trading-calendars.US_EQUITIES, trading-calendars.US_FUTURES]'
                ),
            )

        self.add_event(
            make_eventrule(date_rule, time_rule, cal, half_days),
            func,
        )

    def make_eventrule(date_rule, time_rule, cal, half_days=True):
        """
        Constructs an event rule from the factory api.
        """
        _check_if_not_called(date_rule)
        _check_if_not_called(time_rule)

        if half_days:
            inner_rule = date_rule & time_rule
        else:
            inner_rule = date_rule & time_rule & NotHalfDay()

        opd = OncePerDay(rule=inner_rule)
        # This is where a scheduled function's rule is associated with a calendar.
        opd.cal = cal
        return opd


from datetime import datetime
import pandas as pd,abc

from zipline.errors import (
    AccountControlViolation,
    TradingControlViolation,
)
from zipline.utils.input_validation import (
    expect_bounded,
    expect_types,
)

class TradingControl(abc.ABC):
    """
    Abstract base class representing a fail-safe control on the behavior of any
    algorithm.
    """

    def __init__(self, on_error, **kwargs):
        """
        Track any arguments that should be printed in the error message
        generated by self.fail.
        """
        self.on_error = on_error
        self.__fail_args = kwargs

    @abc.abstractmethod
    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime,
                 algo_current_data):
        """
        Before any order is executed by TradingAlgorithm, this method should be
        called *exactly once* on each registered TradingControl object.

        If the specified asset and amount do not violate this TradingControl's
        restraint given the information in `portfolio`, this method should
        return None and have no externally-visible side-effects.

        If the desired order violates this TradingControl's contraint, this
        method should call self.fail(asset, amount).
        """
        raise NotImplementedError

    def _constraint_msg(self, metadata):
        constraint = repr(self)
        if metadata:
            constraint = "{constraint} (Metadata: {metadata})".format(
                constraint=constraint,
                metadata=metadata
            )
        return constraint

    def handle_violation(self, asset, amount, datetime, metadata=None):
        """
        Handle a TradingControlViolation, either by raising or logging and
        error with information about the failure.

        If dynamic information should be displayed as well, pass it in via
        `metadata`.
        """
        constraint = self._constraint_msg(metadata)

        if self.on_error == 'fail':
            raise TradingControlViolation(
                asset=asset,
                amount=amount,
                datetime=datetime,
                constraint=constraint)
        elif self.on_error == 'log':
            log.error("Order for {amount} shares of {asset} at {dt} "
                      "violates trading constraint {constraint}",
                      amount=amount, asset=asset, dt=datetime,
                      constraint=constraint)

    def __repr__(self):
        return "{name}({attrs})".format(name=self.__class__.__name__,
                                        attrs=self.__fail_args)


class MaxOrderCount(TradingControl):
    """
    TradingControl representing a limit on the number of orders that can be
    placed in a given trading day.
    """

    def __init__(self, on_error, max_count):

        super(MaxOrderCount, self).__init__(on_error, max_count=max_count)
        self.orders_placed = 0
        self.max_count = max_count
        self.current_date = None

    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime,
                 algo_current_data):
        """
        Fail if we've already placed self.max_count orders today.
        """
        algo_date = algo_datetime.date()

        # Reset order count if it's a new day.
        if self.current_date and self.current_date != algo_date:
            self.orders_placed = 0
        self.current_date = algo_date

        if self.orders_placed >= self.max_count:
            self.handle_violation(asset, amount, algo_datetime)
        self.orders_placed += 1


class RestrictedListOrder(TradingControl):
    """TradingControl representing a restricted list of assets that
    cannot be ordered by the algorithm.

    Parameters
    ----------
    restrictions : zipline.finance.asset_restrictions.Restrictions
        Object representing restrictions of a group of assets.
    """

    def __init__(self, on_error, restrictions):
        super(RestrictedListOrder, self).__init__(on_error)
        self.restrictions = restrictions

    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime,
                 algo_current_data):
        """
        Fail if the asset is in the restricted_list.
        """
        if self.restrictions.is_restricted(asset, algo_datetime):
            self.handle_violation(asset, amount, algo_datetime)


class MaxOrderSize(TradingControl):
    """
    TradingControl representing a limit on the magnitude of any single order
    placed with the given asset.  Can be specified by share or by dollar
    value.
    """

    def __init__(self, on_error, asset=None, max_shares=None,
                 max_notional=None):
        super(MaxOrderSize, self).__init__(on_error,
                                           asset=asset,
                                           max_shares=max_shares,
                                           max_notional=max_notional)
        self.asset = asset
        self.max_shares = max_shares
        self.max_notional = max_notional

        if max_shares is None and max_notional is None:
            raise ValueError(
                "Must supply at least one of max_shares and max_notional"
            )

        if max_shares and max_shares < 0:
            raise ValueError(
                "max_shares cannot be negative."
            )

        if max_notional and max_notional < 0:
            raise ValueError(
                "max_notional must be positive."
            )

    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime,
                 algo_current_data):
        """
        Fail if the magnitude of the given order exceeds either self.max_shares
        or self.max_notional.
        """

        if self.asset is not None and self.asset != asset:
            return

        if self.max_shares is not None and abs(amount) > self.max_shares:
            self.handle_violation(asset, amount, algo_datetime)

        current_asset_price = algo_current_data.current(asset, "price")
        order_value = amount * current_asset_price

        too_much_value = (self.max_notional is not None and
                          abs(order_value) > self.max_notional)

        if too_much_value:
            self.handle_violation(asset, amount, algo_datetime)


class MaxPositionSize(TradingControl):
    """
    TradingControl representing a limit on the maximum position size that can
    be held by an algo for a given asset.
    """

    def __init__(self, on_error, asset=None, max_shares=None,
                 max_notional=None):
        super(MaxPositionSize, self).__init__(on_error,
                                              asset=asset,
                                              max_shares=max_shares,
                                              max_notional=max_notional)
        self.asset = asset
        self.max_shares = max_shares
        self.max_notional = max_notional

        if max_shares is None and max_notional is None:
            raise ValueError(
                "Must supply at least one of max_shares and max_notional"
            )

        if max_shares and max_shares < 0:
            raise ValueError(
                "max_shares cannot be negative."
            )

        if max_notional and max_notional < 0:
            raise ValueError(
                "max_notional must be positive."
            )

    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime,
                 algo_current_data):
        """
        Fail if the given order would cause the magnitude of our position to be
        greater in shares than self.max_shares or greater in dollar value than
        self.max_notional.
        """

        if self.asset is not None and self.asset != asset:
            return

        current_share_count = portfolio.positions[asset].amount
        shares_post_order = current_share_count + amount

        too_many_shares = (self.max_shares is not None and
                           abs(shares_post_order) > self.max_shares)
        if too_many_shares:
            self.handle_violation(asset, amount, algo_datetime)

        current_price = algo_current_data.current(asset, "price")
        value_post_order = shares_post_order * current_price

        too_much_value = (self.max_notional is not None and
                          abs(value_post_order) > self.max_notional)

        if too_much_value:
            self.handle_violation(asset, amount, algo_datetime)


class LongOnly(TradingControl):
    """
    TradingControl representing a prohibition against holding short positions.
    """

    def __init__(self, on_error):
        super(LongOnly, self).__init__(on_error)

    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime,
                 algo_current_data):
        """
        Fail if we would hold negative shares of asset after completing this
        order.
        """
        if portfolio.positions[asset].amount + amount < 0:
            self.handle_violation(asset, amount, algo_datetime)


class AssetDateBounds(TradingControl):
    """
    TradingControl representing a prohibition against ordering an asset before
    its start_date, or after its end_date.
    """

    def __init__(self, on_error):
        super(AssetDateBounds, self).__init__(on_error)

    def validate(self,
                 asset,
                 amount,
                 portfolio,
                 algo_datetime,
                 algo_current_data):
        """
        Fail if the algo has passed this Asset's end_date, or before the
        Asset's start date.
        """
        # If the order is for 0 shares, then silently pass through.
        if amount == 0:
            return

        normalized_algo_dt = pd.Timestamp(algo_datetime).normalize()

        # Fail if the algo is before this Asset's start_date
        if asset.start_date:
            normalized_start = pd.Timestamp(asset.start_date).normalize()
            if normalized_algo_dt < normalized_start:
                metadata = {
                    'asset_start_date': normalized_start
                }
                self.handle_violation(
                    asset, amount, algo_datetime, metadata=metadata)
        # Fail if the algo has passed this Asset's end_date
        if asset.end_date:
            normalized_end = pd.Timestamp(asset.end_date).normalize()
            if normalized_algo_dt > normalized_end:
                metadata = {
                    'asset_end_date': normalized_end
                }
                self.handle_violation(
                    asset, amount, algo_datetime, metadata=metadata)


class AccountControl(with_metaclass(abc.ABCMeta)):
    """
    Abstract base class representing a fail-safe control on the behavior of any
    algorithm.
    """

    def __init__(self, **kwargs):
        """
        Track any arguments that should be printed in the error message
        generated by self.fail.
        """
        self.__fail_args = kwargs

    @abc.abstractmethod
    def validate(self,
                 _portfolio,
                 _account,
                 _algo_datetime,
                 _algo_current_data):
        """
        On each call to handle data by TradingAlgorithm, this method should be
        called *exactly once* on each registered AccountControl object.

        If the check does not violate this AccountControl's restraint given
        the information in `portfolio` and `account`, this method should
        return None and have no externally-visible side-effects.

        If the desired order violates this AccountControl's contraint, this
        method should call self.fail().
        """
        raise NotImplementedError

    def fail(self):
        """
        Raise an AccountControlViolation with information about the failure.
        """
        raise AccountControlViolation(constraint=repr(self))

    def __repr__(self):
        return "{name}({attrs})".format(name=self.__class__.__name__,
                                        attrs=self.__fail_args)


class MaxLeverage(AccountControl):
    """
    AccountControl representing a limit on the maximum leverage allowed
    by the algorithm.
    """

    def __init__(self, max_leverage):
        """
        max_leverage is the gross leverage in decimal form. For example,
        2, limits an algorithm to trading at most double the account value.
        """
        super(MaxLeverage, self).__init__(max_leverage=max_leverage)
        self.max_leverage = max_leverage

        if max_leverage is None:
            raise ValueError(
                "Must supply max_leverage"
            )

        if max_leverage < 0:
            raise ValueError(
                "max_leverage must be positive"
            )

    def validate(self,
                 _portfolio,
                 _account,
                 _algo_datetime,
                 _algo_current_data):
        """
        Fail if the leverage is greater than the allowed leverage.
        """
        if _account.leverage > self.max_leverage:
            self.fail()


class MinLeverage(AccountControl):
    """AccountControl representing a limit on the minimum leverage allowed
    by the algorithm after a threshold period of time.

    Parameters
    ----------
    min_leverage : float
        The gross leverage in decimal form.
    deadline : datetime
        The date the min leverage must be achieved by.

    For example, min_leverage=2 limits an algorithm to trading at minimum
    double the account value by the deadline date.
    """

    @expect_types(
        __funcname='MinLeverage',
        min_leverage=(int, float),
        deadline=datetime
    )
    @expect_bounded(__funcname='MinLeverage', min_leverage=(0, None))
    def __init__(self, min_leverage, deadline):
        super(MinLeverage, self).__init__(min_leverage=min_leverage,
                                          deadline=deadline)
        self.min_leverage = min_leverage
        self.deadline = deadline

    def validate(self,
                 _portfolio,
                 account,
                 algo_datetime,
                 _algo_current_data):
        """
        Make validation checks if we are after the deadline.
        Fail if the leverage is less than the min leverage.
        """
        if (algo_datetime > self.deadline and
                account.leverage < self.min_leverage):
            self.fail()
