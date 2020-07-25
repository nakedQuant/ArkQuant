# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from toolz import merge,groupby,keyfilter
from functools import partial
from multiprocessing import Pool
from itertools import chain
from .loader.loader import  PricingLoader


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

    Parameter:

    _get_loader : PricingLoader
    ump_picker : strategy for putting positions
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
        engine_terms = set(_pickers._poll_pickers + _inner_terms)
        self._get_loader = PricingLoader(engine_terms)
        self.ump_pickers = _pickers
        self.pipelines = pipelines

    def _lru_cache(self,dts):
        """
        Register a Pipeline default for pipeline on every day.
        :param dts: initialize attach pipeline and cache metadata for engine
        :return:
        """
        #init pipelines
        for pipeline in self.pipelines:
            pipeline.attach_default(dts)
        # default --- pipeline sids
        default = [pipeline.default for pipeline in self.pipelines]
        # _cache_metada
        cache = self._get_loader.load_pipeline_arrays(dts,default)
        return cache

    def _pipeline_impl(self,pipeline,metadata,alternatives):
        """
        ----------
        pipeline : zipline.pipeline.Pipeline
            The pipeline to run.
        """
        yield pipeline.to_execution_plan(metadata,alternatives)

    def run_pipelines(self,pipeline_metadata):
        """
        Compute values for  pipelines on a specific date.
        Parameters
        ----------
        pipeline_metadata : cache data for pipeline
        """
        workers = len(self.pipelines)
        _implement = partial(self._pipeline_impl,
                             metadata = pipeline_metadata,
                             alternatives = workers)

        with Pool(processes = workers)as pool:
            results = [pool.apply_async(_implement, pipeline)
                      for pipeline in self.pipelines]
            outputs = merge(results)
        return outputs

    def run_pickers(self,holdings,_ump_metadata):
        type_mappings = groupby(lambda x : x.inner_position.asset.asset_type,holdings)
        ump_outputs = {}
        for name , positions in type_mappings.items():
            result = self.ump_pickers[name].evalute(positions,_ump_metadata)
            # position.name ---- which pipeline create position
            if result:
                {ump_outputs.update({p.name:p}) for p in result}
        return ump_outputs

    def _resovle_conflicts(self, puts, calls, holdings):
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
        holding_mappings = groupby(lambda x : x.inner_position.name,holdings)
        extra = set(calls) - set(holding_mappings)
        intersection = set(puts) & set(calls)
        conflicts = [ name for name in intersection if puts[name] == calls[name]]
        assert not conflicts,ValueError('name : %r have conflicts between ump and pipeline '%conflicts)
        # 卖出持仓,买入标的
        result = [(puts[name],calls[name]) for name in intersection]
        # capital 买入标的
        supplement = keyfilter(lambda x : x in extra,calls)
        return result,supplement

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
        metadata = self._lru_cache(dts[0])
        #剔除配股的仓位
        right_holdings = ledger.get_rights_positions(dts[0])
        left_holdings = set(holdings) - set(right_holdings)
        #执行算法逻辑
        polls = self.run_pickers(left_holdings,metadata)
        pipes = self.run_pipelines(metadata)
        # --- 如果selectors 与 outs 存在交集
        negs_pos,supplement = self._resovle_conflicts(polls,pipes,left_holdings)
        # cache
        self._pipeline_cache[dts[0]] = (negs_pos,supplement)
        return negs_pos,supplement,capital,dts[0]


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
