# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import numpy as np
from toolz import keyfilter,merge,groupby
from functools import partial
from multiprocessing import Pool
from itertools import chain


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
