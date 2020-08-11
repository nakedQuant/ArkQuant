# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from toolz import groupby, keyfilter, valmap
from functools import partial
from multiprocessing import Pool
from itertools import chain
from abc import ABC, abstractmethod
from .loader.loader import PricingLoader
from finance.restrictions import UnionRestrictions


class Engine(ABC):
    """
        set asset range which means all asset with some restrictions
        --- engine process should be automatic without much manual interface
    """

    def _init(self, pipelines, pickers):
        inner_terms = chain([pipeline.terms for pipeline in pipelines])
        inner_pickers = pickers.pickers
        engine_terms = set(inner_terms + inner_pickers)
        # get_loader
        _get_loader = PricingLoader(engine_terms, self._data_portal)
        return pipelines, pickers, _get_loader

    def _compute_default(self, dts):
        """
        Register a Pipeline default for pipeline on every day.
        :param dts: initialize attach pipeline and cache metadata for engine
        :return:
        """
        equities = self.asset_finder.retrieve_type_assets('equity')
        default = self._restricted_rule.is_restricted(equities, dts)
        open_pct = self._data_portal.get_open_pct(dts, default)
        # set pipeline metadata
        history_metadata = self._get_loader.load_pipeline_arrays(dts, default)
        return default, history_metadata, open_pct

    def _prepare_for_execution(self,ledger):
        # 判断所有仓位的更新时间 -- 保持一致
        dts = ledger.synchronized_clock
        capital = ledger.porfolio.cash
        holdings = set(ledger.positions) - set(ledger.get_rights_positions(dts))
        # default
        default, metadata, open_pct = self._compute_default(dts)
        return holdings, default, metadata, open_pct, capital

    @staticmethod
    def resolve_pipeline(pipeline_mappings, open_pct):
        pipeline_assets = valmap(lambda x: [element.asset for element in x
                                            if open_pct[element.asset.sid] < 9.90], pipeline_mappings)
        selector = valmap(lambda x: x[0] if x else None, pipeline_assets)
        return selector

    def _run_pipeline(self, pipeline, metadata, default):
        """
        ----------
        pipeline : zipline.pipeline.Pipeline
            The pipeline to run.
        """
        yield pipeline.to_execution_plan(metadata, self.alternatives, default)

    def run_pipelines(self, pipeline_metadata, default, open_pct):
        """
        Compute values for  pipelines on a specific date.
        Parameters
        ----------
        pipeline_metadata : cache data for pipeline
        default : pipeline --- default asset list
        open_pct : intend to filter asset which can not be traded
        """
        _impl = partial(self._run_pipeline,
                        metadata=pipeline_metadata,
                        default=default)

        with Pool(processes=len(self.pipelines))as pool:
            results = [pool.apply_async(_impl, pipeline)
                       for pipeline in self.pipelines]
            outputs = chain(* results)
        # output -- element(PIPE)
        mappings = groupby(lambda x: x.name, outputs)
        mappings_sorted = valmap(lambda x: x.sort(key=lambda k: k.priority), mappings)
        yield self.resolve_pipeline(mappings_sorted, open_pct)

    def run_pickers(self, picker_metadata, holdings):
        """
            umps --- based on different asset type --- (symbols , etf , bond)
                    to determine withdraw strategy
        """
        hold_mappings = groupby(lambda x: x.asset.asset_type, holdings)
        outputs = []
        for asset_type, positions in hold_mappings.items():
            result = self.ump_pickers[asset_type].evalute(positions, picker_metadata)
            outputs.extend(result)
        dct = {{p.tag: p} for p in outputs}
        return dct

    def execute_algorithm(self,ledger):
        """
            calculate pipelines and ump
        """
        # 计算所需要数据
        holdings, default, meta, open_pct, capital = self._prepare_for_execution(ledger)
        # 执行算法逻辑
        pipeline_mappings = self.run_pipelines(meta, default, open_pct)
        poll_mappings = self.run_pickers(meta, holdings)
        # 处理冲突
        yield self.resolve_conflicts(poll_mappings, pipeline_mappings, holdings), capital

    @staticmethod
    @abstractmethod
    def resolve_conflicts(*args):
        """
            param args: pipeline outputs , ump outputs holdings
            return: target asset which can be simulate into orders

            instructions:
                防止策略冲突 当pipeline的结果与ump的结果出现重叠 --- 说明存在问题，正常情况退出策略与买入策略应该不存交集

                1. engine共用一个ump ---- 解决了不同策略产生的相同标的可以同一时间退出
                2. engine --- 不同的pipeline对应不同的ump,产生1中的问题，相同的标的不会在同一时间退出是否合理（冲突）

                退出策略 --- 针对标的，与标的是如何产生的不存在直接关系;只能根据资产类别的有关 --- 1
                如果产生冲突 --- 当天卖出标的与买入标的产生重叠 说明策略是有问题的ump --- pipelines 对立的
                symbol ,etf 的退出策略可以相同，但是bond 属于T+0 机制不一样

                建仓逻辑 --- 逐步建仓 1/2 原则 --- 1 优先发生信号先建仓 ，后发信号仓位变为剩下的1/2（为了提高资金利用效率）
                                                2 如果没新的信号 --- 在已经持仓的基础加仓（不管资金是否足够或者设定一个底层资金池）
                ---- 变相限定了单次单个标的最大持仓为1/2
                position + pipeline - ledger ---  (当ledger为空 --- position也为空)

                关于ump --- 只要当天不是一直在跌停价格，以全部出货为原则，涉及一个滑价问题（position的成交额 与前一周的成交额占比
                评估滑价），如果当天没有买入，可以适当放宽（开盘的时候卖出大部分，剩下的等等） ；
                如果存在买入标的的行为则直接按照全部出货原则以open价格最大比例卖出 ，一般来讲集合竞价的代表主力卖入意愿强度）
                ---- 侧面解决了卖出转为买入的断层问题 transfer1
        """
        raise NotImplementedError()


class SimplePipelineEngine(Engine):
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
    __slots__ = [
        'asset_finder'
        '_reader',
        '_restricted_rule'
        'alternatives',
        '_pipeline_cache'
        ]

    def __init__(self,
                 pipelines,
                 ump_pickers,
                 asset_finder,
                 data_portal,
                 restrictions,
                 alternatives=10):
        self.asset_finder = asset_finder
        self._data_portal = data_portal
        self._restricted_rule = UnionRestrictions(restrictions)
        self.alternatives = alternatives
        self.pipelines, self.ump_pickers, self._get_loader = self._init(pipelines, ump_pickers)
        self._pipeline_cache = {}

    def __setattr__(self, key, value):
        raise NotImplementedError()

    @staticmethod
    def resolve_conflicts(puts, calls, holdings):
        # common pipeline name
        common_pipe_name = set(puts) & set(calls)
        conflicts = [name for name in common_pipe_name if puts[name] == calls[name]]
        assert not conflicts, ValueError('name : %r have conflicts between ump and pipeline ' % conflicts)
        # pipeline_name : holding
        holding_mappings = groupby(lambda x: x.tag, holdings)
        # 直接卖出持仓，无买入标的
        direct_negatives = set(puts) - set(common_pipe_name)
        # 卖出持仓买入对应标的
        dual = [(puts[name],calls[name]) for name in common_pipe_name]
        # 基于capital执行买入标的的对应的pipeline_name
        extra = set(calls) - set(holding_mappings)
        direct_positives = keyfilter(lambda x: x in extra, calls)
        return direct_negatives, dual, direct_positives


class NoEngineRegistered(Exception):
    """
    Raised if a user tries to call pipeline_output in an algorithm that hasn't
    set up a pipeline engine.
    """
