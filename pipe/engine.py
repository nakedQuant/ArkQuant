# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from toolz import keyfilter, valmap
from functools import partial
from multiprocessing import Pool
from itertools import chain
from abc import ABC, abstractmethod
from pipe.loader.loader import PricingLoader
from finance.restrictions import UnionRestrictions
from gateway.asset._finder import init_finder


class Engine(ABC):
    """
        engine process should be automatic without much manual interface
    """
    @staticmethod
    def _init_loader(pipelines):
        pipelines = pipelines if isinstance(pipelines, list) else [pipelines]
        inner_terms = list(chain(pipeline.terms for pipeline in pipelines))
        inner_pickers = list(chain(pipeline.ump_terms for pipeline in pipelines))
        engine_terms = set(inner_terms + inner_pickers)
        # get_loader
        _get_loader = PricingLoader(engine_terms)
        return pipelines, _get_loader

    def compute_mask(self, dts):
        # default assets
        equities = self.asset_finder.retrieve_type_assets('equity')
        # save the high priority and asset which can be traded
        default_mask = self.restricted_rules.is_restricted(equities, dts)
        return default_mask

    def _init_metadata(self, dts):
        # 判断ledger 是否update
        mask = self.compute_mask(dts)
        metadata = self._get_loader.load_pipeline_arrays(dts, mask)
        return metadata, mask

    def _divide_positions(self, ledger, dts):
        """
        Register a Pipeline default for pipe on every day.
        :param dts: initialize attach pipe and cache metadata for engine
        9:25 --- 9:30
        """
        # violate risk management
        violate_positions = ledger.get_violate_risk_positions()
        # 配股持仓
        righted_positions = ledger.get_rights_positions(dts)
        # expires positions
        expired_positions = ledger.get_expired_positions(dts)
        # 剔除的持仓
        if self.disallowed_righted and self.disallowed_violation:
            remove_positions = set(righted_positions) | set(violate_positions)
        elif self.disallowed_violation:
            remove_positions = violate_positions
        elif self.disallowed_righted:
            remove_positions = righted_positions
        else:
            remove_positions = set()
        # 剔除配股的持仓
        remove_positions = set(remove_positions) | set(expired_positions)
        traded_positions = set(ledger.positions) - remove_positions
        return traded_positions, remove_positions

    @staticmethod
    def resolve_pipeline_final(outputs):
        group_sorted = dict()
        for item in outputs:
            group_sorted.update(item)
        finals = valmap(lambda x: x[0], group_sorted)
        return finals

    def _run_pipeline(self, pipeline, metadata, mask):
        """
        ----------
        pipe : zipline.pipe.Pipeline
            The pipe to run.
        """
        yield pipeline.to_execution_plan(metadata, self.alternatives, mask)

    def run_pipeline(self, pipeline_metadata, mask):
        """
        Compute values for  pipelines on a specific date.
        Parameters
        ----------
        pipeline_metadata : cache data for pipe
        mask : default asset list
        ----------
        return --- event
        """
        _pipe_func = partial(self._run_pipeline,
                             mask=mask,
                             metadata=pipeline_metadata)

        with Pool(processes=len(self.pipelines))as pool:
            results = [pool.apply_async(_pipe_func, pipeline)
                       for pipeline in self.pipelines]

        yield self.resolve_pipeline_final(results)

    @staticmethod
    def _run_ump(pipeline, position, metadata):
        output = pipeline.to_withdraw_plan(position, metadata)
        return output

    def run_ump(self, metadata, positions):
        """
            umps --- based on different asset type --- (symbols , etf , bond)
                    to determine withdraw strategy
            return position list
        """
        _ump_func = partial(self._run_ump, metadata=metadata)
        # proxy -- positions : pipeline
        proxy_position = {p.name: p for p in positions}
        proxy_pipeline = {pipe.name: pipe for pipe in self.pipelines}

        with Pool(processes=len(proxy_pipeline))as pool:
            results = [pool.apply_async(_ump_func, proxy_pipeline[name], p)
                       for name, p in proxy_position.items()]
        output = [r for r in results if r]
        return output

    def execute_algorithm(self, ledger, dts):
        """
            calculate pipelines and ump
        """
        metadata, default_mask = self._init_metadata(dts)
        traded_positions, removed_positions = self._divide_positions(ledger, dts)
        # 执行算法逻辑
        pipe_proxy = self.run_pipeline(metadata, default_mask)
        # 剔除righted positions, violate_positions, expired_positions
        ump_positions = set(self.run_ump(metadata, traded_positions)) | removed_positions
        yield self.resolve_conflicts(pipe_proxy, ump_positions, ledger.positions)

    @staticmethod
    @abstractmethod
    def resolve_conflicts(calls, puts, holding):
        """
        :param calls: dict --- pipe_name : asset --- all pipeline
        :param puts: (ump position) + righted position + violate position + expired position
        :param holdings: ledger positions

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
            position + pipe - ledger ---  (当ledger为空 --- position也为空)

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

    1、Determine the domain of the pipe.The domain determines the top-level
        set of dates and field that serves as row and column ---- data needed
        to compute the pipe

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
       c. 考虑需要剔除的持仓（配股持仓 或者 risk management)

    Parameter:

    _get_loader : PricingLoader
    ump_picker : strategy for putting positions
    """
    # __slots__ = [
    #     'asset_finder',
    #     'alternatives',
    #     'disallowed_righted',
    #     'disallowed_violation',
    #     'restricted_rules'
    # ]

    def __init__(self,
                 pipelines,
                 restrictions,
                 alternatives=10,
                 disallow_righted=True,
                 disallow_violation=True):
        self.asset_finder = init_finder()
        self.alternatives = alternatives
        self.disallowed_righted = disallow_righted
        self.disallowed_violation = disallow_violation
        self.restricted_rules = UnionRestrictions(restrictions)
        self.pipelines, self._get_loader = self._init_loader(pipelines)

    @staticmethod
    def resolve_conflicts(calls, puts, holdings):
        # mappings {pipe_name:position}
        put_proxy = {r.name: r for r in puts}
        hold_proxy = {p.name: p for p in holdings}
        # 基于capital执行直接买入标的的
        extra = set(calls) - set(hold_proxy)
        if extra:
            direct_positives = keyfilter(lambda x: x in extra, calls)
        else:
            direct_positives = dict()
        # common pipe name
        common_pipe = set(put_proxy) & set(calls)
        # 直接卖出持仓，无买入标的
        negatives = set(put_proxy) - set(common_pipe)
        direct_negatives = keyfilter(lambda x: x in negatives, put_proxy)
        # 卖出持仓买入对应标的 --- (position, asset)
        if common_pipe:
            conflicts = [name for name in common_pipe if put_proxy[name].asset == calls[name]]
            assert not conflicts, ValueError('name : %r have conflicts between ump and pipe ' % conflicts)
            dual = [(put_proxy[name], calls[name]) for name in common_pipe]
        else:
            dual = set()
        return direct_positives, direct_negatives, dual


class NoEngineRegistered(Exception):
    """
    Raised if a user tries to call pipeline_output in an algorithm that hasn't
    set up a pipe engine.
    """


__all__ = [
    'SimplePipelineEngine',
    'NoEngineRegistered',
]
