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
from pipe.loader.loader import PricingLoader
from finance.restrictions import UnionRestrictions
from gateway.asset._finder import init_finder


class Engine(ABC):
    """
        set asset range which means all asset with some restrictions
        --- engine process should be automatic without much manual interface
    """
    def _init_loader(self, pipelines):
        inner_terms = list(chain(pipeline.terms for pipeline in pipelines))
        print('inner_terms', inner_terms)
        inner_pickers = list(chain(pipeline.ump_terms for pipeline in pipelines))
        print('inner_picker', inner_pickers)
        engine_terms = set(inner_terms + inner_pickers)
        print('engine_terms', engine_terms)
        # get_loader
        _get_loader = PricingLoader(engine_terms)
        return pipelines, _get_loader

    def _compute_mask(self, dts):
        # default assets
        equities = self.asset_finder.retrieve_type_assets('equity')
        # save the high priority and asset which can be traded
        default_mask = self.restricted_rules.is_restricted(equities, dts)
        return default_mask

    def _compute_exclude(self, ledger):
        """
        Register a Pipeline default for pipe on every day.
        :param dts: initialize attach pipe and cache metadata for engine
        9:25 --- 9:30
        """
        # 判断ledger 是否update
        dts = ledger.synchronized_clock
        # capital
        capital = ledger.porfolio.cash
        # 配股持仓
        righted_positions = ledger.get_rights_positions(dts)
        # violate risk management
        violate_positions = ledger.get_violate_risk_positions()
        # expires positions
        expired_positions = ledger.get_expired_positions(dts)
        # 剔除的持仓
        if self.allowed_righted and self.allowed_violation:
            remove_positions = set(righted_positions) | set(violate_positions)
        elif self.allowed_violation:
            remove_positions = violate_positions
        elif self.allowed_righted:
            remove_positions = righted_positions
        else:
            remove_positions = set()
        # 剔除配股的持仓
        traded_positions = set(expired_positions) + set(ledger.positions) - set(remove_positions)
        return traded_positions, remove_positions

    def _prepare(self, ledger):
        # 判断ledger 是否update
        dts = ledger.synchronized_clock
        mask = self._compute_mask(dts)
        # metadata
        metadata = self._get_loader.load_pipeline_arrays(dts, mask)

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
        _impl = partial(self._run_pipeline,
                        metadata=pipeline_metadata,
                        mask=mask)

        with Pool(processes=len(self.pipelines))as pool:
            results = [pool.apply_async(_impl, pipeline)
                       for pipeline in self.pipelines]
            outputs = chain(* results)
        # return pipe_name : asset
        yield self.resolve_pipeline_final(outputs)

    @staticmethod
    def resolve_pipeline_final(outputs):
        # to find out the final asset of each pipe , notice ---  NamedPipe list
        # group by pipe name
        sample_by_pipe = groupby(lambda x: x.event.name, outputs)
        # priority 0 --- high ,diminish by increase number
        group_sorted = valmap(lambda x: x.sort(key=lambda k: k.priority), sample_by_pipe)
        # namedPipe --- event priority
        finals = valmap(lambda x: x[0].event.asset if x else None, group_sorted)
        return finals

    @staticmethod
    def _run_ump(ump_pipe, position, metadata):
        output = ump_pipe.to_withdraw_plan(position, metadata)
        return output

    def run_ump(self, metadata, positions):
        """
            umps --- based on different asset type --- (symbols , etf , bond)
                    to determine withdraw strategy
            return position list
        """
        proxy = {{pipe.name: pipe} for pipe in self.pipelines}
        _impl = partial(self._run_ump, metadata=metadata)

        with Pool(processes=len(positions))as pool:
            results = [pool.apply_async(_impl, proxy[position.name], position)
                       for position in positions]
        output = [r for r in results if r]
        return output

    def execute_algorithm(self, ledger):
        """
            calculate pipelines and ump
        """
        # default pipe_metadata --- pipe ; positions_not_righted pipe_metadata --- ump ;
        default_mask, pipe_metadata, traded_positions, removed_positions, dts, capital = self._compute_default(ledger)
        # 执行算法逻辑
        pipe_proxy = self.run_pipeline(pipe_metadata, default_mask)
        # 退出算法包含righted position
        ump_positions = self.run_ump(pipe_metadata, traded_positions) + removed_positions
        # 买入的event , 卖出的ump_positions , 总持仓（剔除配股持仓）
        yield ledger.porfolio.to_dict(), dts, capital, self.resolve_conflicts(pipe_proxy, ump_positions, traded_positions)

    @staticmethod
    @abstractmethod
    def resolve_conflicts(*args):
        """
            param args: pipe outputs , ump outputs holdings
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
    __slots__ = [
        'asset_finder'
        '_data_portal',
        '_restricted_rule'
        'alternatives',
        'allowed_righted',
        'allowed_violation'
        ]

    def __init__(self,
                 pipelines,
                 restrictions,
                 alternatives=10,
                 allow_righted=False,
                 allowed_violation=True):
        self.asset_finder = init_finder()
        # SecurityListRestrictions  AvailableRestrictions
        self.restricted_rules = UnionRestrictions(restrictions)
        self.alternatives = alternatives
        self.allowed_righted = allow_righted
        self.allowed_violation = allowed_violation
        self.pipelines, self._get_loader = self._init([pipelines])

    @staticmethod
    def resolve_conflicts(calls, puts, holdings):
        """
        :param calls: dict --- pipe_name : asset
        :param puts: Position list (ump position + righted position
        :param holdings: ledger positions (exclude righted positions)
        不同的pipeline 可以相同持仓，但是不一定同一时间卖出
        其他的pipeline产生相同的持仓，持仓动态变动，所以只能做收盘的分析 或者每个minute产生一个account view
        """
        # return mappings {name:position}
        put_proxy = {{r.name: r} for r in puts}
        # common pipe name
        common_pipe = set(put_proxy) & set(calls)
        # 直接卖出持仓，无买入标的
        direct_negatives = set(put_proxy) - set(common_pipe)
        if common_pipe:
            conflicts = [name for name in common_pipe if put_proxy[name].asset == calls[name]]
            assert not conflicts, ValueError('name : %r have conflicts between ump and pipe ' % conflicts)
            # 卖出持仓买入对应标的 --- element (position, asset)
            dual = [(put_proxy[name], calls[name]) for name in common_pipe]
        else:
            dual = set()
        # pipeline_name : holding groupby 可能存在相同的持仓但是由不同的pipeline产生
        hold_proxy = {{p.name: p} for p in holdings}
        # 基于capital执行买入标的的对应的pipeline_name
        extra = set(calls) - set(hold_proxy)
        if extra:
            direct_positives = keyfilter(lambda x: x in extra, calls)
        else:
            direct_positives = dict()
        # unchanged positions unchanged_pipe = set(hold_proxy) - set(put_proxy)
        return direct_negatives, dual, direct_positives


class NoEngineRegistered(Exception):
    """
    Raised if a user tries to call pipeline_output in an algorithm that hasn't
    set up a pipe engine.
    """


__all__ = [
    'SimplePipelineEngine',
    'NoEngineRegistered',
]
