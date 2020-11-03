# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from toolz import keyfilter
from functools import partial
from itertools import chain
from abc import ABC, abstractmethod
from pipe.loader.loader import PricingLoader
from finance.restrictions import UnionRestrictions
from gateway.asset.finder import asset_finder


class Engine(ABC):
    """
        engine process should be automatic without much manual interface
    """
    @staticmethod
    def _init_loader(pipelines):
        pipelines = pipelines if isinstance(pipelines, list) else [pipelines]
        inner_terms = list(chain(*[pipeline.terms for pipeline in pipelines]))
        # print('inner terms', inner_terms)
        inner_pickers = list(chain(*[pipeline.ump_terms for pipeline in pipelines]))
        # print('inner_pickers', inner_pickers)
        engine_terms = set(inner_terms + inner_pickers)
        # print('engine_terms', engine_terms)
        # get_loader
        _get_loader = PricingLoader(engine_terms)
        return pipelines, _get_loader

    def _calculate_universe(self, dts):
        # default assets
        asset_finder.synchronize()
        # equities = asset_finder.retrieve_type_assets('equity')
        equities = list(asset_finder.retrieve_type_assets('equity'))[:10]
        # print('pipeline restricted_rules', self.restricted_rules.sub_restrictions)
        default_mask = self.restricted_rules.is_restricted(equities, dts)
        return default_mask

    def _initialize_metadata(self, ledger, dts):
        # 判断ledger 是否update
        universe_mask = self._calculate_universe(dts)
        # print('universe_mask', universe_mask)
        # mask
        # print('positions mask', set(ledger.positions))
        mask = set(universe_mask) | set(ledger.positions)
        # print('mask', mask)
        metadata = self._get_loader.load_pipeline_arrays(dts, mask, 'daily')
        # print('engine metadata sids', set(metadata))
        # 过滤没有数据的sid
        from toolz import valfilter
        metadata = valfilter(lambda x: not x.empty, metadata)
        # print('metadata symbols', set(metadata))
        # mask = [m for m in mask if m.sid in set(metadata) and not metadata[m.sid].empty]
        engine_mask = [symbol for symbol in list(mask) if symbol.sid in set(metadata)]
        # print('engine mask', engine_mask)
        return metadata, engine_mask

    def _split_positions(self, ledger, dts):
        """
        Register a Pipeline default for pipe on every day.
        :param dts: initialize attach pipe and cache metadata for engine
        9:25 --- 9:30
        """
        # violate risk management
        violate_positions = ledger.get_violate_risk_positions()
        # print('engine violate_positions', violate_positions)
        # 配股持仓
        righted_positions = ledger.get_rights_positions(dts)
        # print('engine righted_positions', righted_positions)
        # expires positions
        expired_positions = ledger.get_expired_positions(dts)
        # print('engine expired_positions', expired_positions)
        # 剔除的持仓
        if self.disallowed_righted and self.disallowed_violation:
            remove_positions = set(righted_positions) | set(violate_positions)
        elif self.disallowed_violation:
            remove_positions = violate_positions
        elif self.disallowed_righted:
            remove_positions = righted_positions
        else:
            remove_positions = set()
        # print('engine removed position', remove_positions)
        # 剔除配股的持仓
        remove_positions = set(remove_positions) | set(expired_positions)
        # print('removed position', remove_positions)
        traded_positions = set(ledger.positions.values()) - remove_positions
        # print('traded_positions', traded_positions)
        return traded_positions, remove_positions

    def _run_pipeline(self, pipeline, metadata, mask):
        """
        ----------
        pipe : zipline.pipe.Pipeline
            The pipe to run.
        """
        out = pipeline.to_execution_plan(metadata, mask, self.final)
        return out

    def run_pipeline(self, pipeline_metadata, mask):
        """
        Compute values for  pipelines on a specific date.
        Parameters
        ----------
        pipeline_metadata : cache data for pipe
        mask : default asset list
        ----------
        return --- assets which tag by pipeline name
        """
        _partial_func = partial(self._run_pipeline,
                                mask=mask,
                                metadata=pipeline_metadata)

        results = []
        for pipeline in self.pipelines:
            out = _partial_func(pipeline)
            # print('out', out)
            results.append(out)
        results = [r for r in results if r]
        # print('run pipeline output', results)
        return results

    @staticmethod
    def _run_ump(pipeline, position, metadata):
        # print('ump_picker', pipeline.ump_terms)
        # output --- bool or position
        result = pipeline.to_withdraw_plan(position, metadata)
        return result

    def run_ump(self, metadata, positions):
        """
            umps --- based on different asset type --- (symbols , etf , bond)
                    to determine withdraw strategy
            return position list
        """
        output = []
        if positions:
            # print('ump positions', positions)
            _ump_func = partial(self._run_ump, metadata=metadata)
            # proxy -- positions : pipeline
            proxy_position = {p.asset.tag: p for p in positions}
            # print('proxy_position', proxy_position)
            proxy_pipeline = {pipe.name: pipe for pipe in self.pipelines}
            # print('proxy_pipeline', proxy_pipeline)
            for proxy in proxy_position:
                res = _ump_func(
                    proxy_pipeline[proxy],
                    proxy_position[proxy]
                )
                output.append(res)
            # ump position
            output = [r for r in output if r]
        # print('run ump result', output)
        return output

    def execute_algorithm(self, ledger, dts):
        """
            calculate pipelines and ump
        """
        metadata, default_mask = self._initialize_metadata(ledger, dts)
        traded_positions, removed_positions = self._split_positions(ledger, dts)
        # 执行算法逻辑
        pipes = self.run_pipeline(metadata, default_mask)
        # 剔除righted positions, violate_positions, expired_positions
        ump_positions = self.run_ump(metadata, traded_positions)
        ump_positions = set(ump_positions) | removed_positions
        # print('ump_positions', ump_positions)
        # yield self.resolve_conflicts(pipes, ump_positions, ledger.positions)
        return self.resolve_conflicts(pipes, ump_positions, ledger.positions)

    @staticmethod
    @abstractmethod
    def resolve_conflicts(calls, puts, holding):
        """
        :param calls: dict --- pipe_name : asset --- all pipeline
        :param puts: (ump position) + righted position + violate position + expired position
        :param holding: ledger positions
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
        max_holding_num : defined by the num of pipelines
    """
    __slots__ = [
        'disallowed_righted',
        'disallowed_violation',
        'restricted_rules'
    ]

    def __init__(self,
                 pipelines,
                 final_model,
                 restrictions,
                 disallow_righted=True,
                 disallow_violation=True):
        self.disallowed_righted = disallow_righted
        self.disallowed_violation = disallow_violation
        self.restricted_rules = UnionRestrictions(restrictions)
        self.final = final_model
        self.pipelines, self._get_loader = self._init_loader(pipelines)

    @staticmethod
    def resolve_conflicts(calls, puts, holdings):
        """
        :param calls: buy assets list
        :param puts:  sell positions list
        :param holdings: dict , ledger holdings , asset : position
        :return: list
        """
        # 判断买入标的的sid与卖出持仓的sid是否存在冲突 --- 主要多策略组合与多策略并行的的区别
        positive_sids = [r.sid for r in calls] if calls else []
        negatives_sids = [p.asset.sid for p in puts] if puts else []
        union_sids = set(positive_sids) & set(negatives_sids)
        assert not union_sids, 'buy and sell the targeted sid on a day is not allowed'
        # asset tag name means pipeline_name
        call_proxy = {r.tag: r for r in calls} if calls else {}
        hold_proxy = {p.name: p for p in holdings.values()} if holdings else {}
        # 基于capital执行直接买入标的的
        extra = set(call_proxy) - set(hold_proxy)
        if extra:
            extra_mappings = keyfilter(lambda x: x in extra, call_proxy)
        else:
            extra_mappings = dict()
        extra_positives = list(extra_mappings.values())
        # print('engine extra_positives', extra_positives)
        # pipeline --- 产生相同的asset对象（算法自动加仓）
        common = set(call_proxy) & set(hold_proxy)
        increment_positives = [call_proxy[c] for c in common if call_proxy[c] == hold_proxy[c].asset]
        # print('engine increment_positives', increment_positives)
        # direct_positives = set(extra_positives) | set(increment_positives)
        direct_positives = list(set(extra_positives) | set(increment_positives))
        # print('engine direct_positives', direct_positives)
        # 基于持仓卖出 --- 分为2种 ， 1.直接卖出 ， 2.卖出买入 ，基于pipeline name
        # 2种 --- 一个pipeline同时存在买入和卖出行为
        put_proxy = {r.name: r for r in puts} if puts else {}
        common_pipe = set(call_proxy) & set(put_proxy)
        if common_pipe:
            # duals (position, asset)
            conflicts = [name for name in common_pipe if put_proxy[name].asset == call_proxy[name]]
            assert not conflicts, ValueError('name : %r have conflicts between ump and pipe ' % conflicts)
            dual = [(put_proxy[name], call_proxy[name]) for name in common_pipe]
        else:
            dual = set()
        # print('engine dual', dual)
        # 1种 --- 直接卖出
        negatives = set(put_proxy) - set(common_pipe)
        negative_puts = keyfilter(lambda x: x in negatives, put_proxy)
        direct_negatives = list(negative_puts.values())
        # print('engine direct_negatives', direct_negatives)
        # asset positions duals
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
