# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd
from contextlib import ExitStack
from utils.api_support import ZiplineAPI
from gens.clock import MinuteSimulationClock
from gens import (
    SESSION_START,
    SESSION_END,
    BEFORE_TRADING_START_BAR
)


class AlgorithmSimulator(object):
    """
        simulation start:
                        a.初始化相关模块以及参数
        before trading:
                        a. 针对ledger预处理
                        b. pipelineEngine计算的结果
                        c. 9:25（撮合价格） --- cancelPolicy过滤pipelineEngine的标的集合 -- 筛选可行域
                        d. 实施具体的执行计划 （非bid_mechnasim --- 创建订单）
        session start:
                        a.基于可行域 -- 调用blotter模块(orders --- transactions)
                        b.transactions --- update ledger
        session end:
                        a.调用metrics_tracker --- generate metrics_perf
        simulation_end:
                        a. 收尾
    """

    EMISSION_TO_PERF_KEY_MAP = {
        'minute': 'minute_perf',
        'daily': 'daily_perf'
    }

    def __init__(self, algorithm, sim_params):
        # ==============
        # Setup Algo
        # ==============
        self.algorithm = algorithm
        # ==============
        # Setup Clock
        # ==============
        self.clock = MinuteSimulationClock(sim_params)

    def transform(self):
        """
        Main generator work loop.
        """
        broker = self.algorithm.broke_class
        algorithm_engine = self.algorithm.pipeline_engine
        ledger = self.algorithm.ledger

        def once_a_day():
            broker.carry_out(algorithm_engine, ledger)

        def on_exit():
            # Remove references to algo, data portal, et al to break cycles
            # and ensure deterministic cleanup of these objects when the
            # simulation finishes.
            self.algorithm = None

        with ExitStack() as stack:
            """
            由于已注册的回调是按注册的相反顺序调用的，因此最终行为就好像with 已将多个嵌套语句与已注册的一组回调一起使用。
            这甚至扩展到异常处理-如果内部回调抑制或替换异常，则外部回调将基于该更新状态传递自变量。
            enter_context  输入一个新的上下文管理器，并将其__exit__()方法添加到回调堆栈中。返回值是上下文管理器自己的__enter__()方法的结果。
            callback（回调，* args，** kwds ）接受任意的回调函数和参数，并将其添加到回调堆栈中。
            """
            stack.callback(on_exit)
            stack.enter_context(ZiplineAPI(self.algorithm))
            clock = self.algorithm.clock

            self.metrics_tracker.handle_start_of_simulation()
            # 涉及 minutes metrics

            for session_label, action in clock:
                if action == BEFORE_TRADING_START_BAR:
                    self.metrics_tracker.handle_market_open(session_label)
                elif action == SESSION_START:
                    once_a_day()
                elif action == SESSION_END:
                    # End of the session.
                    daily_perf_metrics = self._get_daily_message(session_label)
                    yield daily_perf_metrics

            risk_message = self.metrics_tracker.handle_simulation_end()
            yield risk_message

    def _get_daily_message(self, session_label):
        """
        Get a perf message for the given datetime.
        """
        if isinstance(session_label, pd.Timestamp):
            session_label = session_label.strftime('%Y%m%d')
        perf_message = self.metrics_tracker.handle_market_close(session_label)
        return perf_message
