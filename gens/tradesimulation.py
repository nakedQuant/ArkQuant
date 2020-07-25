#
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
from contextlib import ExitStack
from functools import partial
from .protocol import  BarData
from utils.api_support import  ZiplineAPI

BAR = 0
SESSION_START = 1
SESSION_END = 2
MINUTE_END = 3
BEFORE_TRADING_START_BAR = 4

class AlgorithmSimulator(object):

    EMISSION_TO_PERF_KEY_MAP = {
        'minute': 'minute_perf',
        'daily': 'daily_perf'
    }

    def __init__(self,
                 algo,
                 sim_params,
                 data_portal,
                 restriction,
                 benchmark):
        # ==============
        # Algo Setup
        # ==============
        self.algo = algo
        # ==============
        # Simulation
        # Param Setup
        # ==============
        self.sim_params = sim_params
        self.data_portal = data_portal
        self.restrictions = restriction
        self.benchmark = benchmark
        # We don't have a datetime for the current snapshot until we
        # receive a message.
        # This object is the way that user algorithms interact with OHLCV data,
        # fetcher data, and some API methods like `data.can_trade`.
        self.current_data = self._create_bar_data()

    #获取日数据，封装为一个API(fetch process flush other api)
    def _create_bar_data(self):
        return BarData(
            data_portal=self.data_portal,
            data_frequency=self.sim_params.data_frequency,
            trading_calendar=self.algo.trading_calendar,
            restrictions=self.restrictions,
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
            #生成交易订单
            txns = MatchUp.carry_out(algo.engine,ledger)
            #处理交易订单
            ledger.process_transaction(txns)

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
            stack.enter_context(ZiplineAPI(self.algo))

            metrics_tracker.handle_start_of_simulation(self.benchmark)

            daily_perf_metrics = partial(self._get_daily_message,metrics_tracker = metrics_tracker)

            for dt, action in self.clock:
                if action == BEFORE_TRADING_START_BAR:
                    # algo.before_trading_start(self.current_data)
                    metrics_tracker.handle_market_open(dt)
                elif action == SESSION_START:
                    once_a_day(dt)
                elif action == SESSION_END:
                    # End of the session.
                    yield daily_perf_metrics

            risk_message = metrics_tracker.handle_simulation_end()

            yield risk_message

    def _get_daily_message(self,metrics_tracker):
        """
        Get a perf message for the given datetime.
        """
        perf_message = metrics_tracker.handle_market_close()
        return perf_message
