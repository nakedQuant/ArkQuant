# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from threading import Thread
from queue import Queue
from collections import defaultdict


class EventEngine(object):
    """
        定义策略引擎将不同的算法按照顺序放到一个队列里面依次进行执行，算法对应事件，可以注册、剔除事件
    """
    def __init__(self):

        self._queue = Queue()
        self._thread = Thread(target=self._run)
        # 以计时模块为例，换成其他的需求添加都队列里面
        self._timer = Thread(target=self._run_timer)
        self._handlers = defaultdict(list)
        self._general = []

    def _run(self):
        while self._active:
            try:
                algo = self._queue.get(block=True, timeout=1)
                self._process(algo)
            except Empty:
                pass

    def _process(self, algo):
        if algo._type in self._handlers:
            [handler(algo) for handler in self._handlers[algo._type]]

        if self._general:
            [handler(algo) for handler in self._general]

    def _run_timer(self):

        while self._active:
            # sleep(self._interval)
            event = Event('timer')
            self.put(event)

    def start(self):

        self._active = True
        self._thread.start()
        self._timer.start()

    def stop(self):
        self._active = False
        self._timer.join()
        self._thread.join()

    def put(self, event):
        self._queue.put(event)