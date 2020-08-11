# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

@object.__new__
class nop_context(object):
    """
        a nop context manager
    """
    def __enter__(self):
        pass

    def __exit__(self):
        pass


def _nop(args, **kwargs):

    pass


class _ManagedcallbackContext(object):

    def __init__(self, pre, post, args, kwargs):
        self._pre = pre
        self._post = post
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        return self._pre(*self._args, **self._kwargs)

    def __enter__(self):
        self.post(*self._args, **self._kwargs)


class CallbackManager(object):
    """
        create a context manager for a pre-execution callback and a post-execution callback
        context 里面嵌套 context
    """
    def __init__(self, pre=None, post=None):
        self.pre = pre if pre is not None else _nop
        self.post = post if post is not None else _nop

    def __call__(self,*args, **kwargs):
        return _MangedCallbackContext(self.pre, self.post, *args, **kwargs)

    def __enter__(self):
        return self.pre

    def __exit__(self, *exec_info):
        self.post()


# 事件 rule callback  用于scedule module
class EventManager(object):
    """
        manage a list of event objects
        checking the rule and dispatch the handle_data to the events
        event --- rule ,trigger ,function : handle_data
    """
    def __init__(self, create_context=None):
        self._events = []
        # 要不然函数或者类的__call__ 方法
        self._create_context = (
            create_context if create_context is not None
            else  lambda *_ : nop_context
        )

    def add_event(self, event, prepend=False):
        if prepend:
            self._event.insert(0, event)
        else:
            self._events.append(event)

    # evnet hanle_data方法保持一致
    def handle_data(self, context, data, dt):
        with self._create_context(data):
            for event in self._events:
                event.handle_data(
                        context,
                        data,
                        dt)

# event_manager.add_event(
#     Event(
#         Always(),
#         # We pass handle_data.__func__ to get the unbound method.
#         # We will explicitly pass the algorithm to bind it again.
#         handle_data.__func__,
#     ),
#     prepend=True,
# )
