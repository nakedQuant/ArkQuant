# -*- coding :utf-8 -*-

from abc import ABC , abstractmethod
from collections import namedtuple


#上下文context
class nop_context(object):
    """
        a nop context manager
    """
    def __enter__(self):
        pass

    def __exit__(self):
        pass

def _nop(args,**kwargs):

    pass


class CallbackManager(object):
    """
        create a context manager for a pre-execution callback and a post-execution callback
        context 里面嵌套 context
    """
    def __init__(self,pre = None , post = None):
        self.pre = pre if pre is not None else _nop
        self.post = post if post is not None else _nop

    def __call__(self,*args , **kwargs):
        return _MangedCallbackContext(self.pre,self.post,*args,**kwargs)

    def __enter__(self):
        return self.pre

    def __exit__(self,*exec_info):
        self.post()

class _ManagedcallbackContext(object):

    def __init__(self,pre,post,args,kwargs):
        self._pre = pre
        self._post = post
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        return self._pre(*self._args,**self._kwargs)

    def __enter__(self):
        self.post(*self._args,**self._kwargs)


# event_manager.add_event(
#     Event(
#         Always(),
#         # We pass handle_data.__func__ to get the unbound method.
#         # We will explicitly pass the algorithm to bind it again.
#         handle_data.__func__,
#     ),
#     prepend=True,
# )


#事件 rule callback  用于scedule module
class EventManager(object):
    """
        manage a list of event objects
        checking the rule and dispatch the handle_data to the events
        event --- rule ,trigger ,function : handle_data
    """
    def __init__(self,create_context = None):
        self._events = []
        # 要不然函数或者类的__call__ 方法
        self._create_context = (
            create_context if create_context is not None
            else  lambda *_ : nop_context
        )

    def add_event(self,event,prepend = False):
        if prepend:
            self._event.insert(0,event)
        else:
            self._events.append(event)

    #与evnet hanle_data方法保持一致
    def handle_data(self,context,data,dt):
            with self._create_context(data):
                for event in self._events:
                    event.handle_data(
                        context,
                        data,
                        dt,
                    )


class Event(namedtuple('Event',['rule','callback'])):
    """
        event consists of rule and callback
        when rule is triggered ,then callback
    """
    def __new__(cls,rule,callback=None):
        callback = callback or (lambda *args,**kwargs : None)
        return super(cls,cls).__new__(cls,rule = rule,callback = callback)

    def handle_data(self,context,data,dt):
        if self.rule.should_trigger(dt):
            self.callback(context,data)


class EventRule(ABC):
    """
        event --- rule
    """
    _cal = None

    @property
    def cal(self):
        return self._cal

    @cal.setter
    def cal(self,value):
        self._cal = value

    @abstractmethod
    def should_trigger(self,dt):
        raise NotImplementedError

class StatelessRule(EventRule):
    """
        a stateless rule can be composed to create new rule
    """
    def and_(self,rule):
        """
            trigger only when both rules trigger
        :param rule:
        :return:
        """
        return ComposedRule(self,rule,ComposedRule.lazy_and)

    __and__ = and_

class ComposedRule(StatelessRule):
    """
     compose two rule with some composing function
    """
    def __init__(self,first,second,composer):
        if not (isinstance(first,StatelessRule) and isinstance(second,StatelessRule)):
            raise ValueError('only StatelessRule can be composed')

        self.first = first
        self.second = second
        self.composer = composer

    def should_trigger(self,dt):

        return self.composer(self.first,self.second)

    @staticmethod
    def lazy_and(first_trigger,second_trigger,dt):
        """
            lazy means : lazy_and will not call whenwhen first_trigger is not Stateless
        :param first_trigger:
        :param second_trigger:
        :param dt:
        :return:
        """
        return first_trigger.should_trigger(dt) and second_trigger.should_trigger(dt)

    @staticmethod
    def cal(self):
        return self.first.cal

    @cal.setter
    def cal(self,value):
        self.first.cal = self.second.cal = value


class Always(StatelessRule):

    @staticmethod
    def always_trigger(dt):
        return True

    should_trigger = always_trigger


class Never(StatelessRule):

    @staticmethod
    def never_trigger():
        return False

    should_trigger = never_trigger