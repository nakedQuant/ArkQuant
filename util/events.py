# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod
from collections import namedtuple
from util.context_tricks import nop_context


# --- event manager 用于处理ledger(righted violated expired postion)
class EventManager(object):
    """Manages a list of Event objects.
    This manages the logic for checking the rules and dispatching to the
    handle_data function of the Events.

    Parameters
    ----------
    create_context : (BarData) -> context manager, optional
        An optional callback to produce a context manager to wrap the calls
        to handle_data. This will be passed the current BarData.
    """
    def __init__(self, create_context=None):
        self._events = []
        # _create_context CallbackManager
        self._create_context = (
            create_context
            if create_context is not None else
            lambda *_: nop_context
        )

    def add_event(self, event, prepend=False):
        """
        Adds an event to the manager.
        """
        if prepend:
            self._events.insert(0, event)
        else:
            self._events.append(event)

    # __enter__ __exit_ 调用 __call__
    def handle_data(self, context, data, dt):
        with self._create_context(data):
            for event in self._events:
                event.handle_data(
                    context,
                    data,
                    dt,
                )


class Event(namedtuple('Event', ['rule', 'callback'])):
    """
    An event is a pairing of an EventRule and a callable that will be invoked
    with the current algorithm context, data, and datetime only when the rule
    is triggered.
    """
    # 实例之前(__init__) 调用__new__
    def __new__(cls, rule, callback=None):
        callback = callback or (lambda *args, **kwargs: None)
        return super(cls, cls).__new__(cls, rule=rule, callback=callback)

    def handle_data(self, context, data, dt):
        """
        Calls the callable only when the rule is triggered.
        """
        if self.rule.should_trigger(dt):
            self.callback(context, data)


class EventRule(ABC):
    """A rule defining when a scheduled function should execute.
    """
    # Instances of EventRule are assigned a calendar instance when scheduling
    # a function.
    _cal = None

    @property
    def cal(self):
        return self._cal

    @cal.setter
    def cal(self, value):
        self._cal = value

    @abstractmethod
    def should_trigger(self, dt):
        """
        Checks if the rule should trigger with its current state.
        This method should be pure and NOT mutate any state on the object.
        """
        raise NotImplementedError('should_trigger')


class StatelessRule(EventRule):
    """
    A stateless rule has no observable side effects.
    This is reentrant and will always give the same result for the
    same datetime.
    Because these are pure, they can be composed to create new rules.
    """
    def should_trigger(self, dt):
        raise NotImplementedError

    def and_(self, rule):
        """
        Logical and of two rules, triggers only when both rules trigger.
        This follows the short circuiting rules for normal and.
        """
        return ComposedRule(self, rule, ComposedRule.lazy_and)
    __and__ = and_


class ComposedRule(StatelessRule):
    """
    A rule that composes the results of two rules with some composing function.
    The composing function should be a binary function that accepts the results
    first(dt) and second(dt) as positional arguments.
    For example, operator.and_.
    If lazy=True, then the lazy composer is used instead. The lazy composer
    expects a function that takes the two should_trigger functions and the
    datetime. This is useful of you don't always want to call should_trigger
    for one of the rules. For example, this is used to implement the & and |
    operators so that they will have the same short circuit logic that is
    expected.
    """
    def __init__(self, first, second, composer):
        if not (isinstance(first, StatelessRule) and
                isinstance(second, StatelessRule)):
            raise ValueError('Only two StatelessRules can be composed')

        self.first = first
        self.second = second
        self.composer = composer

    def should_trigger(self, dt):
        """
        Composes the two rules with a lazy composer.
        """
        return self.composer(
            self.first.should_trigger,
            self.second.should_trigger,
            dt
        )

    @staticmethod
    def lazy_and(first_should_trigger, second_should_trigger, dt):
        """
        Lazily ands the two rules. This will NOT call the should_trigger of the
        second rule if the first one returns False.
        """
        return first_should_trigger(dt) and second_should_trigger(dt)

    @property
    def cal(self):
        return self.first.cal

    @cal.setter
    def cal(self, value):
        # Thread the calendar through to the underlying rules.
        self.first.cal = self.second.cal = value


class Always(StatelessRule):
    """
    A rule that always triggers.
    """
    @staticmethod
    def always_trigger(dt):
        """
        A should_trigger implementation that will always trigger.
        """
        return True
    should_trigger = always_trigger


class Never(StatelessRule):
    """
    A rule that never triggers.
    """
    @staticmethod
    def never_trigger(dt):
        """
        A should_trigger implementation that will never trigger.
        """
        return False
    should_trigger = never_trigger


__all__ = [
    'EventManager',
    'Event',
    'EventRule',
    'StatelessRule',
    'ComposedRule',
    'Always',
    'Never',
]
