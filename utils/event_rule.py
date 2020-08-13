# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod


class EventRule(ABC):
    """
        event --- rule
    """
    _cal = None

    @property
    def cal(self):
        return self._cal

    @cal.setter
    def cal(self, value):
        self._cal = value

    @abstractmethod
    def should_trigger(self, dt):
        raise NotImplementedError


class StatelessRule(EventRule):
    """
        a stateless rule can be composed to create new rule
    """
    def and_(self, rule):
        """
            trigger only when both rules trigger
        :param rule:
        :return:
        """
        return ComposedRule(self, rule, ComposedRule.lazy_and)

    __and__ = and_

    def should_trigger(self):
        pass


class ComposedRule(StatelessRule):
    """
     compose two rule with some composing function
    """
    def __init__(self, first, second, composer):
        if not (isinstance(first, StatelessRule) and isinstance(second, StatelessRule)):
            raise ValueError('only StatelessRule can be composed')

        self.first = first
        self.second = second
        self.composer = composer

    def should_trigger(self, dt):

        return self.composer(self.first, self.second)

    @staticmethod
    def lazy_and(first_trigger, second_trigger, dt):
        """
            lazy means : lazy_and will not call whenwhen first_trigger is not Stateless
        :param first_trigger:
        :param second_trigger:
        :param dt:
        :return:
        """
        return first_trigger.should_trigger(dt) and second_trigger.should_trigger(dt)

    @property
    def cal(self):
        return self.first.cal

    @cal.setter
    def cal(self, value):
        self.first.cal = self.second.cal = value


class Always(StatelessRule):

    @staticmethod
    def always_trigger(dt):
        return True

    should_trigger = always_trigger


class Never(StatelessRule):

    @staticmethod
    def never_trigger(dt):
        return False

    should_trigger = never_trigger
