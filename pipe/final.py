# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC, abstractmethod


class BaseFinal(ABC):
    """
        Final seek the most of appropriate asset of  pipeline output
        intended for long operation
    """

    @abstractmethod
    def resolve_final(self, finals):
        # 基于最后一个节点的assets --- 选出合适的标的
        raise NotImplementedError


class Final(BaseFinal):

    def resolve_final(self, finals):
        # 扩展 --- 最后一个节点的assets --- 选出合适的标的
        # outputs = [asset.source_id(self.name) for asset in final[:alternative]]
        # asset tag pipeline_name --- 可能存在相同的持仓但是由不同的pipeline产生
        return finals[0]


class OptFinal(BaseFinal):

    def resolve_final(self, finals):
        """
           在胜率与空仓抉择
        """


__all__ = ['Final', 'OptFinal']

