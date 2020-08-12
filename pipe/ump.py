# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from pipe.term import Term
import numpy as np


class UmpPickers(object):
    """
        包括 --- 止损策略
        Examples:
            FeatureUnion(_name_estimators(transformers),weight = weight)
        裁决模块 基于有效的特征集，针对特定的asset进行投票抉择
        关于仲裁逻辑：
        H0假设 --- 标的退出
            迭代选股序列因子 --- 一旦任何因子投出反对票无法通过HO假设
        基于一个因子判断是否退出股票有失偏颇
    """
    def __init__(self, pickers):
        self._validate_features(pickers)

    def _validate_features(self, features):
        for feature in features:
            assert isinstance(feature, Term), 'must term type'
            if feature.dtype != bool:
                raise Exception('bool term needed for ump')
        self._poll_pickers = features

    @property
    def pickers(self):
        return self._poll_pickers

    def __setattr__(self, key, value):
        raise NotImplementedError

    def _evaluate_for_position(self, position, metadata):
        votes = [picker.compute([position.asset], metadata)
                 for picker in self.pickers]
        if np.all(votes):
            return position
        return False

    def evaluate(self, position, metadata):
        vote = self._evaluate_for_position(position, metadata)
        return vote
