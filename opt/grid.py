#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import operator
from collections import Mapping
from functools import reduce, partial
from itertools import product


class ParameterGrid(object):
    """
      scipy.optimize.min(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None,
                         constraints=(), tol=None, callback=None, options=None)
      method: str or callable, optional, Nelder - Mead, (see here)
      Powell,, CG, BFGS, Newton - CG, L - BFGS - B, TNC, COBYLA, SLSQP, dogleg, trust - ncg,
               options: dict, optional
      maxiter: int.Maximum number of iterations to perform. disp: bool Constraints definition(only for COBYLA and SLSQP)
      type: eq for equality, ineq for inequality.fun: callable.jac: optional(only for SLSQP)
      args: sequence, optional
    """

    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        self.param_grid = param_grid

    def __iter__(self):
        """迭代参数组合实现"""
        for p in self.param_grid:
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """参数组合长度实现"""
        product_mul = partial(reduce, operator.mul)
        return sum(product_mul(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)
