#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
import operator
from collections import Mapping
from functools import reduce
from itertools import product
import numpy as np

class ParameterGrid(object):

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

    def __getitem__(self, ind):
        """通过index方式获取某个参数组合实现"""
        for sub_grid in self.param_grid:
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')


def _check_param_grid(param_grid):
    """检测迭代序列是否可进行grid"""



# noinspection PyAttributeOutsideInit
class GridSearch(object):
    """
        grid search , iter the combination of args to run the trade module
        steps : 1、product the args
                2、run trade
                3、metric the result of trade
    """

    def __init__(self, read_cash, choice_symbols, stock_pickers_product=None,
                 buy_factors_product=None, sell_factors_product=None, score_weights=None, metrics_class=None):
        """
        :param read_cash: 初始化资金数(int)
        :param choice_symbols: 初始备选交易对象序列
        :param stock_pickers_product: 选股因子product之后的序列
        :param buy_factors_product: 买入因子product之后的序列
        :param sell_factors_product: 卖出因子product之后的序列
        :param score_weights: make_scorer中设置的评分权重
        :param metrics_class: make_scorer中设置的度量类
        """

    def fit(self, score_class=WrsmScorer, n_jobs=-1):
        """
        开始寻找最优因子参数组合，费时操作，迭代所有因子组合进行交易回测，回测结果进行评分
        :param score_class: 对回测结果进行评分的评分类，AbuBaseScorer类型，非对象，只传递类信息
        :param n_jobs: 默认回测并行的任务数，默认-1, 即启动与cpu数量相同的进程数
        :return: (scores: 评分结果dict， score_tuple_array: 因子组合序列)
        """


def grid_mul_func(read_cash, benchmark, buy_factors, sell_factors, stock_pickers, choice_symbols, kl_pd_manager=None):
    """
    针对输入的买入因子，卖出因子，选股因子及其它参数，进行两年历史交易回测，返回结果包装AbuScoreTuple对象
    :param read_cash: 初始化资金数(int)
    :param benchmark: 交易基准对象，AbuBenchmark实例对象
    :param buy_factors: 买入因子序列
    :param sell_factors: 卖出因子序列
    :param stock_pickers: 选股因子序列
    :param choice_symbols: 初始备选交易对象序列
    :param kl_pd_manager: 金融时间序列管理对象，AbuKLManager实例
    :return: AbuScoreTuple对象
    """


def cartesian(arrays, out=None):
    """
        参数组合 ，不同于product
    """

    arrays = [np.asarray(x) for x in arrays]
    print('arrays',arrays)
    shape = (len(x) for x in arrays)
    print('shape',shape)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    print('ix',ix)
    ix = ix.reshape(len(arrays), -1).T
    print('ix_:',ix)

    if out is None:
        out = np.empty_like(ix, dtype=dtype)
        print('out',out.shape)

    for n, arr in enumerate(arrays):
        print('array',arrays[n])
        print(ix[:,n])
        out[:, n] = arrays[n][ix[:, n]]
        print(out[:,n])

    return out
#
class Opt:
    '''
      scipy.optimize.min(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None,
                         constraints=(), tol=None, callback=None, options=None)
      method: str or callable, optional, Nelder - Mead, (see here)
      Powell,, CG, BFGS, Newton - CG, L - BFGS - B, TNC, COBYLA, SLSQP, dogleg, trust - ncg,
               options: dict, optional
      maxiter: int.Maximum number of iterations to perform. disp: bool Constraints definition(only for COBYLA and SLSQP)
      type: eq for equality, ineq for inequality.fun: callable.jac: optional(only for SLSQP)
      args: sequence, optional
    '''
