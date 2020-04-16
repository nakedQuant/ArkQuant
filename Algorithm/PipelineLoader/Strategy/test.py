# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from abc import ABC,abstractmethod
import importlib,pandas as pd

from Algorithm.Feature.Scorer import BaseScorer

class MIFeature(ABC):
    """
        strategy composed of features which are logically arranged
        input : feature_list
        return : asset_list
        param : _n_field --- all needed field ,_max_window --- upper window along the window args
        core_part : _domain --- logically combine all features
    """
    _n_fields  = []
    _max_window = []
    _feature_params = {}

    def _load_features(self,name):
        try:
            feature_class = importlib.__import__(name, 'Algorithm.Feature')
        except:
            raise ValueError('%s feature not implemented'%name)
        return feature_class

    def _verify_params(self,params):
        if isinstance(params,dict):
            for name,p in params:
                feature = self._load_features(name)
                if hasattr(feature,'_n_fields') and feature._n_fields != p['fields']:
                    raise ValueError('fields must be same with feature : %s'%name)
                if feature.windowed and p['window'] is None:
                    raise ValueError('window of feature  is not None : %s'%name)
                if feature._pairwise and not isinstance(p['window'],(tuple,list)):
                    raise ValueError('when pairwise is True ,the length of window must be two')
                if hasattr(feature,'_triple') and not isinstance(p['window'],dict):
                    raise ValueError('triple means three window , it specify macd --- fast,slow,period')
        else:
            raise TypeError('params must be dict type')

    def _set_params(self,params):
        self._verify_params(params)
        return params


    def _eval_feature(self,raw,name,p:dict):
        """
            特征分为主体、部分，其中部分特征只是作为主体特征的部分逻辑
        """
        feature_class = self._load_features(name)
        if 'field' in p.keys():
            print('filed exists spceify this feature should be initialized')
            if 'window' in p.key():
                result = feature_class.calc_feature(raw[p['field']],p['window'])
            else:
                result = feature_class.calc_feature(raw['field'])
        else:
            print('field not exists spceify this feature is just  a middle process used by outer faeture function')
            result = None
        return result


    def _fit_main_features(self,raw):
        """
            计算每个标的的所有特征
        """
        filter_nan = {}
        for name in self._n_features:
            res = self._eval_feature(raw,name,self._feature_params[name])
            if res:
                filter_nan.update({name:res})
        return filter_nan


    def _execute_main(self, trade_date,stock_list):
        feature_res = {}
        for code in stock_list:
            event = Event(trade_date,code)
            req = GateReq(event, field=self._n_fields, window=self._max_window)
            raw = feed.addBars(req)
            res = self._fit_main_features(raw)
            feature_res.update({code:res})
        return feature_res

    @abstractmethod
    def _domain(self,input):
        """
            MIFeature（构建有特征组成的接口类），特征按照一定逻辑组合处理为策略
            实现： 逻辑组合抽象为不同的特征的逻辑运算，具体还是基于不同的特征的运行结果
        """
        NotImplemented


    def run(self,trade_dt,stock_list:list) -> list:
        exec_info= self._execute_main(trade_dt,stock_list)
        filter_order = self._domain(exec_info)
        return filter_order



class MyStrategy(MIFeature):
    """
        以MyStrategy为例进行实现
    """

    _n_features = ['DMA','Reg']

    def __init__(self,params):
        self._feature_params = super()._set_params(params)
        self._n_fields = [ v['field'] for k,v in params.items() if 'field' in v.keys()]
        self._max_window = [ v['window'] for k,v in params.items() if 'window' in v.keys()].max()

    def __enter__(self):
        return self


    def _domain(self,input):
        """
            策略核心逻辑： DMA --- 短期MA大于的长期MA概率超过80%以及收盘价处于最高价与最低价的形成夹角1/2位以上，则asset有效
            return ranked_list
        """
        df = pd.DataFrame.from_dict(input)
        result = df.T
        hit_rate = result['DMA'].applymap(lambda x : len(x[x>0])/len(x) > 0.75)
        reg = result['Reg'].map(lambda x : x > 0.6)
        # union = set(reg.index) & set(hit_rate.index)
        input = (pd.DataFrame([hit_rate,reg])).T
        union = BaseScorer().calc_feature(input)
        return union

    def __exit__(self,exc_type,exc_val,exc_tb):
        """
            exc_type,exc_value,exc_tb(traceback), 当with 后面语句执行错误输出
        """
        if exc_val :
            print('strategy fails to complete')
        else:
            print('successfully process')