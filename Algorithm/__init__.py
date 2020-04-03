# -*- coding:utf-8 -*-

import importlib,numpy as np
from abc import ABC,abstractmethod
from joblib import Parallel,delayed,Memory
from functools import reduce

from Tool.Wrapper import _validate_type

__all__ = ['Ump','Pipeline', 'FeatureUnion']


class _BaseComposition(ABC):
    """
        将不同的算法通过串行或者并行方式形成算法工厂 ，筛选过滤最终得出目标目标标的组合
        串行：
            1、串行特征工厂借鉴zipline或者scikit_learn Pipeline
            2、串行理论基础：现行的策略大多数基于串行，比如多头排列、空头排列、行业龙头战法、统计指标筛选
            3、缺点：确定存在主观去判断特征顺序，前提对于市场有一套自己的认识以及分析方法
        并行：
            1、并行的理论基础借鉴交集理论
            2、基于结果反向分类strategy
        难点：
            不同算法的权重分配
        input : stategies ,output : list or tuple of filtered assets
    """
    @classmethod
    def _load_from_name(cls,name):
        """Generate names for estimators, if it is instance(already initialized) just return,else return the class  """
        try:
            strat = importlib.__import__(name, 'Algorithm.Strategy')
        except:
            raise ValueError('some of features not implemented')
        return strat

    def _validate_steps(self,steps):
        for item in steps:
            if not hasattr(item, 'fit'):
                raise TypeError('all steps must have calc_feature method')

    @abstractmethod
    def _set_params(self,**params):
        '''
        For this, it enables setting parameters of the various steps using their names and
        params : dict of string -> object
        '''
        raise NotImplemented

    def set_params(self,**kwargs):
        """Set the parameters of estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self._set_params(**kwargs)
        return self

    def register(self, strategy):
        if strategy not in self._n_features:
            self._n_features.append(strategy)

    def unregister(self, feature):
        if feature in self._n_features:
            self._n_features.remove(feature)
        else:
            raise ValueError('特征没有注册')

    @abstractmethod
    def _fit(self,step,res):
        """
        run algorithm  it means already has instance ;else it need to initialize
        """
        raise NotImplemented

    @abstractmethod
    def decision_function(self):
        """Apply transforms, and decision_function of the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the Pipeline.
        Returns
        -------
        output array_like ,stockCode list
        """
        pass


class Pipeline(_BaseComposition):
    """
        pipe of strategy to fit targeted asset
        Parameters
        -----------
        steps :list
            List of strategies
            wgts: List,str or list , default : 'average'
        wgts: List
            List of (name,weight) tuples that allocate the weight of steps means the
            importance, average wgts avoids the unbalance of steps
        memory : joblib.Memory interface used to cache the fitted transformers of
            the Pipeline. By default,no caching is performed. If a string is given,
            it is the path to the caching directory. Enabling caching triggers a clone
            of the transformers before fitting.Caching the transformers is advantageous
            when fitting is time consuming.
    """
    _required_parameters = ['steps']

    def __init__(self,steps,memory = None):
        super()._validate_steps(steps)
        self.steps = steps
        self.cachedir = memory
        self._pipe_params=dict()

    def __len__(self):
        '''
        return the length of Pipeline
        '''
        return len(self.steps)

    def _set_params(self,**params):
        '''
        For this, it enables setting parameters of the various steps using their names and
        params : dict of string -> object
        '''
        for pname ,pval in params.items():
            self._pipeline_params[pname] = pval
        if len(self._pipeline_params) != len(self.steps):
            raise ValueError('all strategies must have params to initialize')

    def register(self, strategy):
        if strategy not in self.steps:
            self.steps.append(strategy)
        else:
            raise ValueError('%s already registered in Pipeline'%strategy)

    def unregister(self, strategy):
        if strategy in self.steps:
            self.steps.remove(strategy)
        else:
            raise ValueError('%s has not registered in Pipeline'%strategy)

    def _fit(self,step) -> list:
        """
        run algorithm ,if param is passthrough , it means already has instance ;else it need to initialize
        """
        strategy = self._load_from_name(step)
        res = strategy(self._self._pipeline_params[step]).run()
        return res

    # Estimator interface
    def _fit_cache(self, step:str,res:list):
        # Setup the memory
        memory = Memory(self.cachedir)
        if hasattr(memory, 'location'):
            if memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                print('no caching is done and the memory object is completely transparent')
        # memory the function
        fit_cached = memory.cache(self._fit)
        # Fit or load from cache the current transfomer.This is necessary when loading the transformer
        out = fit_cached(input,step,res)
        return out

    @_validate_type(_type=(list,tuple))
    def decision_function(self,portfilio:list):
        """
        Based on the steps of algorithm ,we can conclude to predict target assetCode.
        Apply transforms, and predict_proba | predict_log_proba of the final estimator
        If parallel is False(Pipeline),apply all the estimator sequentially by data,then
        predict the target
        """
        for idx,name in enumerate(self.steps):
            portfilio = self._fit_cache(name,portfilio)
        return portfilio


class FeatureUnion(_BaseComposition):
    """
    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters
    ----------
    transformer_list : List of transformer objects to be applied to the data
    n_jobs : int --- Number of jobs to run in parallel,
            -1 means using all processors.`
    allocation: str(default=average) ,dict , callable

    Examples:
        FeatureUnion(_name_estimators(transformers),weight = weight)
    """
    _required_parameters = ["transformer_list"]

    def __init__(self, transformer_list,mode='average'):
        super()._validate_steps(transformer_list)
        self.transformer_list = transformer_list
        self._n_jobs = len(transformer_list)
        self._feature_weight(mode)

    @property
    def _feature_weight(self,mode):
        self.transformer_allocation = dict()
        if not isinstance(mode,(dict , str)):
            raise TypeError('unidentified type')
        elif  isinstance(mode, str) and mode == 'average':
            wgt = 1/len(self.transformer_list)
            self.transformer_allocation = {{name:wgt} for name in self.transformer_list}
        else:
            self.transformer_allocation = mode

    def _set_params(self,**params):
        '''
        For this, it enables setting parameters of the various steps using their names and
        params : dict of string -> object
        '''
        for pname ,pval in params.items():
            self._featureUnion_params[pname] = pval
        if len(self.self._featureUnion_param) != len(self.transformer_list):
            raise ValueError('all  strategies must have params to be initialized')

    def register(self, strategy):
        if strategy not in self.transformer_list:
            self.transformer_list.append(strategy)
        else:
            raise ValueError('%s already registered in featureUnion'%strategy)

    def unregister(self, strategy):
        if strategy in self.transformer_list:
            self.transformer_list.remove(strategy)
        else:
            raise ValueError('%s has not registered in featureUnion'%strategy)

    def _parallel_func(self,porfilio):
        """Runs func in parallel on X and y"""
        return Parallel(n_jobs = self._n_jobs)(delayed(self._fit(name,porfilio)) for name in self.transformer_list)

    def _fit(self,name):
        strategy = self._load_from_name(name)
        outcome = strategy(self._featureUnion_params[name]).run()
        ordered = self._fit_score(name,outcome)
        return (ordered,outcome)

    def _fit_score(self,idx,res):
        """
            Apply score estimator with output
            input : list of ordered assets
            output :
        """
        align = pd.DataFrame(list(range(1, len(res) + 1)), index=res, columns=[idx])
        align_rank = align.rank() * self._feature_weight[idx]
        return align_rank

    def _fit_score_union(self,assets):
        '''
            two action judge if over half of tranformer_list has nothing in common ,that is means union is empty,
            the selection of tranformer_list is not appropriate ,switch to _update_tranformer_list,del tranformer which
            has not intersection with union
        '''
        score_union = pd.DataFrame()
        def union(x,y):
            internal = set(x) | set(y)
            return internal

        r =  self._parallel_func(assets)
        aligning,res = zip(* r)
        intersection = reduce(union,res)
        score_union = [score_union.append(item) for item in aligning]
        return intersection,score_union.sum(axis =1)

    @_validate_type(_type=(list,tuple))
    def decision_function(self):
        assets,scores = self._fit_score_union()
        if not len(targets):
            raise ValueError('union set is empty means strategies need to be modified --- args or change strategy')
        sorted_assets = scores.loc[assets].sort_values(ascending = False)
        return list(sorted_assets.index)


class Ump(_BaseComposition):
    """
        裁决模块 基于有效的特征集，针对特定的asset进行投票抉择
        关于仲裁逻辑：
            普通选股：针对备选池进行选股，迭代初始选股序列，在迭代中再迭代选股因子，选股因子决定是否对
            symbol投出反对票，一旦一个因子投出反对票，即筛出序列
    """

    def __init__(self,poll_workers,thres = 0.8):
        super()._validate_steps(poll_workers)
        self.voters = poll_workers
        self._poll_picker = dict()
        self.threshold = thres

    def _set_params(self,**params):
        for pname ,pval in params.items():
            self._poll_picker[pname] = pval


    def poll_pick(self,res,v):
        """
           vote for feature and quantity the vote action
           simple poll_pick --- calculate rank pct
           return bool
        """
        formatting = pd.Series(range(1,len(res)+1),index = res)
        pct_rank = formatting.rank(pct = True)
        polling = True if pct_rank[v] > self.thres else False
        return polling

    def _fit(self,worker,target):
        '''因子对象针对每一个交易目标的投票结果'''
        picker = super()._load_from_name(worker)
        fit_result = picker(self._poll_picker[worker]).fit()
        poll = self.poll_pick(fit_result,target)
        return poll

    def decision_function(self,asset):
        vote_poll = dict()
        for picker in self.voters:
            vote_poll.update({picker:self._fit(picker,asset)})
        decision = np.sum(list(vote_poll.values))/len(vote_poll)
        return decision