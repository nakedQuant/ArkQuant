from abc import ABC

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


class Ump(_BaseComposition):
    """
        裁决模块 基于有效的特征集，针对特定的asset进行投票抉择
        关于仲裁逻辑：
            普通选股：针对备选池进行选股，迭代初始选股序列，在迭代中再迭代选股因子，选股因子决定是否对
            symbol投出反对票，一旦一个因子投出反对票，即筛出序列
    """

    def __init__(self, poll_workers, thres=0.8):
        super()._validate_steps(poll_workers)
        self.voters = poll_workers
        self._poll_picker = dict()
        self.threshold = thres

    def _set_params(self, **params):
        for pname, pval in params.items():
            self._poll_picker[pname] = pval

    def poll_pick(self, res, v):
        """
           vote for feature and quantity the vote action
           simple poll_pick --- calculate rank pct
           return bool
        """
        formatting = pd.Series(range(1, len(res) + 1), index=res)
        pct_rank = formatting.rank(pct=True)
        polling = True if pct_rank[v] > self.thres else False
        return polling

    def _fit(self, worker, target):
        '''因子对象针对每一个交易目标的投票结果'''
        picker = super()._load_from_name(worker)
        fit_result = picker(self._poll_picker[worker]).fit()
        poll = self.poll_pick(fit_result, target)
        return poll

    def decision_function(self, asset):
        vote_poll = dict()
        for picker in self.voters:
            vote_poll.update({picker: self._fit(picker, asset)})
        decision = np.sum(list(vote_poll.values)) / len(vote_poll)
        return decision
