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


    裁决模块 基于有效的特征集，针对特定的asset进行投票抉择
    关于仲裁逻辑：
        普通选股：针对备选池进行选股，迭代初始选股序列，在迭代中再迭代选股因子，选股因子决定是否对
        symbol投出反对票，一旦一个因子投出反对票，即筛出序列


    组合不同算法---策略
    返回 --- Order对象
    initialize
    handle_data
    before_trading_start
        1.判断已经持仓是否卖出
        2.基于持仓限制确定是否执行买入操作

    MIFeature（构建有特征组成的接口类），特征按照一定逻辑组合处理为策略
            实现： 逻辑组合抽象为不同的特征的逻辑运算，具体还是基于不同的特征的运行结果
        strategy composed of features which are logically arranged
        input : feature_list
        return : asset_list
        param : _n_field --- all needed field ,_max_window --- upper window along the window args
        core_part : _domain --- logically combine all features

# 核心思想 --- 细微的异常预测短期走向 ； 接力行为


"""
