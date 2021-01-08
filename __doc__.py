
'''
   宏观层面分析：
   针对当前交易环境，开发对应的系统，系统基于不同的特征组成
   关键：定义当前环境 --- 环境变量、不同变量之间的作用方式 ，关键：存在一个时间滞后问题，环境的延续性算法特征

   quant平台：分为两大类 一、基于算法主动筛选标的；二、针对于特定标的，基于仲裁算法对标的进行裁定
   事件分为buy sell hold 三个类别，对应的handler不同的处理方法
   借鉴事件引擎思路：多层过滤，采用事件驱动引擎中，队列，开始，事件注册、事件取消、通用事件，将通过的过滤算法排列组合成构成过滤引擎，采用asynico框架
   执行算法主要为：penetrate（渗透算法）
   订单处理系统OMS 基于事件的触发，更新账户
执行买入算法：
1、宏观层面的过滤，动态通过短期涨幅对股票进行Label,并进行划分，寻找强势股；行业过滤，但是由于行业划分不准确，存在较大的误差，只能作为参考；
基本面过滤（集中在股权集中度、并购、借壳等，短期涨幅之后，回调之后，后期延续涨势）
2、通过算法Algo（统计套利） 统计显著，全A股，一共3700多只股票，聚类交易、形态交易，内在逻辑：Filter  Rank
具体实现逻辑：从大范围筛选，逐级降低，最后一层基于排序算法，每一个Layer过滤都有一个权重（算法的权重逐级降低，每个一个算法之后的权重）复合权重，作为最后一层的
排序的排名，目标得出股票集合（设定个数）
3、配对交易做法：（指数套利），选取一个指数，以及对应的成分股票，存在均值回归的倾向，内在逻辑：指数的由成分股的根据一定比例计算出来的，但是偏离度与指数的偏离度的比例偏离不会
太远，当指数突破的时候，突出的成分股必然会涨；（协整检验）：针对每一个指数，以指数里面的每一个股票做协整检验，寻找最显著的股票；（可转债套利）：当股票停盘后，对应的可转债的价格可能存在套利空间；
4、低价股策略（A股特色）：主要集中在从基本面考察，股权集中度、以及持仓股东的关系网，关联持股公司的网络，Relation 策略（涉及爬虫，找相关的内容），当资金充裕时
做长期投资，不太适合短期选股

订单系统OMS：
表单系统，基于orm实现更新create update query ，字段名：balance codeName position cash ,以对象的形式保存（每个交易日都是一个对象），将时间区间的对象集
进行bulk处理，写入到文件里或者数据库

回测模块：
1、分析评价模块 区间收益、最大回撤、波动率，胜率等做为评价集合，基于算法给出评分
2、参数优化，基于不同的参数组合的评价模块为目标，决定相应的最优参数
3、参数泛化，将2中的最优参数偏离一些，看最终的评价结果偏离度，如果偏差很大，说明参数过度优化，分析原因可能：
   3.1、过滤条件或者排序算法过多，3.2、不同涉及算法中的可变参数，3.3、评价方法（主观因素导致的评价方法）
4、回测的目标函数：确定算法的持仓个数num、周期holding_period、止损点loss_point、止盈点 win_point

量化交易系统:
    a.策略识别（搜索策略 ， 挖掘优势 ， 交易频率）
    b.回溯测试（获取数据 ， 分析策略性能 ，剔除偏差）
    c.交割系统（经纪商接口 ，交易自动化 ， 交易成本最小化）
    d.风险管理（最优资本配置 ， 最优赌注或者凯利准则 ， 海龟仓位管理）

监控模块：
1、运行状态：进程间通信
2、基于日志监控:滚动日志

数据高速接口：
gateway

实盘系统：
XTP（中泰证券）：
最终目标基于中泰的接口，实行实盘交易

统计交易策略：
1、原理：all - industry - code ，关键字：行业价格指数算法、技术指标筛选算法、相似度算法
    layer 1：
        获取全量的A股以及对应的行业的字段，每年3月1号作为时间点（理由A股新年标志）计算一级行业指数（只选择一级的原因：由于概念的不断增加，按照概念
    进行划分增加计算负载、无形增加波动率，信号不稳定，很多的概念都是由主业衍生出来的，根据主要做目标相对来说稳妥，同时容易抓住长线的趋势.
    具体实现方式：
    以上个时间点的加权市值/市值波动率、涨幅（收益率）作为评价指标对行业的股票进行排序得出排序权重，最终计算行业的排序权重价格指数
    layer 2:
        基于layer 1的行业价格指数，计算EMA指标或者其他的技术指标(Techincal) ,筛选出目标源行业
    layer 3:
        基于layer 2中的行业，计算相似性算法，找出与行业价格变动关联度最高的股票集（保证一定存在股票可以之后交易日买入）
    关联度的算法：基础的相关性，联动效应
    总结：基于行业价格指数筛选股票 ，1、计算每个行业的偏离度，2、选出最大正向偏离度的行业，3、基于行业筛选出相似度最大的股票
    industry atr,权重不变，但是不是价格的，而是atr由特征转为策略 ，特征抽象化

2、原理：all - algo 1 - algo 2 -algo 3 ------ algo n ，对全面的股票进行算法排序，不同的算法赋予一个权重乘以对应算法的排序号，最终将不同算法排序进行
加权，得出最终排序值，将排序值的top bucket（一揽子股票）作为标的对象（保证一定存在可以交易），借鉴因子暴露度概念推导出算法暴露度，不同的算法代表不同的因子，
关键字：算法的个数、算法的顺序更新时间，其中算法的个数可以通过PCA来确定，算法的顺序代表算法的优先等级；换一个角度看，算法侧重于哪个指标

3、原理：借鉴与所谓的多头或者空头排列衍生出得贝叶斯算法：排列----多个指标同时发出的买入或者卖出信号，作为最后的决策信号；在某种信号发生做为条件，特定事件作为触发器
，以此作为算法（事件关联算法）

4、原理：配对交易--不同标的之间存在稳定的关系，但由于外界因素短暂偏离，经过一段时间会回复原有的相对稳定关系，算法主要是协整检验，平稳性检验，关键点：不建议基于个股
之间做配对交易，因为个股本身容易收到外间干扰的因素太多，推荐指数，因为指数相对来说波动比较平稳，但是发生变动的容易作为目标指示对象，实现逻辑：基于指数，计算指数
内所有成分股与指数本身的偏离度，筛选出最大的偏离度的股票，风险：黑天鹅事件

5、强势趋势分析：
   将所有强势的股票放在篮子里，通过分析他们的情况来判断市场热度
   筛选出前一段时间（比如一个月或者某个固定时间），涨幅巨大超过一定程度，回撤在一定范围的股票，所谓区间龙虎榜，依旧原理：国内的A股的巨大惯性，
   涨福类比于质量，质量越大，惯性越大；前一段阶段的涨幅过程积蓄了确定方向的能量，惯性效应显著的。细节：跌幅被上涨的趋势填平，所以才会上涨。
   存在一种观点：如果股价在爆发点之后的1-3周内就上涨了20%，在这种情况下，持有比如8周或者n周，股价存在上升100-200%的动能因此需要持有更长的时间
   以分享更多的收益。

6、绝对套利：
1、可转债套路：对于A股来将，一些突发的利好消息，导致股票无法买入（涨跌停限制），但是如果对应有可转债的话，买入可转债分享对应的收益
2、A/H股 A/L股，对于同一主体在不同市场上市会产生溢价或折价

7、黑天鹅事件：
本质：1、公司由于流动性、或者管理原因，导致市值断崖式下跌，2、所处于行业具有技术门槛或者优势或者处于生活必须品位建立不可能一步到位，根据市值的理解，建立仓位；
资金必须是自由资金、作为长期的投资，需要时间突发的事件导致股价巨大跌幅
缺点：违法行为，存在退市风险，仔细筛选黑天鹅公司

8、低价股策略：
1、A股特色，当资金充裕的时候，选取一些市值低于20亿一下的个股，同时股权相对来说比较集中（具体达到一定比例例如70%），一般时间会长，但是可以突破的时候，做波动

9、基于期货价格、股票价格联动：
1、基于公司盈利模式，如果公司主营的业务的材料价格已经处于地位或者顶部徘徊，那么以此关联的公司的利润会产生较大变大，例子：钢铁价格之前处于地位，由于国家产业合并，
共供给减少，价格上升，对应的ST钢铁行业上市公司，摘牌就是大概率事件（之前的ST八钢）；反向例子：方大炭素

10、基于可转债的逻辑：
当股票行情转好的，对应的可转债也同向，产生套利机制；如果股票由于利好消息停盘，可以买入对应的可转债，前提标的具有对应的可转债

卖出策略：
原理：卖出指标，前提：指标最好保持时效性，尽量避免滞后性，最大限度的保持利润，鉴于A股的高波动性以及惯性，在盈利的情况下，盈利回撤的比例；在不盈利的情况下，
止损点优先点 ；或者突发利空消息（比如违法违规）立即退出，不能存在侥幸心理
1、优先级：止损点大于止盈点（利润奔跑，克服心理上的恐惧），个人认为止盈点没有实际意义
2、低于买入价7-8%就坚决止损割肉，对于许多投资者来讲是很困难的，毕竟对许多人来说，承认自己犯了错误是比较困难的。投资最重要的就在于当你犯错误时迅速认识到错误
并将损失控制在最小，这是7%止损规则产生的原因。通过研究发现40%的大牛股在爆发之后最终往往回到最初的爆发点。同样的研究也发现，在关键点位下跌7-8%的股票未来有较
好表现的机会较小。投资者应注意不要只看见少数的大跌后股票大涨的例子。长期来看，持续的将损失控制在最小范围内投资将会获得较好收益。在短期内，涨幅过高，回调一定程度


参数回测：
1、网格搜索，对参数设立一个范围，确立变动的步长，利用枚举的方式，对一个参数组合进行回测，最后将不同的参数对应的回测结果进行比较，找出最优的参数
，缺点：1、计算量过于巨大，2、筛选标准、参数范围、步长具有主观性
2、MLE MAP BAYIES，分析最优参数

监控：
1、详细的运行日志记录，滚动的日志
2、进程之间的调度信息，subprogress

高速入口：
1、行情数据daily，tushare、ricequant
2、存量数据，包含退市
3、基本面数据

关于缺失值处理：
1、可用特征填补缺失值
2、特殊值补缺失值，如0、-1等
3、忽略
4、相似样本的均值填补缺失值
5、机器学习算法预测缺失值

框架结构：
事件驱动、异步框架、挂载app

价差的波动性，如果基本面没有改善，峰度会变化
典型的价格摆动指标将证券的经过平滑的价格与n期前价格作出比较
测算平均的趋势区间，基于这个估计指标的具体参数（自适应）
基于市值的角度，分析处于什么区间的市值容易爆发，市值的波动情况的，市值的联动动态，市值的极端变化，分布情况

周期如果太短 --- 噪音 ； 如果太长 --- 特征性能同化（在一个大的跨期内，所有指标都是有效的 ；同时避免样本量不足

回测的效果与实盘效果差距太大，原因分析：1、泛化能力（适应新的样本的能力，背后逻辑的合理性 ，每一个策略对应研究者对于策略的认识和理解
如同学习研究一样，不断的推翻、验证
2、策略都是周期性，回撤周期，长时间的回撤会产生动摇（本身策略是没问题的）
3、真实的回测要以环境限制，手续费、滑价，以及很重要的冲击成本（冲击成本很难衡量的）
4、策略评价效果，必须关注策略的信号频率


Toolz 可以用于编写分析大型数据流脚本，它支持通用的分析模式，如通过纯函数来对数据进行筛选（Selection），分组（Grouping），
化简（Reduction）以及连表（Joining）。这些函数通常可以模拟类似其他数据分析平台（如SQL和Panda）的类似操作行为

如果在一个方法前面加两个下划线为私有方法（类本身可以调用，子类不能调用），不能直接调用，必须通过构造另一个函数来调用私有方法;
单下划线—_为保护方法（本身、子类都可以调用）

Six is a Python 2 and 3 compatibility library. It provides utility functions
for smoothing over the differences between the Python versions with the goal
of writing Python code that is compatible on both Python versions. See the
documentation for more information on what is provided.

对数收益率 --- 数学推导近似等于复利收益率，
特征：1、向上波动于向下波动具有对称性而算术收益率不对称，2、平滑效应，通过对数处理，差异缩小更加平滑，3、区间的收益率即为连加（避免进行除法运算），
4、对数收益率可以转为算术收益率
策略参考：分析对数收益率的波动性以及与股价波动性的关系、相互作用关系
波动率分布：波动率是一个均值回复过程，如何处理分布以及预测，特征：回复、聚集、长记忆性
不同周期的交叉验证、均线流方向、趋势持续性、阶段高点变化趋势

5、 基于强势股票集合特征 ： 惯性强，运行节奏 但是中间回调，再次上车

RL(强化学习）:1、agent参与-行动改变环境，2、environment环境-反馈（react),3、研报提取
加权移动平均线成交量乘以成交量增量计算各期的成交率比率

apply_along_axis(func, axis, arr, *args, **kwargs) axis = 0 / 1, func
对array处理np.raval array_like to 1_D raval(x, order='C' / 'F')
return view C row index; F column index  ‘K’ means to read the elements in the order they occur in memory, except for reversing the
data when strides are negative return np.flatten similar to raval return a copy np.squeeze Remove single - dimensional entries from
the shape of an array

sigmoid序列处理，大的更大，小的更小，分辨不清晰的极值: 1.0 / (1+ np.exp(-arr))
softmax: np.exp(X) / np.sum(np.exp(X), axis=1)

SVD 奇异值分解 将数据映射到低维上 np.linalg.svd  生产U Simga(只有对角元素，其他为0 ，看出类别） VT

朴素贝叶斯算法(监督学习，类别)，算法： p(c / w) = p(w / c) * p(c) / p(w), 而p(w / c) = p(w1 / c) * p(w2 / c)...p(wn / c)
假设各个w都是独立的特征
分类的逻辑： 条件概率最大的所属类别
实现方式：1、贝努利模型，2、多项式模型实现

分类树
熵：li = -log2(pxi)  H = - sum(pxi * li) 基于无序程度来划分

class Adaboost:
    """
        基于不同的弱分类算法的错误度，计算alpha作为弱分类的权重，正确的分类的样本权重变小，错误分类的样本权重变大，迭代逻辑：指数损失
        a = 1/2 * log((1-e)/e)
    """
    pass

cauchy(柯西分布):
柯西分布的平均值、方差或者矩都没有定义，它的众数与中值有定义都等于
x0。如果U与V是期望值为0、方差为
1的两个独立正态分布随机变量的话，那么比值U / V为柯西分布。标准柯西分布（loc = 0, r = 1) 为学生t - 分布自由度为1的特殊情况。
中心极限定理中的有限变量假设： 如果X1, …, Xn是分别符合柯西分布的相互独立同分布随机变量，（X1 + … + Xn） / n
有同样的柯西分布洛仑兹线性分布更适合于那种比较扁、宽的曲线高斯线性分布则适合较高、较窄的曲线很多情况下，采用的是两者各占一定比例的
from scipy.stats import cauchy
rvs(loc=0, scale=1, size=1, random_state=None)
pdf / logpdf(x, loc=0, scale=1)
cdf / logcdf(x, loc=0, scale=1)
sf(x, loc=0, scale=1)
Survival function(also defined as 1 - cdf) gamma:gamma与possion在函数上高度一致 ，连续性与离散型
from scipy.stats import gamma
阶乘函数, 参数同上beta:from scipy.stats import beta可以有gamma函数

# apply --- function axis = 0 or axis = 1
# applymap ---- function value of dataframe
# map ---- function on series

#python 随机数生产需要指定种子seed ， 相同的种子产生相同的随机数
import random
random.seed(-------)

response = requests.get(new_url, **self.requests_kwargs)

content_length = 0

for chunk in response.iter_content(self.CONTENT_CHUNK_SIZE,
                                   decode_unicode=True):
    if content_length > self.MAX_DOCUMENT_SIZE:
        raise Exception('Document size too big.')
    if chunk:
        content_length += len(chunk)
        yield chunk


mysql tool：
    # select host,user,authentication_string from mysql.user;
    # set password for user@localhost = newpassword;
    # flush privileges;
    # create user c_test@locahost identified by password;
    # drop user c_test@localhost;
    # grant select,update | all privileges on orm.* to guest@localhost;
    # 当忘记密码:  mysqld --skip-grant-tables,use mysql,set password
    # primary key
    # constraint
    # foreign key references
    # alter add constraint
    # alter  drop foreign key

mysql --- between and (include edge)

# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

OMS --- 订单处理系统
slippage --- 针对回测用的，
ArkQuant trading --- 将订单拆分小的订单 --- transaction
open price （最高价-最低价）波动率 ，simlation 当天的概率分布，在收盘的时候将没有成交的订单通过集合竞价成交
价格范围 0.9preclose  1.1preclose
order_plan_amount 分为 per_order_amount ; order_num;
per_order_amount : np.ceiling(20000/0.9preclose) ; order_num : np.floor(order_plan_amount / per_order_amount);
left_order_amount : order_plan_amount - per_order_amount * order_num
order_price :
    1. random(0.9preclose , 1.1preclose)  不可取 成本太高
    2. 以open价格为中心，波动率 --- 为历史的波动率 ，3Q法则 ， 采用正态分布
    3. 以open - preclose 偏离价格为中心 ，偏离值的波动率 ， 采用正态分布
    4. 计算股票短期振幅的波动率 ，
概率统计 --- gamma函数 gamma(x) = (x -1)!
            beta函数  beta(x,y) = gamma(x) * gamma(y) / gamma(x+y)
            --- 微积分以及递推可以推导的
            gamma分布 --- gamma(a,b) 期望a/b ,方差a/ b**2 , (b**a)*(x**(a-1) * (e** -b*x) / gamma(a) 当a <= 1为递减函数;a > 1为单峰函数；
            扩展  --- 指数分布 卡方分布
            beta分布 --- beta(a,b) 期望a/(a+b) 方差 ab/[(a+b) ** 2 * (a +b +1)]
            possion --- (r)  (e** -r) r** k / k! , 离散 --- 当n很大p很小，r = np ---- 一段时间内发生的平均发生次数作为r参数 ,泊松分布的期望和方差均为r
            指数分布 --- r* e** (-r *x) ,期望 1/r ,方差 1/r**2 --- 指数分布（也称为负指数分布）是描述泊松过程中的事件之间的时间的概率分布，
            其中λ > 0是分布的一个参数，常被称为率参数（rate parameter）。即每单位时间内发生某事件的次数。
            即事件以恒定平均速率连续且独立地发生的过程。 这是伽马分布的一个特殊情况  gamma(1,b),重要性质 --- 记忆性 ；许多电子产品的寿命分布一般服从指数分布。
            它在可靠性研究中是最常用的一种分布形式；记忆性表明 --- 已经过去的时间没有意义就是干扰项 ，证明无记忆通过条件概率微积分，不同事件之间的
            间隔也就是完成单次事件的事件，这个服从指数分布
            pareto ---- 概率分布(x/ x.min()) ** -k，求导即为分布概率 --- 递减函数 xmin 为边界
            cachy --- cachy(x0,r)类似于 1/ (1 **2 +x**2) ,系数 r/pai ，无期望， 无方差 --- 1. 正态分布比值为柯西分布；2.到时性质 参数= /(r**2 + x**2)
            3.均匀分布x , y = tan(x) ,y服从标准柯西分布 ，tan(x) -- 导数 1/(x**2 +1)
            卡方分布 --- 自由度 v ,期望v 方差 2v --- 当自由度 n 越大, 密度曲线越趋于对称, n越小, 曲线越不对称. 当 n = 1, 2 时曲线是单调下降趋于 0.
            当 n ≥ 3时曲线有单峰, 从 0 开始先单调上升, 在一定位置达到峰值, 然后单下降趋向于 0 ，独立的卡方分布具有可加性
            norm --- 偏度 --- 三阶中心矩 （正态分布 0 ） ； 峰度 --- 四阶中心矩（正态分布 为3）
            t_test T-分布  x / sqrt(y /n) --- x 服从标准正态分布，y服从自由为n的卡方分布，*** (n-1)*样本方差与总体方差之比服从自由度为 n-1 的卡方分布 ，
            证明推论 x1,x2,x3,------ xn独立分布 N(a,o2) , sqrt(n)(e(x) - a) / s(样本标准差） 服从t(n-1) ，计算统计量
           （t-distribution）用于根据小样本来估计呈正态分布且方差未知的总体的均值, 以样本均值来估计总体的均值，不同的样本的均值的均值等于总体样本的均值
            （样本方差 --- * n / (n-1)，* 估计方差的时候要减去1/n个方差，所以方差的无偏估计自由度为n-1）
            f_test F-分布 --- 两个卡方分布比 x/m / y/n，其中x,y是自由度为m,n的卡方分布 ， t分布的平方为F(1,n) ; 特征1.倒数表明自由度调换, 特征2.
            f(n,m)(1-a) = 1/f(m,n)(a)

        amount --- 订单金额(正负)
        filled --- 订单成交比例
        委托和订单
        除了“红马甲（出市代表指证券交易所内的证券交易员，又叫出市员）”，大多数交易者并不能直接将交易指令提交到交易所，
        而是需要经过中间环节代为处理。即交易者提出“委托申请”，中间商将委托提交到交易所形成"订单(Order)"。

        为了区分订单，交易所会为每一笔订单分配一个合同号(Order ID)。

        在交易所撮合的过程中，订单的状态不断变化，交易者根据订单状态了解自己的委托是否成交。

        订单状态 OrderStatus
        由于要经过交易者、中间环节、交易所三级的处理，需要有很多状态区分订单处于哪个阶段。
        目前国内的交易系统中，订单状态主要区分为一下几种：

        待报: 也称为“未报”，交易者提交委托后的初始状态

        正报: 中间环节已经收到委托，但还未报到交易所，或者虽然报到交易所但是没有收到交易所的委托确认

        已报: 交易所确认委托

        部成: 经过撮合，订单部分成交

        已成: 经过撮合，订单全部成交

        废单: 交易所认定订单无效

        待撤: 交易者提交撤单指令

        正撤: 中间环节收到撤单指令，但还未报到交易所，或者未收到交易所的确认

        部撤: 订单部分成交，部分撤销

        已撤: 订单全部撤销

        撤废: 无法撤单，通常是因为已经撮合成交

        有些系统中，待撤会根据当前成交状态，进一步细分为 已报待撤和部成待撤。

        按照委托价格的指定方式，可以分为限价委托(LimitOrder)和市价委托(MarketOrder)。
        其中，市价委托又分为几种方式。

        限价委托(LMT, limit), 以指定的价格报单

        对手方最优价格委托（BOC, best of counterparty)

        如果有对手方报价，则以买一/卖一价格成交，未能完全成交的按成交价转为限价单；
        如果没有对手方报价，自动撤单。

        本方最优价格委托(BOP, best of party)

        如果有买一/买一价格，则跟随报价；如果没有自动撤单。

        即时成交剩余撤销委托(ITC, immediately then cancel)

        自动追涨买入/杀跌卖出，直到涨停/跌停。还有剩余的自动撤单。

        最优五档即时成交剩余撤销委托(B5TC, best 5 then cancel)

        是对ITC的改进，只与对手方的前五档价格成交。

        全额成交或撤销委托(FOK, fill or kill)

        类似ITC，但是不用指定数量，自动全仓操作

        最优五档剩余转限价委托(B5TL,best 5 then limit)

        类似B5TC, 区别是剩余的不撤单，而是转为限价

        目前，深交所支持 LMT, BOC, BOP, ITC, B5TC, FOK；上交所支持LMT、B5TC、B5TL。

        Stop Orders (STP, stop-loss order) 市价止损/止盈单
        Stop Limit Orders (STPLMT) 限价止损/止盈单
        Market If Touched Orders (MIT)
        Limit If Touched Orders (LIT)
        One Cancels the Other (OCO)


# Mapping MutableMapping
# hash __hash__ __eq__ or __cmp__
# Functions are applied from right to left so that compose(f, g, h)(x, y) is the same as f(g(h(x, y)))
# Identity function. Return x
# NamedTemporaryFile has a visble name in file system can be retrieved from the name attribute , delete ---
  True means delete as file closed

default_extension : bool, optional
    Should the default zipline extension be loaded. This is found at
    ``$ZIPLINE_ROOT/extension.py``
extensions : iterable[str], optional
    The names of any other extensions to load. Each element may either be
    a dotted module path like ``a.b.c`` or a path to a python file ending
    in ``.py`` like ``a/b/c.py``.
strict_extensions : bool, optional
    Should the run fail if any extensions fail to load. If this is false,
    a warning will be raised instead.
environ : mapping[str -> str], optional
    The os environment to use. Many extensions use this to get parameters.
    This defaults to ``os.environ``.

    extend :
        __get__(self, instance, owner):调用一个属性时,触发
        __set__(self, instance, value):为一个属性赋值时,触发
        __delete__(self, instance):采用del删除属性时,触发

创业板:
1、申报数量100股及其整数倍；

2、限价申报不超过30万股；

3、市价申报不超过15万股；

4、盘后定价申报的单笔申报数量不得超过100万股。


规则变化之后，自上市首日起就可以作为融资融券标的。
“N”---上市首日；

“C”上市次日至第五日；

“U”发行人未盈利，若发行人首次实现盈利，该特别标识取消；

“W”发行人具有表决权差异安排，若发行人不再具有表决权差异安排，该特别标识取消；

“V”发行人具有协议控制架构或类似特殊安排，若上市后不再具有相关安排，该特别标识取消；


优化:
1. price_loader ---- all equity assets consumer too much time
2. register --- 类注册为抽象类

1 存在价格笼子
2 无跌停限制但是存在竞价机制（10%基准价格），以及临时停盘制度
有存在竞价限制，科创板2% ，或者可转债10%
第十八条 债券现券竞价交易不实行价格涨跌幅限制。
　　             第十九条 债券上市首日开盘集合竞价的有效竞价范围为发行价的上下 30%，连续竞价、收盘集合竞价的有效竞价范围为最近成交价的上下 10%；
非上市首日开盘集合竞价的有效竞价范围为前收盘价的上下 10%，连续竞价、收盘集合竞价的有效竞价范围为最近成交价的上下 10%。
 一、可转换公司债券竞价交易出现下列情形的，本所可以对其实施盘中临时停牌措施：
　　（一）盘中成交价较前收盘价首次上涨或下跌达到或超过20%的；
　　（二）盘中成交价较前收盘价首次上涨或下跌达到或超过30%的。

按照价格在10% 至 -10%范围内基于特定的统计分布模拟价格 --- 方向为开盘的涨跌幅 ，不适用于科创板（竞价机制要求）---买入价格不能超过基准价格（卖一的
102%，卖出价格不得低于买入价格98%，申报最小200股，递增可以以1股为单位 ；设立市价委托必须设立最高价以及最低价 ；
A股主板，中小板首日涨幅最大为44%而后10%波动，而科创板前5个交易日不设立涨跌停而后20%波动但是30%，60%临时停盘10分钟，如果超过2.57(复盘)；
科创板盘后固定价格交易 --- 以后15:00收盘价格进行交易 --- 15:00 -- 15:30(按照时间优先原则，逐步撮合成交）

principle --- 只要发出卖出信号的最大限度的卖出，如果没有完全卖出直接转入下一个交易日继续卖出
订单 --- priceOrder TickerOrder Intime
engine --- xtp_vnpy or simulate(slippage_factor = self.slippage.calculate_slippage_factor)
dual -- True 双方向
      -- False 单方向（提交订单）
eager --- True 最后接近收盘时候集中将为成交的订单成交撮合成交保持最大持仓
      --- False 将为成交的订单追加之前由于restrict_rule里面的为成交订单里面
具体逻辑：
    当产生执行卖出订单时一旦成交接着执行买入算法，要求卖出订单的应该是买入Per买入标的的times，
    保证一次卖出成交金额可以覆盖买入标的
优势：提前基于一定的算法将订单根据时间或者价格提前设定好，在一定程度避免了被监测的程度。
成交的订单放入队列里面，不断的get
针对于put orders 生成的买入ticker_orders （逻辑 --- 滞后的订单是优先提交，主要由于订单生成到提交存在一定延迟)
订单优先级 --- Intime (first) > TickerOrder > priceOrder
基于asset计算订单成交比例
获取当天实时的ticer实点的数据，并且增加一些滑加，+ /-0.01

按照价格在10% 至 -10%范围内基于特定的统计分布模拟价格 --- 方向为开盘的涨跌幅 ，不适用于科创板（竞价机制要求）---买入价格不能超过基准价格（卖一的
102%，卖出价格不得低于买入价格98%，申报最小200股，递增可以以1股为单位 ；设立市价委托必须设立最高价以及最低价 ；
A股主板，中小板首日涨幅最大为44%而后10%波动，而科创板前5个交易日不设立涨跌停而后20%波动但是30%，60%临时停盘10分钟，如果超过2.57(复盘)；
科创板盘后固定价格交易 --- 以后15:00收盘价格进行交易 --- 15:00 -- 15:30(按照时间优先原则，逐步撮合成交）

撮合成交
如果open_pct 达到10% --- 是否买入
分为不同的模块 创业板，科创板，ETF
包含 --- sell orders buy orders 同时存在，但是buy_orders --- 基于sell orders 和 ledger
通过限制买入capital的pct实现分布买入
但是卖出订单 --- 通过追加未成交订单来实现
如何连接卖出与买入模块

由capital --- calculate orders 应该属于在统一模块 ，最后将订单 --- 引擎生成交易 --- 执行计划式的，非手动操作类型的
剔除ReachCancel --- 10%
剔除SwatCancel --- 黑天鹅

防止策略冲突 当pipeline的结果与ump的结果出现重叠 --- 说明存在问题，正常情况退出策略与买入策略应该不存交集

1. engine共用一个ump ---- 解决了不同策略产生的相同标的可以同一时间退出
2. engine --- 不同的pipeline对应不同的ump,产生1中的问题，相同的标的不会在同一时间退出是否合理（冲突）

退出策略 --- 针对标的，与标的是如何产生的不存在直接关系;只能根据资产类别的有关 --- 1
如果产生冲突 --- 当天卖出标的与买入标的产生重叠 说明策略是有问题的ump --- pipelines 对立的
symbol ,etf 的退出策略可以相同，但是bond不行属于T+0
return ---- name : [position , [pipeline_output]]

两个部分 pipelines - ledger
        positions -

建仓逻辑 --- 逐步建仓 1/2 原则 --- 1 优先发生信号先建仓 ，后发信号仓位变为剩下的1/2（为了提高资金利用效率）
                                2 如果没新的信号 --- 在已经持仓的基础加仓（不管资金是否足够或者设定一个底层资金池）
---- 变相限定了单次单个标的最大持仓为1/2
position + pipe - ledger ---  (当ledger为空 --- position也为空)

关于ump --- 只要当天不是一直在跌停价格，以全部出货为原则，涉及一个滑价问题（position的成交额 与前一周的成交额占比
评估滑价），如果当天没有买入，可以适当放宽（开盘的时候卖出大部分，剩下的等等） ；
如果存在买入标的的行为则直接按照全部出货原则以open价格最大比例卖出 ，一般来讲集合竞价的代表主力卖入意愿强度）
---- 侧面解决了卖出转为买入的断层问题 transfer1

holding , asset ,dts
基于触发器构建 通道 基于策略 卖出 --- 买入
principle --- 只要发出卖出信号的最大限度的卖出，如果没有完全卖出直接转入下一个交易日继续卖出
订单 --- priceOrder TickerOrder Intime
engine --- xtp_vnpy or simulate(slippage_factor = self.slippage.calculate_slippage_factor)
dual -- True 双方向
      -- False 单方向（提交订单）
eager --- True 最后接近收盘时候集中将为成交的订单成交撮合成交保持最大持仓
      --- False 将为成交的订单追加之前由于restrict_rule里面的为成交订单里面
具体逻辑：
    当产生执行卖出订单时一旦成交接着执行买入算法，要求卖出订单的应该是买入Per买入标的的times，
    保证一次卖出成交金额可以覆盖买入标的
优势：提前基于一定的算法将订单根据时间或者价格提前设定好，在一定程度避免了被监测的程度。
成交的订单放入队列里面，不断的get
针对于put orders 生成的买入ticker_orders （逻辑 --- 滞后的订单是优先提交，主要由于订单生成到提交存在一定延迟)
订单优先级 --- Intime (first) > TickerOrder > priceOrder
基于asset计算订单成交比例
获取当天实时的ticer实点的数据，并且增加一些滑加，+ /-0.01
卖出标的 --- 对应买入标的 ，闲于的资金


'''

