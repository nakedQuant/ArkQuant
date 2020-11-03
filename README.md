
    		ArkQuant is a event_diver algorithm which concentrates on China Stock. It adopts the form of reality trading to simulate 
	backtest engine and bridge the tunnel between backtest and live trading.

    		ArkQuant是基于事件驱动引擎开发的针对国内市场A股、ETF的仿真回测交易平台。它借鉴了Zipline的设计理念并吸收合并了
	国内主流的VNPY，BackTrader等框架等优势,通过将市场交易机制嵌入到回测引擎保证了分析结果的有效性同时降低了与真实交易环境的偏差。
    	ArkQuant参照自动化交易系统以算法为参数入口（包括：策略算法，风险控制算法（仓位控制，资金分配算法，平仓算法），
	交易控制算法（比如滑价算法，交易成本算法，订单类型，订单拆分算法，限制条件算法)，订单撮合成交算法，评价算法(metrics_tracker) ,
	优化算法), 以市场交易机制为驱动引擎，实现了回测、实盘、自动化交易的无缝衔接，其中将不同模块做到最大程度的结耦而且抽象成算法进行自由组合。
	所以ArKQuant具备高度扩展行性、可塑性，同时能根据市场机制的变化进行实时调整保证分析结果的准备性。
	
    		国内主流的量化回测平台存在以下问题:  1、侧重于提供全面的数据接口与友好的界面交互;  2、回测体系精度很差与真实情况相差太大; 
	3、在滑价，交易成本等参数设定具有随意性与实际情况脱节;  4、由算法发出指令 --- 订单 --- 交易环节上完全封闭默认直接成交忽略真实交易的情况; 
	5、回测平台没有将市场的交易制度嵌入到回测系统由此导致回测的结果都是不可靠的，一旦市场制度调整原有的框架失效;  6、无法有效嫁接回测系统与实盘接口；
	7、对自动化(半自动化)交易系统的框架体系鲜有涉及;  8、对于量化策略缺乏深入的拆解以及抽象。

组件：

    Gateway --- (数据接口 module)
        1. assets (标的对象）
            1.1	Assets ：equity , etf , convertible，attribute : sid, exchange, first_traded , last_traded  and is_alive , can_be_traded (method) 
            1.2 _finder(asset_finder module) ： finder assets according to specific condition (ex: by district , by sector , by exchange , by broker and so on 
        2.database (数据库） 
            2.1	Migration : to transfer database via alembic module
            2.2 	db_schema , db_writer via orm 
        3.driver (api）
            3.1  bar_reader : daily and minute bar  ---- minute_bar retrieve from bcolz
            4.spider(爬虫）
            4.1 基础数据 行情，股权结构，分红配股，除权 市场数据 大宗交易 解禁数据 股权变动
            4.2 爬取基础数据，市场数据 基于多线程并发（或者asynico）
	    
    Pipeline --- (交易算法 module)
        策略目的从全市场获取特定标的或者标的集合，ArkQuant采用拓扑结构的思想对策略进行抽象将其拆分为N个子可计算节点，并按照特定方式进行组合。
    具体的实现方式：子节点包含（计算逻辑，依赖，数据类型）其中依赖的含义节点输入是另一个节点的输出，数据类型为节点的domain，不同的节点通过依赖的方式方式组成整个策略。底层的实现方式主要通过networkx的graph 的in_degree ，out_degree概念进行实施。
        1.Domain 
            1.1 domain (field , window) , __or__ ,用于表示节点的计算数据信息
        2.term --- 节点logic(算法逻辑)
            2.1 dependence (依赖)，params(参数) 
        3.Graph 
            3.1 基于networkx模块 , term按照dependence组合成拓扑结构的有向图
        4.Pipeline
            4.1 graph每个节点代表一个term , 不同节点按照特定的方式组合构建够管道算法
        5.Loader 
            5.1 loader data for pipeline (通过将不同的节点的domain通过__or__内置方式得出pipeline的domain,
            warpper by( gateway的api)去获取整个pipeline所需数据
        6.Engine 
            6.1 算法引擎将整个pipeline流程组合算法引擎
	    
    Finance --- (交易流程 module）
        1.Order ---- 订单对象
            1.1  price_order (asset , price , amount )
            1.2  ticker_order(asset , ticker, amount)
            1.3  order(asset, price, ticker, amount)
        2.Slippage --- 滑价对象
            2.1  滑价算法 e.g :通过度量买入成交量与标的前期的成交量关系动态设置滑价系数
        3.Commission ---- 成本对象
            3.1  真实模拟交易成本(印花税，过户费，交易佣金）以及最低交易成本
        4.Trading_controls ---- 控制对象
            4.1  设立交易过程control, e.g. 最大持仓量，最大持仓比例 ，做多(LongOnly) 
        5.Restrictions ---- 限制对象
            5.1  对标的进行的限制,e.g. 特定标的集限制, 上市不满足一个月，剔除停盘
        6.Execution ---- 执行对象
            6.1  设立订单类型, e.g. 市价单, 限价单，最高价单，最低价单
        7.Transaction --- 交易对象
            7.1  transaction( order object and commission cost) 
        8.Position --- 仓位
            8.1  inner_position 对象
            8.2  position( asset ,amount, last_sync_price, last_sync_date, closed,returns)
        9.Position_tracker ---- 仓位追踪
            9.1  handle_on_splits
            9.2  update transactions
            9.3  synchronize 
        10.Ledger ---- 账簿对象
            10.1  manipulate position_tracker 
            10.2  update portfolio 
        11.Portfolio ---- 组合对象
            11.1  record and analyse positions 
            11.2  portfolio_value , portfolio_cash, portfolio_position, position_weights 
        12.Account ---- 账户对象
            12.1  to cast portfolio on account
	    
    Risk --- (风险控制 module)
        1.	 Alert --- 警告对象
            1.1  针对单个持仓的caution , e.g. 持仓最大跌幅,持仓最大回撤
        2.  Allocation --- 分配对象
            2.1  资金分配算法 e.g.  equal or delta(类似于海龟交易法则，基于波动性分配仓位)
        3.  Fuse  --- 保险丝对象
            3.1  当组合净值低于一定阈值 ，是否强行平仓 ，作为一种保护措施
	    
    Pb --- (撮合交易 module)
        1.	Underneath ---- 隐藏算法
            将amount基于算法按照时间或者价格分割N个tiny amount
            1.1  uncover by ticker 
            1.2	 uncover by price 
        2.	Division ---- 拆分对象
            Used as wrapper for underneath
            2.1  买入行为拆分
            2.2  卖出行为拆分
        3.	Blotter ---- 模拟交易
            3.1	 validate method :transfer ticker_order or price_order into order
            3.2	 check_trigger method: order via slippage and execution to trigger_order
            3.3  create_transaction:tranform trigger_order into transaction via commission
        4.	Generator ---- 交易生成器
            4.1 	combine division and blotter together to create transactions
        5. 	Broker --- 经纪对象
            5.1  由算法、交易生产器、资金分配算法组成的broker 
	    
    Metrics --- (评价分析 module)
        1. metrics 分析指标 
            1.1  daily  每天的计算指标或者特定账户信息
            1.2.  analyzer 基于收益率时间序列计算的统计指标
        2. Metric_tracker 
            2.1  基于时间节点分析指标, e.g. start_of_simulation , start_of_session,end_of_session, end_of_simulation 
        3. Tear_sheet / pdf  回测指标输出特定的格式 
	
    Trading --- (交易执行 module)
        1.sim_params 参数模块 e.g. sessions , capital , loan , benchmark 
        2.Clock 时间触发器 
            2.1  每天按照特定时间节点去执行对应的逻辑, e.g:  before._trading_start, Session_start, session_end 
        3.trading_simulation 回测执行入口(迭代器)
	
    Opt --- (参数优化 module)
        1.Grid 网络搜索
        2.Spread 参数泛化(有效的策略不能针对特定的参数值，但参数变化时分析对应的净值曲线变化评价参数)
        3.借鉴机器学习理论对pipeline进行优化(主要针对计算节点组合方式以及节点的重要性分配)
        4.Similarity 分析不同的策略的关联性
	
    Reality ---- 实盘交易
        1  xtp(中泰证券快速交易系统) --- 基于cython调用底层接口
        1  scanner(掘金的扫单系统）---- 把策略生成的委托信号以 SQLite DB文件作为中转，不需要接口，直接接入掘金做仿真和实盘交易。
特征：

    1.通过模拟真实交易的流程进行回测，最大程度的降低回测曲线与实际交易曲线的误差，更佳合理有效的评估策略的有效性，实现了回测引擎到实盘引擎无缝对接（交易流程保持一致）
    2.基于算法对全市场标的进行过滤筛选，撮合交易，订单处理，实现了自动化交易的基本框架，输入参数为市场交易机制的算法
    1.所有模块之间做到最大程度结耦合，不同的模块存在一个抽象类或者基类，重新实现抽象方法作为算法的变种。
    3.框架的所有的数据都是前赋权数据，因为后复权数据将未来的除权分红信息进行贴现不可避免的造成后验误差，ArkQuant进行回测过程中，对每个时间进行前复权运算（更加复合真实的交易情况以及理念）
    4.通过trading_control, restriction等模块对交易过程进行控制，e.g. 系统发一个特定信号买入信号，为了保证策略在最大可能范围内适应历史的情况，买入数据不能超过过去几个交易日均成交量的特定阈值比例(0.1) , 卖出同理
    5.通过设立risk_model 风控模块控制持仓的账户风险， e.g. 仓位最大回撤，仓位的止赢止损, 投资组合的净值限制(是否平仓)
    6.为了回测结果更加符合实际情况引入虚拟撮合成交的模块，包含订单拆分模块，撮合算法
    7.针对分析模块借鉴了zipline的metric_tracker按照交易时点(before_start, session_start, session_end)进行分析，与simulation(回测运行逻辑分离的)
    8.回测引擎通过yield方式返回daily的perf(分析结果)，内部实现机制封装在trade模块里面避免了对外暴露。
    9.Pipeline ---- ArkQuant策略机制实现的抽象框架，策略定义：通过一系列的算法对市场进行过滤筛选最终对出特定标的或者标的集合，也就是说策略由不同的逻辑算法通过算法之间的相互作用最终生产的集合。Term(计算节点)作为逻辑算法，Term与Term之间链接方式就是Pipeline. Ark quant 采用了拓扑结构的有向图的实现方式，Term节点之间通过依赖方式链接。
    10.ArkQuant将XTP(中泰证券的快速交易系统)的底层python接口纳入了系统
    11.Extension : smtp , parallel, schedule等功能





