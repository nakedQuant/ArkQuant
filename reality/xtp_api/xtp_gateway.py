# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from .vnxtpquote import QuoteApi

MARKET_XTP2VT: Dict[int, Exchange] = {
    1: Exchange.SZSE,
    2: Exchange.SSE
}
MARKET_VT2XTP: Dict[Exchange, int] = {v: k for k, v in MARKET_XTP2VT.items()}

EXCHANGE_XTP2VT: Dict[int, Exchange] = {
    1: Exchange.SSE,
    2: Exchange.SZSE,
}
EXCHANGE_VT2XTP: Dict[Exchange, int] = {v: k for k, v in EXCHANGE_XTP2VT.items()}

DIRECTION_STOCK_XTP2VT: Dict[int, Any] = {
    1: (Direction.LONG, Offset.NONE),
    2: (Direction.SHORT, Offset.NONE),
    21: (Direction.LONG, Offset.OPEN),
    22: (Direction.SHORT, Offset.OPEN),
    24: (Direction.LONG, Offset.CLOSE),
    23: (Direction.SHORT, Offset.CLOSE)
}
DIRECTION_STOCK_VT2XTP: Dict[Any, int] = {v: k for k, v in DIRECTION_STOCK_XTP2VT.items()}

DIRECTION_OPTION_XTP2VT: Dict[int, Direction] = {
    1: Direction.LONG,
    2: Direction.SHORT
}
DIRECTION_OPTION_VT2XTP: Dict[Direction, int] = {v: k for k, v in DIRECTION_OPTION_XTP2VT.items()}

POSITION_DIRECTION_XTP2VT = {
    0: Direction.NET,
    1: Direction.LONG,
    2: Direction.SHORT,
    3: Direction.SHORT
}

ORDERTYPE_XTP2VT: Dict[int, OrderType] = {
    1: OrderType.LIMIT,
    2: OrderType.MARKET
}
ORDERTYPE_VT2XTP: Dict[OrderType, int] = {v: k for k, v in ORDERTYPE_XTP2VT.items()}

PROTOCOL_VT2XTP: Dict[str, int] = {
    "TCP": 1,
    "UDP": 2
}

STATUS_XTP2VT: Dict[int, Status] = {
    0: Status.SUBMITTING,
    1: Status.ALLTRADED,
    2: Status.PARTTRADED,
    3: Status.CANCELLED,
    4: Status.NOTTRADED,
    5: Status.CANCELLED,
    6: Status.REJECTED,
    7: Status.SUBMITTING
}

PRODUCT_XTP2VT: Dict[int, Product] = {
    0: Product.EQUITY,
    1: Product.INDEX,
    2: Product.FUND,
    3: Product.BOND,
    4: Product.OPTION,
    5: Product.EQUITY,
    6: Product.OPTION
}

OFFSET_VT2XTP: Dict[Offset, int] = {
    Offset.NONE: 0,
    Offset.OPEN: 1,
    Offset.CLOSE: 2,
    Offset.CLOSETODAY: 4,
    Offset.CLOSEYESTERDAY: 5
}
OFFSET_XTP2VT: Dict[int, Offset] = {v: k for k, v in OFFSET_VT2XTP.items()}

BUSINESS_VT2XTP: Dict[Any, int] = {
    "CASH": 0,
    Offset.NONE: 0,
    "MARGIN": 4,
    Offset.OPEN: 4,
    Offset.CLOSE: 4,
    "OPTION": 10,
}

symbol_name_map: Dict[str, str] = {}
symbol_exchange_map: Dict[str, Exchange] = {}


class MdApi(QuoteApi):

    def __init__(self, gateway: BaseGateway):
        """"""
        super().__init__()

        self.gateway: BaseGateway = gateway
        self.gateway_name: str = gateway.gateway_name

        self.userid: str = ""
        self.password: str = ""
        self.client_id: int = 0
        self.server_ip: str = ""
        self.server_port: int = 0
        self.protocol: int = 0
        self.session_id: int = 0

        self.connect_status: bool = False
        self.login_status: bool = False

    def onDisconnected(self, reason: int) -> None:
        """"""
        self.connect_status = False
        self.login_status = False
        self.gateway.write_log(f"行情服务器连接断开, 原因{reason}")

        n = self.login()

        if n:
            self.session_id = n
            self.connect_status = True
            self.login_status = True

            self.gateway.write_log(f"交易服务器登录成功，会话编号：{self.session_id}")
        else:
            self.gateway.write_log("行情服务器登录失败")

    def onError(self, error: dict) -> None:
        """"""
        self.gateway.write_error("行情接口报错", error)

    def onSubMarketData(self, data: dict, error: dict, last: bool) -> None:
        """"""
        if not error or not error["error_id"]:
            return

        self.gateway.write_error("行情订阅失败", error)

    def onUnSubMarketData(self, data: dict, error: dict, last: bool) -> None:
        """"""
        pass

    def onDepthMarketData(self, data: dict) -> None:
        """"""
        timestamp = str(data["data_time"])
        dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S%f")

        tick = TickData(
            symbol=data["ticker"],
            exchange=EXCHANGE_XTP2VT[data["exchange_id"]],
            datetime=dt,
            volume=data["qty"],
            last_price=data["last_price"],
            limit_up=data["upper_limit_price"],
            limit_down=data["lower_limit_price"],
            open_price=data["open_price"],
            high_price=data["high_price"],
            low_price=data["low_price"],
            pre_close=data["pre_close_price"],
            gateway_name=self.gateway_name
        )

        tick.bid_price_1, tick.bid_price_2, tick.bid_price_3, tick.bid_price_4, tick.bid_price_5 = data["bid"][0:5]
        tick.ask_price_1, tick.ask_price_2, tick.ask_price_3, tick.ask_price_4, tick.ask_price_5 = data["ask"][0:5]
        tick.bid_volume_1, tick.bid_volume_2, tick.bid_volume_3, tick.bid_volume_4, tick.bid_volume_5 = data["bid_qty"][
                                                                                                        0:5]
        tick.ask_volume_1, tick.ask_volume_2, tick.ask_volume_3, tick.ask_volume_4, tick.ask_volume_5 = data["ask_qty"][
                                                                                                        0:5]

        tick.name = symbol_name_map.get(tick.vt_symbol, tick.symbol)
        self.gateway.on_tick(tick)

    def onSubOrderBook(self, data: dict, error: dict, last: bool) -> None:
        """"""
        pass

    def onUnSubOrderBook(self, data: dict, error: dict, last: bool) -> None:
        """"""
        pass

    def onOrderBook(self, data: dict) -> None:
        """"""
        pass

    def onSubTickByTick(self, data: dict, error: dict, last: bool) -> None:
        """"""
        pass

    def onUnSubTickByTick(self, data: dict, error: dict, last: bool) -> None:
        """"""
        pass

    def onTickByTick(self, data: dict) -> None:
        """"""
        pass

    def onSubscribeAllMarketData(self, data: dict, error: dict) -> None:
        """"""
        pass

    def onUnSubscribeAllMarketData(self, data: dict, error: dict) -> None:
        """"""
        pass

    def onSubscribeAllOrderBook(self, data: dict, error: dict) -> None:
        """"""
        pass

    def onUnSubscribeAllOrderBook(self, data: dict, error: dict) -> None:
        """"""
        pass

    def onSubscribeAllTickByTick(self, data: dict, error: dict) -> None:
        """"""
        pass

    def onUnSubscribeAllTickByTick(self, data: dict, error: dict) -> None:
        """"""
        pass

    def onQueryAllTickers(self, data: dict, error: dict, last: bool) -> None:
        """"""
        contract = ContractData(
            symbol=data["ticker"],
            exchange=EXCHANGE_XTP2VT[data["exchange_id"]],
            name=data["ticker_name"],
            product=PRODUCT_XTP2VT[data["ticker_type"]],
            size=1,
            pricetick=data["price_tick"],
            min_volume=data["buy_qty_unit"],
            gateway_name=self.gateway_name
        )
        self.gateway.on_contract(contract)

        symbol_name_map[contract.vt_symbol] = contract.name

        if contract.product != Product.INDEX:
            symbol_exchange_map[contract.symbol] = contract.exchange

        if last:
            self.gateway.write_log(f"{contract.exchange.value}合约信息查询成功")

    def onQueryTickersPriceInfo(self, data: dict, error: dict, last: bool) -> None:
        """"""
        pass

    def onSubscribeAllOptionMarketData(self, data: dict, error: dict) -> None:
        """"""
        pass

    def onUnSubscribeAllOptionMarketData(self, data: dict, error: dict) -> None:
        """"""
        pass

    def onSubscribeAllOptionOrderBook(self, data: dict, error: dict) -> None:
        """"""
        pass

    def onUnSubscribeAllOptionOrderBook(self, data: dict, error: dict) -> None:
        """"""
        pass

    def onSubscribeAllOptionTickByTick(self, data: dict, error: dict) -> None:
        """"""
        pass

    def onUnSubscribeAllOptionTickByTick(self, data: dict, error: dict) -> None:
        """"""
        pass

    def subscrbie(self, req: SubscribeRequest) -> None:
        """"""
        if self.login_status:
            xtp_exchange = EXCHANGE_VT2XTP.get(req.exchange, "")
            self.subscribeMarketData(req.symbol, 1, xtp_exchange)

    def query_contract(self) -> None:
        """"""
        for exchange_id in EXCHANGE_XTP2VT.keys():
            self.queryAllTickers(exchange_id)

    def connect(
            self,
            userid: str,
            password: str,
            client_id: int,
            server_ip: str,
            server_port: int,
            quote_protocol: int
    ) -> None:
        """"""
        self.userid = userid
        self.password = password
        self.client_id = client_id
        self.server_ip = server_ip
        self.server_port = server_port
        self.protocol = PROTOCOL_VT2XTP[quote_protocol]

        # Create API object
        if not self.connect_status:
            # path = str(get_folder_path(self.gateway_name.lower()))
            import os
            path = os.getcwd()
            # 创建QuoteApi
            # @param client_id （必须输入）用于区分同一用户的不同客户端，由用户自定义
            # @param save_file_path （必须输入）存贮订阅信息文件的目录，请设定一个有可写权限的真实存在的路径
            # @param log_level 日志输出级别
            # @return 创建出的UserApi
            # @remark 如果一个账户需要在多个客户端登录，请使用不同的client_id，系统允许一个账户同时登录多个客户端，
            # 但是对于同一账户，相同的client_id只能保持一个session连接，后面的登录在前一个session存续期间，无法连接
            self.createQuoteApi(self.client_id, path)
            # 1.1.6测试函数
            # 设置心跳检测时间间隔，单位为秒
            # @param interval 心跳检测时间间隔，单位为秒
            # @remark 此函数必须在Login之前调用
            self.setHeartBeatInterval(2)

            # 设置采用UDP方式连接时的接收缓冲区大小
            # @remark 需要在Login之前调用，默认大小和最小设置均为64MB。此缓存大小单位为MB，请输入2的次方数，例如128MB请输入128。
            self.setUDPBufferSize(128)

            # 使用UDP接收行情时，设置接收行情线程绑定的cpu
            # @param cpu_no 设置绑定的cpu，例如绑定cpu 0，可以设置0，绑定cpu 2，可以设置2，建议绑定后面的cpu
            # @remark 此函数可不调用，如果调用则必须在Login之前调用，否则不会生效
            self.setUDPRecvThreadAffinity(2)

            # 使用UDP接收行情时，设置解析行情线程绑定的cpu
            # @param cpu_no 设置绑定的cpu，例如绑定cpu 0，可以设置0，绑定cpu 2，可以设置2，建议绑定后面的cpu
            # @remark 此函数可不调用，如果调用则必须在Login之前调用，否则不会生效
            self.setUDPParseThreadAffinity(2)

            # 设定UDP收行情时是否输出异步日志
            # @param flag 是否输出标识，默认为true，如果不想输出“udpseq”开头的异步日志，请设置此参数为false
            # @remark 此函数可不调用，如果调用则必须在Login之前调用，否则不会生效
            self.setUDPSeqLogOutPutFlag(1)
            self.login_server()

    def login_server(self) -> None:
        # 用户登录请求
        # @return 登录是否成功，“0”表示登录成功，“-1”表示连接服务器出错，此时用户可以调用GetApiLastError()来获取错误代码，
        # “-2”表示已存在连接，不允许重复登录，如果需要重连，请先logout，“-3”表示输入有错误
        # @param ip 服务器ip地址，类似“127.0.0.1”
        # @param port 服务器端口号
        # @param user 登陆用户名
        # @param password 登陆密码
        # @param sock_type “1”代表TCP，“2”代表UDP
        # @param local_ip 本地网卡地址，类似“127.0.0.1”
        # @remark 此函数为同步阻塞式，不需要异步等待登录成功，当函数返回即可进行后续操作，此api只能有一个连接
        n = self.login(
            self.server_ip,
            self.server_port,
            self.userid,
            self.password,
            self.protocol
        )

        if not n:
            self.connect_status = True
            self.login_status = True
            msg = "行情服务器登录成功"
            self.query_contract()
            self.init()
        else:
            msg = f"行情服务器登录失败，原因：{n}"

        self.gateway.write_log(msg)

    def close(self) -> None:
        """"""
        if self.connect_status:
            self.exit()
