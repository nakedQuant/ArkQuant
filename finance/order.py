class StyleType(Enum):
    """
        Market Price (市价单）
    """

    LMT = 'lmt'
    BOC = 'boc'
    BOP = 'bop'
    ITC = 'itc'
    B5TC = 'b5tc'
    B5TL = 'b5tl'
    FOK =  'fok'
    FAK =  'fak'


class Order(ABC):

    def make_id(self):
        return  uuid.uuid4().hex()

    @property
    def open_amount(self):
        return self.amount - self.filled

    @property
    def sid(self):
        # For backwards compatibility because we pass this object to
        # custom slippage models.
        return self.asset.sid

    @property
    def status(self):
        self._status = OrderStatus.OPEN

    @status.setter
    def status(self,status):
        self._status = status

    def to_dict(self):
        dct = {name : getattr(self.name)
               for name in self.__slots__}
        return dct

    def __repr__(self):
        """
        String representation for this object.
        """
        return "Order(%s)" % self.to_dict().__repr__()

    def __getstate__(self):
        """ pickle -- __getstate__ , __setstate__"""
        return self.__dict__()

    @abstractmethod
    def check_trigger(self,price,dt):
        """
        Given an order and a trade event, return a tuple of
        (stop_reached, limit_reached).
        For market orders, will return (False, False).
        For stop orders, limit_reached will always be False.
        For limit orders, stop_reached will always be False.
        For stop limit orders a Boolean is returned to flag
        that the stop has been reached.

        Orders that have been triggered already (price targets reached),
        the order's current values are returned.
        """
        raise NotImplementedError


class TickerOrder(Order):
    # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    """
        Parameters
        ----------
        asset : AssetEvent
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        dt : str, optional
            The date created order.

        市价单 --- 针对与卖出 --- 被动算法 ，基于时刻去卖出，这样避免被检测到 --- 将大订单拆分多个小订单然后基于时点去按照市价卖出

    """
    __slot__ = ['asset','_created_dt','capital']

    def __init__(self,asset,ticker,capital):
        self.asset = asset
        self._created_dt = ticker
        self.order_capital = capital
        self.direction = math.copysign(1,capital)
        self.filled = 0.0
        self.broker_order_id = self.make_id()
        self.order_type = StyleType.BOC

    def check_trigger(self,dts):
        if dts >= self._created_dt:
            return True
        return False


class RealtimeOrder(Order):
    # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    """
        Parameters
        ----------
        asset : AssetEvent
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        dt : str, optional
            The date created order.

        市价单 --- 针对与卖出 --- 被动算法 ，基于时刻去卖出，这样避免被检测到 --- 将大订单拆分多个小订单然后基于时点去按照市价卖出
        实时订单
    """
    __slot__ = ['asset', 'capital']

    def __init__(self, asset, capital):
        self.asset = asset
        self.order_capital = capital
        self.direction = math.copysign(1, capital)
        self.filled = 0.0
        self.broker_order_id = self.make_id()
        self.order_type = StyleType.BOC

    def check_trigger(self, dts):
        return True


class PriceOrder(Order):
    # using __slots__ to save on memory usage --- __dict__.  Simulations can create many
    # Order objects and we keep them all in memory, so it's worthwhile trying
    # to cut down on the memory footprint of this object.
    """
        Parameters
        ----------
        asset : AssetEvent
            The asset that this order is for.
        amount : int
            The amount of shares to order. If ``amount`` is positive, this is
            the number of shares to buy or cover. If ``amount`` is negative,
            this is the number of shares to sell or short.
        dt : str, optional
            The date created order.

        限价单 --- 执行买入算法， 如果确定标的可以买入，偏向于涨的概率，主动买入而不是被动买入

        买1 价格超过卖1，买方以卖1价成交
    """
    __slot__ = ['asset','amount','lmt']

    def __init__(self,asset,amount,price):
        self.asset = asset
        self.amount = amount
        self.lmt_price = price
        self._created_dt = dt
        self.direction = math.copysign(1,self.amount)
        self.filled = 0.0
        self.broker_order_id = self.make_id()
        self.order_type = StyleType.LMT

    def check_trigger(self,bid):
        if bid <= self.lmt_price:
            return True
        return False
