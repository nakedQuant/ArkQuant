from sqlalchemy import select,and_,cast ,Numeric,desc,Integer

class Core:
    """
        sqlalchemy core 操作
    """
    def __init__(self):

        self.db = DataLayer()
        self.tables = self.db.metadata.tables

    def _proc(self,ins):
        rp = self.db.engine.execute(ins)
        return rp

    def load_calendar(self,sdate,edate):
        """获取交易日"""
        sdate = sdate.replace('-','')
        edate = edate.replace('-','')
        table = self.tables['ashareCalendar']
        ins = select([table.c.trade_dt]).where(table.c.trade_dt.between(sdate,edate))
        rp = self._proc(ins)
        trade_dt = [r.trade_dt for r in rp]
        return trade_dt

    def is_calendar(self,dt):
        """判断是否为交易日"""
        dt = dt.replace('-','')
        table = self.tables['ashareCalendar']
        ins = select([table.c.trade_dt])
        rp = self._proc(ins)
        trade_dt = [r.trade_dt for r in rp]
        flag = dt in trade_dt
        return flag

    def load_calendar_offset(self,date,sid):
        date = date.replace('-','')
        table = self.tables['ashareCalendar']
        if sid > 0 :
            ins = select([table.c.trade_dt]).where(table.c.trade_dt > date)
            ins = ins.order_by(table.c.trade_dt)
        else :
            ins = select([table.c.trade_dt]).where(table.c.trade_dt < date)
            ins = ins.order_by(desc(table.c.trade_dt))
        ins = ins.limit(abs(sid))
        rp = self._proc(ins)
        trade_dt = [r.trade_dt for r in rp]
        return trade_dt


class BarReader:

    def __init__(self):
        self.loader = Core()
        self.ts = TushareClient()
        self.extra = spider_engine.ExtraOrdinary()

    def _verify_fields(self,f,asset):
        """如果asset为空，fields必须asset"""
        field = f.copy()
        if not isinstance(field,list):
            raise TypeError('fields must be list')
        elif asset is None:
            field.append('code')
        return field

    @staticmethod
    def calendar_foramtted(t):
        """将eg 20120907 --- 2012-09-07"""
        trans = ('-').join([t[:4],t[4:6],t[6:8]])
        return trans

    def load_trading_calendar(self, sdate, edate):
        """
            返回交易日列表 ， 类型为array
            session in range
        """
        calendar = self.loader.load_calendar(sdate, edate)
        trade_dt = list(map(self.calendar_foramtted,calendar))
        return trade_dt

    def is_market_caledar(self,dt):
        """判断是否交易日"""
        flag = self.loader.is_calendar(dt)
        return flag

    def load_calendar_offset(self, dt, window):
        """
            获取交易日偏移量
        """
        calendar_offset = self.loader.load_calendar_offset(dt, window)
        trade_dt = list(map(self.calendar_foramtted,calendar_offset))
        return trade_dt[-1]

    def load_stock_status(self,code):
        """返回股票是否退市或者暂停上市"""
        raw = self.loader.load_stock_status(code)
        df = pd.DataFrame(raw,columns = ['code','status'])
        df = df if len(df) else None
        return df
