# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""

class Initialize:
    """
        initialize module to prepare the specified industry weight
        1、determine the period to calculate weight
        2、the key of weight depends on the growth and status property ,growth lies in mkv ; status lies in mkv,that is
           have to allocate between growth and status
        3、industry_weight dump to json and according to date tuple
        key : period --- months
    """
    _allocation = [0.3,0.7]

    def __init__(self,period = 12):
        self.indus_component = self._init_industry()
        self._period = period
        self._date_str_tuple = self._split_date()

    def _init_industry(self):
        """
            return DataFrame ,index : stockCode , columns : industry , toMarketTime
        """
        event = Event(dt.datetime.strftime(dt.datetime.now(),'%Y%m%d'))
        req = GateReq(event,field = 'Industry')
        industry = feed.addFundamental(req)
        return industry

    def _split_date(self):
        start_datestr = '20000101'
        start_datetime = dt.datetime.strptime(start_datestr,'%Y%m%d')
        datetime_tuple = [start_datetime + relativedelta(months = self._period * i ) for i in range(1,dt.datetime.now().year - start_datetime.year)]
        datestr_tuple = [dt.datetime.strftime(dt,'%Y%m%d') for dt in datetime_tuple]
        return list(zip(datestr_tuple[:-1],datestr_tuple[1:]))


    def _filter_industry(self,indus,date):
        """
            input : self.indus_component
            return stock_list
            algo : filter by date
        """
        pass


    def calc_weight(self,start_date,end_date,asset):
        """
            shift before one year or specify year to determine the weight of asset of specify category
            weight based on mkt and growth ,mkv means stable and status ,growth --- pct of year means prospection
        """
        event = Event([start_date,end_date],asset)
        req = GateReq(event,field = ['close','mkv'])
        raw = feed.addBars(req)
        raw = raw.fillna(method = 'ffill')
        pct = raw['close'][-1] / raw['close'][0] -1
        stablility = EMA.calc_feature(raw['mkv'],5)[-1]
        return {asset:{'growth':pct,'status':stablility}}


    def _get_indus_weight(self,s_date,e_date,indus):
        assets = self._filter_industry(indus,end_date)
        mid = {}
        for asset in assets:
            mid.update(self.calc_weight(s_date,e_date,asset))
        mid_df = (pd.DataFrame.from_dict(mid)).T
        rank = mid_df.rank().apply(lambda x : x * self._allocation,axis = 1)
        indus_weight = rank / rank.sum()
        return {indus:indus_weight}

    def _eval_composite(self,start_date,end_date):
        indus_dict = {}
        for indus in self.indus_component.index:
            indus_dict.update(self._get_indus_weight(start_date,end_date,indus))
        file_name = ('-').join([start_date,end_date])
        path = file_name + '.json'
        self._init_json(path,indus_dict)


    def _init_json(self,path,data):

        with open(path,'w') as f:
            json.dump(f,data)

    def compute(self):
        for date_tuple in self._date_str_tuple:
            self._eval_composite(*date_tuple)