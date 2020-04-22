# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import json
from gateWay import Feed,GateReq,Event

feed = Feed()

class Analyse:
    """
        select module : 1、self contruct compound index simliary to ETF or index
                2、measure index motivation
                3、select the most significant direction and filter the compound which is at high correlation
        main part is select specify category (e.g. industry --------)
        input :industry
        output : timeseries
    """
    _field = 'close'

    def _get_indus_index(self,dt ,window):
        _index = {}
        indus_weight = self._parse_json(dt)
        for k,v in indus_weight.items():
            raw_arrays = self._load_kl_pd(v.keys(),dt,window)
            indus_index = (raw_arrays * v).sum(axis =1)
            _index.update({k:indus_index})
        return _index

    def _load_kl_pd(self,assets,dt,window):
        event = Event(dt,assets)
        req = GateReq(event,field = self._field,window = window)
        raw = feed.addBars(req)
        return raw


    @staticmethod
    def get_file_by_dt(dt):
        """
            according file_name(right part) -- select
        """
        pass

    def _parse_json(self,dt):
        file = self.get_file_by_dt(dt)
        result = json.load(file)
        return result

    def _add_proc(self,res):
        """
            solve how tranform the indus_index  into sign which is used to select
            1、select indus based on the compositeAnalyse
            2、selec targeted asset based on the selected indus
        """
        pass

    def run(self,dt,window):
        portfilio_indus_index = self._get_indus_index(dt,window)
        selected_asset = self._add_proc(portfilio_indus_index)
        return selected_asset