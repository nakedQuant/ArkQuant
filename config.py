# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
from collections import namedtuple

from _calendar.trading_calendar import calendar

start_date = '2000-01-01'
end_date =  '2018-12-01'

sim_params = namedtuple('capital_base sessions benchmark')
sim_params.capital_base = 500000
sim_params.sessions = calendar.session_in_range(start_date=start_date, end_date=end_date, include=True)
sim_params.benchmark = '000001'
