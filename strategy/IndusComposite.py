# -*- coding : utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""


class Composite:
    """
        initialize module to prepare the specified industry weight
        1、determine the period to calculate weight
        2、the key of weight depends on the growth and status property ,growth lies in mkv ; status lies in mkv,that is
           have to allocate between growth and status
        3、industry_weight dump to json and according to date tuple
        key : period --- months

        shift before one year or specify year to determine the weight of asset of specify category
        weight based on mkt and growth ,mkv means stable and status ,growth --- pct of year means prospection
    """