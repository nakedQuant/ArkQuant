#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:39:46 2019

@author: python
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib,sys,time

from GateWay.Spider import Ancestor
from GateWay.Spider.Tushare import  TushareClient
from GateWay.Financial.MarketValue import MarketValue

class Orient:
    """
        Orient 模块用于初始化数据库并基于Spider批量运行入库操作
        数据库里面ashareEquity里面1900-01-01，5条错误记录 ，
        splitsDivdend --- 进度为实施:
        000001 --- 除权除息日1991-04-03  化为 1991-05-02 ，登记日 1991-03-12 调整为 1991-04-30
        000001 --- 删除1900-01-01 ，增加 修订 (1992-03-04 5 0 2  1992-03-23 1992-03-20)
        000007 --- 1900-01-01 修订 1992-10-22
    """
    nowdays = time.strftime('%Y-%m-%d',time.localtime())

    @classmethod
    def set_mode(cls,init = False):
        if init:
            """初始化将spider 里面 table drop 重新创建"""
            Ancestor._init_db(init)
        else:
            Ancestor.frequency = 'daily'
        cls.frequency = Ancestor.frequency

    def __init__(self):
        self.module_names = ['Index', 'Astock', 'ETF', 'Convertible','ExtraOrdinary']

    @staticmethod
    def import_cls(name):
        """获取Crawler里面的类"""
        sys.path.append('/Users/python/PycharmProjects/git/Alert/GateWay/Spider')
        module = importlib.import_module('Crawler', 'GateWay.Spider')
        cls = getattr(module, name)
        return cls

    def _parallel(self):
        restart = dict()
        def run(attr):
            module = self.import_cls(attr)
            cls = module()
            r = cls.run_bulks()
            restart.update(r)

        with ThreadPoolExecutor(max_workers = 3) as executor:
            to_do = []
            # 线程池
            for name in self.module_names:
                future = executor.submit(run, name)
                to_do.append(future)
            # 线程处理
            for f in as_completed(to_do):
                f.result()
        return restart

    def rerun(self,name,objects):
        """基于跑失败的参数重新执行爬虫"""
        module = self.import_cls(name)
        print('---------load module:',module)
        cls = module()
        for asset in objects:
            cls._run_session(asset)
        record = cls._failure
        if len(record):
            time.sleep(3)
            self.rerun(name,record)
        else:
            print('module :%s run successfully'%name)

    def update_extra(self):
        """更新非基础数据的模块 --- 市值、市场融资融券、股票增减持、交易日、股票状态"""
        m = MarketValue(self.frequency,self.nowdays)
        m.parallel()
        print('update market value via MarketValue')
        """更新市场融资融券和股东增减持"""
        module = self.import_cls('ExtraOrdinary')
        module().download_market_margin()
        print('update market_margin daily successfully')
        ts =  TushareClient()
        ts.update_via_ts()
        print('update trade_dt and status successfully via TushareClient')
        module().download_ashare_holding()
        print('update ashare_holding daily successfully')

    def initialize(self):
        """
            执行爬虫任务
        """
        targets = self._parallel()
        print(targets)
        if len(targets):
            for _name,obj in targets.items():
                self.rerun(_name,obj)
        # self.update_extra()


if __name__ == '__main__':

    s = time.time()
    # Orient.set_mode(init = True)
    Orient.set_mode()
    Orient().initialize()
    elapsed = time.time() - s
    print('elapsed time %f'%elapsed)