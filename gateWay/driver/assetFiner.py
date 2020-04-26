
import sqlalchemy as sa

asset_db_table_names = frozenset({
    'symbol_naive_price',
    'dual_symbol_price'
    'bond_price',
    'index_price',
    'fund_price',
    'symbol_equity_basics',
    'bond_basics',
    'symbol_splits',
    'symbol_issue',
    'symbol_mcap',
    'symbol_massive',
    'market_margin',
    'version_info',
})

class AssetFinder(object):
    """
        AssetFinder is an interface to a database of Asset metadata written by
        an AssetDBWriter
        Asset is mainly concentrated on a_stock which relates with corresponding h_stock and convertible_bond;
        besides etf , benchmark
        基于 上市时间 注册地 主承商 ，对应H股, 可转债
    """
    def __init__(self,engine):
        self.engine = engine
        metadata = sa.MetaData(bind = engine)
        #反射
        metadata.reflect(only= asset_db_table_names)
        for table_name in asset_db_table_names:
            setattr(self,table_name,metadata.tables[table_name])

    def fuzzy_symbol_ownership_by_district(self,area):
        """
            基于区域地址找到对应的股票代码
        """
        assets_list = sa.select(self.equity_baiscs.c.code).\
                         where(self.equity_basics.c.district == area).\
                         execute().fetchall()
        return assets_list

    def fuzzy_symbol_ownership_by_broker(self,broker):
        """
            基于主承商找到对应的股票代码
        """
        assets_list = sa.select(self.equity_baiscs.c.code).\
                      where(self.equity_basics.c.broker == broker).\
                      execute().fetchall()
        return assets_list

    def fuzzy_symbol_ownership_by_ipo(self,date):
        """
            基于上市时间找到对应的股票代码
        """
        assets_list = sa.select(self.equity_baics.c.code).\
                      where(self.equity_basics.c.initial_date == date).\
                      execute().fetchall()
        return assets_list

    def fuzzy_Hsymbol_ownership_by_code(self,code):
        """
            基于A股代码找到对应的H股代码
        """
        hsymbol = sa.select(self.hk_pricing.hcode).\
                  where(self.hk_pricing.c.code == code).\
                  execute().scalar()
        return hsymbol

    def fuzzy_bond_ownership_by_code(self,code):
        """
            基于A股代码找到对应的可转债数据
        """
        bond_id = sa.select(self.bond_basics.bond_id).\
                  where(self.bond_basics.stock_id == code).\
                  execute().fetchall()
        return bond_id

    def retrieve_symbols(self):
        assets = sa.select(self.symbol_basics.c.sid).execute().fetchall()
        return assets

    def retieve_bonds(self):
        bond_assets = sa.select(self.bond_basics.c.sid).execute().fetchall()
        return bond_assets

    def retrieve_funds(self):
        fund_assets = sa.select(self.fund_price.c.code.distinct()).execute().fetchall()
        return fund_assets