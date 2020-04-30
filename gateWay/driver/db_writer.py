from sqlalchemy import MetaData,create_engine
from weakref import WeakValueDictionary
from sqlalchemy import inspect

class DBWriter(object):

    _cache = WeakValueDictionary()

    def __new__(cls,engine_path):
        try:
            return cls._cache[engine_path]
        except KeyError:
            class_ins = cls._cache[engine_path] = \
                             super(DBWriter,cls).__new__()._init_db(engine_path)
            return class_ins

    @staticmethod
    def set_isolation_level(conn,level = "READ COMMITTED"):
        connection = conn.execution_options(
            isolation_level= level
        )
        return connection

    def __enter__(self):
        return self

    def _init_db(self,engine_path):
        self.engine = create_engine(engine_path)
        self.insp = inspect(self.engine)

    def _write_df_to_table(self,conn,tbl,frame,chunksize = 5000):
        expected_cols = [item['name'] for item in self.insp.get_columns(tbl)]
        if frozenset(frame.columns) != frozenset(expected_cols):
            raise ValueError(
                "Unexpected frame columns:\n"
                "Expected Columns: %s\n"
                "Received Columns: %s" % (
                    set(expected_cols),
                    frame.columns.tolist(),
                )
            )
        frame.to_sql(
            tbl,
            conn,
            index=True,
            index_label=None,
            if_exists='append',
            chunksize=chunksize,
        )

    def writer(self,tbl,df):
        with open('db_schema.py','r') as f:
            string_obj = f.read()
        exec(string_obj)

        with self.engine.begin() as conn:
            # Create SQL tables if they do not exist.
            # self.metadata.create_all(bind=engine)
            con = self.set_isolation_level(conn)
            self._write_df_to_table(con,tbl,df)

    def reset(self,overwriter = False):
        meta = MetaData(bind=self.engine)
        if overwriter:
            meta.drop_all()
        else:
            tbls = meta.tables
            for tbl in tbls:
                self.engine.execute(tbl.delete())

    def close(self):
        self.conn.close()

    def __exit__(self, * exc_info):
        self.close()


if __name__ == '__main__':

    engine = create_engine('mysql+pymysql://root:macpython@localhost:3306/spider',
                            pool_size=50,
                            max_overflow=100,
                            pool_timeout=-1,
                            pool_pre_ping=True,
                            isolation_level="READ UNCOMMITTED")
    con = engine.connect().execution_options(isolation_level = "READ UNCOMMITTED")
    print(con.get_execution_options())
