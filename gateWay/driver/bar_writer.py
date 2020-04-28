import bcolz , pandas as pd , json , os

class BarWriter:

    def __init__(self, path):

        self.sid_path = path

    def _write_csv(self, data):
        """
            dump to csv
        """
        if isinstance(data, pd.DataFrame):
            data.to_csv(self.sid_path)
        else:
            with open(self.sid_path, mode='w') as file:
                if isinstance(data, str):
                    file.write(data)
                else:
                    for chunk in data:
                        file.write(chunk)

    def _init_hdf5(self, frames, _complevel=5, _complib='zlib'):
        if isinstance(frames, json):
            frames = json.dumps(frames)
        with pd.HDFStore(self.sid_path, 'w', complevel=_complevel, complib=_complib) as store:
            panel = pd.Panel.from_dict(frames)
            panel.to_hdf(store)
            panel = pd.read_hdf(self.sid_path)
        return panel

    def _init_ctable(self, raw):
        """
            Obtain 、Create 、Append、Attr empty ctable for given path.
            addcol(newcol[, name, pos, move])	Add a new newcol object as column.
            append(cols)	Append cols to this ctable -- e.g. : ctable
            Flush data in internal buffers to disk:
                This call should typically be done after performing modifications
                (__settitem__(), append()) in persistence mode. If you don’t do this,
                you risk losing part of your modifications.

        """
        ctable = bcolz.ctable(rootdir=self.sid_path, columns=None, names= \
            ['open', 'high', 'low', 'close', 'volume'], mode='w')

        if isinstance(raw, pd.DataFrame):
            ctable.fromdataframe(raw)
        elif isinstance(raw, dict):
            for k, v in raw.items():
                ctable.attrs[k] = v
        elif isinstance(raw, list):
            ctable.append([raw])
        #
        ctable.flush()

    @staticmethod
    def load_prices_from_ctable(file):
        """
            bcolz.open return a carray/ctable object or IOError (if not objects are found)
            ‘r’ for read-only
            ‘w’ for emptying the previous underlying data
            ‘a’ for allowing read/write on top of existing data
        """
        sid_path = os.path.join(XML.CTABLE, file)
        table = bcolz.open(rootdir=sid_path, mode='r')
        df = table.todataframe(columns=[
            'open',
            'high',
            'low',
            'close',
            'volume'
        ])
        return df
