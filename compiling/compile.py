import os
import sys
import pandas as pd
import numpy as np
import glob
from pyproj import Transformer

class CompileDatabase:
    def __init__(self, dir_list: list[str]) -> None:
        self.dir_list = dir_list

    def get_dict_ages(self, tab_file) -> dict:
        ages = pd.read_csv(tab_file, header=None, sep='\t', names=['file', 'age'])
        return dict(zip(ages['file'], ages['age']))

    def compile(self) -> None:
        for dir_path in self.dir_list:

            raw_files = glob.glob(f'{dir_path}/raw/*.*')

            _, ext = os.path.splitext(raw_files[0])

            ages = self.get_dict_ages(f'{dir_path}/IRH_ages.tab')
            original_new_columns = pd.read_csv(f'{dir_path}/original_new_column_names.csv')

            if ext == '.tab':
                sep='\t'
            elif ext == '.csv':
                sep=','
            else:
                print('File type not supported, exiting ...')
                sys.exit()

            for _, file in enumerate(raw_files):
                print(file)
                # Read the CSV file
                ds = pd.read_csv(file, comment="#", header=0, sep=sep)
                _, file_name = os.path.split(file)
                file_name_, ext = os.path.splitext(file_name)

                ds = ds[original_new_columns.columns.values]  # Selecting columns of interest
                ds.columns = original_new_columns.iloc[0].values  # renaming the columns

                age = ages[file_name_]
                ds = ds.rename(columns={'IRHdepth': age})

                ds['Trace_ID'] = ds['Trace_ID'].astype(str)
                ds['Trace_ID'] = ds['Trace_ID'].str.replace(r'/\s+', '_') # Replace slashes with underscores, otherwise the paths can get messy
                ds['Trace_ID'] = ds['Trace_ID'].str.replace('/', '_')

                ds.set_index('Trace_ID', inplace=True)

                if 'x' not in ds.columns and 'y' not in ds.columns:
                    if 'lon' in ds.columns and 'lat' in ds.columns:
                        transformer = Transformer.from_proj(
                            "EPSG:4326",  # source: WGS84 (lon/lat)
                            "+proj=stere +lon_0=0 +lat_0=-90 +lat_ts=-71 +datum=WGS84 +units=m +no_defs",  # target: polar
                            always_xy=True
                        )
                        ds['x'], ds['y'] = transformer.transform(ds['lon'].values, ds['lat'].values)
                elif 'lon' not in ds.columns and 'lat' not in ds.columns:
                    if 'x' in ds.columns and 'y' in ds.columns:
                        inverse_transformer = Transformer.from_proj(
                            "+proj=stere +lon_0=0 +lat_0=-90 +lat_ts=-71 +datum=WGS84 +units=m +no_defs",  # source: polar
                            "EPSG:4326",  # target: WGS84 (lon/lat)
                            always_xy=True
                        )
                        ds['lon'], ds['lat'] = inverse_transformer.transform(ds['x'].values, ds['y'].values)
                elif 'lon' in ds.columns and 'lat' in ds.columns and 'x' in ds.columns and 'y' in ds.columns:
                    pass
                else:
                    print('No coordinates found in the dataset, exiting ....')
                    sys.exit()

                for trace_id in np.unique(ds.index):
                    ds_trace = ds.loc[trace_id]
                    ds_trace_file = f'{dir_path}/pkl/{trace_id}/{file_name_}.pkl'

                    os.makedirs(f'{dir_path}/pkl/{trace_id}' , exist_ok=True)
                    ds_trace.to_pickle(ds_trace_file)
                    print(ds_trace_file)
