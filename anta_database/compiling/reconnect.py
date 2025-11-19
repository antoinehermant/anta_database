import os
import pandas as pd
import numpy as np
import h5py
import xarray as xr
from multiprocessing import Pool
from functools import partial
from scipy.spatial import cKDTree
from tqdm import tqdm

from anta_database.database.database import Database, MetadataResult

class Reconnect:
    def __init__(self, db: 'Database', query_to_reconnect: 'MetadataResult', query_connect_with: 'MetadataResult') -> None:
        self.query_to_reconnect = query_to_reconnect
        self.query_connect_with = query_connect_with
        self._db = db
        self.var_list = ['ICE_THK', 'BED_ELEV', 'SURF_ELEV', 'BASAL_UNIT']

    def reconnect(self, cpus: int = 1):
        all_dfs = []
        print('Loading dataset where to get the variables from\n')
        for ds, _ in tqdm(self._db.data_generator(self.query_connect_with), total=len(self.query_connect_with['flight_id']), desc='Loading', unit='file'):
            all_dfs.append(ds)
        df_connect = pd.concat(all_dfs)
        tree = cKDTree(df_connect[['PSX', 'PSY']].values)

        file_paths = self._db._get_file_paths_from_metadata(metadata=self.query_to_reconnect)
        file_paths = np.unique(file_paths)

        num_tasks = len(file_paths)
        num_workers = min(num_tasks, cpus)

        print('\nReconnecting variables in individual flight lines:\n')
        if num_workers > 1:
            with Pool(num_workers) as pool:
                partial_process = partial(self.process, df_connect, tree)
                for _ in tqdm(pool.imap_unordered(partial_process, file_paths), total=len(file_paths)):
                    pass
        else:
            for file_path in tqdm(file_paths, desc='Processing'):
                self.process(df_connect, tree, file_path)


    def process(self, df_connect: pd.DataFrame, tree, file_path: str):
        data_dir = self._db.db_dir
        full_path = os.path.join(data_dir, file_path)
        with h5py.File(full_path, 'a') as f:
            with xr.open_dataset(f, engine='h5netcdf') as ds:

                ds_psx = ds['PSX'].values
                ds_psy = ds['PSY'].values
                coords = np.column_stack((ds_psx, ds_psy))

                point_shape = len(ds['point'].values)
                if point_shape < 1e6:
                    chunk_size_point = min(100, point_shape)
                elif point_shape < 1e7:
                    chunk_size_point = 1000
                else:
                    chunk_size_point = 10000

                # Find nearest neighbors
                _, idx = tree.query(coords, k=1)

                encoding = {
                    'PSX': {'zlib': True, 'complevel': 1},
                    'PSY': {'zlib': True, 'complevel': 1},
                }

                for var in ['BED_ELEV', 'ICE_THK', 'SURF_ELEV']:
                    if var not in ds.variables and var in df_connect.columns:
                        ds[var] = (['point'], df_connect.iloc[idx][var].values)

                for var in self.var_list:
                    if var in ds.variables:
                        ds[var] = ds[var].chunk({'point': chunk_size_point})
                        encoding[var] = {'zlib': True, 'complevel': 1, 'chunksizes': (chunk_size_point)}

                ds = ds[list(ds.variables)]
                ds.to_netcdf(full_path, engine='h5netcdf', mode='a', encoding=encoding)
