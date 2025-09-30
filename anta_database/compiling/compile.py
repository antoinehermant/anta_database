import os
import sys
import pandas as pd
import numpy as np
import glob
from pyproj import Transformer
from typing import Union
from multiprocessing import Pool, cpu_count

class CompileDatabase:
    def __init__(self, dir_list: Union[str, list[str]], file_type: str = 'layer', wave_speed: Union[None, float] = None, firn_correction: Union[None, float] = None, post: bool = True) -> None:
        self.dir_list = dir_list
        self.wave_speed = wave_speed
        self.firn_correction = firn_correction
        self.file_type = file_type
        self.post = post

    def get_dict_ages(self, tab_file) -> dict:
        ages = pd.read_csv(tab_file, header=None, sep='\t', names=['file', 'age'])
        return dict(zip(ages['file'], ages['age']))

    def compile(self, cpus: int = cpu_count()-1) -> None:
        if not isinstance(self.dir_list, list):
            self.dir_list = [self.dir_list]

        num_tasks = len(self.dir_list)
        num_workers = min(num_tasks, cpus)

        print('\n',
                'Will start compiling', len(self.dir_list), 'directories\n'
                '\n   ', num_workers, 'worker(s) allocated out of', cpu_count(), 'available cpus\n')

        if num_workers > 1:
            with Pool(num_workers) as pool:
                pool.map(self._compile, self.dir_list)
        else:
            for dir_path in self.dir_list:
                self._compile(dir_path=dir_path)

        if self.post:
            self._post_compilation()

    def _compile(self, dir_path: str) -> None:
        if not os.path.exists(f'{dir_path}/raw/'):
            print(f"{dir_path}/raw/ does not exists")
            return

        raw_files = glob.glob(f'{dir_path}/raw/*.*')

        _, ext = os.path.splitext(raw_files[0])

        ages = self.get_dict_ages(f'{dir_path}/IRH_ages.tab')
        original_new_columns = pd.read_csv(f'{dir_path}/original_new_column_names.csv')

        if ext == '.tab':
            sep='\t'
        elif ext == '.csv':
            sep=','
        else:
            print(f"{ext}: File type not supported...")
            return

        for i, file in enumerate(raw_files):
            print('Processing', file, f'({i}/{len(raw_files)})')
            ds = pd.read_csv(file, comment="#", header=0, sep=sep, na_values=['-9999', 'NaN'])
            _, file_name = os.path.split(file)
            file_name_, ext = os.path.splitext(file_name)

            ds = ds[ds.columns.intersection(original_new_columns.columns)]
            ds.columns = original_new_columns[ds.columns].iloc[0].values  # renaming the columns

            if 'IceThk' in ds.columns and 'SurfElev' in ds.columns and not 'BedElev' in ds.columns:
                ds['BedElev'] = ds['SurfElev'] - ds['IceThk']
            if 'IceThk' in ds.columns and 'BedElev' in ds.columns and not 'SurfElev' in ds.columns:
                ds['SurfElev'] = ds['BedElev'] + ds['IceThk']
            if 'SurfElev' in ds.columns and 'BedElev' in ds.columns and not 'IceThk' in ds.columns:
                ds['IceThk'] = ds['SurfElev'] - ds['BedElev']

            if self.wave_speed:
                for var in ['IceThk', 'BedElev']:
                    if var in ds.columns:
                        ds[var] *= self.wave_speed
            if self.firn_correction:
                for var in ['IceThk', 'BedElev']:
                    if var in ds.columns:
                        ds[var] += self.firn_correction

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
                print('No coordinates found in the dataset')
                return

            if self.file_type == 'layer':
                age = str(ages[file_name_])
                ds = ds.rename(columns={'IRHdepth': 'IRHDepth'})
                if self.wave_speed:
                    ds[age] *= self.wave_speed
                if self.firn_correction:
                    ds[age] += self.firn_correction

                ds['Trace_ID'] = ds['Trace_ID'].astype(str)
                ds['Trace_ID'] = ds['Trace_ID'].str.replace(r'/\s+', '_') # Replace slashes with underscores, otherwise the paths can get messy
                ds['Trace_ID'] = ds['Trace_ID'].str.replace('/', '_')

                ds.set_index('Trace_ID', inplace=True)

                for trace_id in np.unique(ds.index):
                    ds_trace = ds.loc[trace_id].copy()
                    if 'distance' not in ds_trace.columns:
                        x = ds_trace[['x', 'y']]
                        distances = np.sqrt(np.sum(np.diff(x, axis=0)**2, axis=1))
                        cumulative_distance = np.concatenate([[0], np.cumsum(distances)])
                        ds_trace['distance'] = cumulative_distance
                    elif 'Distance [km]' in original_new_columns.columns:
                        ds_trace['distance'] *= 1000 # if distance in km, convert to meters

                    if age in ['IceThk', 'BedElev', 'SurfElev', 'BasalUnit']:
                        ds_trace_file = f'{dir_path}/pkl/{trace_id}/{age}.pkl' # if var instead of age, call the file as var.pkl
                    else:
                        ds_trace_file = f'{dir_path}/pkl/{trace_id}/{file_name_}.pkl' # else use the same file name.pkl


                    os.makedirs(f'{dir_path}/pkl/{trace_id}' , exist_ok=True)
                    ds_trace.to_pickle(ds_trace_file)
                    print(ds_trace_file)

                    for var in ['IceThk', 'BedElev', 'SurfElev']:
                        if var in ds.columns:
                            ds_var = ds_trace[['x', 'y', 'distance', var]]
                            ds_var_file = f'{dir_path}/pkl/{trace_id}/{var}.pkl'
                            if os.path.exists(ds_var_file):
                                var_data = pd.read_pickle(ds_var_file)
                                merged_data = pd.concat([var_data, ds_var]).drop_duplicates(subset=['x', 'y'])

                                merged_data.to_pickle(ds_var_file)
                                print(ds_var_file)
                            else:
                                ds_var.to_pickle(ds_var_file)
                                print(ds_var_file)

            elif self.file_type == 'trace':
                if 'distance' not in ds.columns:
                    x = ds[['x', 'y']]
                    distances = np.sqrt(np.sum(np.diff(x, axis=0)**2, axis=1))
                    cumulative_distance = np.concatenate([[0], np.cumsum(distances)])
                    ds['distance'] = cumulative_distance
                elif 'Distance [km]' in original_new_columns.columns:
                    ds['distance'] *= 1000 # if distance in km, convert to meters

                trace_id = file_name_
                os.makedirs(f'{dir_path}/pkl/{trace_id}' , exist_ok=True)

                for var in ['IceThk', 'BedElev', 'SurfElev', 'BasalUnit']:
                    if var in ds.columns:
                        ds_var = ds[['x', 'y', 'distance', var]]
                        ds_var_file = f'{dir_path}/pkl/{trace_id}/{var}.pkl'
                        if os.path.exists(ds_var_file):
                            var_data = pd.read_pickle(ds_var_file)
                            merged_data = pd.concat([var_data, ds_var]).drop_duplicates(subset=['x', 'y'])

                            merged_data.to_pickle(ds_var_file)
                            print(ds_var_file)
                        else:
                            ds_var.to_pickle(ds_var_file)
                            print(ds_var_file)

                ages = {key: ages[key] for key in ds.columns if key in ages}

                for IRH in ages:
                    age = str(ages.get(IRH))
                    ds_IRH = ds[IRH]
                    ds_IRH = pd.DataFrame({
                        'lon': ds['lon'],
                        'lat': ds['lat'],
                        'x': ds['x'],
                        'y': ds['y'],
                        'distance': ds['distance'],
                        'IRHDepth': ds_IRH,
                    })
                    if self.wave_speed:
                        ds_IRH['IRHDepth'] *= self.wave_speed
                    if self.firn_correction:
                        ds_IRH += self.firn_correction

                    for var in ['IceThk', 'BedElev', 'SurfElev', 'BasalUnit']:
                        if var in ds.columns:
                            ds_IRH[var] = ds[var]

                    ds_trace_file = f'{dir_path}/pkl/{trace_id}/{IRH}.pkl'
                    ds_IRH.to_pickle(ds_trace_file)
                    print(ds_trace_file)

    def compute_density(self, trace_dir) -> None:
        files = glob.glob(f"{trace_dir}/*.pkl")

        unwanted = {'IceThk.pkl', 'SurfElev.pkl', 'BasalUnit', 'BedElev'}
        files = [f for f in glob.glob(f"{trace_dir}/*.pkl") if os.path.basename(f) not in unwanted]
        dfs = [pd.read_pickle(f) for f in files]
        combined = pd.concat(dfs)
        valid = combined.dropna(subset=['IRHDepth'])
        density = valid.groupby(['x', 'y']).size().reset_index(name='IRHDensity')

        density_file = f'{trace_dir}/IRHDensity.pkl'
        density.to_pickle(density_file)
        print(density_file)

    def _post_compilation(self, cpus:int = cpu_count()-1) -> None:
        all_dirs = []
        for dir_ in self.dir_list:
            dirs = [d for d in glob.glob(f"{dir_}/pkl/*") if os.path.isdir(d)]
            all_dirs.extend(dirs)

        num_tasks = len(all_dirs)
        num_workers = min(num_tasks, cpus)

        print('\n',
                'Will start post compilation of', len(all_dirs), 'traces\n'
                '\n   ', num_workers, 'worker(s) allocated out of', cpu_count(), 'available cpus\n')

        if num_workers > 1:
            with Pool(num_workers) as pool:
                pool.map(self.compute_density, all_dirs)
        else:
            for trace_dir in all_dirs:
                self.compute_density(trace_dir=trace_dir)
