import os
import time
import pandas as pd
import numpy as np
import glob
from pyproj import Transformer
from typing import Union
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

class CompileDatabase:
    def __init__(self, dir_list: Union[str, list[str]], file_type: str = 'layer', wave_speed: Union[None, float] = None, firn_correction: Union[None, float] = None, comp: bool = True, post: bool = True) -> None:
        self.dir_list = dir_list
        self.wave_speed = wave_speed
        self.firn_correction = firn_correction
        self.file_type = file_type
        self.comp = comp
        self.post = post

    def get_dict_ages(self, tab_file) -> dict:
        ages = pd.read_csv(tab_file, header=None, sep='\t', names=['file', 'age'])
        return dict(zip(ages['file'], ages['age']))

    def _pre_compile_checks(self, dir_list: list[str]) -> bool:
        missing = False
        for dir_path in dir_list:
            raw_dir = f"{dir_path}/raw/"
            if not os.path.exists(raw_dir):
                print(f"{raw_dir} does not exist")
                missing = True
        return not missing

    def compile(self, cpus: int = cpu_count()-1) -> None:
        if not isinstance(self.dir_list, list):
            self.dir_list = [self.dir_list]

        start_time = time.time()
        if not self._pre_compile_checks(self.dir_list):
            return

        if self.comp is True:
            all_files_list = []
            for dir_ in self.dir_list:
                files = glob.glob(f"{dir_}/raw/*.*")
                for file_path in files:
                    all_files_list.append({
                        'dir_path': dir_,
                        'file': os.path.basename(file_path),
                        'file_path': file_path
                    })

            num_tasks = len(all_files_list)
            num_workers = min(num_tasks, cpus)

            print('\n',
                    'Will start compiling', num_tasks, 'raw files\n'
                    '\n   ', num_workers, 'worker(s) allocated out of', cpu_count(), 'available cpus\n')

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(pool.imap_unordered(self._compile, all_files_list), total=num_tasks):
                        pass
            else:
                for file_dict in tqdm(all_files_list, desc="Processing"):
                    self._compile(file_dict=file_dict)

        if self.post is True:
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
                    for _ in tqdm(pool.imap_unordered(self._post_compilation, all_dirs), total=num_tasks):
                        pass
            else:
                for trace_dir in tqdm(all_dirs, desc='Processing'):
                    self._post_compilation(trace_dir=trace_dir)

        elapsed = time.time() - start_time
        print(f"\nCompilation completed in {elapsed:.2f} seconds")

    def _compile(self, file_dict) -> None:

        _, ext = os.path.splitext(file_dict['file'])
        ages = self.get_dict_ages(f'{file_dict['dir_path']}/IRH_ages.tab')
        original_new_columns = pd.read_csv(f'{file_dict['dir_path']}/original_new_column_names.csv')

        if ext == '.tab':
            sep='\t'
        elif ext == '.csv':
            sep=','
        else:
            print(f"{ext}: File type not supported...")
            return

        ds = pd.read_csv(file_dict['file_path'], comment="#", header=0, sep=sep, na_values=['-9999', 'NaN', ''])
        _, file_name = os.path.split(file_dict['file_path'])
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
            if self.wave_speed:
                ds['IRHDepth'] *= self.wave_speed
            if self.firn_correction:
                ds['IRHDepth'] += self.firn_correction

            ds['Trace_ID'] = ds['Trace_ID'].astype(str)
            ds['Trace_ID'] = ds['Trace_ID'].str.replace(r'/\s+', '_') # Replace slashes with underscores, otherwise the paths can get messy
            ds['Trace_ID'] = ds['Trace_ID'].str.replace('/', '_')

            ds.set_index('Trace_ID', inplace=True)

            for trace_id in np.unique(ds.index):
                ds_trace = ds.loc[trace_id].copy()
                ds_trace = ds_trace.drop_duplicates(subset=['x', 'y']) # Some datasets showed duplicated data
                if 'distance' not in ds_trace.columns:
                    x = ds_trace[['x', 'y']]
                    distances = np.sqrt(np.sum(np.diff(x, axis=0)**2, axis=1))
                    cumulative_distance = np.concatenate([[0], np.cumsum(distances)])
                    ds_trace['distance'] = cumulative_distance
                elif 'Distance [km]' in original_new_columns.columns:
                    ds_trace['distance'] *= 1000 # if distance in km, convert to meters

                if age in ['IceThk', 'BedElev', 'SurfElev', 'BasalUnit']:
                    ds_trace = ds_trace.rename(columns={'IRHDepth': age})
                    ds_trace_file = f'{file_dict['dir_path']}/pkl/{trace_id}/{age}.pkl' # if var instead of age, call the file as var.pkl
                else:
                    ds_trace_file = f'{file_dict['dir_path']}/pkl/{trace_id}/{file_name_}.pkl' # else use the same file name.pkl

                os.makedirs(f'{file_dict['dir_path']}/pkl/{trace_id}' , exist_ok=True)
                ds_trace.to_pickle(ds_trace_file)

        elif self.file_type == 'trace':
            if 'distance' not in ds.columns:
                x = ds[['x', 'y']]
                distances = np.sqrt(np.sum(np.diff(x, axis=0)**2, axis=1))
                cumulative_distance = np.concatenate([[0], np.cumsum(distances)])
                ds['distance'] = cumulative_distance
            elif 'Distance [km]' in original_new_columns.columns:
                ds['distance'] *= 1000 # if distance in km, convert to meters

            trace_id = file_name_
            os.makedirs(f'{file_dict['dir_path']}/pkl/{trace_id}' , exist_ok=True)

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
                    ds_IRH['IRHDepth'] += self.firn_correction

                for var in ['IceThk', 'BedElev', 'SurfElev', 'BasalUnit']:
                    if var in ds.columns:
                        ds_IRH[var] = ds[var]

                ds_trace_file = f'{file_dict['dir_path']}/pkl/{trace_id}/{IRH}.pkl'
                ds_IRH.to_pickle(ds_trace_file)

    def compute_irh_density(self, trace_dir) -> None:
        files = glob.glob(f"{trace_dir}/*.pkl")

        unwanted = {'IceThk.pkl', 'SurfElev.pkl', 'BasalUnit.pkl', 'BedElev.pkl', 'IRHDensity.pkl'}
        files = [f for f in glob.glob(f"{trace_dir}/*.pkl") if os.path.basename(f) not in unwanted]
        if len(files) > 1:
            dfs = [pd.read_pickle(f) for f in files]
            dfs = pd.concat(dfs)
        elif len(files) == 1:
            dfs = pd.read_pickle(files[0])
        else:
            return

        dfs = dfs[['x','y','IRHDepth']]
        valid = dfs.dropna(subset=['IRHDepth'])
        density = valid.groupby(['x', 'y']).size().reset_index(name='IRHDensity')

        density_file = f'{trace_dir}/IRHDensity.pkl'
        density.to_pickle(density_file)

    def extract_vars(self, trace_dir: str) -> None:
        files = glob.glob(f"{trace_dir}/*.pkl")

        unwanted = {'IceThk.pkl', 'SurfElev.pkl', 'BasalUnit.pkl', 'BedElev.pkl', 'IRHDensity.pkl'}
        files = [f for f in glob.glob(f"{trace_dir}/*.pkl") if os.path.basename(f) not in unwanted]
        if len(files) > 1:
            dfs = [pd.read_pickle(f) for f in files]
            dfs = pd.concat(dfs).drop_duplicates(subset=['x', 'y'])
        elif len(files) == 1:
            dfs = pd.read_pickle(files[0])
        else:
            return

        for var in ['IceThk', 'BedElev', 'SurfElev']:
            if var in dfs.columns:
                ds_var = dfs[['x', 'y', 'distance', var]]
                var_file = f'{trace_dir}/{var}.pkl'
                ds_var.to_pickle(var_file)

    def _post_compilation(self, trace_dir: str) -> None:
        self.extract_vars(trace_dir)
        self.compute_irh_density(trace_dir)

