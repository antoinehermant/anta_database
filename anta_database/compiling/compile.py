import warnings
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

    def file_metadata(self, file_path) -> pd.DataFrame:
        table = pd.read_csv(file_path, header=0, sep=',')
        table = table.fillna({
            'acquisition_year': 0,
            'age': 0,
            'age_unc': 0,
        })
        table = table.astype({
            'raw_file': 'str',
            'author': 'str',
            'institute': 'str',
            'project': 'str',
            # 'acquisition_year': 'int32',
            # 'age': 'int32',
            # 'age_unc': 'int32',
        })
        table.set_index('raw_file', inplace=True)
        return table

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
                    for _ in tqdm(pool.imap_unordered(self._compile, all_files_list), total=num_tasks, desc="Processing"):
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

    def convert_col_to_Int32(self, df):
        df = df.fillna(pd.NA)
        df = df.round(0)
        df = df.astype('Int32')
        return df

    def convert_col_to_num(self, df):
        df = pd.to_numeric(df, errors='coerce')
        return df

    def _compile(self, file_dict) -> None:

        _, ext = os.path.splitext(file_dict['file'])
        table = self.file_metadata(f'{file_dict['dir_path']}/raw_files_md.csv')
        original_new_columns = pd.read_csv(f'{file_dict['dir_path']}/original_new_column_names.csv')

        if ext == '.tab':
            sep='\t'
        elif ext == '.csv':
            sep=','
        elif ext == '.txt':
            sep=' '
        else:
            print(f"{ext}: File type not supported...")
            return

        def warn_with_file_path(message, category, filename, lineno, file=None, line=None):
                print(f"Warning in file: {file_dict['file_path']}")
                print(f"Warning message: {message}")

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = warn_with_file_path

            ds = pd.read_csv(file_dict['file_path'],
                            comment="#",
                            header=0,
                            sep=sep,
                            # usecols=original_new_columns.columns,
                            na_values=['-9999', '-9999.0', 'NaN', 'nan', ''],
                            dtype=str
                            )

        _, file_name = os.path.split(file_dict['file_path'])
        file_name_, ext = os.path.splitext(file_name)

        ds = ds[ds.columns.intersection(original_new_columns.columns)]
        ds.columns = original_new_columns[ds.columns].iloc[0].values  # renaming the columns

        pattern_values = original_new_columns[1:]
        pattern_header = original_new_columns.iloc[0]
        pattern_values.columns = pattern_header

        if self.file_type == 'layer':
            ds = ds.astype({'Trace_ID': str})

        for var in ['IceThk', 'SurfElev', 'BedElev', 'IRHDepth', 'x', 'distance', 'y', 'lat', 'lon']:
            if var in ds.columns:
                ds[var] = self.convert_col_to_num(ds[var])

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
        elif 'x' in ds.columns and 'y' in ds.columns:
            pass
        else:
            print('No coordinates found in the dataset')
            return

        if self.file_type == 'layer':
            if 'age' in table.columns:
                age = table.loc[file_name_]['age']
            else:
                age = pd.NA
            if self.wave_speed:
                ds['IRHDepth'] *= self.wave_speed
            if self.firn_correction:
                ds['IRHDepth'] += self.firn_correction

            ds['Trace_ID'] = ds['Trace_ID'].str.replace(r'/\s+', '_') # Replace slashes with underscores, otherwise the paths can get messy
            ds['Trace_ID'] = ds['Trace_ID'].str.replace('/', '_')
            ds.set_index('Trace_ID', inplace=True)

            unique_trace_ids = np.unique(ds.index)
            converted = pd.to_numeric(unique_trace_ids, errors='coerce')
            converted = pd.Series(converted)
            trace_id = []
            trace_id_flag = 'original'

            if 'trace_id' in table.columns:
                if not pd.isna(table.loc[file_name_]['trace_id']):
                    trace_id = table.loc[file_name_]['trace_id']
                    trace_id_flag = 'not_provided'

            if 'trace_id_prefix' in table.columns:
                if not pd.isna(table.loc[file_name_]['trace_id_prefix']):
                    ds.index = [f"{table.loc[file_name_]['trace_id_prefix']}_{x}" for x in ds.index]
                    trace_id_flag = 'project_acq-year_number'

            author = table.loc[file_name_]['author']
            institute = table.loc[file_name_]['institute']
            institute_flag = 'original'
            project = table.loc[file_name_]['project']
            project_flag = 'original'
            acq_year = table.loc[file_name_]['acquisition_year']
            acq_year_flag = 'original'

            trace_ids = np.unique(ds.index)
            if trace_id_flag == 'not_provided':
                trace_ids = [trace_id]

            def extract_year(time: str, pattern: str):
                position = pattern.find('YYYY')
                return time[position:position+4]

            for trace_id in trace_ids:
                if trace_id_flag == 'not_provided':
                    ds_trace = ds.copy()
                else:
                    ds_trace = ds.loc[trace_id].copy()

                if 'acq_year' in pattern_values.columns:
                    pattern = pattern_values.iloc[0]['acq_year']
                    if not pd.isna(pattern):
                        ds_trace['acq_year'] = ds_trace['acq_year'].apply(lambda x: extract_year(x, pattern))
                        unique_time = np.unique(ds_trace['acq_year'])
                        if len(unique_time) > 1:
                            raise ValueError(f'trace {trace_id} in {file_name_} contains {len(unique_time)} different acquisition year. Current code does not support this')
                        else:
                            position = pattern.find('YYYY')
                            acq_year = unique_time[0][position:position+4]
                        ds_trace.drop(columns='acq_year', inplace=True)

                ds_trace = ds_trace.drop_duplicates(subset=['x', 'y']) # Some datasets showed duplicated data

                if trace_id == 'nan':
                    trace_id = f'{project}_{acq_year}'
                    trace_id_flag = 'not_provided'

                if 'distance' not in ds_trace.columns and author not in ['BEDMAP1', 'BEDMAP2']:
                    x = ds_trace[['x', 'y']]
                    distances = np.sqrt(np.sum(np.diff(x, axis=0)**2, axis=1))
                    cumulative_distance = np.concatenate([[0], np.cumsum(distances)])
                    ds_trace['distance'] = cumulative_distance
                elif 'Distance [km]' in original_new_columns.columns:
                    ds_trace['distance'] *= 1000 # if distance in km, convert to meters

                for col in ds_trace.columns:
                    ds_trace[col] = self.convert_col_to_Int32(ds_trace[col])

                if age is not pd.NA and age in ['IceThk', 'BedElev', 'SurfElev', 'BasalUnit']:
                    ds_trace = ds_trace.rename(columns={'IRHDepth': age})
                    ds_trace_file = f'{file_dict['dir_path']}/pkl/{trace_id}/{age}.pkl' # if var instead of age, call the file as var.pkl
                    if ds_trace[age].isna().all(): # If trace contains only nan, skip it
                        continue
                else:
                    ds_trace_file = f'{file_dict['dir_path']}/pkl/{trace_id}/{file_name_}.pkl' # else use the same file name.pkl

                if 'IRHDepth' in ds_trace.columns:
                    if ds_trace['IRHDepth'].isna().all(): # If trace contains only nan, skip it
                        continue

                os.makedirs(f'{file_dict['dir_path']}/pkl/{trace_id}' , exist_ok=True)
                ds_trace.to_pickle(ds_trace_file)

                trace_metadata = f'{file_dict['dir_path']}/pkl/{trace_id}/trace_md.csv'
                if not os.path.exists(trace_metadata):
                    if not pd.isna(acq_year) and acq_year == 0:
                        acq_year = pd.NA
                    trace_md = pd.DataFrame({
                        'author': [author, 'original'],
                        'trace_id': [trace_id, trace_id_flag],
                        'institute': [institute, institute_flag],
                        'project': [project, project_flag],
                        'acq_year': [acq_year, acq_year_flag]
                    })
                    trace_md.set_index('trace_id', inplace=True)
                    trace_md.to_csv(trace_metadata)

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
            ages = {key: int(table.loc[key]['age']) for key in ds.columns if key in table.index}

            for IRH in ages:
                age = str(ages.get(IRH))
                ds_IRH = ds[IRH]
                ds_IRH = pd.DataFrame({
                    'x': ds['x'],
                    'y': ds['y'],
                    'distance': ds['distance'],
                    'IRHDepth': ds_IRH,
                })

                ds_IRH['IRHDepth'] = self.convert_col_to_num(ds_IRH['IRHDepth'])
                if self.wave_speed:
                    ds_IRH['IRHDepth'] *= self.wave_speed
                if self.firn_correction:
                    ds_IRH['IRHDepth'] += self.firn_correction

                for var in ['IceThk', 'BedElev', 'SurfElev', 'BasalUnit']:
                    if var in ds.columns:
                        ds_IRH[var] = ds[var]

                for col in ds_IRH.columns:
                    ds_IRH[col] = self.convert_col_to_Int32(ds_IRH[col])

                ds_trace_file = f'{file_dict['dir_path']}/pkl/{trace_id}/{IRH}.pkl'
                ds_IRH.to_pickle(ds_trace_file)

                author = table.loc[IRH]['author']
                institute = table.loc[IRH]['institute']
                institute_flag = 'original'
                project = table.loc[IRH]['project']
                project_flag = 'original'
                acq_year = table.loc[IRH]['acquisition_year']
                acq_year_flag = 'original'
                trace_id_flag = 'original'

                trace_metadata = f'{file_dict['dir_path']}/pkl/{trace_id}/trace_md.csv'
                if not os.path.exists(trace_metadata):
                    trace_md = pd.DataFrame({
                        'author': [author, 'original'],
                        'trace_id': [trace_id, trace_id_flag],
                        'institute': [institute, institute_flag],
                        'project': [project, project_flag],
                        'acq_year': [acq_year, acq_year_flag]
                    })
                    trace_md.set_index('trace_id', inplace=True)
                    trace_md.to_csv(trace_metadata)

    def compute_irh_density(self, trace_dir: str) -> None:
        unwanted = {'IceThk.pkl', 'SurfElev.pkl', 'BasalUnit.pkl', 'BedElev.pkl', 'IRHDensity.pkl', 'total_x.pkl', 'total_y.pkl'}
        files = [f for f in glob.glob(f"{trace_dir}/*.pkl") if os.path.basename(f) not in unwanted]
        if len(files) > 1:
            dfs = [pd.read_pickle(f) for f in files]
            dfs = pd.concat(dfs)
        elif len(files) == 1:
            dfs = pd.read_pickle(files[0])
        else:
            return

        if 'IRHDepth' in dfs.columns:
            dfs = dfs[['x','y','IRHDepth']]
            valid = dfs.dropna(subset=['IRHDepth'])
            density = valid.groupby(['x', 'y']).size().reset_index(name='IRHDensity')

            density_file = f'{trace_dir}/IRHDensity.pkl'
            density.to_pickle(density_file)

    def compute_fractional_depth(self, trace_dir: str) -> None:
        unwanted = {'IceThk.pkl', 'SurfElev.pkl', 'BasalUnit.pkl', 'BedElev.pkl', 'IRHDensity.pkl', 'total_x.pkl', 'total_y.pkl'}
        files = [f for f in glob.glob(f"{trace_dir}/*.pkl") if os.path.basename(f) not in unwanted]

        for f in files:
            df = pd.read_pickle(f)
            if 'IceThk' in df.columns and 'IRHDepth' in df.columns:
                df['FracDepth'] = df['IRHDepth'] / df['IceThk'] * 100
                df.to_pickle(f)

    def compute_total_extend(self, trace_dir: str) -> None:
        files = glob.glob(f"{trace_dir}/*.pkl")

        if len(files) > 1:
            dfs = [pd.read_pickle(f) for f in files]
            dfs = pd.concat(dfs)
        elif len(files) == 1:
            dfs = pd.read_pickle(files[0])
        else:
            return

        dfs = dfs.drop_duplicates(subset=['x', 'y'])
        total_x = dfs['x']
        total_y = dfs['y']
        total_x.to_pickle(f"{trace_dir}/total_x.pkl")
        total_y.to_pickle(f"{trace_dir}/total_y.pkl")

    def extract_vars(self, trace_dir: str) -> None:
        unwanted = {'IceThk.pkl', 'SurfElev.pkl', 'BasalUnit.pkl', 'BedElev.pkl', 'IRHDensity.pkl', 'total_x.pkl', 'total_y.pkl'}
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
                ds_var = dfs[dfs.columns.intersection(['x', 'y', 'distance', var])]
                if ds_var[var].isna().all(): # If trace contains only nan, skip it
                    continue
                var_file = f'{trace_dir}/{var}.pkl'
                ds_var.to_pickle(var_file)

    def clean_IRH_arrays(self, trace_dir: str) -> None:
        unwanted = {'IceThk.pkl', 'SurfElev.pkl', 'BasalUnit.pkl', 'BedElev.pkl', 'IRHDensity.pkl', 'total_x.pkl', 'total_y.pkl'}
        files = [f for f in glob.glob(f"{trace_dir}/*.pkl") if os.path.basename(f) not in unwanted]
        for f in files:
            df = pd.read_pickle(f)
            if 'IRHDepth' not in df.columns:
                os.remove(f)
            else:
                cols = ['x', 'y', 'distance', 'IRHDepth', 'FracDepth']
                df = df[[col for col in cols if col in df.columns]]
                df.reset_index(drop=True, inplace=True)
                df.to_pickle(f)

        wanted = {'IceThk.pkl', 'SurfElev.pkl', 'BasalUnit.pkl', 'BedElev.pkl', 'IRHDensity.pkl'}
        files = [f for f in glob.glob(f"{trace_dir}/*.pkl") if os.path.basename(f) in wanted]
        for f in files:
            df = pd.read_pickle(f)
            cols = ['x', 'y', 'distance', 'IceThk', 'SurfElev', 'BasalUnit', 'BedElev', 'IRHDensity']
            df = df[[col for col in cols if col in df.columns]]
            df.reset_index(drop=True, inplace=True)
            df.to_pickle(f)

    def _post_compilation(self, trace_dir: str) -> None:
        self.extract_vars(trace_dir)
        self.compute_irh_density(trace_dir)
        self.compute_fractional_depth(trace_dir)
        self.compute_total_extend(trace_dir)
        self.clean_IRH_arrays(trace_dir)
