import warnings
import h5py
import os
import shutil
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import xarray as xr
import glob
from pyproj import Transformer
from typing import Union, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from importlib.resources import files
from scipy.spatial.distance import cdist

class CompileDatabase:
    def __init__(self,
                 dir_list: Union[str, list[str]],
                 wave_speed: Optional[float] = None,
                 firn_correction: Optional[float] = None,
                 extract_data: bool = True,
                 hdf5: bool = True,
                 compute_distance: bool = True,
                 compute_IMBIE_basins: bool = True,
                 compute_IRH_density: bool = True,
                 set_attributes: bool = True,
                 shapefiles: bool = True,
                 remove_tmp_files: bool = True,
                 var_attrs_json: Optional[str] = None) -> None:
        self.dir_list = dir_list
        self.wave_speed = wave_speed
        self.firn_correction = firn_correction
        self.extract_data = extract_data
        self.compute_distance = compute_distance
        self.compute_IRH_density = compute_IRH_density
        self.compute_IMBIE_basins = compute_IMBIE_basins
        self.set_attributes = set_attributes
        self.hdf5 = hdf5
        self.shapefiles = shapefiles
        self.remove_tmp_files = remove_tmp_files
        imbie_path = files('anta_database.data').joinpath('ANT_Basins_IMBIE2_v1.6.shp')
        self.basins = gpd.read_file(imbie_path)
        self.var_list = ['ICE_THCK', 'BED_ELEV', 'SURF_ELEV', 'BASAL_UNIT']
        self.var_attrs_json = var_attrs_json

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

        if self.extract_data is True:
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
                    'Will start extracting data from', num_tasks, 'raw files\n'
                    '\n   ', num_workers, 'worker(s) allocated out of', cpu_count(), 'available cpus\n')

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(pool.imap_unordered(self.extract, all_files_list), total=num_tasks, desc="Processing", unit="file"):
                        pass
            else:
                for file_dict in tqdm(all_files_list, desc="Processing", unit="file"):
                    self.extract(file_dict=file_dict)

        if self.hdf5 is True:
            all_dirs = []
            for dir_ in self.dir_list:
                dirs = [d for d in glob.glob(f"{dir_}/pkl/*") if os.path.isdir(d)]
                all_dirs.extend(dirs)

            num_tasks = len(all_dirs)
            num_workers = min(num_tasks, cpus)

            print('\n',
                    'Will start creating the hdf5 for', len(all_dirs), 'traces\n'
                    '\n   ', num_workers, 'worker(s) allocated out of', cpu_count(), 'available cpus\n')

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(pool.imap_unordered(self.combine_dfs, all_dirs), total=num_tasks):
                        pass
            else:
                for trace_dir in tqdm(all_dirs, desc='Processing'):
                    self.combine_dfs(trace_dir)


        all_h5 = []
        for dir_ in self.dir_list:
            h5_files = glob.glob(f"{dir_}/h5/*.h5")
            all_h5.extend(h5_files)

        num_tasks = len(all_h5)
        num_workers = min(num_tasks, cpus)

        if self.compute_distance:

            print('\n',
                    'Computing distances along transects for', num_tasks, 'h5 files\n'
                    '\n   ', num_workers, 'worker(s) allocated out of', cpu_count(), 'available cpus\n')

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(pool.imap_unordered(self.compute_distances, all_h5), total=num_tasks):
                        pass
            else:
                for h5_file in tqdm(all_h5, desc='Processing'):
                    self.compute_distances(h5_file)

        if self.compute_IRH_density:

            print('\n',
                    'Computing IRH density in', num_tasks, 'h5 files\n'
                    '\n   ', num_workers, 'worker(s) allocated out of', cpu_count(), 'available cpus\n')

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(pool.imap_unordered(self.compute_irh_density, all_h5), total=num_tasks):
                        pass
            else:
                for h5_file in tqdm(all_h5, desc='Processing'):
                    self.compute_irh_density(h5_file)

        if self.compute_IMBIE_basins:

            print('\n',
                    'Computing IMBIE basins for', num_tasks, 'h5 files\n'
                    '\n   ', num_workers, 'worker(s) allocated out of', cpu_count(), 'available cpus\n')

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(pool.imap_unordered(self.compute_imbie_basins, all_h5), total=num_tasks):
                        pass
            else:
                for h5_file in tqdm(all_h5, desc='Processing'):
                    self.compute_imbie_basins(h5_file)

        if self.set_attributes:

            print('\n',
                    'Settings variable attributes in', num_tasks, 'h5 files\n'
                    '\n   ', num_workers, 'worker(s) allocated out of', cpu_count(), 'available cpus\n')

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(pool.imap_unordered(self.set_attrs, all_h5), total=num_tasks):
                        pass
            else:
                for h5_file in tqdm(all_h5, desc='Processing'):
                    self.set_attrs(h5_file)

            if self.remove_tmp_files:
                print('\nRemoving temporary directories ...')
                num_tasks = len(self.dir_list)
                num_workers = min(num_tasks, cpus)
                if num_workers > 1:
                    with Pool(num_workers) as pool:
                        for _ in tqdm(pool.imap_unordered(self._cleanup, self.dir_list), total=num_tasks, unit='directory'):
                            pass
                else:
                    for ds_dir in tqdm(self.dir_list, desc='Removing', unit='directory'):
                        self._cleanup(ds_dir)

        if self.shapefiles is True:
            num_tasks = len(self.dir_list)
            num_workers = min(num_tasks, cpus)

            print('\n',
                    'Will start creating the shapefiles for', len(self.dir_list), 'datasets\n'
                    '\n   ', num_workers, 'worker(s) allocated out of', cpu_count(), 'available cpus\n')

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(pool.imap_unordered(self.make_shapefile, self.dir_list), total=num_tasks):
                        pass
            else:
                for ds_dir in tqdm(self.dir_list, desc='Processing'):
                    self.make_shapefile(ds_dir)

        elapsed = time.time() - start_time
        print(f"\nCompilation completed in {elapsed:.2f} seconds")

    def _cleanup(self, dataset_dir: str) -> None:
        dir_path = f'{dataset_dir}/pkl/'

        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

    def convert_col_to_Int32(self, df):
        df = df.fillna(pd.NA)
        df = df.round(0)
        df = df.astype('Int32')
        return df

    def convert_col_to_num(self, df):
        df = pd.to_numeric(df, errors='coerce')
        return df

    def extract(self, file_dict) -> None:

        _, ext = os.path.splitext(file_dict['file'])
        raw_md = pd.read_json(f'{file_dict['dir_path']}/raw_md.json')
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

        for var in ['ICE_THCK', 'SURF_ELEV', 'BED_ELEV', 'IRH_DEPTH', 'PSX', 'DIST', 'PSY', 'lat', 'lon']:
            if var in ds.columns:
                ds[var] = self.convert_col_to_num(ds[var])

        if 'ICE_THCK' in ds.columns and 'SURF_ELEV' in ds.columns and not 'BED_ELEV' in ds.columns:
            ds['BED_ELEV'] = ds['SURF_ELEV'] - ds['ICE_THCK']
        if 'ICE_THCK' in ds.columns and 'BED_ELEV' in ds.columns and not 'SURF_ELEV' in ds.columns:
            ds['SURF_ELEV'] = ds['BED_ELEV'] + ds['ICE_THCK']
        if 'SURF_ELEV' in ds.columns and 'BED_ELEV' in ds.columns and not 'ICE_THCK' in ds.columns:
            ds['ICE_THCK'] = ds['SURF_ELEV'] - ds['BED_ELEV']

        if self.wave_speed:
            for var in ['ICE_THCK', 'BED_ELEV']:
                if var in ds.columns:
                    ds[var] *= self.wave_speed
        if self.firn_correction:
            for var in ['ICE_THCK', 'BED_ELEV']:
                if var in ds.columns:
                    ds[var] += self.firn_correction

        if 'PSX' not in ds.columns and 'PSY' not in ds.columns:
            if 'lon' in ds.columns and 'lat' in ds.columns:
                transformer = Transformer.from_proj(
                    "EPSG:4326",  # source: WGS84 (lon/lat)
                    "+proj=stere +lon_0=0 +lat_0=-90 +lat_ts=-71 +datum=WGS84 +units=m +no_defs",  # target: polar
                    always_xy=True
                )
                ds['PSX'], ds['PSY'] = transformer.transform(ds['lon'].values, ds['lat'].values)
        elif 'PSX' in ds.columns and 'PSY' in ds.columns:
            pass
        else:
            print('No coordinates found in the dataset')
            return

        if 'raw file' in raw_md.columns:
            raw_md.set_index('raw file', inplace=True)
            raw_md = raw_md.loc[file_dict['file']]

            if 'age' in raw_md.index:
                age = raw_md['age']
            else:
                age = pd.NA
            if self.wave_speed:
                ds['IRH_DEPTH'] *= self.wave_speed
            if self.firn_correction:
                ds['IRH_DEPTH'] += self.firn_correction

            ds = ds.astype({'Flight_ID': str})
            ds['Flight_ID'] = ds['Flight_ID'].str.replace(r'/\s+', '_') # Replace slashes with underscores, otherwise the paths can get messy
            ds['Flight_ID'] = ds['Flight_ID'].str.replace('/', '_')
            ds.set_index('Flight_ID', inplace=True)

            unique_flight_ids = np.unique(ds.index)
            converted = pd.to_numeric(unique_flight_ids, errors='coerce')
            converted = pd.Series(converted)
            flight_id = []
            flight_id_flag = 'original'

            if 'flight ID' in raw_md.index:
                if not pd.isna(raw_md['flight ID']):
                    flight_id = raw_md['flight ID']
                    flight_id_flag = 'not_provided'

            if 'flight_id_prefix' in raw_md.index:
                if not pd.isna(raw_md['flight ID prefix']):
                    ds.index = [f"{raw_md['flight ID prefix']}_{x}" for x in ds.index]
                    flight_id_flag = 'project_acq-year_number'

            dataset = raw_md['dataset']
            institute = raw_md['institute']
            if pd.isna(institute):
                institute = 'nan'
            institute_flag = 'original'
            project = raw_md['project']
            project_flag = 'original'
            acq_year = raw_md['acquisition year']
            acq_year_flag = 'original'

            flight_ids = np.unique(ds.index)
            if flight_id_flag == 'not_provided':
                flight_ids = [flight_id]

            def extract_year(time: str, pattern: str):
                position = pattern.find('YYYY')
                return time[position:position+4]

            for flight_id in flight_ids:
                if flight_id_flag == 'not_provided':
                    ds_trace = ds.copy()
                else:
                    ds_trace = ds.loc[flight_id].copy()

                if 'acq_year' in pattern_values.columns:
                    pattern = pattern_values.iloc[0]['acq_year']
                    if not pd.isna(pattern):
                        ds_trace['acq_year'] = ds_trace['acq_year'].apply(lambda x: extract_year(x, pattern))
                        unique_time = np.unique(ds_trace['acq_year'])
                        if len(unique_time) > 1:
                            raise ValueError(f'flight {flight_id} in {file_name_} contains {len(unique_time)} different acquisition year. Current code does not support this')
                        else:
                            position = pattern.find('YYYY')
                            acq_year = unique_time[0][position:position+4]
                        ds_trace.drop(columns='acq_year', inplace=True)

                if flight_id == 'nan':
                    flight_id = f'{project}_{acq_year}'
                    flight_id_flag = 'not_provided'

                if age is not pd.NA:
                    # age = str(age)
                    ds_trace = ds_trace.rename(columns={'IRH_DEPTH': age})
                    if institute != 'nan':
                        ds_trace_file = f'{file_dict['dir_path']}/pkl/{institute}_{flight_id}/{age}.pkl' # if var instead of age, call the file as var.pkl
                    else:
                        ds_trace_file = f'{file_dict['dir_path']}/pkl/{flight_id}/{age}.pkl' # if var instead of age, call the file as var.pkl
                else:
                    if institute != 'nan':
                        ds_trace_file = f'{file_dict['dir_path']}/pkl/{institute}_{flight_id}/{file_name_}.pkl' # else use the same file name.pkl
                    else:
                        ds_trace_file = f'{file_dict['dir_path']}/pkl/{flight_id}/{file_name_}.pkl' # else use the same file name.pkl

                # if age in ds_trace.columns:
                #     if ds_trace[age].isna().all(): # If trace contains only nan, skip it
                #         continue

                flight_dir, _ = os.path.split(ds_trace_file)
                os.makedirs(flight_dir, exist_ok=True)
                ds_trace.reset_index(drop=True, inplace=True)
                ds_trace.to_pickle(ds_trace_file)

                trace_metadata = f'{flight_dir}/metadata.csv'
                if not os.path.exists(trace_metadata):
                    if not pd.isna(acq_year) and acq_year == 0:
                        acq_year = pd.NA
                    trace_md = pd.DataFrame({
                        'dataset': [dataset, 'original'],
                        'flight_id': [flight_id, flight_id_flag],
                        'institute': [institute, institute_flag],
                        'project': [project, project_flag],
                        'acq_year': [acq_year, acq_year_flag]
                    })
                    trace_md.set_index('flight_id', inplace=True)
                    trace_md.to_csv(trace_metadata)

        # FIXME: how to handle data type flight line from raw md file now
        elif raw_md.iloc[0]['data type'] == 'flight line':
            IRH_md = pd.read_json(f'{file_dict['dir_path']}/raw_md.json')

            flight_id = file_name_
            IRH_md.set_index('IRH name', inplace=True)

            for IRH, row in IRH_md.iterrows():
                if IRH in ds.columns:
                    age = row['age']
                    ds_IRH = ds[IRH]
                    ds_IRH = pd.DataFrame({
                        'PSX': ds['PSX'],
                        'PSY': ds['PSY'],
                        age: ds_IRH,
                    })

                    ds_IRH[age] = self.convert_col_to_num(ds_IRH[age])
                    if self.wave_speed:
                        ds_IRH[age] *= self.wave_speed
                    if self.firn_correction:
                        ds_IRH[age] += self.firn_correction

                    for var in self.var_list:
                        if var in ds.columns:
                            ds_IRH[var] = ds[var]

                    dataset = row['dataset']
                    institute = row['institute']
                    if pd.isna(institute):
                        institute = 'nan'
                    institute_flag = 'original'
                    project = row['project']
                    project_flag = 'original'
                    acq_year = row['acquisition year']
                    acq_year_flag = 'original'
                    flight_id_flag = 'original'

                    if institute != 'nan':
                        ds_trace_file = f'{file_dict['dir_path']}/pkl/{institute}_{flight_id}/{IRH}.pkl'
                    else:
                        ds_trace_file = f'{file_dict['dir_path']}/pkl/{flight_id}/{IRH}.pkl'
                    flight_dir, _ = os.path.split(ds_trace_file)
                    os.makedirs(flight_dir, exist_ok=True)
                    ds_IRH.reset_index(drop=True, inplace=True)
                    ds_IRH.to_pickle(ds_trace_file)

                    trace_metadata = f'{flight_dir}/metadata.csv'
                    if not os.path.exists(trace_metadata):
                        trace_md = pd.DataFrame({
                            'dataset': [dataset, 'original'],
                            'flight_id': [flight_id, flight_id_flag],
                            'institute': [institute, institute_flag],
                            'project': [project, project_flag],
                            'acq_year': [acq_year, acq_year_flag]
                        })
                        trace_md.set_index('flight_id', inplace=True)
                        trace_md.to_csv(trace_metadata)


    def combine_dfs(self, trace_dir: str) -> None:
        files = glob.glob(f"{trace_dir}/*.pkl")
        if not files:
            return

        if len(files) > 1:
            dfs = [pd.read_pickle(f) for f in files]

            dfs_prep = []
            for df in dfs:
                df = df.set_index(['PSX', 'PSY'])
                df = df[~df.index.duplicated(keep='first')]
                dfs_prep.append(df)

            merged_df = pd.concat(dfs_prep, axis=1, join='outer')
            merged_df.reset_index(inplace=True)
            merged_df.columns = merged_df.columns.astype(str)

            for col in self.var_list:
                matching_cols = [c for c in merged_df.columns if col.lower() in c.lower()]
                if matching_cols:
                    temp_col = f"temp_{col}"
                    merged_df[temp_col] = merged_df[matching_cols].bfill(axis=1).iloc[:, 0]
                    merged_df = merged_df.drop(columns=matching_cols)
                    merged_df = merged_df.rename(columns={temp_col: col})

        else:
            merged_df = pd.read_pickle(files[0])

        x = merged_df['PSX']
        y = merged_df['PSY']
        data_vars = {
            'x': (['point'], x),
            'y': (['point'], y),
        }

        ages = [col for col in merged_df.columns if str(col).isdigit()]

        for var in self.var_list:
            if var in merged_df.columns:
                data_vars[var] = (['point'], merged_df[var].values)

        pkl_dir, trace_id = os.path.split(trace_dir)
        dataset_dir, pkl = os.path.split(pkl_dir)
        trace_md = pd.read_csv(f'{trace_dir}/metadata.csv')
        institute = str(trace_md.iloc[0]['institute'])
        flight_id = str(trace_md.iloc[0]['flight_id'])
        ds = xr.Dataset(
            coords={
                'point': np.arange(len(merged_df)),
            },
            data_vars=data_vars,
            attrs={
                'dataset': str(trace_md.iloc[0]['dataset']),
                'institute': institute,
                'project': str(trace_md.iloc[0]['project']),
                'acq_year': str(trace_md.iloc[0]['acq_year']),
                'flight_id': str(trace_md.iloc[0]['flight_id']),
            }
        )
        point_shape = len(merged_df)
        if point_shape < 1e6:
            chunk_size_point = min(100, point_shape)
        elif point_shape < 1e7:
            chunk_size_point = 1000
        else:
            chunk_size_point = 10000

        encoding = {
            'x': {'zlib': True, 'complevel': 1},
            'y': {'zlib': True, 'complevel': 1},
        }

        if ages:
            sorted_ages = sorted(ages, key=lambda x: int(x))
            sorted_ages_int = [int(age) for age in sorted_ages]
            depth_data = merged_df[sorted_ages]
            data_vars['IRH_DEPTH'] = (['point', 'age'], depth_data)  # Use 'age' as the dimension name
            ds = ds.assign_coords(age=sorted_ages_int)
            ds['IRH_DEPTH'] = (['point', 'age'], depth_data)

            if 'IRH_DEPTH' in ds.variables:
                ds['IRH_DEPTH'] = ds['IRH_DEPTH'].chunk({'age': 1, 'point': chunk_size_point})
                encoding['IRH_DEPTH'] = {'zlib': True, 'complevel': 1, 'chunksizes': (chunk_size_point, 1)}

            md = pd.read_json(f'{dataset_dir}/raw_md.json')
            if 'age' in md.columns and 'age uncertainty' in md.columns:
                age_unc = md[['age', 'age uncertainty']]
                age_unc = age_unc.drop_duplicates()
                # age_unc = age_unc.astype(int)
                age_unc = age_unc.set_index('age')
                age_unc = age_unc.loc[sorted_ages_int]['age uncertainty'].values
                age_uncertainties = xr.DataArray(
                        data=age_unc,
                        dims=['age'],
                        coords={'age': sorted_ages_int},
                        attrs={'units': 'years'}
                    )
                ds['age_uncertainty'] = age_uncertainties


        for var in self.var_list:
            if var in ds.variables:
                ds[var] = ds[var].chunk({'point': chunk_size_point})
                encoding[var] = {'zlib': True, 'complevel': 1, 'chunksizes': (chunk_size_point)}

        h5_dir = os.path.join(dataset_dir, 'h5')
        os.makedirs(h5_dir, exist_ok=True)
        if institute != 'nan':
            h5_file = f'{h5_dir}/{institute}_{flight_id}.h5'
        else:
            h5_file = f'{h5_dir}/{flight_id}.h5'

        ds.to_netcdf(h5_file, engine='h5netcdf', encoding=encoding, mode='w')

    def compute_fractional_depth(self, h5_file: str) -> None:
        with h5py.File(h5_file, 'a') as f:
            with xr.open_dataset(f, engine='h5netcdf') as ds:

                if 'ICE_THCK' in ds.variables and 'IRH_DEPTH' in ds.variables:

                    ds['IRH_FRAC_DEPTH'] = ds.IRH_DEPTH/ds.ICE_THCK*100

                    point_shape = len(ds.point.values)
                    if point_shape < 1e6:
                        chunk_size_point = min(100, point_shape)
                    elif point_shape < 1e7:
                        chunk_size_point = 1000
                    else:
                        chunk_size_point = 10000
                    ds['IRH_FRAC_DEPTH'] = ds['IRH_DEPTH'].chunk({'age': 1, 'point': chunk_size_point})
                    encoding = {'IRH_FRAC_DEPTH': {'zlib': True, 'complevel': 1, 'chunksizes': (chunk_size_point, 1)}}

                    ds = ds[list(ds.variables)]
                    ds.to_netcdf(h5_file, engine='h5netcdf', mode='a', encoding=encoding)

    def compute_irh_density(self, h5_file: str) -> None:
        with h5py.File(h5_file, 'a') as f:
            with xr.open_dataset(f, engine='h5netcdf') as ds:

                if 'IRH_DEPTH' in ds.variables:
                    ds['IRH_DENS'] = (~np.isnan(ds['IRH_DEPTH'])).sum(dim='age')

                    point_shape = len(ds.point.values)
                    if point_shape < 1e6:
                        chunk_size_point = min(100, point_shape)
                    elif point_shape < 1e7:
                        chunk_size_point = 1000
                    else:
                        chunk_size_point = 10000

                    ds['IRH_DENS'] = ds['IRH_DENS'].chunk({'point': chunk_size_point})
                    encoding = {'IRH_DENS': {'zlib': True, 'complevel': 1, 'chunksizes': (chunk_size_point)}}

                    ds = ds[list(ds.variables)]
                    ds.to_netcdf(h5_file, engine='h5netcdf', mode='a', encoding=encoding)

    def order_points(self, h5_file: str) -> None:
        with h5py.File(h5_file, 'a') as f:
            with xr.open_dataset(f, engine='h5netcdf') as ds:
                x = ds['x'].values
                y = ds['y'].values
                coords = np.column_stack((x, y))

                distances = cdist(coords, coords, metric='euclidean')

                np.random.seed(0)
                start_idx = 0
                ordered_indices = [start_idx]

                remaining_indices = set(range(len(coords)))
                remaining_indices.remove(start_idx)

                while remaining_indices:
                    last_idx = ordered_indices[-1]
                    nearest_idx = min(remaining_indices, key=lambda idx: distances[last_idx, idx])
                    ordered_indices.append(nearest_idx)
                    remaining_indices.remove(nearest_idx)

                ds = ds.isel(point=ordered_indices)
                ds = ds.assign_coords(point=np.arange(len(ds.point)))

                ds = ds[list(ds.variables)]
                ds.to_netcdf(h5_file, engine='h5netcdf', mode='a')

    def compute_distances(self, h5_file: str) -> None:
        with h5py.File(h5_file, 'a') as f:
            with xr.open_dataset(f, engine='h5netcdf') as ds:
                if 'distance' not in ds.variables:
                    x = ds['x'].values
                    y = ds['y'].values

                    coords = np.column_stack((x, y))

                    diff = np.diff(coords, axis=0)
                    DISTs = np.sqrt(np.einsum('ij,ij->i', diff, diff))
                    cumulative_DIST = np.concatenate([[0], np.cumsum(DISTs)])

                    ds['distance'] = (['point'], cumulative_DIST)

                    point_shape = len(ds.point.values)
                    if point_shape < 1e6:
                        chunk_size_point = min(100, point_shape)
                    elif point_shape < 1e7:
                        chunk_size_point = 1000
                    else:
                        chunk_size_point = 10000

                    ds['distance'] = ds['distance'].chunk({'point': chunk_size_point})
                    encoding = {'distance': {'zlib': True, 'complevel': 1, 'chunksizes': (chunk_size_point)}}

                    ds = ds[list(ds.variables)]
                    ds.to_netcdf(h5_file, engine='h5netcdf', mode='a', encoding=encoding)

    def compute_imbie_basins(self, h5_file: str) -> None:

        with h5py.File(h5_file, 'a') as f:
            with xr.open_dataset(f, engine='h5netcdf') as ds:
                x = ds['x'].values
                y = ds['y'].values

                coords = np.column_stack((x, y))
                geometry = [Point(xy) for xy in zip(x, y)]
                points = gpd.GeoDataFrame(coords, geometry=geometry, crs=self.basins.crs)
                joined = gpd.sjoin(points, self.basins, how="inner", predicate="within")
                lookup_df = joined[['Subregion', 'Regions']].drop_duplicates()
                lookup_df.reset_index(drop=True, inplace=True)
                basin_mapping = dict(zip(lookup_df['Subregion'], lookup_df['Regions']))
                import json
                ds.attrs['basins'] = json.dumps(basin_mapping)

                ds = ds[list(ds.variables)]
                ds.to_netcdf(h5_file, engine='h5netcdf', mode='a')

    def set_attrs(self, h5_file: str) -> None:

        with h5py.File(h5_file, 'a') as f:
            with xr.open_dataset(f, engine='h5netcdf') as ds:
                if self.var_attrs_json:
                    var_attrs = pd.read_json(self.var_attrs_json)
                    var_attrs.set_index('variable', inplace=True)

                    for var, row in var_attrs.iterrows():
                        if var in ds.variables:
                            ds[var].attrs['units'] = row['units']
                            ds[var].attrs['long_name'] = row['long_name']
                            ds[var].attrs['description'] = row['description']
                        elif var in ds.coords:
                            ds[var].attrs['units'] = row['units']
                            ds[var].attrs['long_name'] = row['long_name']
                            ds[var].attrs['description'] = row['description']

                ds = ds[list(ds.variables)]
                ds.to_netcdf(h5_file, engine='h5netcdf', mode='a')

    def make_shapefile(self, dataset_dir: str) -> None:
        h5files = glob.glob(f'{dataset_dir}/h5/*.h5')

        all_dfs = []
        for h5f in h5files:
            with h5py.File(h5f, 'r') as f:
                x = f["x"][:]
                y = f["y"][:]
                N = f["IRH_DENS"][:]
                depth = f["IRH_DEPTH"][:, -1]
                flight_id = f.attrs["flight_id"]

            df = pd.DataFrame({
                "x": x,
                "y": y,
                "N": N,
                "DL": depth,
                "flight_id": flight_id
            })

            all_dfs.append(df)

        df_all = pd.concat(all_dfs, ignore_index=True)

        geometry = [Point(xy) for xy in zip(df_all.x, df_all.y)]
        gdf = gpd.GeoDataFrame(df_all, geometry=geometry, crs="EPSG:3031")
        combined_gdf = gpd.GeoDataFrame(gdf, crs="EPSG:3031")
        dataset = os.path.basename(os.path.normpath(dataset_dir))
        shape_dir = f"{dataset_dir}/shap/"
        os.makedirs(shape_dir, exist_ok=True)
        combined_gdf.to_file(f"{shape_dir}/{dataset}.shp")
