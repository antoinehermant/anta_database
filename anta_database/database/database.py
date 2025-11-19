import os
import sqlite3
import pandas as pd
import numpy as np
import xarray as xr
import h5py
from typing import Union, List, Dict, Tuple, Optional, Generator

from anta_database.plotting.plotting import Plotting

class Database:
    def __init__(self, database_dir: str, file_db: str = 'AntADatabase.db', include_BEDMAP: bool = False, max_displayed_flight_ids: Optional[int] = 50) -> None:
        self.db_dir = database_dir
        self.file_db = file_db
        self.file_db_path = os.path.join(self.db_dir, file_db)
        self.md = None
        self._plotting = None
        self.max_displayed_flight_ids = max_displayed_flight_ids
        self.include_BM = include_BEDMAP
        self.excluded = {
            'dataset': [],
            'institute': [],
            'project': [],
            'acq_year': [],
            'age': [],
            'var': [],
            'flight_id': [],
            'region': [],
            'IMBIE_basin': [],
            'radar_instrument': [],
        }

    def _build_query_and_params(self,
                                age: Optional[Union[str, List[str]]] = None,
                                var: Optional[Union[str, List[str]]] = None,
                                dataset: Optional[Union[str, List[str]]] = None,
                                institute: Optional[Union[str, List[str]]] = None,
                                project: Optional[Union[str, List[str]]] = None,
                                acq_year: Optional[Union[str, List[str]]] = None,
                                line: Optional[Union[str, List[str]]] = None,
                                region: Optional[Union[str, List[str]]] = None,
                                IMBIE_basin: Optional[Union[str, List[str]]] = None,
                                radar_instrument: Optional[Union[str, List[str]]] = None,
                                select_clause='') -> Tuple[str, List[Union[str, int]]]:
        """
        Helper method to build the SQL query and parameters for filtering.
        Returns the query string and parameters list.
        """
        query = f'''
            SELECT {select_clause}
            FROM datasets d
            JOIN sources a ON d.dataset = a.id
        '''
        conditions = []
        params = []
        for field, column in [
                (age, 'd.age'),
                (var, 'd.var'),
                (dataset, 'a.name'),
                (institute, 'd.institute'),
                (project, 'd.project'),
                (acq_year, 'd.acq_year'),
                (line, 'd.flight_id'),
                (region, 'd.region'),
                (IMBIE_basin, 'd.IMBIE_basin'),
                (radar_instrument, 'd.radar_instrument')
        ]:
            if field is not None:
                if isinstance(field, list):
                    # For lists, use IN for exact matches, or LIKE for wildcards
                    like_conditions = []
                    in_values = []
                    range_conditions = []
                    for item in field:
                        if '%' in item:
                            like_conditions.append(f"{column} LIKE ?")
                            params.append(item)
                        elif self._is_range_query(item):
                            op, val = self._parse_range_query(item)
                            range_conditions.append(f"{column} {op} ?")
                            params.append(val)
                        else:
                            in_values.append(item)
                    if like_conditions:
                        conditions.append('(' + ' OR '.join(like_conditions) + ')')
                    if in_values:
                        placeholders = ','.join(['?'] * len(in_values))
                        conditions.append(f"{column} IN ({placeholders})")
                        params.extend(in_values)
                    if range_conditions:
                        conditions.append('(' + ' OR '.join(range_conditions) + ')')
                else:
                    if '%' in field:
                        conditions.append(f"{column} LIKE ?")
                        params.append(field)
                    elif self._is_range_query(field):
                        op, val = self._parse_range_query(field)
                        conditions.append(f"{column} {op} ?")
                        params.append(val)
                    else:
                        conditions.append(f"{column} = ?")
                        params.append(field)

        for field, column in [
                ('age', 'd.age'),
                ('var', 'd.var'),
                ('dataset', 'a.name'),
                ('institute', 'd.institute'),
                ('project', 'd.project'),
                ('acq_year', 'd.acq_year'),
                ('flight_id', 'd.flight_id'),
                ('region', 'd.region'),
                ('IMBIE_basin', 'd.IMBIE_basin'),
                ('radar_instrument', 'd.radar_instrument')
        ]:
            if self.excluded[field]:
                not_like_conditions = []
                not_in_values = []
                not_range_conditions = []
                for item in self.excluded[field]:
                    if '%' in item:
                        not_like_conditions.append(f"{column} NOT LIKE ?")
                        params.append(item)
                    elif self._is_range_query(item):
                        op, val = self._parse_range_query(item)
                        inverted_op = self._invert_range_operator(op)
                        not_range_conditions.append(f"{column} {inverted_op} ?")
                        params.append(val)
                    else:
                        not_in_values.append(item)
                if not_like_conditions:
                    conditions.append('(' + ' AND '.join(not_like_conditions) + ')')
                if not_in_values:
                    if len(not_in_values) == 1:
                        conditions.append(f"{column} != ?")
                        params.append(not_in_values[0])
                    else:
                        placeholders = ','.join(['?'] * len(not_in_values))
                        conditions.append(f"{column} NOT IN ({placeholders})")
                        params.extend(not_in_values)
                if not_range_conditions:
                    conditions.append('(' + ' AND '.join(not_range_conditions) + ')')

        if not self.include_BM:
            conditions.append("a.name NOT LIKE ?")
            params.append('%BEDMAP%')

        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        query += ' ORDER BY CAST(d.age AS INTEGER) ASC'
        return query, params

    def _is_year_in_range(self, year: str, range_str: str) -> bool:
        """Check if a year is within a stored range (e.g., '2016-2020')."""
        if '-' not in range_str:
            return year == range_str  # Exact match for non-range values
        start, end = map(int, range_str.split('-'))
        return int(year) >= start and int(year) <= end

    def _is_range_value(self, s: str) -> bool:
        """Check if the string is a range value (e.g., '1999-2003')."""
        return '-' in s and all(part.isdigit() for part in s.split('-'))

    def _parse_range_value(self, s: str) -> Tuple[int, int]:
        """Parse a range value string into start and end years."""
        start, end = s.split('-')
        return int(start), int(end)

    def _is_range_query(self, s: str) -> bool:
        """Check if the string is a range query (e.g., '>2000', '<=2010')."""
        return s.startswith(('>', '<', '=')) and any(c.isdigit() for c in s)

    def _parse_range_query(self, s: str) -> Tuple[str, Union[str, int]]:
        """Parse a range query string into operator and value."""
        op = ''.join(c for c in s if c in ('>', '<', '='))
        val = s[len(op):]
        try:
            val = int(val)
        except ValueError:
            pass
        return op, val

    def _invert_range_operator(self, op: str) -> str:
        """Invert the range operator for NOT conditions."""
        invert_map = {
            '>': '<=',
            '<': '>=',
            '>=': '<',
            '<=': '>',
            '=': '!=',
        }
        return invert_map.get(op, op)

    def filter_out(
            self,
            age: Optional[Union[str, List[str]]] = None,
            var: Optional[Union[str, List[str]]] = None,
            dataset: Optional[Union[str, List[str]]] = None,
            institute: Optional[Union[str, List[str]]] = None,
            project: Optional[Union[str, List[str]]] = None,
            acq_year: Optional[Union[str, List[str]]] = None,
            flight_id: Optional[Union[str, List[str]]] = None,
            region: Optional[Union[str, List[str]]] = None,
            IMBIE_basin: Optional[Union[str, List[str]]] = None,
            radar_instrument: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """
        Add values to exclude from the query results.
        Example: filter_out(dataset='Cavitte', project='OldProject')
        """
        # Map arguments to their corresponding fields
        field_mapping = {
            'age': age,
            'var': var,
            'dataset': dataset,
            'institute': institute,
            'project': project,
            'acq_year': acq_year,
            'flight_id': flight_id,
            'region': region,
            'IMBIE_basin': IMBIE_basin,
            'radar_instrument': radar_instrument,
        }

        for field, value in field_mapping.items():
            if value is not None:
                if isinstance(value, list):
                    self.excluded[field] = value
                else:
                    self.excluded[field] = [value]
            else:
                self.excluded[field] = []

    def _get_file_metadata(self, file_path) -> Dict:
        """
        Helper method to build the SQL query and parameters for filtering.
        Returns the query string and parameters list.
        """
        select_clause = 'a.name, \
                        a.citation, \
                        a.DOI_dataset, \
                        a.DOI_publication, \
                        d.institute, \
                        d.project, \
                        d.acq_year, \
                        d.age, \
                        d.age_unc, \
                        d.var, \
                        d.flight_id, \
                        d.region, \
                        d.IMBIE_basin, \
                        d.radar_instrument \
        '
        query = f'''
            SELECT {select_clause}
            FROM datasets d
            JOIN sources a ON d.dataset = a.id
        '''
        conditions = []
        params = []

        if file_path is not None:
            if isinstance(file_path, list):
                placeholders = ','.join(['?'] * len(file_path))
                conditions.append(f'd.file_path IN ({placeholders})')
                params.extend(file_path)
            else:
                conditions.append(f'd.file_path = ?')
                params.append(file_path)

        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)

        conn = sqlite3.connect(self.file_db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        metadata = {
            'dataset': results[0][0],
            'var': results[0][9],
            'age': results[0][7],
            'flight_id': results[0][10],
            'DOI_dataset': results[0][2],
            'DOI_publication': results[0][3],
            'institute': results[0][4],
            'project': results[0][5],
            'acq_year': results[0][6],
            'age_unc': results[0][8],
            'region': results[0][11],
            'IMBIE_basin': results[0][12],
            'radar_instrument': results[0][13],
            'reference': results[0][1],
            'file_path': file_path,
            'database_path': self.db_dir,
            'file_db': self.file_db,
        }
        return metadata

    def query(self,
              age: Optional[Union[str, List[str]]] = None,
              var: Optional[Union[str, List[str]]] = None,
              dataset: Optional[Union[str, List[str]]] = None,
              institute: Optional[Union[str, List[str]]] = None,
              project: Optional[Union[str, List[str]]] = None,
              acq_year: Optional[Union[str, List[str]]] = None,
              flight_id: Optional[Union[str, List[str]]] = None,
              region: Optional[Union[str, List[str]]] = None,
              IMBIE_basin: Optional[Union[str, List[str]]] = None,
              radar_instrument: Optional[Union[str, List[str]]] = None,
              retain_query: Optional[bool] = True,
              ) -> 'MetadataResult':

        select_clause = 'a.name, \
                        a.citation, \
                        a.DOI_dataset, \
                        a.DOI_publication, \
                        d.institute, \
                        d.project, \
                        d.acq_year, \
                        d.age, \
                        d.age_unc, \
                        d.var, \
                        d.flight_id, \
                        d.region, \
                        d.IMBIE_basin, \
                        d.radar_instrument \
        '
        query, params = self._build_query_and_params(age, var, dataset, institute, project, acq_year, flight_id, region, IMBIE_basin, radar_instrument, select_clause)

        conn = sqlite3.connect(self.file_db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        metadata = {
            'dataset': [],
            'institute': [],
            'project': [],
            'acq_year': [],
            'age': [],
            'age_unc': [],
            'var': [],
            'reference': [],
            'DOI_dataset': [],
            'DOI_publication': [],
            'flight_id': [],
            'region': [],
            'IMBIE_basin': [],
            'radar_instrument': [],
            '_query_params': {'dataset': dataset, 'institute': institute, 'project': project, 'acq_year': acq_year, 'age': age, 'var': var, 'flight_id': flight_id, 'region': region, 'IMBIE_basin': IMBIE_basin, 'radar_instrument': radar_instrument},
            '_filter_params': {'dataset': self.excluded['dataset'], 'institute': self.excluded['institute'], 'project': self.excluded['project'], 'acq_year': self.excluded['acq_year'], 'age': self.excluded['age'], 'var': self.excluded['var'], 'flight_id': self.excluded['flight_id'], 'region': self.excluded['region'], 'IMBIE_basin': self.excluded['IMBIE_basin'], 'radar_instrument': self.excluded['radar_instrument']},
            'database_path': self.db_dir,
            'file_db': self.file_db,
        }
        ages_list = []
        ages_unc_list = []
        vars_list = []
        institutes_list = []
        projects_list = []
        acq_years_list = []
        radar_list = []
        for dataset_name, citations, DOI_dataset, DOI_publication, institutes, projects, acq_years, ages, ages_unc, vars, flight_id, regions, basins, radar_instruments in results:
            metadata['dataset'].append(dataset_name)
            metadata['reference'].append(citations)
            metadata['DOI_dataset'].append(DOI_dataset)
            metadata['DOI_publication'].append(DOI_publication)
            metadata['flight_id'].append(flight_id)
            metadata['region'].append(regions)
            metadata['IMBIE_basin'].append(basins)
            # Check if the age is numeric
            if ages is not None and ages.isdigit():
                ages_list.append(int(ages))
                if ages_unc is not None and ages_unc.isdigit():
                    ages_unc_list.append(int(ages_unc))
                else:
                    ages_unc_list.append('-')
            if vars is not None:
                vars_list.append(vars)
            if institutes is not None:
                institutes_list.append(institutes)
            else:
                institutes_list.append('-')
            if radar_instruments is not None:
                radar_list.append(radar_instruments)
            else:
                radar_list.append('-')
            if projects is not None:
                projects_list.append(projects)
            else:
                projects_list.append('-')
            if acq_years is not None:
                acq_years_list.append(acq_years)
            else:
                acq_years_list.append('-')


        paired = sorted(zip(ages_list, ages_unc_list), key=lambda x: x[0])

        unique_pairs = []
        seen = set()
        for age, unc in paired:
            if age not in seen:
                seen.add(age)
                unique_pairs.append((age, unc))

        sorted_ages, sorted_age_unc = zip(*unique_pairs) if unique_pairs else ([], [])
        metadata['age'] = [str(age) for age in sorted_ages]
        metadata['age_unc'] = [str(age_unc) for age_unc in sorted_age_unc]
        metadata['var'] = sorted(set(vars_list))
        metadata['institute'] = sorted(set(institutes_list))
        metadata['project'] = sorted(set(projects_list))
        metadata['acq_year'] = sorted(set(acq_years_list))
        metadata['dataset'] = list(dict.fromkeys(metadata['dataset']))
        metadata['reference'] = list(dict.fromkeys(metadata['reference']))
        metadata['DOI_dataset'] = list(dict.fromkeys(metadata['DOI_dataset']))
        metadata['DOI_publication'] = list(dict.fromkeys(metadata['DOI_publication']))
        metadata['flight_id'] = list(set(metadata['flight_id']))
        metadata['region'] = list(set(metadata['region']))
        metadata['IMBIE_basin'] = list(set(metadata['IMBIE_basin']))
        metadata['radar_instrument'] = sorted(set(radar_list))

        if retain_query:
            self.md = metadata

        return MetadataResult(metadata, self.max_displayed_flight_ids)

    def _get_file_paths_from_metadata(self, metadata) -> List:

        query_params = metadata['_query_params']
        age = query_params.get('age')
        var = query_params.get('var')
        dataset = query_params.get('dataset')
        institute = query_params.get('institute')
        project = query_params.get('project')
        acq_year = query_params.get('acq_year')
        line = query_params.get('flight_id')
        region = query_params.get('region')
        basin = query_params.get('IMBIE_basin')
        radar = query_params.get('radar_instrument')

        select_clause = 'd.file_path'
        query, params = self._build_query_and_params(age, var, dataset, institute, project, acq_year, line, region, basin, radar, select_clause)

        conn = sqlite3.connect(self.file_db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        file_paths = [row[0] for row in cursor.fetchall()]
        conn.close()

        return file_paths

    def get_files(
        self,
        metadata: Union[None, Dict, 'MetadataResult'] = None,
        data_dir: Union[None, str] = None,
    ):

        md = metadata or self.md
        if not md:
            print('Please provide metadata of the files you want to generate the data from. Exiting...')
            return

        data_dir = data_dir or self.db_dir
        if not data_dir:
            print('No data directory provided. Exiting...')
            return

        file_paths = self._get_file_paths_from_metadata(metadata=md)
        file_paths = np.unique(file_paths)
        full_paths = [os.path.join(data_dir, fp) for fp in file_paths]


        return full_paths

    def data_generator(
        self,
        metadata: Union[None, Dict, 'MetadataResult'] = None,
        data_dir: Optional[str] = None,
        downscale_factor: Optional[str] = None,
    ) -> Generator[Tuple[pd.DataFrame, Dict]]:
        """
        Generates xarray Datasets from HDF5 files, one at a time, with lazy loading.

        Args:
            metadata: Metadata for filtering files.
            data_dir: Directory containing the data files.
            vars_to_load: List of variables to load from each file.

        Yields:
            Tuple[xr.Dataset, Dict]: A lazy-loaded xarray Dataset and its metadata.
        """
        # Resolve metadata
        md = metadata or self.md
        if not md:
            print('Please provide metadata of the files you want to generate the data from. Exiting...')
            return

        # Resolve data directory
        data_dir = data_dir or self.db_dir
        if not data_dir:
            print('No data directory provided. Exiting...')
            return

        file_paths = self._get_file_paths_from_metadata(metadata=md)
        file_paths = np.unique(file_paths) # Be carefull as many pointers point to the same file

        for file_path in file_paths:
            full_path = os.path.join(data_dir, file_path)
            file_md = self._get_file_metadata(file_path)
            for var in md['var']:
                if var == 'IRH_DEPTH':
                    for age in md['age']:
                        ds = h5py.File(full_path, 'r')
                        irh_values = ds['IRH_AGE'][:]
                        irh_index = np.where(irh_values == int(age))[0]
                        if len(irh_index) == 0:
                            continue
                        irh_index = irh_index[0]
                        df = pd.DataFrame({'PSX': ds['PSX'][::downscale_factor],
                                        'PSY': ds['PSY'][::downscale_factor],
                                        'Distance': ds['Distance'][::downscale_factor],
                                        var: ds[var][::downscale_factor, irh_index]})

                        metadata = {
                            'dataset': file_md['dataset'],
                            'var': var,
                            'age': age,
                            'flight_id': file_md['flight_id'],
                            'institute': file_md['institute'],
                            'project': file_md['project'],
                            'acq_year': file_md['acq_year'],
                            'age_unc': file_md['age'],
                            'reference': file_md['reference'],
                            'DOI_dataset': file_md['DOI_dataset'],
                            'DOI_publication': file_md['DOI_publication'],
                            'flight_id': file_md['flight_id'],
                            'region': file_md['region'],
                            'IMBIE_basin': file_md['IMBIE_basin'],
                            'radar_instrument': file_md['radar_instrument'],
                        }

                        yield df, metadata

                else:
                    ds = h5py.File(full_path, 'r')
                    df = pd.DataFrame({'PSX': ds['PSX'][::downscale_factor],
                                    'PSY': ds['PSY'][::downscale_factor],
                                    'Distance': ds['Distance'][::downscale_factor],
                                    var: ds[var][::downscale_factor]})

                    metadata = {
                        'dataset': file_md['dataset'],
                        'var': var,
                        'age': None,
                        'flight_id': file_md['flight_id'],
                        'institute': file_md['institute'],
                        'project': file_md['project'],
                        'acq_year': file_md['acq_year'],
                        'age_unc': file_md['age'],
                        'reference': file_md['reference'],
                        'DOI_dataset': file_md['DOI_dataset'],
                        'DOI_publication': file_md['DOI_publication'],
                        'flight_id': file_md['flight_id'],
                        'region': file_md['region'],
                        'IMBIE_basin': file_md['IMBIE_basin'],
                        'radar_instrument': file_md['radar_instrument'],
                    }

                    yield df, metadata

    def return_dataset(
        self,
        metadata: Union[None, Dict, 'MetadataResult'] = None,
        data_dir: Optional[str] = None,
        downscale_factor: Optional[str] = None,
    ) -> Generator[Tuple[pd.DataFrame, Dict]]:
        """
        Generates xarray Datasets from HDF5 files, one at a time, with lazy loading.

        Args:
            metadata: Metadata for filtering files.
            data_dir: Directory containing the data files.
            vars_to_load: List of variables to load from each file.

        Yields:
            Tuple[xr.Dataset, Dict]: A lazy-loaded xarray Dataset and its metadata.
        """
        # Resolve metadata
        md = metadata or self.md
        if not md:
            print('Please provide metadata of the files you want to generate the data from. Exiting...')
            return

        # Resolve data directory
        data_dir = data_dir or self.db_dir
        if not data_dir:
            print('No data directory provided. Exiting...')
            return

        if len(md['var']) > 1:
            print(f'WARNING: you requested multiple variables ({md['var']}), this will return individual dataframes for each variable in each transect.')

        file_paths = self._get_file_paths_from_metadata(metadata=md)
        file_paths = np.unique(file_paths) # Be carefull as many pointers point to the same file

        for file_path in file_paths:
            full_path = os.path.join(data_dir, file_path)
            file_md = self._get_file_metadata(file_path)
            for var in md['var']:
                if var == 'IRH_DEPTH':
                    for age in md['age']:
                        with h5py.File(full_path, 'r') as ds:
                            irh_values = ds['age'][:]
                            irh_index = np.where(irh_values == int(age))[0]
                            if len(irh_index) == 0:
                                continue
                            irh_index = irh_index[0]
                            df = pd.DataFrame({'x': ds['x'][::downscale_factor],
                                            'y': ds['y'][::downscale_factor],
                                            'distance': ds['distance'][::downscale_factor],
                                            var: ds[var][::downscale_factor, irh_index]})

                            metadata = {
                                'dataset': file_md['dataset'],
                                'var': var,
                                'age': age,
                                'flight_id': file_md['flight_id'],
                                'institute': file_md['institute'],
                                'project': file_md['project'],
                                'acq_year': file_md['acq_year'],
                                'age_unc': file_md['age'],
                                'reference': file_md['reference'],
                                'DOI_dataset': file_md['DOI_dataset'],
                                'DOI_publication': file_md['DOI_publication'],
                                'flight_id': file_md['flight_id'],
                                'region': file_md['region'],
                                'IMBIE_basin': file_md['IMBIE_basin'],
                                'radar_instrument': file_md['radar_instrument'],
                            }

                            yield df, metadata

                    else:
                        ds = h5py.File(full_path, 'r')
                        df = pd.DataFrame({'PSX': ds['PSX'][::downscale_factor],
                                        'PSY': ds['PSY'][::downscale_factor],
                                        'Distance': ds['Distance'][::downscale_factor],
                                        var: ds[var][::downscale_factor]})

                        metadata = {
                            'dataset': file_md['dataset'],
                            'var': var,
                            'age': None,
                            'flight_id': file_md['flight_id'],
                            'institute': file_md['institute'],
                            'project': file_md['project'],
                            'acq_year': file_md['acq_year'],
                            'age_unc': file_md['age'],
                            'reference': file_md['reference'],
                            'DOI_dataset': file_md['DOI_dataset'],
                            'DOI_publication': file_md['DOI_publication'],
                            'flight_id': file_md['flight_id'],
                            'region': file_md['region'],
                            'IMBIE_basin': file_md['IMBIE_basin'],
                            'radar_instrument': file_md['radar_instrument'],
                        }

                        yield df, metadata

    @property
    def plot(self):
        if self._plotting is None:
            self._plotting = Plotting(self)
        return self._plotting


class MetadataResult:
    def __init__(self, metadata, max_displayed_flight_ids):
        self._metadata = metadata
        self.max_displayed_flight_ids = max_displayed_flight_ids

    def __getitem__(self, key):
        return self._metadata[key]

    def __repr__(self):
        """Pretty-print the metadata, truncating long flight_id lists."""
        md = self._metadata
        output = []
        output.append("Metadata from query:")

        flight_ids = md['flight_id']
        if len(flight_ids) > self.max_displayed_flight_ids:
            first_20 = flight_ids[:self.max_displayed_flight_ids//2]
            last_20 = flight_ids[-self.max_displayed_flight_ids//2:]
            flight_id_str = ", ".join(first_20) + f", [ ... ] , " + ", ".join(last_20) + f" (found {len(flight_ids)}, displayed {self.max_displayed_flight_ids})"
        else:
            flight_id_str = ", ".join(flight_ids)

        output.append(f"\n  - dataset: {', '.join(md['dataset'])}")
        output.append(f"\n  - institute: {', '.join(md['institute'])}")
        output.append(f"\n  - project: {', '.join(md['project'])}")
        output.append(f"\n  - acq_year: {', '.join(md['acq_year'])}")
        output.append(f"\n  - age: {', '.join(map(str, md['age']))}")
        output.append(f"\n  - age_unc: {', '.join(map(str, md['age_unc']))}")
        output.append(f"\n  - var: {', '.join(md['var'])}")
        output.append(f"\n  - region: {', '.join(md['region'])}")
        output.append(f"\n  - IMBIE_basin: {', '.join(md['IMBIE_basin'])}")
        output.append(f"\n  - radar_instrument: {', '.join(md['radar_instrument'])}")
        output.append(f"\n  - flight_id: {flight_id_str}")
        output.append(f"\n  - reference: {', '.join(md['reference'])}")
        output.append(f"  - dataset DOI: {', '.join(md['DOI_dataset'])}")
        output.append(f"  - publication DOI: {', '.join(md['DOI_publication'])}")
        output.append(f"\n  - database: {md['database_path']}/{md['file_db']}")
        output.append(f"  - query params: {md['_query_params']}")
        output.append(f"  - filter params: {md['_filter_params']}")

        return "\n".join(output)

    def to_dict(self):
        """Return the raw metadata dictionary."""
        return self._metadata
