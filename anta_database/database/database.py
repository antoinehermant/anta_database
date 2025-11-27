import os
import sqlite3
import pandas as pd
import numpy as np
import xarray as xr
import h5py
from typing import Union, List, Dict, Tuple, Optional, Generator
from tqdm import tqdm

from anta_database.plotting.plotting import Plotting

class Database:
    def __init__(self, database_dir: str, file_db: str = 'AntADatabase.db', include_BEDMAP: bool = False, max_displayed_flight_ids: Optional[int] = 50) -> None:
        self._db_dir = database_dir
        self._file_db = file_db
        self._file_db_path = os.path.join(self._db_dir, file_db)
        self._md = None
        self._plotting = None
        self._max_displayed_flight_ids = max_displayed_flight_ids
        self._include_BM = include_BEDMAP
        self._disable_tqdm = os.getenv("JUPYTER_BOOK_BUILD", False)
        self._excluded = {
            'dataset': [],
            'institute': [],
            'project': [],
            'acquisition_year': [],
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
                                acquisition_year: Optional[Union[str, List[str]]] = None,
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
            FROM
                datasets d
            JOIN
                sources s ON d.dataset = s.id
            INNER JOIN (
                    SELECT
                        di.dataset_id,
                        GROUP_CONCAT(i.name, ', ') AS projects
                    FROM
                        dataset_projects di
                    JOIN
                        projects i ON di.project_id = i.id
                    GROUP BY
                        di.dataset_id
                ) AS projects_list ON d.id = projects_list.dataset_id
            INNER JOIN (
                    SELECT
                        di.dataset_id,
                        GROUP_CONCAT(i.name, ', ') AS institutes
                    FROM
                        dataset_institutes di
                    JOIN
                        institutes i ON di.institute_id = i.id
                    GROUP BY
                        di.dataset_id
                ) AS institutes_list ON d.id = institutes_list.dataset_id
            INNER JOIN (
                    SELECT
                        di.dataset_id,
                        GROUP_CONCAT(i.name, ', ') AS radar_instruments
                    FROM
                        dataset_radar_instruments di
                    JOIN
                        radar_instruments i ON di.radar_instrument_id = i.id
                    GROUP BY
                        di.dataset_id
                ) AS radar_instruments_list ON d.id = radar_instruments_list.dataset_id
            LEFT JOIN
                dataset_variables dv ON d.id = dv.dataset_id
            LEFT JOIN
                variables v ON dv.variable_id = v.id
            LEFT JOIN
                dataset_ages da ON d.id = da.dataset_id AND v.name = 'IRH_DEPTH'
            LEFT JOIN
                ages a ON da.age_id = a.id
            LEFT JOIN
                dataset_regions dr ON d.id = dr.dataset_id
            LEFT JOIN
                regions r ON dr.region_id = r.id
        '''
        conditions = []
        params = []

        if institute:
            if isinstance(institute, list):
                like_conditions = []
                for inst in institute:
                    like_conditions.append("institutes_list.institutes LIKE ?")
                    params.append(f'%{inst}%')
                conditions.append('(' + ' OR '.join(like_conditions) + ')')
            else:
                conditions.append("institutes_list.institutes LIKE ?")
                params.append(f'%{institute}%')

        if radar_instrument:
            if isinstance(radar_instrument, list):
                like_conditions = []
                for radar in radar_instrument:
                    like_conditions.append("radar_instruments_list.radar_instruments LIKE ?")
                    params.append(f'%{radar}%')
                conditions.append('(' + ' OR '.join(like_conditions) + ')')
            else:
                conditions.append("radar_instruments_list.radar_instruments LIKE ?")
                params.append(f'%{radar_instrument}%')

        if project:
            if isinstance(project, list):
                like_conditions = []
                for proj in project:
                    like_conditions.append("projects_list.projects LIKE ?")
                    params.append(f'%{proj}%')
                conditions.append('(' + ' OR '.join(like_conditions) + ')')
            else:
                conditions.append("projects_list.projects LIKE ?")
                params.append(f'%{project}%')

        for field, column in [
                    (age, 'a.age'),
                    (var, 'v.name'),
                    (dataset, 's.name'),
                    (acquisition_year, 'd.acq_year'),
                    (line, 'd.flight_id'),
                    (region, 'r.region'),
                    (IMBIE_basin, 'r.IMBIE_basin')
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
                        if op == '>':
                            conditions.append(f"("
                                f"({column} > ?) OR "
                                f"({column} LIKE '%-%' AND CAST(SUBSTR({column}, INSTR({column}, '-')+1) AS INTEGER) > ?)"
                                f")")
                            params.extend([val, val])
                        elif op == '<':
                            conditions.append(f"("
                                f"({column} < ?) OR "
                                f"({column} LIKE '%-%' AND CAST(SUBSTR({column}, 1, INSTR({column}, '-')-1) AS INTEGER) < ?)"
                                f")")
                            params.extend([val, val])
                        elif op == '>=':
                            conditions.append(f"("
                                f"({column} >= ?) OR "
                                f"({column} LIKE '%-%' AND CAST(SUBSTR({column}, INSTR({column}, '-')+1) AS INTEGER) >= ?)"
                                f")")
                            params.extend([val, val])
                        elif op == '<=':
                            conditions.append(f"("
                                f"({column} <= ?) OR "
                                f"({column} LIKE '%-%' AND CAST(SUBSTR({column}, 1, INSTR({column}, '-')-1) AS INTEGER) <= ?)"
                                f")")
                            params.extend([val, val])
                        elif op == '=':
                            conditions.append(f"("
                                f"({column} = ?) OR "
                                f"({column} LIKE '%-%' AND CAST(SUBSTR({column}, 1, INSTR({column}, '-')-1) AS INTEGER) <= ? AND CAST(SUBSTR({column}, INSTR({column}, '-')+1) AS INTEGER) >= ?)"
                                f")")
                            params.extend([val, val, val])
                    elif self._is_range_value(field):
                            start, end = self._parse_range_value(field)
                            conditions.append(f"("
                                f"({column} BETWEEN ? AND ?) OR "
                                f"({column} LIKE '%-%' AND "
                                f"CAST(SUBSTR({column}, 1, INSTR({column}, '-')-1) AS INTEGER) <= ? AND "
                                f"CAST(SUBSTR({column}, INSTR({column}, '-')+1) AS INTEGER) >= ?)"
                                f")")
                            params.extend([start, end, end, start])
                    else:
                        conditions.append(f"{column} = ?")
                        params.append(field)

        for field, column in [
                ('age', 'a.age'),
                ('var', 'v.var'),
                ('dataset', 's.name'),
                ('acquisition_year', 'd.acq_year'),
                ('flight_id', 'd.flight_id'),
                ('region', 'r.region'),
                ('IMBIE_basin', 'r.IMBIE_basin'),
        ]:
            if self._excluded[field]:
                    not_like_conditions = []
                    not_in_values = []
                    not_range_conditions = []
                    for item in self._excluded[field]:
                        if '%' in item:
                            not_like_conditions.append(f"{column} NOT LIKE ?")
                            params.append(item)
                        elif self._is_range_query(item):
                            op, val = self._parse_range_query(item)
                            inverted_op = self._invert_range_operator(op)
                            if inverted_op == '>':
                                conditions.append(f"("
                                    f"({column} > ?) OR "
                                    f"({column} LIKE '%-%' AND CAST(SUBSTR({column}, INSTR({column}, '-')+1) AS INTEGER) > ?)"
                                    f")")
                                params.extend([val, val])
                            elif inverted_op == '<':
                                conditions.append(f"("
                                    f"({column} < ?) OR "
                                    f"({column} LIKE '%-%' AND CAST(SUBSTR({column}, 1, INSTR({column}, '-')-1) AS INTEGER) < ?)"
                                    f")")
                                params.extend([val, val])
                            elif inverted_op == '>=':
                                conditions.append(f"("
                                    f"({column} >= ?) OR "
                                    f"({column} LIKE '%-%' AND CAST(SUBSTR({column}, INSTR({column}, '-')+1) AS INTEGER) >= ?)"
                                    f")")
                                params.extend([val, val])
                            elif inverted_op == '<=':
                                conditions.append(f"("
                                    f"({column} <= ?) OR "
                                    f"({column} LIKE '%-%' AND CAST(SUBSTR({column}, 1, INSTR({column}, '-')-1) AS INTEGER) <= ?)"
                                    f")")
                                params.extend([val, val])
                            elif inverted_op == '=':
                                conditions.append(f"("
                                    f"({column} = ?) OR "
                                    f"({column} LIKE '%-%' AND CAST(SUBSTR({column}, 1, INSTR({column}, '-')-1) AS INTEGER) <= ? AND CAST(SUBSTR({column}, INSTR({column}, '-')+1) AS INTEGER) >= ?)"
                                    f")")
                                params.extend([val, val, val])
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

        if self._excluded.get('institute'):
            if isinstance(self._excluded['institute'], list):
                not_like_conditions = []
                for inst in self._excluded['institute']:
                    not_like_conditions.append("institutes_list.institutes NOT LIKE ?")
                    params.append(f'%{inst}%')
                conditions.append('(' + ' AND '.join(not_like_conditions) + ')')
            else:
                conditions.append("institutes_list.institutes NOT LIKE ?")
                params.append(f'%{self._excluded["institute"]}%')

        if self._excluded.get('project'):
            if isinstance(self._excluded['project'], list):
                not_like_conditions = []
                for proj in self._excluded['project']:
                    not_like_conditions.append("projects_list.projects NOT LIKE ?")
                    params.append(f'%{proj}%')
                conditions.append('(' + ' AND '.join(not_like_conditions) + ')')
            else:
                conditions.append("projects_list.projects NOT LIKE ?")
                params.append(f'%{self._excluded["project"]}%')

        if self._excluded.get('radar_instrument'):
            if isinstance(self._excluded['radar_instrument'], list):
                not_like_conditions = []
                for ri in self._excluded['radar_instrument']:
                    not_like_conditions.append("radar_instruments_list.radar_instruments NOT LIKE ?")
                    params.append(f'%{ri}%')
                conditions.append('(' + ' AND '.join(not_like_conditions) + ')')
            else:
                conditions.append("radar_instruments_list.radar_instruments NOT LIKE ?")
                params.append(f'%{self._excluded["radar_instrument"]}%')

        if not self._include_BM:
            conditions.append("s.name NOT LIKE ?")
            params.append('%BEDMAP%')

        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        query += ' ORDER BY CAST(a.age AS INTEGER) ASC'
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
        start, end = map(int, s.split('-'))
        return start, end

    def _ranges_overlap(self, range1: str, range2: str) -> bool:
        """Check if two ranges overlap."""
        start1, end1 = self._parse_range_value(range1)
        start2, end2 = self._parse_range_value(range2)
        return start1 <= end2 and end1 >= start2

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
            acquisition_year: Optional[Union[str, List[str]]] = None,
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
            'acquisition_year': acquisition_year,
            'flight_id': flight_id,
            'region': region,
            'IMBIE_basin': IMBIE_basin,
            'radar_instrument': radar_instrument,
        }

        for field, value in field_mapping.items():
            if value is not None:
                if isinstance(value, list):
                    self._excluded[field] = value
                else:
                    self._excluded[field] = [value]
            else:
                self._excluded[field] = []

    def _get_file_metadata(self, file_path) -> Dict:
        """
        Helper method to build the SQL query and parameters for filtering.
        Returns the query string and parameters list.
        """
        select_clause = "s.name AS dataset, \
                        s.citation, \
                        s.DOI_dataset, \
                        s.DOI_publication, \
                        institutes_list.institutes, \
                        projects_list.projects, \
                        d.acq_year, \
                        CASE WHEN v.name = 'IRH_DEPTH' THEN a.age ELSE NULL END AS age, \
                        CASE WHEN v.name = 'IRH_DEPTH' THEN a.age_unc ELSE NULL END AS age_unc, \
                        v.name AS var, \
                        d.flight_id, \
                        r.region, \
                        r.IMBIE_basin, \
                        radar_instruments_list.radar_instruments \
        "
        query = f'''
            SELECT {select_clause}
            FROM
                datasets d
            JOIN
                sources s ON d.dataset = s.id
            INNER JOIN (
                    SELECT
                        di.dataset_id,
                        GROUP_CONCAT(i.name, ', ') AS projects
                    FROM
                        dataset_projects di
                    JOIN
                        projects i ON di.project_id = i.id
                    GROUP BY
                        di.dataset_id
                ) AS projects_list ON d.id = projects_list.dataset_id
            INNER JOIN (
                    SELECT
                        di.dataset_id,
                        GROUP_CONCAT(i.name, ', ') AS institutes
                    FROM
                        dataset_institutes di
                    JOIN
                        institutes i ON di.institute_id = i.id
                    GROUP BY
                        di.dataset_id
                ) AS institutes_list ON d.id = institutes_list.dataset_id
            INNER JOIN (
                    SELECT
                        di.dataset_id,
                        GROUP_CONCAT(i.name, ', ') AS radar_instruments
                    FROM
                        dataset_radar_instruments di
                    JOIN
                        radar_instruments i ON di.radar_instrument_id = i.id
                    GROUP BY
                        di.dataset_id
                ) AS radar_instruments_list ON d.id = radar_instruments_list.dataset_id
            LEFT JOIN
                dataset_variables dv ON d.id = dv.dataset_id
            LEFT JOIN
                variables v ON dv.variable_id = v.id
            LEFT JOIN
                dataset_ages da ON d.id = da.dataset_id AND v.name = 'IRH_DEPTH'
            LEFT JOIN
                ages a ON da.age_id = a.id
            LEFT JOIN
                dataset_regions dr ON d.id = dr.dataset_id
            LEFT JOIN
                regions r ON dr.region_id = r.id
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

        conn = sqlite3.connect(self._file_db_path)
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
            'acquisition_year': results[0][6],
            'age_unc': results[0][8],
            'region': results[0][11],
            'IMBIE_basin': results[0][12],
            'radar_instrument': results[0][13],
            'reference': results[0][1],
            'file_path': file_path,
            'database_path': self._db_dir,
            'file_db': self._file_db,
        }
        return metadata

    def query(self,
              age: Optional[Union[str, List[str]]] = None,
              var: Optional[Union[str, List[str]]] = None,
              dataset: Optional[Union[str, List[str]]] = None,
              institute: Optional[Union[str, List[str]]] = None,
              project: Optional[Union[str, List[str]]] = None,
              acquisition_year: Optional[Union[str, List[str]]] = None,
              flight_id: Optional[Union[str, List[str]]] = None,
              region: Optional[Union[str, List[str]]] = None,
              IMBIE_basin: Optional[Union[str, List[str]]] = None,
              radar_instrument: Optional[Union[str, List[str]]] = None,
              retain_query: Optional[bool] = True,
              ) -> 'MetadataResult':

        select_clause = "s.name AS dataset, \
                        s.citation, \
                        s.DOI_dataset, \
                        s.DOI_publication, \
                        institutes_list.institutes, \
                        projects_list.projects, \
                        d.acq_year, \
                        CASE WHEN v.name = 'IRH_DEPTH' THEN a.age ELSE NULL END AS age, \
                        CASE WHEN v.name = 'IRH_DEPTH' THEN a.age_unc ELSE NULL END AS age_unc, \
                        v.name AS var, \
                        d.flight_id, \
                        r.region, \
                        r.IMBIE_basin, \
                        radar_instruments_list.radar_instruments \
        "
        query, params = self._build_query_and_params(age, var, dataset, institute, project, acquisition_year, flight_id, region, IMBIE_basin, radar_instrument, select_clause)

        conn = sqlite3.connect(self._file_db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        metadata = {
            'dataset': [],
            'institute': [],
            'project': [],
            'acquisition_year': [],
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
            '_query_params': {'dataset': dataset, 'institute': institute, 'project': project, 'acquisition_year': acquisition_year, 'age': age, 'var': var, 'flight_id': flight_id, 'region': region, 'IMBIE_basin': IMBIE_basin, 'radar_instrument': radar_instrument},
            '_filter_params': {'dataset': self._excluded['dataset'], 'institute': self._excluded['institute'], 'project': self._excluded['project'], 'acquisition_year': self._excluded['acquisition_year'], 'age': self._excluded['age'], 'var': self._excluded['var'], 'flight_id': self._excluded['flight_id'], 'region': self._excluded['region'], 'IMBIE_basin': self._excluded['IMBIE_basin'], 'radar_instrument': self._excluded['radar_instrument']},
            'database_path': self._db_dir,
            'file_db': self._file_db,
        }
        ages_list = []
        ages_unc_list = []
        vars_list = []
        institutes_list = []
        projects_list = []
        acq_years_list = []
        radar_instruments_list = []
        region_list = []
        basin_list = []
        for dataset_name, citations, DOI_dataset, DOI_publication, institutes, projects, acq_years, ages, ages_unc, vars, flight_id, regions, basins, radar_instruments in results:
            metadata['dataset'].append(dataset_name)
            metadata['reference'].append(citations)
            metadata['DOI_dataset'].append(DOI_dataset)
            metadata['DOI_publication'].append(DOI_publication)
            metadata['flight_id'].append(flight_id)
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
                radar_instruments_list.append(radar_instruments)
            else:
                radar_instruments_list.append('-')
            if projects is not None:
                projects_list.append(projects)
            else:
                projects_list.append('-')
            if acq_years is not None:
                acq_years_list.append(acq_years)
            else:
                acq_years_list.append('-')
            if regions is not None:
                region_list.append(regions)
            else:
                region_list.append('-')
            if basins is not None:
                basin_list.append(basins)
            else:
                basin_list.append('-')


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
        metadata['acquisition_year'] = sorted(set(acq_years_list))
        metadata['dataset'] = list(dict.fromkeys(metadata['dataset']))
        metadata['reference'] = list(dict.fromkeys(metadata['reference']))
        metadata['DOI_dataset'] = list(dict.fromkeys(metadata['DOI_dataset']))
        metadata['DOI_publication'] = list(dict.fromkeys(metadata['DOI_publication']))
        metadata['flight_id'] = list(set(metadata['flight_id']))
        metadata['radar_instrument'] = sorted(set(radar_instruments_list))
        metadata['region'] = sorted(set(region_list))
        metadata['IMBIE_basin'] = sorted(set(basin_list))

        if retain_query:
            self._md = metadata

        return MetadataResult(metadata, self._max_displayed_flight_ids)

    def _get_file_paths_from_metadata(self, metadata) -> List:

        query_params = metadata['_query_params']
        age = query_params.get('age')
        var = query_params.get('var')
        dataset = query_params.get('dataset')
        institute = query_params.get('institute')
        project = query_params.get('project')
        acq_year = query_params.get('acquisition_year')
        line = query_params.get('flight_id')
        region = query_params.get('region')
        basin = query_params.get('IMBIE_basin')
        radar = query_params.get('radar_instrument')

        select_clause = 'd.file_path'
        query, params = self._build_query_and_params(age, var, dataset, institute, project, acq_year, line, region, basin, radar, select_clause)

        conn = sqlite3.connect(self._file_db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        file_paths = [row[0] for row in cursor.fetchall()]
        conn.close()

        return file_paths

    def get_files(
        self,
        metadata: Optional[Union[None, Dict, 'MetadataResult']] = None,
        data_dir: Optional[Union[None, str]] = None,
    ):

        md = metadata or self._md
        if not md:
            print('Please provide metadata of the files you want to generate the data from.')
            return

        data_dir = data_dir or self._db_dir
        if not data_dir:
            print('No data directory provided.')
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
        disable_tqdm: bool = False,
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
        md = metadata or self._md
        if not md:
            print('Please provide metadata of the files you want to generate the data from. Exiting...')
            return

        # Resolve data directory
        data_dir = data_dir or self._db_dir
        if not data_dir:
            print('No data directory provided. Exiting...')
            return

        file_paths = self._get_file_paths_from_metadata(metadata=md)
        file_paths = np.unique(file_paths) # Be carefull as many pointers point to the same file

        if disable_tqdm or self._disable_tqdm:
            disable_tqdm = True

        for file_path in tqdm(file_paths, desc='Generating dataframes', total=len(file_paths), unit='file', disable=disable_tqdm):
            full_path = os.path.join(data_dir, file_path)
            file_md = self._get_file_metadata(file_path)
            ds = h5py.File(full_path, 'r')
            df = pd.DataFrame({'PSX': ds['PSX'][::downscale_factor],
                                'PSY': ds['PSY'][::downscale_factor]})

            if 'Distance' in ds.keys():
                df['Distance'] = ds['Distance'][::downscale_factor]

            var_impl = []
            age_impl = []
            for var in md['var']:
                if var == 'IRH_DEPTH':
                    var_impl.append(var)
                    for age in md['age']:
                        irh_values = ds['IRH_AGE'][:]
                        irh_index = np.where(irh_values == int(age))[0]
                        if len(irh_index) == 0:
                            continue
                        irh_index = irh_index[0]
                        age_impl.append(age)

                        df[int(age)] = ds[var][::downscale_factor, irh_index]

                else:
                    if var in ds.keys():
                        var_impl.append(var)
                        df[var] = ds[var][::downscale_factor]

            metadata = {
                'dataset': file_md['dataset'],
                'var': var_impl,
                'age': age_impl,
                'flight_id': file_md['flight_id'],
                'institute': file_md['institute'],
                'project': file_md['project'],
                'acquisition_year': file_md['acquisition_year'],
                'age_unc': md['age_unc'],
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
        self._max_displayed_flight_ids = max_displayed_flight_ids

    def __getitem__(self, key):
        return self._metadata[key]

    def __repr__(self):
        """Pretty-print the metadata, truncating long flight_id lists."""
        md = self._metadata
        output = []

        flight_ids = md['flight_id']
        if len(flight_ids) > self._max_displayed_flight_ids:
            first_20 = flight_ids[:self._max_displayed_flight_ids//2]
            last_20 = flight_ids[-self._max_displayed_flight_ids//2:]
            flight_id_str = ", ".join(first_20) + f", [ ... ] , " + ", ".join(last_20) + f" (found {len(flight_ids)}, displayed {self._max_displayed_flight_ids})"
        else:
            flight_id_str = ", ".join(flight_ids)

        if not md['dataset']:
            output.append(f"\nNo data found for this query")
            output.append(f"Try something else:")
            output.append(f"\n  - database: {md['database_path']}/{md['file_db']}")
            output.append(f"  - query params: {md['_query_params']}")
            output.append(f"  - filter params: {md['_filter_params']}")

        else:
            output.append("Metadata from query:")
            output.append(f"\n  - dataset: {', '.join(md['dataset'])}")
            output.append(f"\n  - institute: {', '.join(md['institute'])}")
            output.append(f"\n  - project: {', '.join(md['project'])}")
            output.append(f"\n  - acquisition_year: {', '.join(md['acquisition_year'])}")
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
