import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Tuple, Optional

from anta_database.plotting.plotting import Plotting

class Database:
    def __init__(self, database_dir: str, file_db: str = 'AntADatabase.db') -> None:
        self.db_dir = database_dir
        self.file_db = file_db
        self.file_db_path = os.path.join(self.db_dir, file_db)
        self.md = None
        self._plotting = None
        self.excluded = {
            'author': [],
            'institute': [],
            'project': [],
            'acq_year': [],
            'age': [],
            'var': [],
            'flight_id': [],
        }

    def _build_query_and_params(self,
                                age: Optional[Union[str, List[str]]] = None,
                                var: Optional[Union[str, List[str]]] = None,
                                author: Optional[Union[str, List[str]]] = None,
                                institute: Optional[Union[str, List[str]]] = None,
                                project: Optional[Union[str, List[str]]] = None,
                                acq_year: Optional[Union[str, List[str]]] = None,
                                line: Optional[Union[str, List[str]]] = None,
                                select_clause='') -> Tuple[str, List[Union[str, int]]]:
        """
        Helper method to build the SQL query and parameters for filtering.
        Returns the query string and parameters list.
        """
        query = f'''
            SELECT {select_clause}
            FROM datasets d
            JOIN authors a ON d.author = a.id
        '''
        conditions = []
        params = []
        for field, column in [
            (age, 'd.age'),
            (var, 'd.var'),
            (author, 'a.name'),
            (institute, 'd.institute'),
            (project, 'd.project'),
            (acq_year, 'd.acq_year'),
            (line, 'd.flight_id')
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
            ('author', 'a.name'),
            ('institute', 'd.institute'),
            ('project', 'd.project'),
            ('acq_year', 'd.acq_year'),
            ('flight_id', 'd.flight_id')
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

        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        query += ' ORDER BY CAST(d.age AS INTEGER) ASC'
        return query, params

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
        author: Optional[Union[str, List[str]]] = None,
        institute: Optional[Union[str, List[str]]] = None,
        project: Optional[Union[str, List[str]]] = None,
        acq_year: Optional[Union[str, List[str]]] = None,
        flight_id: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """
        Add values to exclude from the query results.
        Example: filter_out(author='Cavitte', project='OldProject')
        """
        # Map arguments to their corresponding fields
        field_mapping = {
            'age': age,
            'var': var,
            'author': author,
            'institute': institute,
            'project': project,
            'acq_year': acq_year,
            'flight_id': flight_id,
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
        select_clause = 'a.name, d.institute, d.project, d.acq_year, d.age, d.age_unc, d.var, d.flight_id, d.file_path'
        query = f'''
            SELECT {select_clause}
            FROM datasets d
            JOIN authors a ON d.author = a.id
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
            'author': results[0][0],
            'institute': results[0][1],
            'project': results[0][2],
            'acq_year': results[0][3],
            'age': results[0][4],
            'age_unc': results[0][5],
            'var': results[0][6],
            'flight_id': results[0][7],
            'file_path': results[0][8],
            'database_path': self.db_dir,
            'file_db': self.file_db,
        }
        return metadata

    def query(self,
              age: Optional[Union[str, List[str]]] = None,
              var: Optional[Union[str, List[str]]] = None,
              author: Optional[Union[str, List[str]]] = None,
              institute: Optional[Union[str, List[str]]] = None,
              project: Optional[Union[str, List[str]]] = None,
              acq_year: Optional[Union[str, List[str]]] = None,
              flight_id: Optional[Union[str, List[str]]] = None) -> 'MetadataResult':
        select_clause = 'a.name, a.citation, a.dataset_doi, a.publication_doi, d.institute, d.project, d.acq_year, d.age, d.age_unc, d.var, d.flight_id'
        query, params = self._build_query_and_params(age, var, author, institute, project, acq_year, flight_id, select_clause)

        conn = sqlite3.connect(self.file_db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        metadata = {
            'author': [],
            'institute': [],
            'project': [],
            'acq_year': [],
            'age': [],
            'age_unc': [],
            'var': [],
            'reference': [],
            'dataset_doi': [],
            'publication_doi': [],
            'flight_id': [],
            '_query_params': {'author': author, 'institute': institute, 'project': project, 'acq_year': acq_year, 'age': age, 'var': var, 'flight_id': flight_id},
            '_filter_params': {'author': self.excluded['author'], 'institute': self.excluded['institute'], 'project': self.excluded['project'], 'acq_year': self.excluded['acq_year'], 'age': self.excluded['age'], 'var': self.excluded['var'], 'flight_id': self.excluded['flight_id']},
            'database_path': self.db_dir,
            'file_db': self.file_db,
        }
        ages_list = []
        ages_unc_list = []
        vars_list = []
        institutes_list = []
        projects_list = []
        acq_years_list = []
        for author_name, citations, dataset_doi, publication_doi, institutes, projects, acq_years, ages, ages_unc, vars, flight_id in results:
            metadata['author'].append(author_name)
            metadata['reference'].append(citations)
            metadata['dataset_doi'].append(dataset_doi)
            metadata['publication_doi'].append(publication_doi)
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
        metadata['author'] = list(dict.fromkeys(metadata['author']))
        metadata['reference'] = list(dict.fromkeys(metadata['reference']))
        metadata['dataset_doi'] = list(dict.fromkeys(metadata['dataset_doi']))
        metadata['publication_doi'] = list(dict.fromkeys(metadata['publication_doi']))
        metadata['flight_id'] = list(set(metadata['flight_id']))

        self.md = metadata
        return MetadataResult(metadata)

    def _get_file_paths_from_metadata(self, metadata) -> List:

        query_params = metadata['_query_params']
        age = query_params.get('age')
        var = query_params.get('var')
        author = query_params.get('author')
        institute = query_params.get('institute')
        project = query_params.get('project')
        acq_year = query_params.get('acq_year')
        line = query_params.get('flight_id')

        select_clause = 'd.file_path'
        query, params = self._build_query_and_params(age, var, author, institute, project, acq_year, line, select_clause)

        conn = sqlite3.connect(self.file_db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        file_paths = [row[0] for row in cursor.fetchall()]
        conn.close()

        return file_paths

    def data_generator(self, metadata: Union[None, Dict, 'MetadataResult'] = None, data_dir: Union[None, str] = None, downscale_factor: Union[None, int] = None, downsample_distance: Union[None, float, int] = None):
        """
        Generates DataFrames and their associated author names from the database based on the provided metadata.

        This method queries the database using the filter parameters stored in the metadata,
        retrieves the file paths and author names, and yields each DataFrame along with its author.

        Args:
            metadata: the results from the query()
        """
        if metadata:
            md = metadata
        elif self.md:
            md = self.md
        else:
            print('Please provide metadata of the files you want to generate the data from. Exiting ...')
            return

        query_params = md['_query_params']
        age = query_params.get('age')
        var = query_params.get('var')
        author = query_params.get('author')
        institute = query_params.get('institute')
        project = query_params.get('project')
        acq_year = query_params.get('acq_year')
        line = query_params.get('flight_id')

        select_clause = 'DISTINCT d.file_path, d.age'
        query, params = self._build_query_and_params(age, var, author, institute, project, acq_year, line, select_clause)

        conn = sqlite3.connect(self.file_db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        if data_dir:
            data_dir = data_dir
        elif self.db_dir:
            data_dir = self.db_dir
        else:
            print('No data dir provided, do not know where to look for data ...')
            return

        for file_path, age in results:
            df = pd.read_pickle(os.path.join(data_dir, file_path))
            if downscale_factor:
                df = df[::downscale_factor]
            if downsample_distance:
                df['bin'] = np.floor(df['distance'] / downsample_distance) * downsample_distance
                df = df.groupby('bin').mean().reset_index()
                df.drop(columns=['bin'], inplace=True)
            metadata = self._get_file_metadata(file_path)
            yield df, metadata

    @property
    def plot(self):
        if self._plotting is None:
            self._plotting = Plotting(self)
        return self._plotting


class MetadataResult:
    def __init__(self, metadata):
        self._metadata = metadata

    def __getitem__(self, key):
        return self._metadata[key]

    def __repr__(self):
        """Pretty-print the metadata."""
        md = self._metadata
        output = []
        output.append("Metadata from query:")
        output.append(f"\n  - author: {', '.join(md['author'])}")
        output.append(f"\n  - institute: {', '.join(md['institute'])}")
        output.append(f"\n  - project: {', '.join(md['project'])}")
        output.append(f"\n  - acq_year: {', '.join(md['acq_year'])}")
        output.append(f"\n  - age: {', '.join(map(str, md['age']))}")
        output.append(f"\n  - age_unc: {', '.join(map(str, md['age_unc']))}")
        output.append(f"\n  - var: {', '.join(md['var'])}")
        output.append(f"\n  - flight_id: {', '.join(md['flight_id'])}")
        output.append(f"\n  - reference: {', '.join(md['reference'])}")
        output.append(f"  - dataset DOI: {', '.join(md['dataset_doi'])}")
        output.append(f"  - publication DOI: {', '.join(md['publication_doi'])}")
        output.append(f"  - database: {md['database_path']}/{md['file_db']}")
        output.append(f"  - query params: {md['_query_params']}")
        output.append(f"  - filter params: {md['_filter_params']}")
        return "\n".join(output)

    def to_dict(self):
        """Return the raw metadata dictionary."""
        return self._metadata
