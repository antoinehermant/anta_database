import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Tuple

from anta_database.plotting.plotting import Plotting

class Database:
    def __init__(self, database_dir: str, file_db: str = 'AntADatabase.db') -> None:
        self.db_dir = database_dir
        self.file_db = file_db
        self.file_db_path = os.path.join(self.db_dir, file_db)
        self.md = None
        self._plotting = None

    def _build_query_and_params(self, age: Union[None, str, List[str]]=None, var: Union[None, str, List[str]]=None, author: Union[None, str, List[str]]=None, line: Union[None, str, List[str]]=None, select_clause='') -> Tuple[str, List[Union[str, int]]]:
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
            (line, 'd.trace_id')
        ]:
            if field is not None:
                if isinstance(field, list):
                    # For lists, use IN for exact matches, or LIKE for wildcards
                    like_conditions = []
                    in_values = []
                    for item in field:
                        if '%' in item:
                            like_conditions.append(f"{column} LIKE ?")
                            params.append(item)
                        else:
                            in_values.append(item)
                    if like_conditions:
                        conditions.append('(' + ' OR '.join(like_conditions) + ')')
                    if in_values:
                        placeholders = ','.join(['?'] * len(in_values))
                        conditions.append(f"{column} IN ({placeholders})")
                        params.extend(in_values)
                else:
                    # For single values, use = or LIKE
                    if '%' in field:
                        conditions.append(f"{column} LIKE ?")
                    else:
                        conditions.append(f"{column} = ?")
                    params.append(field)
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        query += ' ORDER BY CAST(d.age AS INTEGER) ASC'
        return query, params

    def _get_file_metadata(self, file_path) -> Dict:
        """
        Helper method to build the SQL query and parameters for filtering.
        Returns the query string and parameters list.
        """
        select_clause = 'a.name, d.age, d.var, d.trace_id, d.file_path'
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
            'age': results[0][1],
            'var': results[0][2],
            'trace_id': results[0][3],
            'file_path': results[0][4],
            'database_path': self.db_dir,
            'file_db': self.file_db,
        }
        return metadata

    def query(self, age: Union[None, str, List[str]]=None, var: Union[None, str, List[str]]=None,author: Union[None, str, List[str]]=None, trace_id: Union[None, str, List[str]]=None) -> 'MetadataResult':
        select_clause = 'a.name, a.citation, a.doi, d.age, d.var, d.trace_id'
        query, params = self._build_query_and_params(age, var, author, trace_id, select_clause)

        conn = sqlite3.connect(self.file_db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        metadata = {
            'author': [],
            'age': [],
            'var': [],
            'reference': [],
            'doi': [],
            'trace_id': [],
            '_query_params': {'age': age, 'var': var, 'author': author, 'trace_id': trace_id},
            'database_path': self.db_dir,
            'file_db': self.file_db,
        }
        ages_list = []
        vars_list = []
        for author_name, citations, doi, ages, vars, trace_id in results:
            metadata['author'].append(author_name)
            metadata['reference'].append(citations)
            metadata['doi'].append(doi)
            metadata['trace_id'].append(trace_id)
            # Check if the age is numeric
            if ages is not None and ages.isdigit():
                ages_list.append(int(ages))
            if vars is not None:
                vars_list.append(vars)
        metadata['age'] = sorted({str(age) for age in set(ages_list)}, key=int)
        metadata['var'] = sorted(set(vars_list))
        metadata['author'] = list(dict.fromkeys(metadata['author']))
        metadata['reference'] = list(dict.fromkeys(metadata['reference']))
        metadata['doi'] = list(dict.fromkeys(metadata['doi']))
        metadata['trace_id'] = list(set(metadata['trace_id']))

        self.md = metadata
        return MetadataResult(metadata)

    def _get_file_paths_from_metadata(self, metadata) -> List:
        query_params = metadata.get('_query_params', {})
        age = query_params.get('age')
        var = query_params.get('var')
        author = query_params.get('author')
        line = query_params.get('trace_id')

        select_clause = 'd.file_path'
        query, params = self._build_query_and_params(age, var, author, line, select_clause)

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
        trace_id = query_params.get('trace_id')

        select_clause = 'DISTINCT d.file_path, d.age'
        query, params = self._build_query_and_params(age, var, author, trace_id, select_clause)

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
        output.append(f"\n  - age: {', '.join(map(str, md['age']))}")
        output.append(f"\n  - var: {', '.join(md['var'])}")
        output.append(f"\n  - trace_id: {', '.join(md['trace_id'])}")
        output.append(f"\n  - reference: {', '.join(md['reference'])}")
        output.append(f"  - DOIs: {', '.join(md['doi'])}")
        output.append(f"  - database: {md['database_path']}/{md['file_db']}")
        output.append(f"  - query params: {md['_query_params']}")
        return "\n".join(output)

    def to_dict(self):
        """Return the raw metadata dictionary."""
        return self._metadata
