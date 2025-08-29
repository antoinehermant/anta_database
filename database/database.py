import os, sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cmastro as cma
from typing import Union, List, Dict, Tuple

class Database:
    def __init__(self, database_dir: str, file_db: str = 'AntADatabase.db'):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.gl = pd.read_pickle(os.path.join(module_dir, 'plotting', 'GL.pkl'))
        self.db_dir = database_dir
        self.file_db = file_db
        self.file_db_path = os.path.join(self.db_dir, file_db)
        self.md = None

    def _build_query_and_params(self, age: Union[None, str, List[str]]=None, author: Union[None, str, List[str]]=None, line: Union[None, str, List[str]]=None, select_clause='') -> Tuple[str, List[Union[str, int]]]:
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
            (author, 'a.name'),
            (line, 'd.trace_id')
        ]:
            if field is not None:
                if isinstance(field, list):
                    placeholders = ','.join(['?'] * len(field))
                    conditions.append(f'{column} IN ({placeholders})')
                    params.extend(field)
                else:
                    conditions.append(f'{column} = ?')
                    params.append(field)

        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)

        return query, params

    def _get_file_metadata(self, file_path) -> Dict:
        """
        Helper method to build the SQL query and parameters for filtering.
        Returns the query string and parameters list.
        """
        select_clause = 'a.name, d.age, d.trace_id, d.file_path'
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
            'author': set(),
            'age': set(),
            'trace_id': set(),
            'file_path': set(),
            'database_path': self.db_dir,
            'file_db': self.file_db,
        }

        for author_name, age, trace_id, file_path in results:
            metadata['author'].add(author_name)
            metadata['age'].add(age)
            metadata['trace_id'].add(trace_id)
            metadata['file_path'].add(file_path)

        return metadata

    def query(self, age: Union[None, str, List[str]]=None, author: Union[None, str, List[str]]=None, trace_id: Union[None, str, List[str]]=None) -> Dict:
        select_clause = 'a.name, a.citation, a.doi, d.age, d.trace_id'
        query, params = self._build_query_and_params(age, author, trace_id, select_clause)

        conn = sqlite3.connect(self.file_db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        metadata = {
            'author': set(),
            'age': set(),
            'reference': set(),
            'doi': set(),
            'trace_id': set(),
            '_query_params': {'age': age, 'author': author, 'trace_id': trace_id},
            'database_path': self.db_dir,
            'file_db': self.file_db,
        }

        for author_name, citations, doi, ages, trace_id in results:
            metadata['author'].add(author_name)
            metadata['age'].add(ages)
            metadata['reference'].add(citations)
            metadata['doi'].add(doi)
            metadata['trace_id'].add(trace_id)

        self.md = metadata
        return metadata

    def _get_file_paths_from_metadata(self, metadata) -> List:
        query_params = metadata.get('_query_params', {})
        age = query_params.get('age')
        author = query_params.get('author')
        line = query_params.get('trace_id')

        select_clause = 'd.file_path'
        query, params = self._build_query_and_params(age, author, line, select_clause)

        conn = sqlite3.connect(self.file_db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        file_paths = [row[0] for row in cursor.fetchall()]
        conn.close()

        return file_paths

    def data_generator(self, metadata: Union[None, Dict] = None, data_dir: Union[None, str] = None):
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
            sys.exit()

        query_params = md.get('_query_params', {})
        age = query_params.get('age')
        author = query_params.get('author')
        trace_id = query_params.get('trace_id')

        select_clause = 'd.file_path'
        query, params = self._build_query_and_params(age, author, trace_id, select_clause)

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
            print('No data dir provided, do not know where to look for data, exiting ...')
            sys.exit()

        for file_path in results:
            if len(file_path) > 1:
                print('Returned more than 1 file path for current file, this has to be fixed. Exiting ...')
                sys.exit()
            else:
                file_path = next(iter(file_path))
                df = pd.read_pickle(os.path.join(data_dir, file_path))
                metadata = self._get_file_metadata(file_path)
            yield df, metadata

    def plotXY(self,
               metadata: Union[None, Dict] = None,
               title: str = '',
               xlim: tuple = (None, None),
               ylim: tuple = (None, None),
               scale_factor: float = 1.0,
               latex:bool = False,
               save: Union[str, None] = None) -> None:

        if metadata:
            metadata = metadata
        elif self.md:
            metadata = self.md
        else:
            print('Please provide metadata of the files you want to generate the data from. Exiting ...')
            sys.exit()

        cmaps = cma.cmaps['cma:emph_r']
        authors = list(metadata['author'])
        color_indices = np.linspace(0.1, 0.9, len(authors))
        colors = {author: cmaps(i) for i, author in zip(color_indices, authors)}

        if latex:
            from matplotlib import rc
            rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
            rc('text', usetex=True)

        plt.figure()
        plt.plot(self.gl.x/1000, self.gl.y/1000, linewidth=1, color='k')
        for df, md in self.data_generator(metadata):
            author = next(iter(md['author']))
            plt.plot(df.x/1000, df.y/1000, linewidth=0.8, color=colors[author])

        ax = plt.gca()
        x0, x1 = ax.get_xlim() if xlim == (None, None) else xlim
        y0, y1 = ax.get_ylim() if ylim == (None, None) else ylim

        for author in authors:
            citation = self.query(author=author)['reference']
            plt.plot([], [], color=colors[author], label=citation)
        plt.legend(loc='lower left')

        x_extent = x1 - x0
        y_extent = y1 - y0

        aspect_ratio = y_extent / x_extent

        # Set the figure size based on the aspect ratio and a scale factor
        plt.gcf().set_size_inches(8 * scale_factor, 8 * aspect_ratio * scale_factor)
        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        # self.scale_axes(ax, factor=1000)
        plt.title(title, fontsize=30*scale_factor)
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=200)
            plt.close()
            print('Figure saved as', save)
        else:
            plt.show()
            plt.close()
