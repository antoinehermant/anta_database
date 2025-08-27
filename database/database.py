import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cmastro as cma
from typing import Union

class Database:
    def __init__(self, database_dir: str, file_db: str = 'isochrones.db'):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.gl = pd.read_pickle(os.path.join(module_dir, 'plotting', 'GL.pkl'))
        self.db_dir = database_dir
        self.file_db = os.path.join(self.db_dir, file_db)

    def query(self, age=None, author=None, line=None):
        conn = sqlite3.connect(self.file_db)
        cursor = conn.cursor()

        query = '''
            SELECT a.name, a.citation, a.doi, d.age, d.trace_id
            FROM datasets d
            JOIN authors a ON d.author = a.id
        '''

        conditions = []
        params = []
        if age is not None:
            conditions.append('d.age = ?')
            params.append(age)
        if author is not None:
            conditions.append('a.name = ?')
            params.append(author)
        if line is not None:
            conditions.append('d.line = ?')
            params.append(line)

        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)

        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        metadata = {
            'author': set(),
            'age': set(),
            'reference': set(),
            'doi': set(),
            'trace_id': set(),
            '_query_params': {'age': age, 'author': author, 'line': line}
        }

        for author_name, citations, doi, ages, line in results:
            metadata['author'].add(author_name)
            metadata['age'].add(ages)
            metadata['reference'].add(citations)
            metadata['doi'].add(doi)
            metadata['trace_id'].add(line)

        return metadata

    def _get_file_paths_from_metadata(self, metadata):
        """Internal method: Get file paths for all entries in metadata."""
        query_params = metadata.get('_query_params', {})
        age = query_params.get('age')
        author = query_params.get('author')
        line = query_params.get('line')

        conn = sqlite3.connect(self.file_db)
        cursor = conn.cursor()
        query = '''
            SELECT d.file_path
            FROM datasets d
            JOIN authors a ON d.author = a.id
        '''
        conditions = []
        params = []
        if age is not None:
            conditions.append('d.age = ?')
            params.append(age)
        if author is not None:
            conditions.append('a.name = ?')
            params.append(author)
        if line is not None:
            conditions.append('d.line = ?')
            params.append(line)
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        cursor.execute(query, params)
        file_paths = [row[0] for row in cursor.fetchall()]
        conn.close()
        return file_paths

    def data_generator(self, metadata):
        """Generate DataFrames and their associated authors for all entries in metadata."""
        query_params = metadata.get('_query_params', {})
        age = query_params.get('age')
        author = query_params.get('author')
        line = query_params.get('line')

        conn = sqlite3.connect(self.file_db)
        cursor = conn.cursor()
        query = '''
            SELECT d.file_path, a.name
            FROM datasets d
            JOIN authors a ON d.author = a.id
        '''
        conditions = []
        params = []
        if age is not None:
            conditions.append('d.age = ?')
            params.append(age)
        if author is not None:
            conditions.append('a.name = ?')
            params.append(author)
        if line is not None:
            conditions.append('d.line = ?')
            params.append(line)
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        for file_path, author_name in results:
            df = pd.read_pickle(file_path)
            yield df, author_name

    def plotXY(self,
               metadata: dict,
               title: str = '',
               xlim: tuple = (None, None),
               ylim: tuple = (None, None),
               scale_factor: float = 1.0,
               latex:bool = False,
               save: Union[str, None] = None) -> None:

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
        for df, author in self.data_generator(metadata):
            plt.plot(df.x/1000, df.y/1000, linewidth=1, color=colors[author])

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
