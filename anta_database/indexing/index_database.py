import os
from tqdm import tqdm
import glob
import pandas as pd
import sqlite3
import sys

class IndexDatabase:
    def __init__(self, database_dir: str, file_db: str = 'AntADatabase.db', index: str = 'database_index.csv'):
        self.db_dir = database_dir
        self.file_db = os.path.join(self.db_dir, file_db)
        self.file_index = index
        self.index = pd.read_csv(f'{self.db_dir}/{self.file_index}', header=0)

    def file_metadata(self, file_path) -> pd.DataFrame:
        table = pd.read_csv(file_path, header=0, sep=',')
        table.set_index('raw_file', inplace=True)
        return table

    def index_database(self):
        Authors_ages = {}
        pkl_files = []
        for _, row in self.index.iterrows():
            table = self.file_metadata(f"{self.db_dir}/{row.directory}/raw_files_md.csv")
            Authors_ages.update({f"{row.directory}": table})

            pkl_files.extend(list(glob.glob(f'{self.db_dir}/{row.directory}/pkl/**/*.pkl', recursive=False)))

        var_list = pd.read_csv(f"{self.db_dir}/vars.csv").columns

        if os.path.exists(self.file_db):
            os.remove(self.file_db)

        conn = sqlite3.connect(self.file_db)
        cursor = conn.cursor()
        # Create a table for original reference to datasets
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS authors (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                citation TEXT,
                doi TEXT
            )
        ''')

        for _, row in self.index.iterrows():
            try:
                cursor.execute(
                    'INSERT INTO authors (name, citation, doi) VALUES (?, ?, ?)',
                    (row.directory, row.citation, row.doi)
                )
            except sqlite3.IntegrityError:
                # Author already exists, skip
                continue

        # Create a table to store the metadata for each dataset
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY,
                file_path TEXT,
                author TEXT,
                institute TEXT,
                project TEXT,
                acq_year TEXT,
                age TEXT,
                age_unc TEXT,
                var TEXT,
                trace_id TEXT,
                FOREIGN KEY (author) REFERENCES authors (id)
            )
        ''')

        for file in tqdm(pkl_files, desc="Indexing files"):
            dir_name, file_name = os.path.split(file)
            pkl_dir, trace_id = os.path.split(dir_name)
            author_dir, _ = os.path.split(pkl_dir)
            trace_md = pd.read_csv(f'{dir_name}/trace_md.csv')
            trace_md['trace_id'] = trace_md['trace_id'].astype(str)
            trace_md.set_index('trace_id', inplace=True)

            author = os.path.basename(author_dir)
            file_name_, ext = os.path.splitext(file_name)
            relative_file_path = f'{author}/pkl/{trace_id}/{file_name}'

            # Get the author's ID from the authors table
            cursor.execute('SELECT id FROM authors WHERE name = ?', (author,))
            author_id = cursor.fetchone()[0]

            metadata = Authors_ages[author]

            if file_name_ in metadata.index:
                age = int(metadata.loc[file_name_]['age'])
                age_unc = metadata.loc[file_name_]['age_unc']
                if not pd.isna(age_unc):
                    age_unc = int(age_unc)
                else:
                    age_unc = None
            else:
                age = None
                age_unc = None

            if file_name_ in var_list:
                var = file_name_
            else:
                var = None

            institute = trace_md.loc[trace_id]['institute']
            project = trace_md.loc[trace_id]['project']
            acq_year = trace_md.loc[trace_id]['acq_year']

            if not pd.isna(institute):
                institute = str(institute)
            else:
                institute = None
            if not pd.isna(project):
                project = str(project)
            else:
                project = None
            if not pd.isna(acq_year):
                acq_year = str(acq_year)
            else:
                acq_year = None

            cursor.execute('''
                INSERT INTO datasets (file_path, author, institute, project, acq_year, age, age_unc, var, trace_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (relative_file_path, author_id, institute, project, acq_year, age, age_unc, var, trace_id))

        conn.commit()
        conn.close()
