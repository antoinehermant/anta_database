import os
from tqdm import tqdm
import glob
import pandas as pd
import sqlite3

class IndexDatabase:
    def __init__(self, database_dir: str, file_db: str = 'AntADatabase.db', index: str = 'database_index.csv'):
        self.db_dir = database_dir
        self.file_db = os.path.join(self.db_dir, file_db)
        self.file_index = index
        self.index = pd.read_csv(f'{self.db_dir}/{self.file_index}', header=0)

    def get_dict_ages(self, tab_file) -> pd.DataFrame:
        ages = pd.read_csv(tab_file, header=None, sep='\t', names=['file', 'age', 'age_unc'])
        ages.set_index('file', inplace=True)
        return ages

    def index_database(self):
        Authors_ages = {}
        for _, row in self.index.iterrows():
            ages = self.get_dict_ages(f"{self.db_dir}/{row.directory}/IRH_ages.tab")
            Authors_ages.update({f"{row.directory}": ages})

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
                age TEXT,
                age_unc TEXT,
                var TEXT,
                trace_id TEXT,
                FOREIGN KEY (author) REFERENCES authors (id)
            )
        ''')

        pkl_files = list(glob.glob(f'{self.db_dir}/**/**/*.pkl', recursive=True))
        for file in tqdm(pkl_files, desc="Processing files"):
            dir_name, file_name = os.path.split(file)
            pkl_dir, trace_id = os.path.split(dir_name)
            author_dir, _ = os.path.split(pkl_dir)

            author = os.path.basename(author_dir)
            file_name_, ext = os.path.splitext(file_name)
            relative_file_path = f'{author}/pkl/{trace_id}/{file_name}'

            # Get the author's ID from the authors table
            cursor.execute('SELECT id FROM authors WHERE name = ?', (author,))
            author_id = cursor.fetchone()[0]

            ages = Authors_ages[author]

            if file_name_ in ages.index:
                age = int(ages.loc[file_name_]['age'])
                age_unc = ages.loc[file_name_]['age_unc']
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

            cursor.execute('''
                INSERT INTO datasets (file_path, author, age, age_unc, var, trace_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (relative_file_path, author_id, age, age_unc, var, trace_id))

        conn.commit()
        conn.close()
