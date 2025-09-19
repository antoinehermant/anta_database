import os
import glob
import pandas as pd
import sqlite3

class IndexDatabase:
    def __init__(self, database_dir: str, file_db: str = 'AntADatabase.db', index: str = 'database_index.csv'):
        self.db_dir = database_dir
        self.file_db = os.path.join(self.db_dir, file_db)
        self.file_index = index
        self.index = pd.read_csv(f'{self.db_dir}/{self.file_index}', header=0)

    def get_dict_ages(self, tab_file) -> dict:
        ages = pd.read_csv(tab_file, header=None, sep='\t', names=['file', 'age'])
        return dict(zip(ages['file'], ages['age']))

    def get_dict_vars(self, tab_file) -> dict:
        ages = pd.read_csv(tab_file, header=None, sep='\t', names=['file', 'var'])
        return dict(zip(ages['file'], ages['var']))

    def index_database(self):
        Authors_ages = {}
        for _, row in self.index.iterrows():
            ages = self.get_dict_ages(f"{self.db_dir}/{row.directory}/IRH_ages.tab")
            Authors_ages.update({f"{row.directory}": ages})

        Extra_vars = {}
        for _, row in self.index.iterrows():
            vars = self.get_dict_ages(f"{self.db_dir}/{row.directory}/vars.tab")
            Extra_vars.update({f"{row.directory}": vars})

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
                var TEXT,
                trace_id TEXT,
                FOREIGN KEY (author) REFERENCES authors (id)
            )
        ''')

        # Insert the metadata for each dataset into the table
        for file in glob.glob(f'{self.db_dir}/**/**/*.pkl', recursive=True):
            dir_name, file_name = os.path.split(file)
            pkl_dir, trace_id = os.path.split(dir_name)
            author_dir, _ = os.path.split(pkl_dir)

            author = os.path.basename(author_dir)
            file_name_, ext = os.path.splitext(file_name)
            relative_file_path = f'{author}/pkl/{trace_id}/{file_name}'
            print(relative_file_path)

            # Get the author's ID from the authors table
            cursor.execute('SELECT id FROM authors WHERE name = ?', (author,))
            author_id = cursor.fetchone()[0]

            if file_name_ in Authors_ages[author]:
                age = Authors_ages[author][file_name_]
            else:
                age = None

            if file_name_ in Extra_vars[author]:
                var = Extra_vars[author][file_name_]
            else:
                var = None

            cursor.execute('''
                INSERT INTO datasets (file_path, author, age, var, trace_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (relative_file_path, author_id, age, var, trace_id))

        conn.commit()
        conn.close()
