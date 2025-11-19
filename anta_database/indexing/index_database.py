import os
from tqdm import tqdm
import glob
import pandas as pd
import numpy as np
import xarray as xr
import sqlite3
import h5py
import json
import ast

class IndexDatabase:
    def __init__(self, database_dir: str, file_db: str = 'AntADatabase.db', index: str = 'references.json'):
        self.db_dir = database_dir
        self.file_db = os.path.join(self.db_dir, file_db)
        self.file_index = index
        with open(os.path.join(self.db_dir, self.file_index), 'r', encoding='utf-8') as f:
            self.index = json.load(f)

    def file_metadata(self, file_path) -> pd.DataFrame:
        table = pd.read_csv(file_path, header=0, sep=',')
        table.set_index('raw_file', inplace=True)
        return table

    def index_database(self):
        h5_files = []
        for ref in self.index:
            found_files = glob.glob(f'{self.db_dir}/{ref['dataset']}/h5/*.h5', recursive=False)
            if found_files:
                h5_files.extend(found_files)
            else:
                print(f'No h5 file found for {ref['dataset']} dataset')

        print(f"\n Found {len(h5_files)} files to index")

        if os.path.exists(self.file_db):
            os.remove(self.file_db)

        conn = sqlite3.connect(self.file_db)
        cursor = conn.cursor()
        # Create a table for original reference to datasets
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                citation TEXT,
                DOI_dataset TEXT,
                DOI_publication TEXT
            )
        ''')

        for row in self.index:
            try:
                cursor.execute(
                    'INSERT INTO sources (name, citation, DOI_dataset, DOI_publication) VALUES (?, ?, ?, ?)',
                    (row['dataset'], row['citation'], row['DOI_dataset'], row['DOI_publication'])
                )
            except sqlite3.IntegrityError:
                # dataset already exists, skip
                continue

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE,
                dataset INTEGER,
                institute TEXT,
                project TEXT,
                acq_year TEXT,
                radar_instrument TEXT,
                flight_id TEXT,
                FOREIGN KEY (dataset) REFERENCES sources (id)
                )
            ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS variables (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE
                )
            ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ages (
                id INTEGER PRIMARY KEY,
                age TEXT,
                age_unc TEXT
                )
            ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS regions (
                id INTEGER PRIMARY KEY,
                region TEXT,
                IMBIE_basin TEXT
                )
            ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE
                )
            ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS institutes (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE
                )
            ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_variables (
                dataset_id INTEGER,
                variable_id INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (variable_id) REFERENCES variables (id),
                PRIMARY KEY (dataset_id, variable_id)
                )
            ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_ages (
                dataset_id INTEGER,
                age_id INTEGER,
                region_id INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (age_id) REFERENCES ages (id),
                FOREIGN KEY (region_id) REFERENCES regions (id),
                PRIMARY KEY (dataset_id, age_id, region_id)
                )
            ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_projects (
                dataset_id INTEGER,
                project_id INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (project_id) REFERENCES projects (id),
                PRIMARY KEY (dataset_id, project_id)
                )
            ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_institutes (
                dataset_id INTEGER,
                institute_id INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (institute_id) REFERENCES institutes (id),
                PRIMARY KEY (dataset_id, institute_id)
                )
            ''')

        for f in tqdm(h5_files, desc="Indexing files"):
            _, file_name = os.path.split(f)

            with h5py.File(f, 'r') as f:
                flight_id = f.attrs["flight ID"]
                if flight_id in ['nan', 'none']:
                    flight_id = None

                institutes = f.attrs['institute']
                if isinstance(institutes, str):
                    try:
                        institutes = ast.literal_eval(institutes)
                    except (ValueError, SyntaxError):
                        institutes = [institutes]
                if not isinstance(institutes, (list, np.ndarray)):
                    if institutes in ['nan', 'none', None]:
                        institutes = []
                    else:
                        institutes = [institutes]
                else:
                    institutes = list(institutes)

                projects = f.attrs['project']
                if isinstance(projects, str):
                    try:
                        projects = ast.literal_eval(projects)
                    except (ValueError, SyntaxError):
                        projects = [projects]
                if not isinstance(projects, (list, np.ndarray)):
                    if projects in ['nan', 'none', None]:
                        projects = []
                    else:
                        projects = [projects]
                else:
                    projects = list(projects)

                acq_year = f.attrs['acquisition year']
                if acq_year in ['nan', 'none']:
                    acq_year = None

                dataset = f.attrs['dataset']
                if dataset in ['nan', 'none']:
                    dataset = None

                radar = f.attrs['radar instrument']
                if radar in ['nan', 'none']:
                    radar = None

                ds_vars = list(f.keys())

                if 'IRH_AGE' in ds_vars:
                    ages = f['IRH_AGE'][:]
                    age_uncs = f['IRH_AGE_UNC'][:]
                    age_uncs = pd.DataFrame({
                        'age': ages,
                        'age_unc': age_uncs
                    })
                    age_uncs = age_uncs.set_index('age')
                else:
                    ages = None
                    age_uncs = None

                basin_mapping = json.loads(f.attrs['IMBIE_basins'])

            relative_file_path = f'{dataset}/h5/{file_name}'

            # Get the dataset's ID from the dataset table NOTE: This is essential!! Don't remove
            cursor.execute('SELECT id FROM sources WHERE name = ?', (dataset,))
            dataset_id = cursor.fetchone()[0]

            # Insert into datasets and get the new row id
            cursor.execute('''
                INSERT INTO datasets (file_path, dataset, acq_year, radar_instrument, flight_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (relative_file_path, dataset_id, str(acq_year), radar, flight_id))
            dataset_row_id = cursor.lastrowid

            # Insert projects and link to dataset
            for project in projects:
                cursor.execute('INSERT OR IGNORE INTO projects (name) VALUES (?)', (project,))
                project_id = cursor.execute('SELECT id FROM projects WHERE name = ?', (project,)).fetchone()[0]
                cursor.execute('''
                    INSERT OR IGNORE INTO dataset_projects (dataset_id, project_id)
                    VALUES (?, ?)
                ''', (dataset_row_id, project_id))

            # Insert institutes and link to dataset
            for institute in institutes:
                cursor.execute('INSERT OR IGNORE INTO institutes (name) VALUES (?)', (institute,))
                institute_id = cursor.execute('SELECT id FROM institutes WHERE name = ?', (institute,)).fetchone()[0]
                cursor.execute('''
                    INSERT OR IGNORE INTO dataset_institutes (dataset_id, institute_id)
                    VALUES (?, ?)
                ''', (dataset_row_id, institute_id))

            # Insert variables
            var_list = ['ICE_THK', 'SURF_ELEV', 'BED_ELEV', 'BASAL_UNIT', 'IRH_NUM', 'IRH_DEPTH']
            for var in var_list:
                if var in ds_vars or var == 'IRH_DEPTH':
                    cursor.execute('INSERT OR IGNORE INTO variables (name) VALUES (?)', (var,))
                    var_id = cursor.execute('SELECT id FROM variables WHERE name = ?', (var,)).fetchone()[0]
                    cursor.execute('''
                        INSERT OR IGNORE INTO dataset_variables (dataset_id, variable_id)
                        VALUES (?, ?)
                    ''', (dataset_row_id, var_id))

            # Insert ages and regions
            if ages is not None:
                for age, age_unc in age_uncs.iterrows():
                    cursor.execute('INSERT OR IGNORE INTO ages (age, age_unc) VALUES (?, ?)', (str(age), str(age_unc['age_unc'])))
                    age_id = cursor.execute('SELECT id FROM ages WHERE age = ? AND age_unc = ?', (str(age), str(age_unc['age_unc']))).fetchone()[0]
                    for basin, region in basin_mapping.items():
                        cursor.execute('INSERT OR IGNORE INTO regions (region, IMBIE_basin) VALUES (?, ?)', (region, basin))
                        region_id = cursor.execute('SELECT id FROM regions WHERE region = ? AND IMBIE_basin = ?', (region, basin)).fetchone()[0]
                        cursor.execute('''
                            INSERT OR IGNORE INTO dataset_ages (dataset_id, age_id, region_id)
                            VALUES (?, ?, ?)
                        ''', (dataset_row_id, age_id, region_id))

        conn.commit()
        conn.close()
