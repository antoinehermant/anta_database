import os
from tqdm import tqdm
import glob
import pandas as pd
import xarray as xr
import sqlite3
import h5py
import json

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
        dataset_ages = {}
        h5_files = []
        for _, row in self.index.iterrows():
            table = self.file_metadata(f"{self.db_dir}/{row.directory}/raw_files_md.csv")
            dataset_ages.update({f"{row.directory}": table})

            h5_files.extend(list(glob.glob(f'{self.db_dir}/{row.directory}/h5/*.h5', recursive=False)))

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
                dataset_doi TEXT,
                publication_doi TEXT
            )
        ''')

        for _, row in self.index.iterrows():
            try:
                cursor.execute(
                    'INSERT INTO sources (name, citation, dataset_doi, publication_doi) VALUES (?, ?, ?, ?)',
                    (row.directory, row.citation, row.dataset_doi, row.publication_doi)
                )
            except sqlite3.IntegrityError:
                # dataset already exists, skip
                continue

        # Create a table to store the metadata for each dataset
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY,
                file_path TEXT,
                dataset TEXT,
                institute TEXT,
                project TEXT,
                acq_year TEXT,
                age TEXT,
                age_unc TEXT,
                var TEXT,
                flight_id TEXT,
                region TEXT,
                basin TEXT,
                FOREIGN KEY (dataset) REFERENCES sources (id)
            )
        ''')

        for f in tqdm(h5_files, desc="Indexing files"):
            _, file_name = os.path.split(f)

            with h5py.File(f, 'r') as f:
                flight_id = f.attrs["flight_id"]
                if flight_id in ['nan', 'none']:
                    flight_id = None

                institute = f.attrs['institute']
                if institute in ['nan', 'none']:
                    institute = None

                project = f.attrs['project']
                if project in ['nan', 'none']:
                    project = None

                acq_year = f.attrs['acq_year']
                if acq_year in ['nan', 'none']:
                    acq_year = None

                dataset = f.attrs['dataset']
                if dataset in ['nan', 'none']:
                    dataset = None

                ds_vars = list(f.keys())

                if 'age' in ds_vars:
                    ages = f['age'][:]
                    age_uncs = f['age_uncertainty'][:]
                    age_uncs = pd.DataFrame({
                        'age': ages,
                        'age_unc': age_uncs
                    })
                    age_uncs = age_uncs.set_index('age')
                else:
                    ages = None
                    age_uncs = None

                basin_mapping = json.loads(f.attrs['basins'])

            relative_file_path = f'{dataset}/h5/{file_name}'

            # Get the dataset's ID from the dataset table NOTE: This is essential!! Don't remove
            cursor.execute('SELECT id FROM sources WHERE name = ?', (dataset,))
            dataset_id = cursor.fetchone()[0]

            var_list = ['ICE_THCK', 'SURF_ELEV', 'BED_ELEV', 'BASAL_UNIT', 'IRH_DENS']

            if ages is not None:
                for age in ages:
                    if age_uncs is not None:
                        age_unc = age_uncs.loc[age]
                    else:
                        age_unc = None

                    for basin, region in basin_mapping.items():
                        cursor.execute('''
                            INSERT INTO datasets (file_path, dataset, institute, project, acq_year, age, age_unc, var, region, basin, flight_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (relative_file_path, dataset_id, institute, project, str(acq_year), str(age), str(age_unc), 'IRH_DEPTH', region, basin, flight_id))

            for var in var_list:
                if var in ds_vars:
                    for basin, region in basin_mapping.items():
                        cursor.execute('''
                            INSERT INTO datasets (file_path, dataset, institute, project, acq_year, age, age_unc, var, region, basin, flight_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (relative_file_path, dataset_id, institute, project, str(acq_year), None, None, var, region, basin, flight_id))

        conn.commit()
        conn.close()
