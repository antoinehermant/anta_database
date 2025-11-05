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
        dataset_ages = {}
        pkl_files = []
        for _, row in self.index.iterrows():
            table = self.file_metadata(f"{self.db_dir}/{row.directory}/raw_files_md.csv")
            dataset_ages.update({f"{row.directory}": table})

            pkl_files.extend(list(glob.glob(f'{self.db_dir}/{row.directory}/pkl/**/*.pkl', recursive=False)))

        var_list = ['ICE_THCK', 'SURF_ELEV', 'BED_ELEV', 'BASAL_UNIT', 'IRH_DENS', 'IRH_FRAC_DEPTH', 'IRH_DEPTH']

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

        for file in tqdm(pkl_files, desc="Indexing files"):
            dir_name, file_name = os.path.split(file)
            pkl_dir, flight_id_dir = os.path.split(dir_name)
            dataset_dir, _ = os.path.split(pkl_dir)
            trace_md = pd.read_csv(f'{dir_name}/metadata.csv')
            basins_regions = pd.read_pickle(f'{dir_name}/IMBIE.pkl')
            trace_md['flight_id'] = trace_md['flight_id'].astype(str)

            dataset = os.path.basename(dataset_dir)
            file_name_, ext = os.path.splitext(file_name)
            relative_file_path = f'{dataset}/pkl/{flight_id_dir}/{file_name}'

            # Get the dataset's ID from the dataset table
            cursor.execute('SELECT id FROM sources WHERE name = ?', (dataset,))
            dataset_id = cursor.fetchone()[0]

            metadata = dataset_ages[dataset]

            if 'age' in metadata.columns:
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
            else:
                age = None
                age_unc = None

            if file_name_ in var_list:
                var = file_name_
            elif file_name_ in ['TOTAL_PSXPSY', 'IMBIE']:
                continue
            else:
                var = 'IRH_DEPTH'

            flight_id = trace_md.iloc[0]['flight_id']
            institute = trace_md.iloc[0]['institute']
            project = trace_md.iloc[0]['project']
            acq_year = trace_md.iloc[0]['acq_year']

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

            for index, row in basins_regions.iterrows():
                basin = row['Subregion']
                region = row['Regions']

                cursor.execute('''
                    INSERT INTO datasets (file_path, dataset, institute, project, acq_year, age, age_unc, var, flight_id, region, basin)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (relative_file_path, dataset_id, institute, project, acq_year, age, age_unc, var, flight_id, region, basin))

                if var == 'IRH_DEPTH' and os.path.exists(f'{dir_name}/ICE_THCK.pkl'):
                    var = 'IRH_FRAC_DEPTH' # If both IRH DEPTH and ICE THK exist, IRH FRAC DEPTH must have been calculated so index for it
                    cursor.execute('''
                        INSERT INTO datasets (file_path, dataset, institute, project, acq_year, age, age_unc, var, flight_id, region, basin)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (relative_file_path, dataset_id, institute, project, acq_year, age, age_unc, var, flight_id, region, basin))

        conn.commit()
        conn.close()
