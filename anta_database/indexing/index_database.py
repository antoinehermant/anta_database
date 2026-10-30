from multiprocessing import Pool, cpu_count
from functools import partial
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
from typing import Optional, Union


class IndexDatabase:
    def __init__(
        self,
        database_dir: str,
        file_db: str = "AntADatabase.db",
        assign_ages: bool = False,
        cpus: int = cpu_count() - 1,
    ) -> None:
        self._db_dir = database_dir
        self._file_db = os.path.join(self._db_dir, file_db)
        self._disable_tqdm = os.getenv("JUPYTER_BOOK_BUILD", False)
        self._assign_ages = assign_ages
        self.cpus = cpus

    @staticmethod
    def _target_ages(
        f_irh: np.ndarray, md_raw: pd.DataFrame, assign_age_df: Union[pd.DataFrame, None]
    ):
        """
        Ages to assign to the IRHs of a file (in the order of f_irh), from the metadata
        of the dataset (raw_md.json) and the overrides of assign_ages.json (by IRH ID).

        Returns (ids or None, ages, age uncertainties), NaN where unknown.
        """
        inconsistent_irhs = md_raw.groupby("IRH")["age"].nunique()
        if (inconsistent_irhs > 1).any():
            raise ValueError(
                f"Inconsistent ages for IRHs in {md_raw['dataset'].iloc[0]}: {inconsistent_irhs[inconsistent_irhs > 1].index.tolist()}"
            )
        md_raw = md_raw.drop_duplicates(subset="IRH", keep="first")
        # Ensures IRH are in the same order and fill with NaN values if some are missing
        md_raw = md_raw.set_index("IRH").reindex(f_irh, fill_value=np.nan)

        n = len(f_irh)
        ids = md_raw["IRH ID"].to_numpy() if "IRH ID" in md_raw.columns else None
        ages = (
            md_raw["age"].to_numpy(dtype=float)
            if "age" in md_raw
            else np.full(n, np.nan)
        )
        uncs = (
            md_raw["age uncertainty"].to_numpy(dtype=float)
            if "age uncertainty" in md_raw
            else np.full(n, np.nan)
        )

        if ids is not None and assign_age_df is not None:
            for i, irh_id in enumerate(ids):
                if pd.isna(irh_id) or irh_id not in assign_age_df.index:
                    continue
                assigned = assign_age_df.loc[irh_id]
                ages[i] = assigned["age"]
                if "age uncertainty" in assigned and pd.notna(assigned["age uncertainty"]):
                    uncs[i] = assigned["age uncertainty"]

        # Ages are whole years
        for name, values in (("age", ages), ("age uncertainty", uncs)):
            known = values[np.isfinite(values)]
            not_years = known[known != np.round(known)]
            if not_years.size:
                raise ValueError(
                    f"Ages must be whole years (e.g. 10000): {name} {not_years.tolist()} "
                    f"in the metadata of {md_raw['dataset'].dropna().iloc[0]} or in assign_ages.json"
                )
        return ids, ages, uncs

    @staticmethod
    def _same_values(old, new) -> bool:
        """Equal numbers, whatever the dtype (1000 == 1000.0), and NaN equal to NaN."""
        old = np.asarray(old, dtype=float)
        new = np.asarray(new, dtype=float)
        return old.shape == new.shape and np.array_equal(old, new, equal_nan=True)

    @staticmethod
    def _stored(values: np.ndarray) -> np.ndarray:
        """Years as integers, or floats when some are missing (NaN)."""
        if np.all(np.isfinite(values)):
            return values.astype(np.int64)
        return values.astype(float)

    @staticmethod
    def _id_strings(ids) -> list:
        return ["nan" if pd.isna(i) else str(i) for i in ids]

    def _do_assign_ages(
        self, h5_file: str, md_dict: dict, assign_age_df: Union[pd.DataFrame, None]
    ) -> dict:
        """
        Assign the ages of the IRHs of a file, only if they changed.

        The file is first read: when the ages it holds are already the ones to assign, it
        is left untouched (the same bytes, so it is not published or synced again).
        Returns {"file", "dataset", "status": "unchanged" | "updated" | "new", "changes"},
        changes being (IRH, old ID, new ID, old age, new age, old unc, new unc) per IRH.
        """
        with xr.open_dataset(h5_file, engine="h5netcdf") as ds:
            dataset = ds.attrs["dataset"]
            f_irh = ds.IRH.values
            old_ages = ds["IRH_AGE"].values if "IRH_AGE" in ds.variables else None
            old_uncs = ds["IRH_AGE_UNC"].values if "IRH_AGE_UNC" in ds.variables else None
            old_ids = ds["IRH_ID"].values if "IRH_ID" in ds.variables else None

        ids, ages, uncs = self._target_ages(f_irh, md_dict[dataset], assign_age_df)
        result = {"file": h5_file, "dataset": dataset, "changes": []}

        assigned = old_ages is not None and old_uncs is not None
        if ids is not None:
            assigned = assigned and old_ids is not None
        if assigned:
            same_ids = ids is None or self._id_strings(old_ids) == self._id_strings(ids)
            if (
                same_ids
                and self._same_values(old_ages, ages)
                and self._same_values(old_uncs, uncs)
            ):
                result["status"] = "unchanged"
                return result
            old_id_str = self._id_strings(old_ids) if ids is not None else [None] * len(f_irh)
            new_id_str = self._id_strings(ids) if ids is not None else [None] * len(f_irh)
            for i, irh in enumerate(f_irh):
                if (
                    old_id_str[i] != new_id_str[i]
                    or not self._same_values(old_ages[i], ages[i])
                    or not self._same_values(old_uncs[i], uncs[i])
                ):
                    # None for missing values: NaN != NaN would repeat the change per file
                    values = [
                        None if np.isnan(v) else v
                        for v in map(float, (old_ages[i], ages[i], old_uncs[i], uncs[i]))
                    ]
                    result["changes"].append(
                        (irh.item() if hasattr(irh, "item") else irh, old_id_str[i], new_id_str[i], *values)
                    )
            result["status"] = "updated"
        else:
            result["status"] = "new"

        # Read everything before writing: the file must be closed when it is rewritten
        with xr.open_dataset(h5_file, engine="h5netcdf") as ds:
            ds = ds.load()

        ds["IRH_AGE"] = ("IRH", self._stored(ages))
        ds["IRH_AGE_UNC"] = ("IRH", self._stored(uncs))
        if ids is not None:
            ds = ds.assign_coords(IRH_ID=("IRH", ids))
            ds = ds.set_coords(["IRH_ID", "IRH_AGE"])
            ds = ds.swap_dims({"IRH": "IRH_ID"})
            # NOTE: we swap again the dims to keep IRH as
            # primary dim. Not sure why we need to swap twice,
            # but it was not working without the first swap
            ds = ds.swap_dims({"IRH_ID": "IRH"})
        else:
            ds = ds.set_coords("IRH_AGE")

        # A new file, then replaced: the age variables change type when an age becomes
        # missing (NaN), which writing into the existing file cannot do, and a failure
        # never leaves a half-written file
        tmp_file = f"{h5_file}.tmp"
        try:
            ds.to_netcdf(tmp_file, engine="h5netcdf", mode="w")
            os.replace(tmp_file, h5_file)
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        return result

    @staticmethod
    def _report_age_changes(results: list) -> None:
        """Print the age changes once per dataset, and how many files were written."""

        def ka(age, unc):
            if age is None:
                return "no age"
            return f"{age / 1000:g} ka" if unc is None else f"{age / 1000:g} ± {unc / 1000:g} ka"

        by_dataset = {}
        for r in results:
            if r["status"] == "updated":
                entry = by_dataset.setdefault(r["dataset"], {"files": 0, "changes": set()})
                entry["files"] += 1
                entry["changes"].update(r["changes"])

        for dataset, entry in sorted(by_dataset.items()):
            print(f"\n Ages changed in {dataset} ({entry['files']} files):")
            for irh, old_id, new_id, old_age, new_age, old_unc, new_unc in sorted(
                entry["changes"], key=lambda c: (str(c[0]), str(c[1:]))
            ):
                line = f"   IRH {irh}: {ka(old_age, old_unc)} -> {ka(new_age, new_unc)}"
                if old_id != new_id:
                    line += f" (IRH ID {old_id} -> {new_id})"
                print(line)

        counts = {s: sum(r["status"] == s for r in results) for s in ("updated", "new", "unchanged")}
        print(
            f"\n Ages: {counts['updated']} files updated, {counts['new']} newly assigned,"
            f" {counts['unchanged']} unchanged"
        )

    @staticmethod
    def _same_content(db_a: str, db_b: str) -> bool:
        """Same tables and rows (an SQL dump), whatever the SQLite version wrote in the file."""

        def dump(path):
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                return list(conn.iterdump())
            except sqlite3.DatabaseError:
                return None  # not a readable database: replace it
            finally:
                conn.close()

        a = dump(db_a)
        return a is not None and a == dump(db_b)

    def index_database(self):
        h5_files = []

        if os.path.exists(f"{self._db_dir}/assign_ages.json"):
            assign_age_df = pd.read_json(f"{self._db_dir}/assign_ages.json")
            assign_age_df = assign_age_df.set_index("IRH ID")
        else:
            assign_age_df = None

        # Sorted, so the same files always give the same database (row ids), whatever
        # order the file system lists them in
        dirs = sorted(glob.glob(f"{self._db_dir}/*/"))
        self._dataset_list = []
        md_dict = {}

        for d in dirs:
            _, dataset = os.path.split(d.rstrip("/"))
            md_dict[dataset] = pd.read_json(f"{d}/raw_md.json")
            with open(f"{d}/raw_md.json", "r", encoding="utf-8") as f:
                unique_data = []
                seen = set()
                index = json.load(f)
                for item in index:
                    combo = (
                        item["dataset"],
                        item["DOI publication"],
                        item["DOI dataset"],
                    )
                    if combo not in seen:
                        seen.add(combo)
                        unique_data.append(
                            {
                                "dataset": item["dataset"],
                                "citation": item["citation"],
                                "DOI publication": item["DOI publication"],
                                "DOI dataset": item["DOI dataset"],
                            }
                        )
                for unique in unique_data:
                    self._dataset_list.append(unique)

        for ref in self._dataset_list:
            found_files = sorted(
                glob.glob(f'{self._db_dir}/{ref["dataset"]}/h5/*.h5', recursive=False)
            )
            if found_files:
                h5_files.extend(found_files)
            else:
                print(f"No h5 file found for {ref['dataset']} dataset")

        print(f"\n Found {len(h5_files)} files to index")

        if self._assign_ages:
            num_tasks = len(h5_files)
            num_workers = min(num_tasks, self.cpus)
            print(
                "\n assign_ages is True:",
                "will assign ages in",
                num_tasks,
                "isochrone files\n" "\n   ",
                num_workers,
                "worker(s) allocated out of",
                cpu_count(),
                "available cpus\n",
            )

            partial_process = partial(
                self._do_assign_ages, md_dict=md_dict, assign_age_df=assign_age_df
            )
            if num_workers > 1:
                with Pool(num_workers) as pool:
                    results = list(
                        tqdm(
                            pool.imap_unordered(partial_process, h5_files),
                            total=num_tasks,
                            desc="Processing",
                            unit="file",
                        )
                    )
            else:
                results = [
                    partial_process(f)
                    for f in tqdm(h5_files, desc="Processing", unit="file")
                ]
            self._report_age_changes(results)

        # Build into a temporary file, so a failure never leaves a broken or missing database
        tmp_db = f"{self._file_db}.tmp"
        if os.path.exists(tmp_db):
            os.remove(tmp_db)

        print("\n Creating the main table...")

        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        # Create a table for original reference to datasets
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                citation TEXT,
                DOI_dataset TEXT,
                DOI_publication TEXT
            )
        """)

        for row in self._dataset_list:
            try:
                cursor.execute(
                    "INSERT INTO sources (name, citation, DOI_dataset, DOI_publication) VALUES (?, ?, ?, ?)",
                    (
                        row["dataset"],
                        row["citation"],
                        row["DOI dataset"],
                        row["DOI publication"],
                    ),
                )
            except sqlite3.IntegrityError:
                # dataset already exists, skip
                continue

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE,
                dataset INTEGER,
                institute TEXT,
                project TEXT,
                radar_instrument TEXT,
                acq_year TEXT,
                flight_id TEXT,
                FOREIGN KEY (dataset) REFERENCES sources (id)
                )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS variables (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE
                )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ages (
                id INTEGER PRIMARY KEY,
                age TEXT,
                age_unc TEXT
                )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regions (
                id INTEGER PRIMARY KEY,
                region TEXT,
                IMBIE_basin TEXT
                )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE
                )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS institutes (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE
                )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS radar_instruments (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE
                )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_variables (
                dataset_id INTEGER,
                variable_id INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (variable_id) REFERENCES variables (id),
                PRIMARY KEY (dataset_id, variable_id)
                )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_ages (
                dataset_id INTEGER,
                age_id INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (age_id) REFERENCES ages (id),
                PRIMARY KEY (dataset_id, age_id)
                )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_projects (
                dataset_id INTEGER,
                project_id INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (project_id) REFERENCES projects (id),
                PRIMARY KEY (dataset_id, project_id)
                )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_institutes (
                dataset_id INTEGER,
                institute_id INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (institute_id) REFERENCES institutes (id),
                PRIMARY KEY (dataset_id, institute_id)
                )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_radar_instruments (
                dataset_id INTEGER,
                radar_instrument_id INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (radar_instrument_id) REFERENCES radar_instruments (id),
                PRIMARY KEY (dataset_id, radar_instrument_id)
                )
            """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_regions (
                dataset_id INTEGER,
                region_id INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id),
                FOREIGN KEY (region_id) REFERENCES regions (id),
                PRIMARY KEY (dataset_id, region_id)
                )
            """)
        print(" Done.")

        print(
            "\n Adding",
            len(h5_files),
            "files to the table...\n",
        )

        for f in tqdm(h5_files, desc="Indexing files", disable=self._disable_tqdm):
            _, file_name = os.path.split(f)

            with h5py.File(f, "r") as ds:
                flight_id = ds.attrs["flight ID"]
                if flight_id in ["nan", "none"]:
                    flight_id = None

                institutes = ds.attrs["institute"]
                if isinstance(institutes, str):
                    try:
                        institutes = ast.literal_eval(institutes)
                    except (ValueError, SyntaxError):
                        institutes = [institutes]
                if not isinstance(institutes, (list, np.ndarray)):
                    if institutes in ["nan", "none", None]:
                        institutes = []
                    else:
                        institutes = [institutes]
                else:
                    institutes = list(institutes)

                projects = ds.attrs["project"]
                if isinstance(projects, str):
                    try:
                        projects = ast.literal_eval(projects)
                    except (ValueError, SyntaxError):
                        projects = [projects]
                if not isinstance(projects, (list, np.ndarray)):
                    if projects in ["nan", "none", None]:
                        projects = []
                    else:
                        projects = [projects]
                else:
                    projects = list(projects)

                acq_year = ds.attrs["acquisition year"]
                if acq_year in ["nan", "none"]:
                    acq_year = None

                dataset = ds.attrs["dataset"]
                if dataset in ["nan", "none"]:
                    dataset = None

                radar_instruments = ds.attrs["radar instrument"]
                if isinstance(radar_instruments, str):
                    try:
                        radar_instruments = ast.literal_eval(radar_instruments)
                    except (ValueError, SyntaxError):
                        radar_instruments = [radar_instruments]
                if not isinstance(radar_instruments, (list, np.ndarray)):
                    if radar_instruments in ["nan", "none", None]:
                        radar_instruments = []
                    else:
                        radar_instruments = [radar_instruments]
                else:
                    radar_instruments = list(radar_instruments)

                ds_vars = list(ds.keys())

                if "IRH_AGE" in ds_vars:
                    ages = ds["IRH_AGE"][:]
                    age_uncs = ds["IRH_AGE_UNC"][:]
                else:
                    ages = None
                    age_uncs = None

                basin_mapping = json.loads(ds.attrs["IMBIE_basins"])

            relative_file_path = f"{dataset}/h5/{file_name}"

            # Get the dataset's ID from the dataset table NOTE: This is essential!! Don't remove
            cursor.execute("SELECT id FROM sources WHERE name = ?", (dataset,))
            dataset_id = cursor.fetchone()[0]

            # Insert into datasets and get the new row id
            cursor.execute(
                """
                INSERT INTO datasets (file_path, dataset, acq_year, flight_id)
                VALUES (?, ?, ?, ?)
            """,
                (relative_file_path, dataset_id, str(acq_year), flight_id),
            )
            dataset_row_id = cursor.lastrowid

            # Insert projects and link to dataset
            for project in projects:
                cursor.execute(
                    "INSERT OR IGNORE INTO projects (name) VALUES (?)", (project,)
                )
                project_id = cursor.execute(
                    "SELECT id FROM projects WHERE name = ?", (project,)
                ).fetchone()[0]
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO dataset_projects (dataset_id, project_id)
                    VALUES (?, ?)
                """,
                    (dataset_row_id, project_id),
                )

            # Insert institutes and link to dataset
            for institute in institutes:
                cursor.execute(
                    "INSERT OR IGNORE INTO institutes (name) VALUES (?)", (institute,)
                )
                institute_id = cursor.execute(
                    "SELECT id FROM institutes WHERE name = ?", (institute,)
                ).fetchone()[0]
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO dataset_institutes (dataset_id, institute_id)
                    VALUES (?, ?)
                """,
                    (dataset_row_id, institute_id),
                )

            # Insert radar_instruments and link to dataset
            for radar_instrument in radar_instruments:
                cursor.execute(
                    "INSERT OR IGNORE INTO radar_instruments (name) VALUES (?)",
                    (radar_instrument,),
                )
                radar_instrument_id = cursor.execute(
                    "SELECT id FROM radar_instruments WHERE name = ?",
                    (radar_instrument,),
                ).fetchone()[0]
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO dataset_radar_instruments (dataset_id, radar_instrument_id)
                    VALUES (?, ?)
                """,
                    (dataset_row_id, radar_instrument_id),
                )

            # Insert variables
            var_list = [
                "ICE_THK",
                "SURF_ELEV",
                "BED_ELEV",
                "BASAL_UNIT",
                "IRH_NUM",
                "IRH_DEPTH",
            ]
            for var in var_list:
                if var in ds_vars:
                    cursor.execute(
                        "INSERT OR IGNORE INTO variables (name) VALUES (?)", (var,)
                    )
                    var_id = cursor.execute(
                        "SELECT id FROM variables WHERE name = ?", (var,)
                    ).fetchone()[0]
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO dataset_variables (dataset_id, variable_id)
                        VALUES (?, ?)
                    """,
                        (dataset_row_id, var_id),
                    )

            # Insert ages and regions
            if ages is not None:
                for age, age_unc in zip(ages, age_uncs):
                    cursor.execute(
                        "INSERT OR IGNORE INTO ages (age, age_unc) VALUES (?, ?)",
                        (str(age), str(age_unc)),
                    )
                    age_id = cursor.execute(
                        "SELECT id FROM ages WHERE age = ? AND age_unc = ?",
                        (str(age), str(age_unc)),
                    ).fetchone()[0]
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO dataset_ages (dataset_id, age_id)
                        VALUES (?, ?)
                    """,
                        (dataset_row_id, age_id),
                    )

            for basin, region in basin_mapping.items():
                cursor.execute(
                    "INSERT OR IGNORE INTO regions (region, IMBIE_basin) VALUES (?, ?)",
                    (region, basin),
                )
                region_id = cursor.execute(
                    "SELECT id FROM regions WHERE region = ? AND IMBIE_basin = ?",
                    (region, basin),
                ).fetchone()[0]
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO dataset_regions (dataset_id, region_id)
                    VALUES (?, ?)
                """,
                    (dataset_row_id, region_id),
                )

        conn.commit()
        conn.close()

        # Keep the existing database when nothing changed, so it is not published or
        # synced again
        if os.path.exists(self._file_db) and self._same_content(tmp_db, self._file_db):
            os.remove(tmp_db)
            print("\n The database is unchanged, kept as is.")
            return
        os.replace(tmp_db, self._file_db)

        print("\n Done creating the Database table.")
