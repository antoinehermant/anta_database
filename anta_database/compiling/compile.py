import warnings
import h5py
import json
import os
import ast
import shutil
import time
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import xarray as xr
import glob
from pyproj import Transformer
from typing import Union, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from importlib.resources import files
from scipy.spatial import cKDTree

from anta_database.compiling.lines import order_by_files, order_line, out_of_order, path_length, split_into_lines
from anta_database.netcdf import SOURCE_VARIABLES, SOURCES, source_attrs, write

# Lines were called flights before ground-based radar joined the database: the old
# names of raw files and raw_md.json are still read as the new ones.
OLD_COLUMN_NAMES = {"Flight_ID": "Line_ID"}
OLD_RAW_MD_KEYS = {"flight ID": "line ID", "flight ID prefix": "line ID prefix"}

# What the compilation has to do to the raw files of a dataset, kept next to them, and recorded
# in the attribute "processing" of its files:
#   wave_speed             m/ns: the raw depths are two-way travel times (ns), multiplied by it
#   firn_correction        m: added to the depths
#   firn_correction_vars   what it is added to, ["IRH_DEPTH", "ICE_THK"] by default
#   line_ids               "dataset": the raw files have no line IDs, the lines are named after
#                          the dataset (Dataset_1, Dataset_2... when split). With one file per
#                          layer, raw_md.json gives that name as "line ID"
#   split_at_jump_km       cut a file of the "line" structure where consecutive points are further
#                          apart than this; with one file per layer, the points named by "line ID"
#                          are put in order and cut into lines at larger gaps (lines.py)
#   drop_vars              {raw file: [variables]}: values of these files left out (found wrong),
#                          the reason in "note"
#   same_trace_within_m    m: with one file per layer, a point of a file this close to a point of
#                          the files before (each the nearest of the other) is the same trace,
#                          given in another file with other coordinates: it takes theirs
#   note                   anything else to report
SETTINGS_FILE = "compile_settings.json"
DEPTH_VARS = ["IRH_DEPTH", "ICE_THK"]


class CompileDatabase:
    def __init__(
        self,
        dir_list: Union[str, list[str]],
        wave_speed: Optional[float] = None,
        firn_correction: Optional[float] = None,
        extract_data: bool = True,
        compute_distance: bool = True,
        compute_IMBIE_basins: bool = True,
        compute_IRH_density: bool = True,
        set_attributes: bool = True,
        shapefiles: bool = False,
        geopackages: bool = False,
        remove_tmp_files: bool = True,
        var_attrs_json: Optional[str] = None,
    ) -> None:
        self._dir_list = dir_list
        self._wave_speed = wave_speed
        self._firn_correction = firn_correction
        self._extract_data = extract_data
        self._compute_distance = compute_distance
        self._compute_IRH_density = compute_IRH_density
        self._compute_IMBIE_basins = compute_IMBIE_basins
        self._set_attributes = set_attributes
        self._shapefiles = shapefiles
        self._geopackages = geopackages
        self._remove_tmp_files = remove_tmp_files
        imbie_path = files("anta_database.data").joinpath(
            "ANT_Basins_IMBIE2_v1.6.parquet"
        )
        self._basins = gpd.read_parquet(imbie_path)
        self._var_list = ["ICE_THK", "BED_ELEV", "SURF_ELEV", "BASAL_UNIT", *(f"{v}_SOURCE" for v in SOURCE_VARIABLES)]
        self._var_attrs_json = var_attrs_json

    def _settings(self, dir_path: str) -> dict:
        """The settings of a dataset (SETTINGS_FILE), the arguments of the class taking over."""
        path = os.path.join(dir_path, SETTINGS_FILE)
        settings = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                settings = json.load(f)
        if self._wave_speed:
            settings["wave_speed"] = self._wave_speed
        if self._firn_correction:
            settings["firn_correction"] = self._firn_correction
        return settings

    def _processing(self, dir_path: str) -> str:
        """What was done to the raw files, in words, for the attribute "processing"."""
        settings = self._settings(dir_path)
        done = []
        if settings.get("wave_speed"):
            done.append(
                f"Two-way travel times converted to depths with {settings['wave_speed']} m/ns "
                f"({', '.join(DEPTH_VARS)})."
            )
        if settings.get("firn_correction"):
            done.append(
                f"Firn correction of {settings['firn_correction']:+g} m added "
                f"({', '.join(settings.get('firn_correction_vars', DEPTH_VARS))})."
            )
        if settings.get("line_ids") == "dataset" and settings.get("line_names"):
            pass  # named elsewhere: said by the note
        elif settings.get("line_ids") == "dataset":
            split = settings.get("split_at_jump_km")
            done.append(
                "The raw files have no line IDs: lines named after the dataset"
                + (f", cut at jumps of more than {split:g} km." if split else ".")
            )
        elif settings.get("split_at_jump_km"):
            done.append(f"Lines cut at jumps of more than {settings['split_at_jump_km']:g} km.")
        if settings.get("note"):
            done.append(settings["note"])
        return " ".join(done)

    @staticmethod
    def _to_depth(values: pd.Series, var: str, settings: dict) -> pd.Series:
        """Travel times to depths, and the firn correction, as the settings of the dataset ask."""
        if settings.get("wave_speed"):
            values = values * settings["wave_speed"]
        if settings.get("firn_correction") and var in settings.get("firn_correction_vars", DEPTH_VARS):
            values = values + settings["firn_correction"]
        return values

    def _segments(self, df: pd.DataFrame, line_id: str, settings: dict) -> list:
        """(line ID, points) of each line in the file of a line: cut at the jumps if asked."""
        if settings.get("line_ids") == "dataset":
            line_id = settings["dataset"]
        km = settings.get("split_at_jump_km")
        if not km:
            return [(line_id, df)]
        jumps = np.hypot(df["PSX"].diff(), df["PSY"].diff()) > km * 1000
        parts = [part for _, part in df.groupby(jumps.cumsum().values)]
        if len(parts) == 1:
            return [(line_id, df)]
        # The names of the parts given elsewhere than in the raw files (e.g. a table of the article)
        names = settings.get("line_names", {})
        return [(names.get(f"{line_id}_{n}", f"{line_id}_{n}"), part) for n, part in enumerate(parts, start=1)]

    def _split_km(self, trace_dir: str) -> Optional[float]:
        """The jump to cut this line at: only the line of layer files without line IDs."""
        dataset_dir = os.path.dirname(os.path.dirname(trace_dir))
        km = self._settings(dataset_dir).get("split_at_jump_km")
        if not km:
            return None
        raw_md = pd.read_json(f"{dataset_dir}/raw_md.json").rename(columns=OLD_RAW_MD_KEYS)
        if raw_md.iloc[0]["file structure"] != "IRH" or "line ID" not in raw_md:
            return None
        names = raw_md["line ID"].dropna().astype(str)
        return km if os.path.basename(trace_dir) in set(names) else None

    def _pre_compile_checks(self, dir_list: list[str]) -> bool:
        missing = False
        for dir_path in dir_list:
            raw_dir = f"{dir_path}/raw/"
            if not os.path.exists(raw_dir):
                print(f"{raw_dir} does not exist")
                missing = True
        return not missing

    def compile(self, cpus: int = cpu_count() - 1) -> None:
        if not isinstance(self._dir_list, list):
            self._dir_list = [self._dir_list]

        start_time = time.time()
        if not self._pre_compile_checks(self._dir_list):
            return

        if self._extract_data is True:
            all_files_list = []
            for dir_ in self._dir_list:
                files = glob.glob(f"{dir_}/raw/*.*")
                for file_path in files:
                    all_files_list.append(
                        {
                            "dir_path": dir_,
                            "file": os.path.basename(file_path),
                            "file_path": file_path,
                        }
                    )

            num_tasks = len(all_files_list)
            num_workers = min(num_tasks, cpus)

            print(
                "\n",
                "Will start extracting data from",
                num_tasks,
                "raw files\n" "\n   ",
                num_workers,
                "worker(s) allocated out of",
                cpu_count(),
                "available cpus\n",
            )

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(
                        pool.imap_unordered(self.extract, all_files_list),
                        total=num_tasks,
                        desc="Processing",
                        unit="file",
                    ):
                        pass
            else:
                for file_dict in tqdm(all_files_list, desc="Processing", unit="file"):
                    self.extract(file_dict=file_dict)

        all_dirs = []
        for dir_ in self._dir_list:
            dirs = [d for d in glob.glob(f"{dir_}/pkl/*") if os.path.isdir(d)]
            all_dirs.extend(dirs)

        num_tasks = len(all_dirs)
        num_workers = min(num_tasks, cpus)

        print(
            "\n",
            "Will start creating the netCDF files for",
            len(all_dirs),
            "traces\n" "\n   ",
            num_workers,
            "worker(s) allocated out of",
            cpu_count(),
            "available cpus\n",
        )

        # Always in a worker: one directory can make several files (split_into_lines), and the pools
        # below would hang if this process had used the threads of dask before forking
        with Pool(max(num_workers, 1)) as pool:
            for _ in tqdm(
                pool.imap_unordered(self.combine_dfs, all_dirs), total=num_tasks
            ):
                pass

        all_h5 = []
        for dir_ in self._dir_list:
            h5_files = glob.glob(f"{dir_}/nc/*.nc")
            all_h5.extend(h5_files)

        num_tasks = len(all_h5)
        num_workers = min(num_tasks, cpus)

        if self._compute_distance:

            print(
                "\n",
                "Computing distances along transects for",
                num_tasks,
                "files\n" "\n   ",
                num_workers,
                "worker(s) allocated out of",
                cpu_count(),
                "available cpus\n",
            )

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(
                        pool.imap_unordered(self.compute_distances, all_h5),
                        total=num_tasks,
                    ):
                        pass
            else:
                for h5_file in tqdm(all_h5, desc="Processing"):
                    self.compute_distances(h5_file)

        if self._compute_IRH_density:

            print(
                "\n",
                "Computing IRH density in",
                num_tasks,
                "files\n" "\n   ",
                num_workers,
                "worker(s) allocated out of",
                cpu_count(),
                "available cpus\n",
            )

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(
                        pool.imap_unordered(self.compute_irh_density, all_h5),
                        total=num_tasks,
                    ):
                        pass
            else:
                for h5_file in tqdm(all_h5, desc="Processing"):
                    self.compute_irh_density(h5_file)

        if self._compute_IMBIE_basins:

            print(
                "\n",
                "Computing IMBIE basins for",
                num_tasks,
                "files\n" "\n   ",
                num_workers,
                "worker(s) allocated out of",
                cpu_count(),
                "available cpus\n",
            )

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(
                        pool.imap_unordered(self.compute_imbie_basins, all_h5),
                        total=num_tasks,
                    ):
                        pass
            else:
                for h5_file in tqdm(all_h5, desc="Processing"):
                    self.compute_imbie_basins(h5_file)

        if self._set_attributes:

            print(
                "\n",
                "Settings variable attributes in",
                num_tasks,
                "files\n" "\n   ",
                num_workers,
                "worker(s) allocated out of",
                cpu_count(),
                "available cpus\n",
            )

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(
                        pool.imap_unordered(self.set_attrs, all_h5), total=num_tasks
                    ):
                        pass
            else:
                for h5_file in tqdm(all_h5, desc="Processing"):
                    self.set_attrs(h5_file)

        print("\nWriting the final files ...")
        if num_workers > 1:
            with Pool(num_workers) as pool:
                for _ in tqdm(pool.imap_unordered(self.finalise, all_h5), total=num_tasks):
                    pass
        else:
            for h5_file in tqdm(all_h5, desc="Processing"):
                self.finalise(h5_file)

        if self._remove_tmp_files:
            print("\nRemoving temporary directories ...")
            num_tasks = len(self._dir_list)
            num_workers = min(num_tasks, cpus)
            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(
                        pool.imap_unordered(self._cleanup, self._dir_list),
                        total=num_tasks,
                        unit="directory",
                    ):
                        pass
            else:
                for ds_dir in tqdm(self._dir_list, desc="Removing", unit="directory"):
                    self._cleanup(ds_dir)

        if self._shapefiles:
            num_tasks = len(self._dir_list)
            num_workers = min(num_tasks, cpus)

            print(
                "\n",
                "Will start creating the shapefiles for",
                len(self._dir_list),
                "datasets\n" "\n   ",
                num_workers,
                "worker(s) allocated out of",
                cpu_count(),
                "available cpus\n",
            )

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(
                        pool.imap_unordered(self.make_shapefile, self._dir_list),
                        total=num_tasks,
                    ):
                        pass
            else:
                for ds_dir in tqdm(self._dir_list, desc="Processing"):
                    self.make_shapefile(ds_dir)

        if self._geopackages:
            num_tasks = len(self._dir_list)
            num_workers = min(num_tasks, cpus)

            print(
                "\n",
                "Will start creating the geopackages for",
                len(self._dir_list),
                "datasets\n" "\n   ",
                num_workers,
                "worker(s) allocated out of",
                cpu_count(),
                "available cpus\n",
            )

            if num_workers > 1:
                with Pool(num_workers) as pool:
                    for _ in tqdm(
                        pool.imap_unordered(self.make_geopackage, self._dir_list),
                        total=num_tasks,
                    ):
                        pass
            else:
                for ds_dir in tqdm(self._dir_list, desc="Processing"):
                    self.make_geopackage(ds_dir)

        elapsed = time.time() - start_time
        print(f"\nCompilation completed in {elapsed:.2f} seconds")

    def _cleanup(self, dataset_dir: str) -> None:
        dir_path = f"{dataset_dir}/pkl/"

        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

    def convert_col_to_num(self, df):
        df = pd.to_numeric(df, errors="coerce")
        return df

    def extract(self, file_dict) -> None:

        _, ext = os.path.splitext(file_dict["file"])
        raw_md = pd.read_json(f"{file_dict['dir_path']}/raw_md.json").rename(
            columns=OLD_RAW_MD_KEYS
        )
        original_new_columns = pd.read_csv(
            f"{file_dict['dir_path']}/original_new_column_names.csv"
        )

        if ext == ".tab":
            sep = "\t"
        elif ext == ".csv":
            sep = ","
        elif ext == ".txt":
            sep = " "
        else:
            print(f"{ext}: File type not supported...")
            return

        def warn_with_file_path(
            message, category, filename, lineno, file=None, line=None
        ):
            print(f"Warning in file: {file_dict['file_path']}")
            print(f"Warning message: {message}")

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = warn_with_file_path

            df = pd.read_csv(
                file_dict["file_path"],
                comment="#",
                header=0,
                sep=sep,
                # usecols=original_new_columns.columns,
                # na_values=['-9999', '-9999.0', 'NaN', 'nan', ''],
                dtype=str,
            )

        _, file_name = os.path.split(file_dict["file_path"])
        file_name_, ext = os.path.splitext(file_name)

        df = df[df.columns.intersection(original_new_columns.columns)]
        df.columns = (
            original_new_columns[df.columns].iloc[0].values
        )  # renaming the columns
        df = df.rename(columns=OLD_COLUMN_NAMES)

        values_to_replace = ["-9999", "-9999.0", "NaN", "nan", ""]
        # Not a coordinate: 9999 m is a place, but no depth, thickness or elevation
        missing_values = [*values_to_replace, "9999", "9999.0"]

        if "PSX" in df.columns and "PSY" in df.columns:
            # Create a mask for rows where both PSX and PSY are in values_to_replace
            mask = df["PSX"].isin(values_to_replace) & df["PSY"].isin(values_to_replace)
            if mask.any():
                problematic_indices = df.index[mask].tolist()
                warnings.warn(
                    f"Found {len(problematic_indices)} rows where both PSX and PSY are in {values_to_replace}. "
                    f"Indices: {problematic_indices}. Dropping these rows."
                )

            df = df[~mask].copy()

        cols_to_replace = [col for col in df.columns if col not in ["PSX", "PSY"]]
        df[cols_to_replace] = df[cols_to_replace].replace(missing_values, pd.NA)

        pattern_values = original_new_columns[1:]
        pattern_header = original_new_columns.iloc[0]
        pattern_values.columns = pattern_header

        for var in [
            "ICE_THK",
            "SURF_ELEV",
            "BED_ELEV",
            "BASAL_UNIT",
            "IRH_DEPTH",
            "PSX",
            "DIST",
            "PSY",
            "lat",
            "lon",
        ]:
            if var in df.columns:
                df[var] = self.convert_col_to_num(df[var])

        settings = self._settings(file_dict["dir_path"])
        settings["dataset"] = os.path.basename(os.path.normpath(file_dict["dir_path"]))
        df = df.drop(columns=[v for v in settings.get("drop_vars", {}).get(file_dict["file"], []) if v in df.columns])
        # Elevations are not travel times: only the depths and the thickness are converted, before
        # the third of ice thickness, surface and bed is worked out from the other two
        if "ICE_THK" in df.columns:
            df["ICE_THK"] = self._to_depth(df["ICE_THK"], "ICE_THK", settings)

        published = [v for v in SOURCE_VARIABLES if v in df.columns]
        if (
            "ICE_THK" in df.columns
            and "SURF_ELEV" in df.columns
            and not "BED_ELEV" in df.columns
        ):
            df["BED_ELEV"] = df["SURF_ELEV"] - df["ICE_THK"]
        if (
            "ICE_THK" in df.columns
            and "BED_ELEV" in df.columns
            and not "SURF_ELEV" in df.columns
        ):
            df["SURF_ELEV"] = df["BED_ELEV"] + df["ICE_THK"]
        if (
            "SURF_ELEV" in df.columns
            and "BED_ELEV" in df.columns
            and not "ICE_THK" in df.columns
        ):
            df["ICE_THK"] = df["SURF_ELEV"] - df["BED_ELEV"]
        # Per point, with the values: they are merged with them, file by file (combine_dfs)
        for var in SOURCE_VARIABLES:
            if var in df.columns:
                code = SOURCES["published_dataset" if var in published else "computed_from_published"]
                df[f"{var}_SOURCE"] = np.where(df[var].notna(), code, np.nan)

        if "PSX" not in df.columns and "PSY" not in df.columns:
            if "lon" in df.columns and "lat" in df.columns:
                transformer = Transformer.from_proj(
                    "EPSG:4326",  # source: WGS84 (lon/lat)
                    "+proj=stere +lon_0=0 +lat_0=-90 +lat_ts=-71 +datum=WGS84 +units=m +no_defs",  # target: polar
                    always_xy=True,
                )
                df["PSX"], df["PSY"] = transformer.transform(
                    df["lon"].values, df["lat"].values
                )
        elif "PSX" in df.columns and "PSY" in df.columns:
            pass
        else:
            print("No coordinates found in the dataset")
            return

        if raw_md.iloc[0]["file structure"] == "IRH":
            raw_md.set_index("raw file", inplace=True)
            raw_md = raw_md.loc[file_dict["file"]]

            # Not in the files of a single variable (ICE_THK.tab...)
            if "IRH_DEPTH" in df.columns:
                df["IRH_DEPTH"] = self._to_depth(df["IRH_DEPTH"], "IRH_DEPTH", settings)

            # Files without line IDs hold one line, named by "line ID" in raw_md.json
            has_line_ids = "Line_ID" in df.columns
            if has_line_ids:
                df = df.astype({"Line_ID": str})
                df["Line_ID"] = df["Line_ID"].str.replace(
                    r"/\s+", "_", regex=True
                )  # Replace slashes with underscores, otherwise the paths can get messy
                df["Line_ID"] = df["Line_ID"].str.replace("/", "_")
                df.set_index("Line_ID", inplace=True)
            elif pd.isna(raw_md.get("line ID")):
                raise ValueError(
                    f"{file_name}: no Line_ID column and no \"line ID\" in raw_md.json"
                )

            line_id = []
            line_id_flag = "original"
            one_line = False

            if "line ID" in raw_md.index:
                if not pd.isna(raw_md["line ID"]):
                    line_id = raw_md["line ID"]
                    line_id_flag = "not_provided"
                    one_line = True
                    if not has_line_ids and "line_ids" not in settings:
                        line_id_flag = "original"  # given with the data, not made up

            if "line ID prefix" in raw_md.index:
                if not pd.isna(raw_md["line ID prefix"]):
                    df.index = [f"{raw_md['line ID prefix']}{x}" for x in df.index]
                    line_id_flag = "project_acq-year_number"

            dataset = raw_md["dataset"]
            institute = raw_md["institute"]
            if isinstance(institute, list):
                if all(pd.isna(institute)):
                    institute = "nan"
            else:
                if pd.isna(institute):
                    institute = "nan"
            institute_flag = "original"
            project = raw_md["project"]
            project_flag = "original"
            acq_year_flag = "original"
            doi_data = raw_md["DOI dataset"]
            doi_article = raw_md.get("DOI article", raw_md.get("DOI publication"))
            acq_year = raw_md["acquisition year"]
            IRH = raw_md["IRH"]
            if "radar instrument" in raw_md.index:
                radar_inst = raw_md["radar instrument"]
            else:
                radar_inst = "nan"
            platform = self._platform(raw_md)
            firn = self._firn(raw_md, settings)

            line_ids = np.unique(df.index)
            if one_line:
                line_ids = [line_id]

            def extract_year(time: str, pattern: str):
                position = pattern.find("YYYY")
                return time[position : position + 4]

            for line_id in line_ids:
                if one_line:
                    df_trace = df.copy()
                else:
                    df_trace = df.loc[[line_id]].copy()

                if "acq_year" in pattern_values.columns:
                    pattern = pattern_values.iloc[0]["acq_year"]
                    if not pd.isna(pattern):
                        df_trace["acq_year"] = df_trace["acq_year"].apply(
                            lambda x: extract_year(x, pattern)
                        )
                        unique_time = np.unique(df_trace["acq_year"])
                        if len(unique_time) > 1:
                            raise ValueError(
                                f"line {line_id} in {file_name_} contains {len(unique_time)} different acquisition year. Current code does not support this"
                            )
                        else:
                            position = pattern.find("YYYY")
                            acq_year = unique_time[0][position : position + 4]
                        df_trace.drop(columns="acq_year", inplace=True)

                if line_id == "nan":
                    line_id = f"{project}_{acq_year}"
                    line_id_flag = "not_provided"

                # NOTE: This will not work for BEDMAP. We assign a IRH_ID to the data.
                ds_trace_file = f"{file_dict['dir_path']}/pkl/{line_id}/{IRH}.pkl"

                line_dir, _ = os.path.split(ds_trace_file)
                os.makedirs(line_dir, exist_ok=True)
                df_trace.reset_index(drop=True, inplace=True)
                df_trace.to_pickle(ds_trace_file)

                trace_metadata = f"{line_dir}/metadata.csv"
                if not os.path.exists(trace_metadata):
                    if not pd.isna(acq_year) and acq_year == 0:
                        acq_year = pd.NA
                    trace_md = pd.DataFrame(
                        {
                            "dataset": [dataset, "original"],
                            "line_id": [line_id, line_id_flag],
                            "institute": [institute, institute_flag],
                            "project": [project, project_flag],
                            "acq_year": [acq_year, acq_year_flag],
                            "radar_instrument": [radar_inst, None],
                            "platform": [platform, None],
                            "firn_correction": [firn, None],
                            "doi_data": [doi_data, None],
                            "doi_article": [doi_article, None],
                        }
                    )
                    trace_md.set_index("line_id", inplace=True)
                    trace_md.to_csv(trace_metadata)

        elif raw_md.iloc[0]["file structure"] == "line":
            IRH_md = pd.read_json(f"{file_dict['dir_path']}/raw_md.json")

            IRH_md.set_index("IRH", inplace=True)

            for line_id, df in self._segments(df, file_name_, settings):
                for IRH, row in IRH_md.iterrows():
                    if IRH in df.columns:
                        df_IRH = df[IRH]
                        df_IRH = pd.DataFrame(
                            {
                                "PSX": df["PSX"],
                                "PSY": df["PSY"],
                                "IRH_DEPTH": df_IRH,
                            }
                        )

                        df_IRH["IRH_DEPTH"] = self._to_depth(
                            self.convert_col_to_num(df_IRH["IRH_DEPTH"]), "IRH_DEPTH", settings
                        )

                        for var in self._var_list:
                            if var in df.columns:
                                df_IRH[var] = df[var]

                        dataset = row["dataset"]
                        institute = row["institute"]
                        if isinstance(institute, list):
                            if all(pd.isna(institute)):
                                institute = "nan"
                        else:
                            if pd.isna(institute):
                                institute = "nan"
                        institute_flag = "original"
                        project = row["project"]
                        project_flag = "original"
                        acq_year = row["acquisition year"]
                        acq_year_flag = "original"
                        line_id_flag = "not_provided" if settings.get("line_ids") == "dataset" else "original"
                        if line_id in settings.get("line_names", {}).values():
                            line_id_flag = "article"  # named after the article, not the raw files
                        doi_data = row["DOI dataset"]
                        doi_article = row.get("DOI article", row.get("DOI publication"))
                        if "radar instrument" in row.index:
                            radar_inst = row["radar instrument"]
                        else:
                            radar_inst = "nan"
                        platform = self._platform(row)
                        firn = self._firn(row, settings)

                        # NOTE: This will not work for BEDMAP. We assign a IRH_ID to the data.
                        ds_trace_file = f"{file_dict['dir_path']}/pkl/{line_id}/{IRH}.pkl"

                        line_dir, _ = os.path.split(ds_trace_file)
                        os.makedirs(line_dir, exist_ok=True)
                        df_IRH.reset_index(drop=True, inplace=True)
                        df_IRH.to_pickle(ds_trace_file)

                        trace_metadata = f"{line_dir}/metadata.csv"
                        if not os.path.exists(trace_metadata):
                            trace_md = pd.DataFrame(
                                {
                                    "dataset": [dataset, "original"],
                                    "line_id": [line_id, line_id_flag],
                                    "institute": [institute, institute_flag],
                                    "project": [project, project_flag],
                                    "acq_year": [acq_year, acq_year_flag],
                                    "radar_instrument": [radar_inst, None],
                                    "platform": [platform, None],
                                    "firn_correction": [firn, None],
                                    "doi_data": [doi_data, None],
                                    "doi_article": [doi_article, None],
                                }
                            )
                            trace_md.set_index("line_id", inplace=True)
                            trace_md.to_csv(trace_metadata)

    @staticmethod
    def _platform(raw_md: pd.Series) -> list:
        """Airborne and/or ground-based ("platform" of raw_md.json, one or a list); [] if not given."""
        value = raw_md.get("platform")
        if isinstance(value, (list, np.ndarray)):
            return [str(v) for v in value]
        return [] if value is None or pd.isna(value) else [str(value)]

    @staticmethod
    def _firn(raw_md: pd.Series, settings: dict) -> str:
        """The firn correction in the depths of the database: added by the compilation (the published
        depths have none), or by the authors ("firn correction applied" and "firn correction" of
        raw_md.json)."""
        if settings.get("firn_correction"):
            return f"{settings['firn_correction']:g} m, added by the compilation"
        applied, value = raw_md.get("firn correction applied"), raw_md.get("firn correction")
        if applied is None or (not isinstance(applied, bool) and pd.isna(applied)):
            return "not known"
        if not applied:
            return "none"
        return f"{value:g} m, by the authors" if value is not None and not pd.isna(value) else "by the authors, value not known"

    @staticmethod
    def _platform_attr(trace_md: pd.DataFrame) -> list:
        """The platforms of a line from its metadata.csv (written by _platform), as a list."""
        if "platform" not in trace_md.columns or pd.isna(trace_md.iloc[0]["platform"]):
            return []
        return list(ast.literal_eval(str(trace_md.iloc[0]["platform"])))

    @staticmethod
    def _same_trace(df: pd.DataFrame, known: np.ndarray, within: float) -> tuple[pd.DataFrame, np.ndarray]:
        """The points of df within `within` m of a point of known (the points of the files before),
        each the nearest of the other, moved onto it; and known with the points of df."""
        xy = df[["PSX", "PSY"]].to_numpy(dtype=float, copy=True)
        finite = np.flatnonzero(np.isfinite(xy).all(axis=1))
        if len(known) and len(finite):
            dist, near = cKDTree(known).query(xy[finite], distance_upper_bound=within)
            close = (dist > 0) & np.isfinite(dist)
            # Not onto a point that df has already, nor two points of df onto one
            back = cKDTree(xy[finite]).query(known[near[close]])[1]
            mine = finite[close][back == np.flatnonzero(close)]
            target = known[near[close][back == np.flatnonzero(close)]]
            df = df.copy()
            df.loc[df.index[mine], "PSX"], df.loc[df.index[mine], "PSY"] = target[:, 0], target[:, 1]
            xy[mine] = target
        known = np.unique(np.vstack([known, xy[finite]]), axis=0)
        return df, known

    def combine_dfs(self, trace_dir: str) -> None:
        # Sorted: the order of the points must not change from one compilation to the next
        files = sorted(glob.glob(f"{trace_dir}/*.pkl"))
        if not files:
            return

        split_km = self._split_km(trace_dir)
        same_trace = self._settings(os.path.dirname(os.path.dirname(trace_dir))).get("same_trace_within_m")
        dfs_prep = []
        if len(files) > 1:
            known = np.empty((0, 2))
            for f in files:
                df = pd.read_pickle(f)
                if same_trace:
                    df, known = self._same_trace(df, known, same_trace)
                df = df.set_index(["PSX", "PSY"])
                df = df[~df.index.duplicated(keep="first")]
                _, f_name = os.path.split(f)
                IRH, _ = os.path.splitext(f_name)
                df = df.rename(columns={"IRH_DEPTH": IRH})
                dfs_prep.append(df)

            merged_df = pd.concat(dfs_prep, axis=1, join="outer")
            merged_df.reset_index(inplace=True)
            merged_df.columns = merged_df.columns.astype(str)

            for col in self._var_list:
                # The same name only: ICE_THK is not ICE_THK_SOURCE
                matching_cols = [
                    c for c in merged_df.columns if col.lower() == c.lower()
                ]
                if matching_cols:
                    temp_col = f"temp_{col}"
                    merged_df[temp_col] = (
                        merged_df[matching_cols].bfill(axis=1).iloc[:, 0]
                    )
                    merged_df = merged_df.drop(columns=matching_cols)
                    merged_df = merged_df.rename(columns={temp_col: col})

            points = pd.MultiIndex.from_frame(merged_df[["PSX", "PSY"]])
        else:
            f = files[0]
            df = pd.read_pickle(f)
            df = df.set_index(["PSX", "PSY"])
            df = df[~df.index.duplicated(keep="first")]
            dfs_prep.append(df)
            points = df.index
            df = df.reset_index()
            _, f_name = os.path.split(f)
            IRH, _ = os.path.splitext(f_name)
            merged_df = df.rename(columns={"IRH_DEPTH": IRH})

        if not split_km:
            sequences = [points.get_indexer(df.index) for df in dfs_prep]
            # The points of the layer files, merged, follow each other in the order of the files:
            # the points of one layer only come after all the others. Unless every file is in the
            # merged order already (one table for all the layers), put them in order along the
            # line (lines.py), which is never cut
            if any((np.diff(seq) < 0).any() for seq in sequences):
                xy = merged_df[["PSX", "PSY"]].to_numpy(dtype=float)
                order = order_line(xy, sequences)
                # The order that zigzags the least: a line along the same track several times
                # (out and back) can be woven together by order_line
                if path_length(xy[order]) >= path_length(xy):
                    order = np.arange(len(xy))
                # A flight passing again near its own track: the order of the files keeps the
                # passes apart, where the pieces joined by their closest ends go back and forth.
                # Only for a line out of order, and if it zigzags less
                if out_of_order(xy[order]):
                    by_files = order_by_files(xy, sequences)
                    if by_files is not None and path_length(xy[by_files]) < path_length(xy[order]):
                        order = by_files
                merged_df = merged_df.iloc[order].reset_index(drop=True)
            self._write_line(merged_df, trace_dir, None)
            return
        lines = split_into_lines(
            merged_df[["PSX", "PSY"]].to_numpy(dtype=float),
            [points.get_indexer(df.index) for df in dfs_prep],
            split_km * 1000,
        )
        if len(lines) == 1:
            self._write_line(merged_df.iloc[lines[0]].reset_index(drop=True), trace_dir, None)
            return
        line_id = os.path.basename(trace_dir)
        for n, line in enumerate(lines, start=1):
            # Without the layers and variables missing from this part of the data
            part = merged_df.iloc[line].dropna(axis=1, how="all").reset_index(drop=True)
            self._write_line(part, trace_dir, f"{line_id}_{n}")

    def _write_line(self, merged_df: pd.DataFrame, trace_dir: str, name: Optional[str]) -> None:
        """Write the nc file of a line; name: when it is not the one of its directory."""
        x = merged_df["PSX"]
        y = merged_df["PSY"]
        data_vars = {
            "PSX": (["point"], x),
            "PSY": (["point"], y),
        }

        # Only the layers with values on this line: a layer of the dataset that the line does not
        # have is left out, not written as a column of NaN
        IRHS = [col for col in merged_df.columns if str(col).isdigit() and merged_df[col].notna().any()]

        for var in self._var_list:
            if var in merged_df.columns:
                data_vars[var] = (["point"], merged_df[var].values)

        pkl_dir, _ = os.path.split(trace_dir)
        dataset_dir, _ = os.path.split(pkl_dir)
        trace_md = pd.read_csv(f"{trace_dir}/metadata.csv")
        institute = str(trace_md.iloc[0]["institute"])
        if isinstance(institute, str):
            try:
                institute = ast.literal_eval(institute)
            except (ValueError, SyntaxError):
                institute = [institute]
        if not isinstance(institute, (list, np.ndarray)):
            if institute in ["nan", "none", None]:
                institute = "nan"
            else:
                institute = [institute]
        else:
            institute = list(institute)
        line_id = name or str(trace_md.iloc[0]["line_id"])
        ds = xr.Dataset(
            coords={
                "point": np.arange(len(merged_df)),
            },
            data_vars=data_vars,
            attrs={
                "dataset": str(trace_md.iloc[0]["dataset"]),
                "institute": institute,
                "project": str(trace_md.iloc[0]["project"]),
                "radar instrument": str(trace_md.iloc[0]["radar_instrument"]),
                "platform": self._platform_attr(trace_md),
                "firn correction": (
                    str(trace_md.iloc[0]["firn_correction"]) if "firn_correction" in trace_md.columns else "not known"
                ),
                "acquisition year": str(trace_md.iloc[0]["acq_year"]),
                "line ID": line_id,
                "line ID flag": str(trace_md.iloc[1]["line_id"]),
                "DOI dataset": str(trace_md.iloc[0]["doi_data"]),
                "DOI article": str(trace_md.iloc[0]["doi_article"]),
                "processing": self._processing(dataset_dir),
            },
        )
        if not ds.attrs["platform"]:  # not in the metadata of the dataset
            del ds.attrs["platform"]
        point_shape = len(merged_df)
        if point_shape < 1e6:
            chunk_size_point = min(100, point_shape)
        elif point_shape < 1e7:
            chunk_size_point = 1000
        else:
            chunk_size_point = 10000

        encoding = {
            "PSX": {"zlib": True, "complevel": 1},
            "PSY": {"zlib": True, "complevel": 1},
        }

        if IRHS:
            sorted_irhs = sorted(IRHS, key=lambda x: int(x))
            sorted_irhs_int = [int(irh) for irh in sorted_irhs]
            depth_data = merged_df[sorted_irhs]
            data_vars["IRH_DEPTH"] = (["point", "IRH"], depth_data)
            ds = ds.assign_coords(IRH=sorted_irhs_int)
            ds["IRH_DEPTH"] = (["point", "IRH"], depth_data)

            if "IRH_DEPTH" in ds.variables:
                ds["IRH_DEPTH"] = ds["IRH_DEPTH"].chunk(
                    {"IRH": 1, "point": chunk_size_point}
                )
                encoding["IRH_DEPTH"] = {
                    "zlib": True,
                    "complevel": 1,
                    "chunksizes": (chunk_size_point, 1),
                }

        for var in self._var_list:
            if var in ds.variables:
                ds[var] = ds[var].chunk({"point": chunk_size_point})
                encoding[var] = {
                    "zlib": True,
                    "complevel": 1,
                    "chunksizes": (chunk_size_point),
                }

        nc_dir = os.path.join(dataset_dir, "nc")
        os.makedirs(nc_dir, exist_ok=True)
        # Named by line ID, as published: two lines of a dataset never have the same
        nc_file = f"{nc_dir}/{line_id}.nc"
        if os.path.exists(nc_file):
            raise FileExistsError(f"{nc_file}: two lines named {line_id} in {dataset_dir}")

        ds.to_netcdf(nc_file, engine="h5netcdf", encoding=encoding, mode="w")

    def compute_irh_density(self, h5_file: str) -> None:
        with h5py.File(h5_file, "a") as f:
            with xr.open_dataset(f, engine="h5netcdf") as ds:

                if "IRH_DEPTH" in ds.variables:
                    irh_num = (~np.isnan(ds["IRH_DEPTH"])).sum(dim="IRH")
                    irh_num = irh_num.where(irh_num != 0, np.nan)
                    has_nan = np.isnan(irh_num).any().values

                    if has_nan:
                        pass
                    else:
                        irh_num = irh_num.astype("int32")

                    ds["IRH_NUM"] = irh_num

                    point_shape = len(ds.point.values)
                    if point_shape < 1e6:
                        chunk_size_point = min(100, point_shape)
                    elif point_shape < 1e7:
                        chunk_size_point = 1000
                    else:
                        chunk_size_point = 10000

                    ds["IRH_NUM"] = ds["IRH_NUM"].chunk({"point": chunk_size_point})
                    encoding = {
                        "IRH_NUM": {
                            "zlib": True,
                            "complevel": 1,
                            "chunksizes": (chunk_size_point),
                        }
                    }

                    ds = ds[list(ds.variables)]
                    ds.to_netcdf(
                        h5_file, engine="h5netcdf", mode="a", encoding=encoding
                    )

    def compute_distances(self, h5_file: str) -> None:
        with h5py.File(h5_file, "a") as f:
            with xr.open_dataset(f, engine="h5netcdf") as ds:
                if "Distance" not in ds.variables:
                    x = ds["PSX"].values
                    y = ds["PSY"].values

                    coords = np.column_stack((x, y))

                    diff = np.diff(coords, axis=0)
                    DISTs = np.sqrt(np.einsum("ij,ij->i", diff, diff))
                    cumulative_DIST = np.concatenate([[0], np.cumsum(DISTs)])

                    ds["Distance"] = (["point"], cumulative_DIST)

                    point_shape = len(ds.point.values)
                    if point_shape < 1e6:
                        chunk_size_point = min(100, point_shape)
                    elif point_shape < 1e7:
                        chunk_size_point = 1000
                    else:
                        chunk_size_point = 10000

                    ds["Distance"] = ds["Distance"].chunk({"point": chunk_size_point})
                    encoding = {
                        "Distance": {
                            "zlib": True,
                            "complevel": 1,
                            "chunksizes": (chunk_size_point),
                        }
                    }

                    ds = ds[list(ds.variables)]
                    ds.to_netcdf(
                        h5_file, engine="h5netcdf", mode="a", encoding=encoding
                    )

    def compute_imbie_basins(self, h5_file: str) -> None:

        with h5py.File(h5_file, "a") as f:
            with xr.open_dataset(f, engine="h5netcdf") as ds:
                x = ds["PSX"].values
                y = ds["PSY"].values

                coords = np.column_stack((x, y))
                geometry = [Point(xy) for xy in zip(x, y)]
                points = gpd.GeoDataFrame(
                    coords, geometry=geometry, crs=self._basins.crs
                )
                joined = gpd.sjoin(
                    points, self._basins, how="inner", predicate="within"
                )
                lookup_df = joined[joined["Regions"] != "Islands"][
                    ["Subregion", "Regions"]
                ].drop_duplicates()
                lookup_df.reset_index(drop=True, inplace=True)
                basin_mapping = dict(zip(lookup_df["Subregion"], lookup_df["Regions"]))
                ds.attrs["IMBIE_basins"] = json.dumps(basin_mapping)

                ds = ds[list(ds.variables)]
                ds.to_netcdf(h5_file, engine="h5netcdf", mode="a")

    def set_attrs(self, h5_file: str) -> None:

        with h5py.File(h5_file, "a") as f:
            with xr.open_dataset(f, engine="h5netcdf") as ds:
                if self._var_attrs_json:
                    var_attrs = pd.read_json(self._var_attrs_json)
                    var_attrs.set_index("variable", inplace=True)

                    for var, row in var_attrs.iterrows():
                        if var in ds.variables:
                            ds[var].attrs["units"] = row["units"]
                            ds[var].attrs["long_name"] = row["long_name"]
                            ds[var].attrs["description"] = row["description"]
                        elif var in ds.coords:
                            ds[var].attrs["units"] = row["units"]
                            ds[var].attrs["long_name"] = row["long_name"]
                            ds[var].attrs["description"] = row["description"]

                for var in SOURCE_VARIABLES:
                    if f"{var}_SOURCE" in ds.variables:
                        ds[f"{var}_SOURCE"].attrs.update(source_attrs(var))

                ds = ds[list(ds.variables)]
                ds.to_netcdf(h5_file, engine="h5netcdf", mode="a")

    @staticmethod
    def finalise(path: str) -> None:
        """Rewrite a line with the encoding of the database (anta_database.netcdf)."""
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            ds = ds.load()
        write(ds, path)

    def make_shapefile(self, dataset_dir: str) -> None:
        h5files = glob.glob(f"{dataset_dir}/nc/*.nc")

        all_dfs = []
        for h5f in h5files:
            with h5py.File(h5f, "r") as f:
                with xr.open_dataset(f, engine="h5netcdf") as ds:
                    ds = ds.dropna(dim="point", subset=["IRH_NUM"])
                    N = ds.IRH_NUM.values

                    irh_depth = ds.IRH_DEPTH.values
                    max_depth = np.full(len(N), np.nan)
                    age_max_depth = np.full(len(N), np.nan)

                    # NOTE: This compute the oldest traced isochrone. Will not work if ages are not assigned yet.
                    for i in range(len(N)):
                        depth_slice = irh_depth[i, :]
                        if not np.all(
                            np.isnan(depth_slice)
                        ):  # Check if all values are NaN
                            max_idx = np.nanargmax(depth_slice)
                            max_depth[i] = depth_slice[max_idx]
                            age_max_depth[i] = ds.IRH_AGE.values[max_idx]

                    age_max_depth = np.abs(age_max_depth)
                    max_depth = np.abs(max_depth)
                    line_id = ds.attrs["line ID"]
                    acq_year = ds.attrs["acquisition year"]
                    project = ds.attrs["project"]
                    instrument = ds.attrs["radar instrument"]
                    doi_data = ds.attrs["DOI dataset"]

                    df = gpd.GeoDataFrame(
                        {
                            "geometry": [
                                Point(x, y)
                                for x, y in zip(ds.PSX.values, ds.PSY.values)
                            ],
                            "irh_num": N.astype("int8"),
                            "max_age": age_max_depth.astype("int32"),
                            "max_depth": np.rint(max_depth).astype("int16"),
                            # "line_id": [line_id] * len(N),
                            # "doi_data": [doi_data] * len(N),
                        },
                        crs="EPSG:3031",
                    )

                    all_dfs.append(df)

        df_points = gpd.GeoDataFrame(
            pd.concat(all_dfs, ignore_index=True), crs="EPSG:3031"
        )

        dataset = os.path.basename(os.path.normpath(dataset_dir))
        shp_dir = f"{dataset_dir}/shp/"
        os.makedirs(shp_dir, exist_ok=True)

        points_shp_path = f"{shp_dir}/{dataset}.shp"
        df_points.to_file(points_shp_path)

    def make_geopackage(self, dataset_dir: str) -> None:
        h5files = glob.glob(f"{dataset_dir}/nc/*.nc")

        all_points = []
        all_md = []
        for h5f in h5files:
            with h5py.File(h5f, "r") as f:
                with xr.open_dataset(f, engine="h5netcdf") as ds:
                    N = ds.IRH_NUM.values

                    irh_depth = ds.IRH_DEPTH.values
                    max_depth = np.full(len(N), np.nan)
                    age_max_depth = np.full(len(N), np.nan)

                    for i in range(len(N)):
                        depth_slice = irh_depth[i, :]
                        if not np.all(
                            np.isnan(depth_slice)
                        ):  # Check if all values are NaN
                            max_idx = np.nanargmax(depth_slice)
                            max_depth[i] = depth_slice[max_idx]
                            age_max_depth[i] = ds.IRH_AGE.values[max_idx]

                    line_id = ds.attrs["line ID"]
                    acq_year = ds.attrs["acquisition year"]
                    project = ds.attrs["project"]
                    instrument = ds.attrs["radar instrument"]

                    points_gdf = gpd.GeoDataFrame(
                        {
                            "line_id": [line_id] * len(N),
                            "geometry": [
                                Point(x, y)
                                for x, y in zip(ds.PSX.values, ds.PSY.values)
                            ],
                            "IRH_NUM": N,
                            "IRH_DEPTH_MAX": max_depth,
                            "IRHAGE_MAX": age_max_depth,
                        },
                        crs="EPSG:3031",
                    )

                    lines_df = gpd.GeoDataFrame(
                        {
                            "line_id": line_id,
                            "acquisition_year": [acq_year],
                            "project": [project],
                            "instrument": [instrument],
                        }
                    )

                    all_points.append(points_gdf)
                    all_md.append(lines_df)

        df_points = gpd.GeoDataFrame(
            pd.concat(all_points, ignore_index=True), crs="EPSG:3031"
        )
        df_md = gpd.GeoDataFrame(pd.concat(all_md, ignore_index=True))

        dataset = os.path.basename(os.path.normpath(dataset_dir))
        gpkg_dir = f"{dataset_dir}/gpkg/"
        os.makedirs(gpkg_dir, exist_ok=True)

        df_points.to_file(f"{gpkg_dir}/{dataset}.gpkg", layer="points", driver="GPKG")
        df_md.to_file(f"{gpkg_dir}/{dataset}.gpkg", layer="lines", driver="GPKG")
