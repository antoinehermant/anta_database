import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
import shapefile
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import patheffects as path_effects
import colormaps as cmaps
from contextlib import contextmanager
from typing import Union, Dict, TYPE_CHECKING, Optional
from tqdm import tqdm
from importlib.resources import files
from multiprocessing import Pool, cpu_count

if TYPE_CHECKING:
    from anta_database.database.database import Database, MetadataResult


class PISMPlotting:
    def __init__(self, database_instance: "Database") -> None:
        self._db = database_instance
        self._gl_path = files("anta_database.data").joinpath("GL.parquet")
        self._site_coords_path = files("anta_database.data").joinpath(
            "site-coords.parquet"
        )
        self._imbie_path = files("anta_database.data").joinpath(
            "ANT_Basins_IMBIE2_v1.6.shp"
        )
        self._center_coords = files("anta_database.data").joinpath(
            "centeroid_coords_basins.shp"
        )
        self._disable_tqdm = os.getenv("JUPYTER_BOOK_BUILD", False)

    def _pre_plot_check(
        self, metadata: Union[None, Dict, "MetadataResult"] = None
    ) -> bool:

        if not metadata["age"] and not metadata["var"]:
            print(
                "Result from query provided is empty: nothing to plot. Please ensure that the query returns either valid age or var."
            )
            return False
        return True

    def _is_notebook(self) -> bool:
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return True  # Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter

    def _custom_cmap(self, reversed_: bool = False):
        cm1 = cmaps.torch_r
        cm2 = cmaps.deep_r
        cm1_colors = cm1(np.linspace(0.15, 0.8, 256))
        cm2_colors = cm2(np.linspace(0.1, 0.9, 256))
        combined_colors = np.vstack((cm1_colors, cm2_colors))
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", combined_colors, N=512
        )
        return custom_cmap.reversed() if reversed_ else custom_cmap

    def _custom_cmap_density(self):
        return self._custom_cmap(reversed_=True)

    def stratigraphy(
        self,
        PISM_file: str,
        metadata: Union[None, Dict, "MetadataResult"] = None,
        elevation: Optional[bool] = False,
        downsampling_factor: Optional[int] = None,
        cpus: int = 2,
        firn_correction: Optional[int] = 31,
        title: Optional[str] = None,
        xlim: Optional[tuple] = (None, None),
        ylim: Optional[tuple] = (None, None),
        sel_method: Optional[str] = "nearest",
        scale_factor: float = 1.0,
        marker_size: Optional[float] = 2,
        cmap: Optional["LinearSegmentedColormap"] = None,
        grounding_line: Optional[bool] = True,
        basins: Optional[bool] = True,
        stations: Optional[bool] = True,
        save: Optional[str] = None,
        auto_zoom: Optional[bool] = True,
        mini_map: Optional[bool] = True,
    ) -> None:
        """
        Plot the color-coded values of the given variable on Antarcitic map
        """
        self._base_plot(
            color_by="stratigraphy",
            PISM_file=PISM_file,
            elevation=elevation,
            metadata=metadata,
            downsampling_factor=downsampling_factor,
            cpus=cpus,
            firn_correction=firn_correction,
            title=title,
            xlim=xlim,
            ylim=ylim,
            sel_method=sel_method,
            scale_factor=scale_factor,
            marker_size=marker_size,
            cmap=cmap,
            grounding_line=grounding_line,
            basins=basins,
            stations=stations,
            save=save,
            auto_zoom=auto_zoom,
            mini_map=mini_map,
        )

    def mismatch_2D(
        self,
        PISM_file: str,
        metadata: Union[None, Dict, "MetadataResult"] = None,
        elevation: Optional[bool] = False,
        downsampling_factor: Optional[int] = None,
        cpus: int = 2,
        firn_correction: Optional[int] = 31,
        sel_method: Optional[str] = "nearest",
        title: Optional[str] = None,
        xlim: Optional[tuple] = (None, None),
        ylim: Optional[tuple] = (None, None),
        vmin: Optional[int] = -400,
        vmax: Optional[int] = 400,
        scale_factor: float = 1.0,
        marker_size: Optional[float] = 2,
        cmap: Optional["LinearSegmentedColormap"] = None,
        grounding_line: Optional[bool] = True,
        basins: Optional[bool] = True,
        stations: Optional[bool] = True,
        save: Optional[str] = None,
        auto_zoom: Optional[bool] = True,
        mini_map: Optional[bool] = True,
    ) -> None:
        """
        Plot the color-coded values of the given variable on Antarcitic map
        """
        self._base_plot(
            color_by="mismatch_2D",
            PISM_file=PISM_file,
            elevation=elevation,
            metadata=metadata,
            downsampling_factor=downsampling_factor,
            cpus=cpus,
            firn_correction=firn_correction,
            sel_method=sel_method,
            title=title,
            xlim=xlim,
            ylim=ylim,
            vmin=vmin,
            vmax=vmax,
            scale_factor=scale_factor,
            marker_size=marker_size,
            cmap=cmap,
            grounding_line=grounding_line,
            basins=basins,
            stations=stations,
            save=save,
            auto_zoom=auto_zoom,
            mini_map=mini_map,
        )

    @contextmanager
    def _plot_context(self, close=None):
        if close is None:
            close = not self._is_notebook()
        try:
            yield
        finally:
            if close:
                plt.close()

    def _process_mismatch(self, args) -> Union[None, pd.DataFrame]:
        (
            pism_ds,
            f,
            metadata,
        ) = args

        X_ds, Y_ds = np.meshgrid(pism_ds.x, pism_ds.y)
        pism_points = np.column_stack((X_ds.flatten(), Y_ds.flatten()))

        full_path = os.path.join(self._db._db_dir, f)
        all_psx = []
        all_psy = []
        all_depth_diff = []

        with h5py.File(full_path, "r") as irh:
            dataset = irh.attrs["dataset"]
            flight_id = irh.attrs["flight ID"]
            if self.elevation:
                if "SURF_ELEV" not in irh.keys():
                    print(
                        "SURF_ELEV not in",
                        flight_id,
                        "for",
                        dataset,
                        ": cannot compute IRH elevation. Stopping the process here.",
                    )
                    return
            irh_values = irh["IRH_AGE"][:]
            for age in metadata["age"]:
                irh_index = np.where(irh_values == int(age))[0]
                if len(irh_index) > 0:
                    irh_depth = irh["IRH_DEPTH"]

                    PSX_values = irh["PSX"][:: self.downsampling_factor]
                    PSY_values = irh["PSY"][:: self.downsampling_factor]
                    fl_depth = irh_depth[:: self.downsampling_factor, irh_index[0]]

                    all_psx.extend(PSX_values)
                    all_psy.extend(PSY_values)

                    age_seconds = -(int(age) + self.age_offset) * 3600 * 24 * 365
                    fl_line_xy = np.column_stack((PSX_values, PSY_values))

                    ds_depth_interp = griddata(
                        pism_points,
                        pism_ds.isochronal_layer_depth.sel(
                            deposition_time=age_seconds, method=self.sel_method
                        ).values.flatten(),
                        fl_line_xy,
                        method="linear",
                    )

                    if self.elevation:
                        pism_usurf_interp = griddata(
                            pism_points,
                            pism_ds.usurf.values.flatten(),
                            fl_line_xy,
                            method="linear",
                        )
                        ds_depth_interp = pism_usurf_interp - ds_depth_interp

                        fl_depth = (
                            irh["SURF_ELEV"][:: self.downsampling_factor] - fl_depth
                        )

                    results = ds_depth_interp + self.firn_correction - fl_depth

                    all_depth_diff.extend(results)

        df = pd.DataFrame(
            {"PSX": all_psx, "PSY": all_psy, "IRH_DEPTH_DIFF": all_depth_diff}
        )

        return df.dropna()

    def _base_plot(
        self,
        PISM_file: str,
        metadata: Union[None, Dict, "MetadataResult"] = None,
        elevation: Optional[bool] = False,
        fraction_depth: Optional[bool] = False,
        downsampling_factor: Optional[int] = None,
        age_offset: Optional[int] = -1950,
        firn_correction: Optional[int] = 31,
        sel_method: Optional[str] = None,
        cpus: int = 4,
        title: Optional[str] = None,
        xlim: Optional[tuple] = (None, None),
        ylim: Optional[tuple] = (None, None),
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        scale_factor: float = 1.0,
        marker_size: Optional[float] = 0.1,
        save: Optional[str] = None,
        auto_zoom: Optional[bool] = True,
        mini_map: Optional[bool] = True,
        color_by: str = "dataset",  # 'dataset', 'flight_id', 'depth', 'density'
        cmap: Optional["LinearSegmentedColormap"] = None,
        grounding_line: Optional[bool] = True,
        basins: Optional[bool] = True,
        stations: Optional[bool] = True,
        ncol: Optional[int] = None,
    ) -> None:
        # --- Setup ---
        if metadata is None:
            if hasattr(self._db, "_md") and self._db._md:
                metadata = self._db._md
            else:
                print(
                    "Please provide metadata of the files you want to generate the data from..."
                )
                return

        total_traces = len(metadata["flight_id"])

        if not self._pre_plot_check(metadata):
            return

        self.age_offset = age_offset
        self.downsampling_factor = downsampling_factor
        self.firn_correction = firn_correction
        self.elevation = elevation
        self.sel_method = sel_method

        # if save:
        #     matplotlib.use('Agg')
        # else: FIXME: this seems to crash spyder for spyder users
        #     matplotlib.use('TkAgg')

        fig, ax = plt.subplots(constrained_layout=True)

        if basins:
            grounding_line = False
        # --- Plot Grounding Line ---
        if True and color_by != "stratigraphy":  # FIXME
            gl = pd.read_parquet(self._gl_path)
            ax.plot(gl.x / 1000, gl.y / 1000, linewidth=1, color="k")

        # --- Plot Data ---
        colors = {}
        scatter = None
        values = None
        label = None
        extend = None

        if auto_zoom and (
            xlim[0] is not None
            or xlim[1] is not None
            or ylim[0] is not None
            or ylim[1] is not None
        ):
            auto_zoom = False

        if auto_zoom:
            xmin, xmax = None, None
            ymin, ymax = None, None
        else:
            xmin, xmax = xlim
            ymin, ymax = ylim

        if color_by == "stratigraphy":
            mini_map = False

        inset_pos = [0.7, 0.75, 0.25, 0.25]
        inset = fig.add_axes(
            inset_pos,  # [left, bottom, width, height] in figure coordinates
        )
        inset.set_xlim(-2700, 2700)
        inset.set_ylim(-2700, 2700)
        inset.set_facecolor("none")

        if color_by == "mismatch_2D":
            if cmap == None:
                if elevation:
                    cmap = "RdBu"
                else:
                    cmap = "RdBu_r"

            if len(metadata["age"]) > 1:
                print("\nWARNING: found multiple ages in query:", metadata["age"])
            import xarray as xr

            pism_ds = xr.open_dataset(PISM_file, decode_times=False)
            pism_ds = pism_ds.isel(
                deposition_time=np.unique(pism_ds.deposition_time, return_index=True)[1]
            )
            X_ds, Y_ds = np.meshgrid(pism_ds.x, pism_ds.y)
            pism_points = np.column_stack((X_ds.flatten(), Y_ds.flatten()))

            file_paths = self._db._get_file_paths_from_metadata(metadata)
            file_paths = np.unique(file_paths)

            args_list = [(pism_ds, f, metadata) for f in file_paths]

            num_tasks = len(args_list)
            num_workers = min(num_tasks, cpus)

            print(
                "\n",
                "Interpolating",
                num_tasks,
                "transect(s) \n" "\n   ",
                num_workers,
                "worker(s) allocated out of",
                cpu_count(),
                "available cpus\n",
            )

            dfs = []
            if num_workers > 1:
                with Pool(num_workers) as pool:
                    dfs = list(
                        tqdm(
                            pool.imap_unordered(self._process_mismatch, args_list),
                            total=num_tasks,
                            desc="Processing",
                        )
                    )
            else:
                for file_dict in tqdm(args_list, desc="Processing"):
                    dfs.append(self._process_mismatch(args=file_dict))

            df = pd.concat(dfs, ignore_index=True)

            scatter = ax.scatter(
                df["PSX"] / 1000,
                df["PSY"] / 1000,
                c=df["IRH_DEPTH_DIFF"],
                cmap=cmap,
                s=marker_size,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
                rasterized=True,
            )
            inset.scatter(
                df["PSX"] / 1000,
                df["PSY"] / 1000,
                color="darkgreen",
                s=marker_size * 0.5,
                linewidths=0,
                zorder=11,
            )
            extend = "both"

            if auto_zoom:
                psx_min, psx_max = df["PSX"].min() / 1000, df["PSX"].max() / 1000
                psy_min, psy_max = df["PSY"].min() / 1000, df["PSY"].max() / 1000
                xmin = psx_min if xmin is None else min(xmin, psx_min)
                xmax = psx_max if xmax is None else max(xmax, psx_max)
                ymin = psy_min if ymin is None else min(ymin, psy_min)
                ymax = psy_max if ymax is None else max(ymax, psy_max)

        if color_by == "stratigraphy":
            flight_id = list(metadata["flight_id"])
            if len(flight_id) > 1:
                flight_id = flight_id[0]
                print(
                    "Found mutilple flight lines to plot, so will plot the first one: ",
                    flight_id,
                )
            elif len(flight_id) < 1:
                print("No flight line found to plot")
                plt.close()
                return

            metadata_impl = self._db.query(
                flight_id=flight_id, dataset=metadata["dataset"], retain_query=False
            )
            if not title:
                title = f"Transect {metadata_impl['flight_id'][0]} from {metadata_impl['reference'][0]}"

            f = self._db._get_file_paths_from_metadata(metadata_impl)[0]
            full_path = os.path.join(self._db._db_dir, f)
            import xarray as xr

            pism_ds = xr.open_dataset(PISM_file, decode_times=False)
            pism_ds = pism_ds.isel(
                deposition_time=np.unique(pism_ds.deposition_time, return_index=True)[1]
            )

            ds = xr.open_dataset(full_path, engine="h5netcdf")
            fl_line_xy = np.column_stack((ds.PSX, ds.PSY))

            pism_iso = -((pism_ds.deposition_time.values / (3600 * 24 * 365)) - 1950)
            # layer_ages = ds.IRH_AGE
            ages = [int(a) for a in metadata["age"]]
            layer_ages = ages
            common_values = np.intersect1d(layer_ages, pism_iso)

            X_pism, Y_pism = np.meshgrid(pism_ds.x, pism_ds.y)
            pism_points = np.column_stack((X_pism.flatten(), Y_pism.flatten()))

            if "Distance" not in ds.variables:
                print("Distance not in varibles, cannot plot along transect")
                plt.close()
                return

            if elevation:
                if "SURF_ELEV" in ds.variables:
                    ds["IRH_DEPTH"] = ds.SURF_ELEV - ds.IRH_DEPTH

                elif "ICE_THK" in ds.variables and "BED_ELEV" in ds.variables:
                    ds["SURF_ELEV"] = (ds.ICE_THK + ds.BED_ELEV) - ds.IRH_DEPTH
                    ds["IRH_DEPTH"] = ds.SURF_ELEV - ds.IRH_DEPTH
                else:
                    print(
                        "Cannot plot IRH Elevation from the variables in the file, missing either ICE_THK and BED_ELEV or SURF_ELEV"
                    )
                    plt.close()
                    return

                pism_usurf_interp = griddata(
                    pism_points,
                    pism_ds.usurf.values.flatten(),
                    fl_line_xy,
                    method="linear",
                )

            cmap = self._custom_cmap_density()
            colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(metadata_impl["age"]))]
            print(
                "\nInterpolating",
                # len(common_values),
                len(ages),
                "PISM isochronal layers along the transect...",
            )
            # for age, color in zip(list(map(int, common_values)), colors):
            for age, color in zip(list(map(int, ages)), colors):
                if age not in ds.IRH_AGE:
                    print(
                        f"{metadata_impl['flight_id'][0]} does not contain age {age}, skipping"
                    )
                    continue

                ax.scatter(
                    ds.Distance / 1000,
                    ds.IRH_DEPTH.sel(IRH_AGE=age),
                    color=color,
                    s=marker_size,
                    linewidths=0.1,
                )
                ax.plot([], [], color=color, label=age, linewidth=3)

                age_offset = -1950
                age_converted = -(age + age_offset) * (3600 * 24 * 365)
                pism_layer = pism_ds.isochronal_layer_depth.sel(
                    deposition_time=age_converted,
                    method=self.sel_method,
                ).values.flatten()

                depth_offset = 31 + 12 * 0.038
                pism_depth_interp = (
                    griddata(pism_points, pism_layer, fl_line_xy, method="linear")
                    + depth_offset
                )
                if elevation:
                    pism_depth_interp = pism_usurf_interp - pism_depth_interp

                ax.plot(
                    ds.Distance / 1000,
                    pism_depth_interp,
                    linestyle="--",
                    color="k",
                )

            if elevation:
                if "BED_ELEV" in ds.variables:
                    scatter = ax.scatter(
                        ds.Distance / 1000,
                        ds.BED_ELEV,
                        color="k",
                        s=marker_size,
                        linewidths=0.1,
                    )
                    ax.plot([], [], color="k", label="Bed Elevation", linewidth=3)

                elif "ICE_THK" in ds.variables and "SURF_ELEV" in ds.variables:
                    ds["BED_ELEV"] = ds.SURF_ELEV - ds.ICE_THK
                    scatter = ax.scatter(
                        ds.Distance / 1000,
                        ds.BED_ELEV,
                        color="k",
                        s=marker_size,
                        linewidths=0.1,
                    )
                    ax.plot([], [], color="k", label="Bed Elevation", linewidth=3)
                else:
                    print("Cannot plot Bed Elevation from the variables in the file")
                if "SURF_ELEV" in ds.variables:
                    ax.scatter(
                        ds.Distance / 1000,
                        ds.SURF_ELEV,
                        color="darkred",
                        s=marker_size,
                        linewidths=0.1,
                    )
                    ax.plot(
                        [], [], color="darkred", label="Surface Elevation", linewidth=3
                    )
                else:
                    print(
                        "Cannot plot Surface Elevation from the variables in the file"
                    )

                pism_bed_elev_interp = griddata(
                    pism_points,
                    pism_ds.topg.values.flatten(),
                    fl_line_xy,
                    method="linear",
                )

                ax.plot(
                    ds.Distance / 1000,
                    pism_bed_elev_interp,
                    color="gray",
                    linestyle="--",
                    label="PISM Bed Elevation",
                )

                ax.plot(
                    ds.Distance / 1000,
                    pism_usurf_interp,
                    color="darkred",
                    linestyle="--",
                    label="PISM Surface Elevation",
                )

                ax.plot(
                    [],
                    [],
                    color="k",
                    label="PISM Isochrone Elevation",
                    linestyle="--",
                    linewidth=3,
                )

            else:
                if "ICE_THK" in ds.variables:
                    scatter = ax.scatter(
                        ds.Distance / 1000,
                        ds.ICE_THK,
                        color="k",
                        s=marker_size,
                        linewidths=0.1,
                    )
                    ax.plot([], [], color="k", label="Bed Depth", linewidth=3)
                else:
                    print("Cannot plot Bed Depth from the variables in the file")

                pism_bed_depth_interp = griddata(
                    pism_points,
                    pism_ds.thk.values.flatten(),
                    fl_line_xy,
                    method="linear",
                )

                ax.plot(
                    [],
                    [],
                    color="k",
                    label="PISM Isochrone Depth",
                    linestyle="--",
                )

                ax.plot(
                    ds.Distance / 1000,
                    pism_bed_depth_interp,
                    color="gray",
                    linestyle="--",
                    label="PISM Bed Depth",
                )

            if elevation:
                ylim = (
                    (ds.BED_ELEV.min() - 200, ds.SURF_ELEV.max() + 200)
                    if ylim == (None, None)
                    else ylim
                )
            else:
                ylim = (ds.ICE_THK.max() + 200, 0) if ylim == (None, None) else ylim

            if ncol == None:
                if ds.sizes["IRH_AGE"] > 7:
                    ncol = 2
                if ds.sizes["IRH_AGE"] > 15:
                    ncol = 3

        # --- Format Figure ---
        if not self._disable_tqdm:
            print("Formatting ...")

        if color_by == "stratigraphy":
            xmin, xmax = xlim
            ymin, ymax = ylim

        elif auto_zoom:
            xmin = xmin - 10 if xmin is not None else None
            xmax = xmax + 10 if xmax is not None else None
            ymin = ymin - 10 if ymin is not None else None
            ymax = ymax + 10 if ymax is not None else None

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if color_by != "stratigraphy":
            x0_ax, x1_ax = ax.get_xlim()
            y0_ax, y1_ax = ax.get_ylim()
            x0 = x0_ax if xlim[0] == None else xlim[0]
            x1 = x1_ax if xlim[1] == None else xlim[1]
            y0 = y0_ax if ylim[0] == None else ylim[0]
            y1 = y1_ax if ylim[1] == None else ylim[1]

            x_extent = x1 - x0
            y_extent = y1 - y0
            aspect_ratio = y_extent / x_extent
            ax.set_xlabel("x [km]")
            ax.set_ylabel("y [km]")
            ax.set_aspect("equal")
            plt.gcf().set_size_inches(
                10 * scale_factor, 10 * aspect_ratio * scale_factor
            )

        ax.set_title(title, fontsize=24 * scale_factor)

        if ncol == None:
            ncol = 1
        # --- Legend/Colorbar ---
        if color_by == "mismatch_2D" and scatter is not None:
            if values is not None:
                cbar = fig.colorbar(
                    scatter,
                    ax=ax,
                    ticks=values,
                    orientation="horizontal",
                    pad=0.1,
                    fraction=0.04,
                    extend=extend,
                )
            else:
                cbar = fig.colorbar(
                    scatter,
                    ax=ax,
                    orientation="horizontal",
                    pad=0.1,
                    fraction=0.04,
                    extend=extend,
                )
            cbar.ax.xaxis.set_ticks_position("bottom")
            if label:
                cbar.set_label(label)
            else:
                if elevation:
                    cbar.set_label("Isochronal elevation mismatch")
                else:
                    cbar.set_label("Isochronal depth mismatch")

        elif color_by == "stratigraphy":
            ax.legend(ncols=2)
            ax.set_xlabel("Distance along transect [km]")
            if elevation:
                ax.set_ylabel("Elevation above sea level [m]")
            else:
                ax.set_ylabel("Depth below surface [m]")
            plt.gcf().set_size_inches(10 * scale_factor, 10 * 2 / 3)

        # --- Plot IMBIE basins ---
        if basins and color_by != "stratigraphy":
            sf_basins = shapefile.Reader(self._imbie_path)
            basin_patches = []

            for shape_rec in sf_basins.shapeRecords():
                shp = shape_rec.shape
                pts = shp.points
                parts = list(shp.parts) + [len(pts)]  # add sentinel end index

                # build one polygon per part
                for i in range(len(parts) - 1):
                    start, end = parts[i], parts[i + 1]
                    ring = pts[start:end]

                    # scale coordinates
                    scaled = [(x * 0.001, y * 0.001) for x, y in ring]

                    poly = Polygon(scaled, closed=True, fill=False)
                    basin_patches.append(poly)

            pc = PatchCollection(
                basin_patches,
                facecolor="none",
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_collection(pc)

            # ---- CENTERS: read + scale + label ----
            sf_centers = shapefile.Reader(self._center_coords)

            # find the field index for "Subregion" in the DBF
            fields = sf_centers.fields[1:]  # first is DeletionFlag
            field_names = [f[0] for f in fields]
            sub_idx = field_names.index("Subregion")

            for shape_rec in sf_centers.shapeRecords():
                # assuming center_coords are points
                x_raw, y_raw = shape_rec.shape.points[0]
                x = x_raw * 0.001
                y = y_raw * 0.001
                sub = shape_rec.record[sub_idx]

                if x0 <= x <= x1 and y0 <= y <= y1:
                    ax.text(
                        x,
                        y,
                        sub,
                        fontsize=12,
                        color="k",
                        ha="center",
                        path_effects=[
                            path_effects.withStroke(
                                linewidth=5, foreground=(1, 1, 1, 0.7)
                            )
                        ],
                    )

            inset_basin_patches = []
            for shape_rec in sf_basins.shapeRecords():
                shp = shape_rec.shape
                pts = shp.points
                parts = list(shp.parts) + [len(pts)]

                for i in range(len(parts) - 1):
                    start, end = parts[i], parts[i + 1]
                    ring = pts[start:end]
                    scaled = [(x * 0.001, y * 0.001) for x, y in ring]
                    poly = Polygon(scaled, closed=True, fill=True)
                    inset_basin_patches.append(poly)

            pc = PatchCollection(
                inset_basin_patches,
                facecolor="white",
                edgecolor="black",
                linewidth=0.5,
            )
            inset.add_collection(pc)

            # Add rectangle showing current view
            inset.plot(
                [xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
                color="darkred",
                linestyle="--",
            )
            inset.set_xlabel("")
            inset.set_ylabel("")
            inset.set_xticks([])
            inset.set_yticks([])

            inset.set_aspect("equal")
            inset.spines[["right", "left", "top", "bottom"]].set_visible(False)

        if xmin is None and ymin is None and xmax is None and ymax is None:
            inset.remove()
        if not mini_map:
            inset.remove()

        # --- Plot ice core sites ---
        if stations and color_by != "stratigraphy":
            site_coords = pd.read_parquet(self._site_coords_path)
            for i in site_coords.index:
                site = site_coords.loc[i]
                ax.scatter(
                    site["x"] / 1000,
                    site["y"] / 1000,
                    color="red",
                    s=50,
                    marker="^",
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=50,
                )
        # --- Save/Show ---
        with self._plot_context():
            if save:
                plt.savefig(save, dpi=200)
                print("Figure saved as", save)
            else:
                plt.show()
