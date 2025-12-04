import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

if TYPE_CHECKING:
    from anta_database.database.database import Database, MetadataResult

class Plotting:
    def __init__(self, database_instance: 'Database') -> None:
        self._db = database_instance
        self._gl_path = files('anta_database.data').joinpath('GL.pkl')
        self._site_coords_path = files('anta_database.data').joinpath('site-coords.pkl')
        self._imbie_path = files('anta_database.data').joinpath('ANT_Basins_IMBIE2_v1.6.shp')
        self._center_coords = files('anta_database.data').joinpath('centeroid_coords_basins.shp')
        self._disable_tqdm = os.getenv("JUPYTER_BOOK_BUILD", False)

    def _pre_plot_check(self,
                        metadata: Union[None, Dict, 'MetadataResult'] = None
                        ) -> bool:

        if not metadata['age'] and not metadata['var']:
            print('Result from query provided is empty: nothing to plot. Please ensure that the query returns either valid age or var.')
            return False
        return True

    def _is_notebook(self) -> bool:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter

    def _custom_cmap(self, reversed_: bool = False):
        cm1 = cmaps.torch_r
        cm2 = cmaps.deep_r
        cm1_colors = cm1(np.linspace(0.15, 0.8, 256))
        cm2_colors = cm2(np.linspace(0.1, 0.9, 256))
        combined_colors = np.vstack((cm1_colors, cm2_colors))
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', combined_colors, N=512)
        return custom_cmap.reversed() if reversed_ else custom_cmap

    def _custom_cmap_density(self):
        return self._custom_cmap(reversed_=True)

    def dataset(
            self,
            metadata: Union[None, Dict, 'MetadataResult'] = None,
            downsampling_factor: Optional[int] = None,
            title: Optional[str] = None,
            xlim: Optional[tuple] = (None, None),
            ylim: Optional[tuple] = (None, None),
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            scale_factor: float = 1.0,
            marker_size: Optional[float] = 0.5,
            cmap: Optional['LinearSegmentedColormap'] = None,
            grounding_line: Optional[bool] = True,
            basins: Optional[bool] = True,
            stations: Optional[bool] = True,
            save: Optional[str] = None,
    ) -> None:
        """
        Plot the data points on a Antarctic map with color-coded dataset dataset
        """
        self._base_plot(
            color_by='dataset',
            metadata=metadata,
            downsampling_factor=downsampling_factor,
            marker_size=marker_size,
            title=title,
            xlim=xlim,
            ylim=ylim,
            vmin=vmin,
            vmax=vmax,
            scale_factor=scale_factor,
            cmap=cmap,
            grounding_line=grounding_line,
            basins=basins,
            stations=stations,
            save=save,
        )

    def institute(
            self,
            metadata: Union[None, Dict, 'MetadataResult'] = None,
            downsampling_factor: Optional[int] = None,
            title: Optional[str] = None,
            xlim: Optional[tuple] = (None, None),
            ylim: Optional[tuple] = (None, None),
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            scale_factor: float = 1.0,
            marker_size: Optional[float] = 0.1,
            cmap: Optional['LinearSegmentedColormap'] = None,
            grounding_line: Optional[bool] = True,
            basins: Optional[bool] = True,
            stations: Optional[bool] = True,
            save: Optional[str] = None,
    ) -> None:
        """
        Plot the data points on a Antarctic map with color-coded institutes
        """
        self._base_plot(
            color_by='institute',
            metadata=metadata,
            downsampling_factor=downsampling_factor,
            marker_size=marker_size,
            title=title,
            xlim=xlim,
            ylim=ylim,
            vmin=vmin,
            vmax=vmax,
            scale_factor=scale_factor,
            cmap=cmap,
            grounding_line=grounding_line,
            basins=basins,
            stations=stations,
            save=save,
        )

    def flight_id(
            self,
            metadata: Union[None, Dict, 'MetadataResult'] = None,
            downsampling_factor: Optional[int] = None,
            title: Optional[str] = None,
            xlim: Optional[tuple] = (None, None),
            ylim: Optional[tuple] = (None, None),
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            scale_factor: float = 1.0,
            marker_size: Optional[float] = 0.1,
            cmap: Optional['LinearSegmentedColormap'] = None,
            grounding_line: Optional[bool] = True,
            basins: Optional[bool] = True,
            stations: Optional[bool] = True,
            save: Optional[str] = None,
    ) -> None:
        """
        Plot the data points on a Antarctic map with color-coded trace IDs
        """
        self._base_plot(
            color_by='flight_id',
            metadata=metadata,
            downsampling_factor=downsampling_factor,
            scale_factor=scale_factor,
            marker_size=marker_size,
            title=title,
            xlim=xlim,
            ylim=ylim,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            grounding_line=grounding_line,
            basins=basins,
            stations=stations,
            save=save,
        )

    def var(
            self,
            metadata: Union[None, Dict, 'MetadataResult'] = None,
            downsampling_factor: Optional[int] = None,
            fraction_depth: Optional[bool] = False,
            title: Optional[str] = None,
            xlim: Optional[tuple] = (None, None),
            ylim: Optional[tuple] = (None, None),
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            scale_factor: float = 1.0,
            marker_size: Optional[float] = 0.3,
            cmap: Optional['LinearSegmentedColormap'] = None,
            grounding_line: Optional[bool] = True,
            basins: Optional[bool] = True,
            stations: Optional[bool] = True,
            save: Optional[str] = None,
    ) -> None:
        """
        Plot the color-coded values of the given variable on Antarcitic map
        """
        self._base_plot(
            color_by='var',
            metadata=metadata,
            downsampling_factor=downsampling_factor,
            fraction_depth=fraction_depth,
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
        )

    def transect_1D(
            self,
            metadata: Union[None, Dict, 'MetadataResult'] = None,
            elevation: Optional[bool] = False,
            downsampling_factor: Optional[int] = None,
            title: Optional[str] = None,
            xlim: Optional[tuple] = (None, None),
            ylim: Optional[tuple] = (None, None),
            scale_factor: float = 1.0,
            marker_size: Optional[float] = 2,
            cmap: Optional['LinearSegmentedColormap'] = None,
            grounding_line: Optional[bool] = True,
            basins: Optional[bool] = True,
            stations: Optional[bool] = True,
            save: Optional[str] = None,
    ) -> None:
        """
        Plot the color-coded values of the given variable on Antarcitic map
        """
        self._base_plot(
            color_by='transect_1D',
            elevation=elevation,
            metadata=metadata,
            downsampling_factor=downsampling_factor,
            title=title,
            xlim=xlim,
            ylim=ylim,
            scale_factor=scale_factor,
            marker_size=marker_size,
            cmap=cmap,
            grounding_line=grounding_line,
            basins=basins,
            stations=stations,
            save=save,
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

    def _base_plot(
            self,
            metadata: Union[None, Dict, 'MetadataResult'] = None,
            elevation: Optional[bool] = False,
            fraction_depth: Optional[bool] = False,
            downsampling_factor: Optional[int] = None,
            title: Optional[str] = None,
            xlim: Optional[tuple] = (None, None),
            ylim: Optional[tuple] = (None, None),
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            scale_factor: float = 1.0,
            marker_size: Optional[float] = 0.1,
            save: Optional[str] = None,
            color_by: str = 'dataset',  # 'dataset', 'flight_id', 'depth', 'density'
            cmap: Optional['LinearSegmentedColormap'] = None,
            grounding_line: Optional[bool] = True,
            basins: Optional[bool] = True,
            stations: Optional[bool] = True,
            ncol: Optional[int] = None,
    ) -> None:
        # --- Setup ---
        if metadata is None:
            if hasattr(self._db, '_md') and self._db._md:
                metadata = self._db._md
            else:
                print('Please provide metadata of the files you want to generate the data from...')
                return

        total_traces = len(metadata['flight_id'])

        if not self._pre_plot_check(metadata):
            return

        # if save:
        #     matplotlib.use('Agg')
        # else: FIXME: this seems to crash spyder for spyder users
        #     matplotlib.use('TkAgg')

        fig, ax = plt.subplots()

        if basins:
            grounding_line = False
        # --- Plot Grounding Line ---
        if True and color_by != 'transect_1D': # FIXME
            gl = pd.read_pickle(self._gl_path)
            ax.plot(gl.x/1000, gl.y/1000, linewidth=1, color='k')

        # --- Plot Data ---
        colors = {}
        scatter = None
        values = None
        label = None
        extend = None

        if color_by == 'dataset':
            if not title:
                title = f'AntADatabase by datasets'
            if downsampling_factor == None:
                downsampling_factor = 1
            datasets = list(metadata['dataset'])
            bedmap_entries = {'BEDMAP1', 'BEDMAP2', 'BEDMAP3'}
            bedmap_colors = {
                'BEDMAP1': '#e0e0e0',
                'BEDMAP2': '#adaaaf',
                'BEDMAP3': '#828084',
            }
            remaining_dataset = [dataset for dataset in datasets if dataset not in bedmap_entries]
            color_indices = np.linspace(0.1, 0.9, len(remaining_dataset))
            if cmap is None:
                cmap = self._custom_cmap()
            colors = {dataset: cmap(i) for i, dataset in zip(color_indices, remaining_dataset)}
            colors.update(bedmap_colors)

            flight_ids = metadata['flight_id']
            for dataset in tqdm(datasets, desc="Plotting", total=len(datasets), unit='dataset', disable=self._disable_tqdm):
                metadata_impl = self._db.query(dataset=dataset, flight_id=flight_ids, retain_query=False)
                file_paths = self._db._get_file_paths_from_metadata(metadata_impl)
                file_paths = np.unique(file_paths)
                zorder = 0 if dataset in ['BEDMAP1', 'BEDMAP2', 'BEDMAP3'] else 1

                all_x, all_y = [], []
                for f in file_paths:
                    full_path = os.path.join(self._db._db_dir, f)
                    with h5py.File(full_path, 'r') as ds:
                        all_x.append(ds['PSX'][::downsampling_factor])
                        all_y.append(ds['PSY'][::downsampling_factor])
                df = pd.DataFrame({'PSX': np.concatenate(all_x),
                                'PSY': np.concatenate(all_y)})
                plt.scatter(df['PSX']/1000, df['PSY']/1000, color=colors[dataset], s=marker_size, zorder=zorder, linewidths=0)

            for dataset in datasets:
                citation = self._db.query(dataset=dataset, retain_query=False)['reference']
                plt.plot([], [], color=colors[dataset], label=citation, linewidth=3)
            if ncol == None:
                if len(datasets) > 7:
                    ncol = 2
                if len(datasets) > 15:
                    ncol = 3

        if color_by == 'institute':
            if not title:
                title = f'AntADatabase by institutes'
            if downsampling_factor == None:
                downsampling_factor = 1
            institutes = list(metadata['institute'])

            color_indices = np.linspace(0.1, 0.9, len(institutes))
            if cmap is None:
                cmap = self._custom_cmap()
            colors = {dataset: cmap(i) for i, dataset in zip(color_indices, institutes)}

            flight_ids = metadata['flight_id']
            for institute in tqdm(institutes, desc="Plotting", total=len(institutes), unit='institute', disable=self._disable_tqdm):
                metadata_impl = self._db.query(institute=institute, flight_id=flight_ids, retain_query=False)
                file_paths = self._db._get_file_paths_from_metadata(metadata_impl)
                file_paths = np.unique(file_paths)

                all_x, all_y = [], []
                for f in file_paths:
                    full_path = os.path.join(self._db._db_dir, f)
                    with h5py.File(full_path, 'r') as ds:
                        all_x.append(ds['PSX'][::downsampling_factor])
                        all_y.append(ds['PSY'][::downsampling_factor])
                df = pd.DataFrame({'PSX': np.concatenate(all_x),
                                'PSY': np.concatenate(all_y)})
                plt.scatter(df['PSX']/1000, df['PSY']/1000, color=colors[institute], s=marker_size, linewidths=0)

                plt.plot([], [], color=colors[institute], label=institute, linewidth=3)
            if ncol == None:
                if len(institutes) > 7:
                    ncol = 2
                if len(institutes) > 15:
                    ncol = 3

        if color_by == 'var':
            var = list(metadata['var'])
            if len(var) > 1:
                print('Found mutilple variables to plot, chose one: ', var)
                return
            elif len(var) < 1:
                print('No variable found to plot')
                return
            else:
                var = var[0]

            all_dfs = []
            for ds, _ in tqdm(self._db.data_generator(metadata, downsampling_factor=downsampling_factor, disable_tqdm=True, fraction_depth=fraction_depth), desc="Plotting", total=total_traces, unit='trace', disable=self._disable_tqdm):
                all_dfs.append(ds)
            df = pd.concat(all_dfs)

            if var == 'IRH_NUM':
                if not title:
                    title = f'Number of traced IRHs per data point'
                levels = np.linspace(1, 10, 10)
                if cmap == None:
                    cmap = self._custom_cmap_density()

                norm = BoundaryNorm(levels, ncolors=256)
                values = np.arange(1, 11)
                colors = cmap(np.linspace(0, 1, len(values)))
                discrete_cmap = ListedColormap(colors)
                bounds = np.arange(0.5, 11.5)
                norm = BoundaryNorm(bounds, ncolors=discrete_cmap.N)
                label = f'{var} [N]'
                extend = 'max'

                df[var] = df[var].fillna(0)
                df = df.sort_values(by=var)
                unique_values = df[var].unique()
                for i, val in enumerate(unique_values):
                    subset = df[df[var] == val]
                    subset = subset[subset['IRH_NUM'] != 0.0]
                    scatter = ax.scatter(
                        subset['PSX'] / 1000,
                        subset['PSY'] / 1000,
                        c=subset[var],
                        cmap=discrete_cmap,

                        s=marker_size,
                        norm=norm,
                        linewidths=0,
                        zorder=i
                    )

            elif var in ['ICE_THK', 'SURF_ELEV', 'BED_ELEV', 'BASAL_UNIT']:
                label = f'{var} [m]'
                if var == 'BED_ELEV':
                    if cmap == None:
                        cmap = cmaps.bukavu
                    extend = 'both'
                    if vmin is None: vmin = -1000
                    if vmax is None: vmax = 1000
                    if not title:
                        title = f'AntADatabase Bed Elevation'
                elif var == 'ICE_THK':
                    if cmap == None:
                        cmap = cmaps.torch_r
                    extend = 'max'
                    if vmin is None: vmin = 0
                    if vmax is None: vmax = 4000
                    if not title:
                        title = f'AntADatabase Ice Thickness'
                elif var == 'SURF_ELEV':
                    if cmap == None:
                        cmap = cmaps.ice_r
                    extend = 'max'
                    if vmin is None: vmin = 1000
                    if vmax is None: vmax = 4000
                    if not title:
                        title = f'AntADatabase Surface Elevation'
                elif var == 'BASAL_UNIT':
                    if cmap == None:
                        cmap = cmaps.torch_r
                    extend = 'both'
                    if vmin is None: vmin = 2000
                    if vmax is None: vmax = 400
                    if not title:
                        title = f'AntADatabase Basal Unit'
                scatter = plt.scatter(df['PSX']/1000, df['PSY']/1000, c=df[var], cmap=cmap, s=marker_size, vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)

            elif var in ['IRH_DEPTH']:
                if cmap == None:
                    cmap = cmaps.torch_r
                extend = 'both'
                age = list(metadata['age'])
                if not title:
                    if fraction_depth:
                        title = f'AntADatabase IRH Fraction Depth'
                    else:
                        title = f'AntADatabase IRH Depth'
                if len(age) > 1:
                    print('WARNING: Multiple layers provided: ', age,
                        '\nSelect a unique age for better results')
                elif len(age) == 0:
                    print('No layer provided, please provide a valid age.')
                    return
                if fraction_depth:
                    label = r'IRH Fractional Depth [\%]'
                    if vmin is None: vmin = 0
                    if vmax is None: vmax = 100
                else:
                    label = f'IRH Depth [m]'

                for age in metadata['age']:
                    scatter = plt.scatter(df['PSX']/1000, df['PSY']/1000, c=df[age], cmap=cmap, s=marker_size, vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)

        if color_by == 'transect_1D':
            flight_id = list(metadata['flight_id'])
            if len(flight_id) > 1:
                flight_id = flight_id[0]
                print('Found mutilple flight lines to plot, so will plot the first one: ', flight_id)
            elif len(flight_id) < 1:
                print('No flight line found to plot')
                return

            metadata_impl = self._db.query(flight_id=flight_id, dataset=metadata['dataset'], retain_query=False)
            if not title:
                title = f'Transect {metadata_impl['flight_id'][0]} from {metadata_impl['reference'][0]}'

            f = self._db._get_file_paths_from_metadata(metadata_impl)[0]
            full_path = os.path.join(self._db._db_dir, f)
            import xarray as xr
            ds = xr.open_dataset(full_path, engine='h5netcdf')

            if 'Distance' not in ds.variables:
                print('Distance not in varibles, cannot plot along transect')
                return

            if elevation:
                if 'SURF_ELEV' in ds.variables:
                    ds['IRH_DEPTH'] = ds.SURF_ELEV - ds.IRH_DEPTH

                elif 'ICE_THK' in ds.variables and 'BED_ELEV' in ds.variables:
                    ds['SURF_ELEV'] = (ds.ICE_THK + ds.BED_ELEV) - ds.IRH_DEPTH
                    ds['IRH_DEPTH'] = ds.SURF_ELEV - ds.IRH_DEPTH
                else:
                    print('Cannot plot IRH Elevation from the variables in the file, missing either ICE_THK and BED_ELEV or SURF_ELEV')
                    return

            cmap = self._custom_cmap_density()
            colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(metadata_impl['age']))]
            for age, color in zip(list(map(int, metadata['age'])), colors):
                if age not in ds.IRH_AGE:
                    print(f'{metadata_impl['flight_id'][0]} does not contain age {age}, skipping')
                    continue

                ax.scatter(ds.Distance/1000, ds.IRH_DEPTH.sel(IRH_AGE=age),
                        color=color, s=marker_size, linewidths=0.1)
                plt.plot([], [], color=color, label=age, linewidth=3)

            if elevation:
                if 'BED_ELEV' in ds.variables:
                    scatter = ax.scatter(ds.Distance/1000, ds.BED_ELEV, color='k', s=marker_size, linewidths=0.1)
                    plt.plot([], [], color='k', label='Bed Elevation', linewidth=3)
                elif 'ICE_THK' in ds.variables and 'SURF_ELEV' in ds.variables:
                    ds['BED_ELEV'] = ds.SURF_ELEV - ds.ICE_THK
                    scatter = ax.scatter(ds.Distance/1000, ds.BED_ELEV, color='k', s=marker_size, linewidths=0.1)
                    plt.plot([], [], color='k', label='Bed Elevation', linewidth=3)
                else:
                    print('Cannot plot Bed Elevation from the variables in the file')
                ax.scatter(ds.Distance/1000, ds.SURF_ELEV, color='grey', s=marker_size, linewidths=0.1)
                plt.plot([], [], color='grey', label='Surface Elevation', linewidth=3)

            else:
                scatter = ax.scatter(ds.Distance/1000, ds.ICE_THK, color='k', s=marker_size, linewidths=0.1)
                plt.plot([], [], color='k', label='Bed Depth', linewidth=3)

            if elevation:
                ylim = (ds.BED_ELEV.min() - 200, ds.SURF_ELEV.max()+200) if ylim == (None, None) else ylim
            else:
                ylim = (ds.ICE_THK.max() + 200, 0) if ylim == (None, None) else ylim

            if ncol == None:
                if ds.sizes['IRH_AGE'] > 7:
                    ncol = 2
                if ds.sizes['IRH_AGE'] > 15:
                    ncol = 3

        elif color_by == 'flight_id':
            if not title:
                title = f'AntADatabase by flight IDs'
            flight_ids = list(metadata['flight_id'])
            color_indices = np.linspace(0.1, 0.9, len(flight_ids))
            if cmap == None:
                cmap = self._custom_cmap()
            colors = {tid: cmap(i) for i, tid in zip(color_indices, flight_ids)}

            datasets = list(metadata['dataset'])
            flight_ids = metadata['flight_id']
            for flight_id in tqdm(flight_ids, desc="Plotting", total=len(flight_ids), unit='flight_id', disable=self._disable_tqdm):
                metadata_impl = self._db.query(flight_id=flight_id, retain_query=False)
                file_paths = self._db._get_file_paths_from_metadata(metadata_impl)
                file_paths = np.unique(file_paths)
                all_x, all_y = [], []
                for f in file_paths:
                    full_path = os.path.join(self._db._db_dir, f)
                    with h5py.File(full_path, 'r') as ds:
                        all_x.append(ds['PSX'][:])
                        all_y.append(ds['PSY'][:])
                df = pd.DataFrame({'PSX': np.concatenate(all_x),
                                'PSY': np.concatenate(all_y)})
                plt.scatter(df['PSX'][::downsampling_factor]/1000, df['PSY'][::downsampling_factor]/1000, color=colors[flight_id], s=marker_size, linewidths=0)

                plt.plot([], [], color=colors[flight_id], label=flight_id, linewidth=3)
            ncol = 2 if len(flight_ids) > 40 else 1

        # --- Format Figure ---
        if not self._disable_tqdm:
            print('Formatting ...')

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if color_by != 'transect_1D':
            x0, x1 = ax.get_xlim() if xlim == (None, None) else xlim
            y0, y1 = ax.get_ylim() if ylim == (None, None) else ylim
            x_extent = x1 - x0
            y_extent = y1 - y0
            aspect_ratio = y_extent / x_extent
            ax.set_xlabel('x [km]')
            ax.set_ylabel('y [km]')
            ax.set_aspect('equal')
            plt.gcf().set_size_inches(10 * scale_factor, 10 * aspect_ratio * scale_factor)
        plt.title(title, fontsize=24*scale_factor)

        if ncol == None:
            ncol = 1
        # --- Legend/Colorbar ---
        if color_by in ['dataset', 'institute']:
            plt.legend(ncols=ncol, loc='lower left', fontsize=8)
        elif color_by == 'flight_id':
            ax.legend(ncols=ncol, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.gcf().set_size_inches(10 * scale_factor * ncol/1.15, 10 * aspect_ratio * scale_factor)
        elif color_by == 'var' and scatter is not None:
            if values is not None:
                cbar = fig.colorbar(scatter, ax=ax, ticks=values, orientation='horizontal', pad=0.1, fraction=0.04, extend=extend)
            else:
                cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.1, fraction=0.04, extend=extend)
            cbar.ax.xaxis.set_ticks_position('bottom')
            if label:
                cbar.set_label(label)
        elif color_by == 'transect_1D':
            ax.legend(ncols=2)
            ax.set_xlabel('Distance along transect [km]')
            if elevation:
                ax.set_ylabel('Elevation above sea level [m]')
            else:
                ax.set_ylabel('Depth below surface [m]')
            plt.gcf().set_size_inches(10 * scale_factor, 10 * 2/3)

        plt.tight_layout()

        # --- Plot IMBIE basins ---
        if basins and color_by != 'transect_1D':
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
                facecolor='none',
                edgecolor='black',
                linewidth=0.5,
            )
            ax.add_collection(pc)

            # ---- CENTERS: read + scale + label ----
            sf_centers = shapefile.Reader(self._center_coords)

            # find the field index for "Subregion" in the DBF
            fields = sf_centers.fields[1:]  # first is DeletionFlag
            field_names = [f[0] for f in fields]
            sub_idx = field_names.index('Subregion')

            for shape_rec in sf_centers.shapeRecords():
                # assuming center_coords are points
                x_raw, y_raw = shape_rec.shape.points[0]
                x = x_raw * 0.001
                y = y_raw * 0.001
                sub = shape_rec.record[sub_idx]

                if x0 <= x <= x1 and y0 <= y <= y1:
                    ax.text(
                        x, y, sub, fontsize=12, color='k', ha='center',
                        path_effects=[path_effects.withStroke(
                            linewidth=5, foreground=(1, 1, 1, 0.7)
                        )]
                    )
        # --- Plot ice core sites ---
        if stations and color_by != 'transect_1D':
            site_coords = pd.read_pickle(self._site_coords_path)
            for i in site_coords.index:
                site = site_coords.loc[i]
                ax.scatter(site['x']/1000, site['y']/1000, color='red', s=50, marker='^', edgecolor='black', linewidth=1.5, zorder=50)
        # --- Save/Show ---
        with self._plot_context():
            if save:
                plt.savefig(save, dpi=200)
                print('Figure saved as', save)
            else:
                plt.show()
