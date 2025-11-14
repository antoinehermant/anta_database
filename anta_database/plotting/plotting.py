import os
import h5py
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
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
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.gl_path = files('anta_database.data').joinpath('GL.pkl')
        self.site_coords_path = files('anta_database.data').joinpath('site-coords.pkl')
        self.imbie_path = files('anta_database.data').joinpath('ANT_Basins_IMBIE2_v1.6.shp')
        self.center_coords = files('anta_database.data').joinpath('centeroid_coords_basins.shp')

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
            downscale_factor: Optional[int] = None,
            title: str = '',
            xlim: tuple = (None, None),
            ylim: tuple = (None, None),
            scale_factor: float = 1.0,
            marker_size: Optional[float] = 0.1,
            latex: bool = False,
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
            downscale_factor=downscale_factor,
            marker_size=marker_size,
            title=title,
            xlim=xlim,
            ylim=ylim,
            scale_factor=scale_factor,
            latex=latex,
            cmap=cmap,
            grounding_line=grounding_line,
            basins=basins,
            stations=stations,
            save=save,
        )

    def institute(
            self,
            metadata: Union[None, Dict, 'MetadataResult'] = None,
            downscale_factor: Optional[int] = None,
            title: str = '',
            xlim: tuple = (None, None),
            ylim: tuple = (None, None),
            scale_factor: float = 1.0,
            marker_size: Optional[float] = 0.1,
            latex: bool = False,
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
            downscale_factor=downscale_factor,
            marker_size=marker_size,
            title=title,
            xlim=xlim,
            ylim=ylim,
            scale_factor=scale_factor,
            latex=latex,
            cmap=cmap,
            grounding_line=grounding_line,
            basins=basins,
            stations=stations,
            save=save,
        )

    def flight_id(
            self,
            metadata: Union[None, Dict, 'MetadataResult'] = None,
            downscale_factor: Optional[int] = None,
            title: str = '',
            xlim: tuple = (None, None),
            ylim: tuple = (None, None),
            scale_factor: float = 1.0,
            marker_size: Optional[float] = 0.1,
            latex: bool = False,
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
            downscale_factor=downscale_factor,
            scale_factor=scale_factor,
            marker_size=marker_size,
            title=title,
            xlim=xlim,
            ylim=ylim,
            latex=latex,
            cmap=cmap,
            grounding_line=grounding_line,
            basins=basins,
            stations=stations,
            save=save,
        )

    def var(
            self,
            metadata: Union[None, Dict, 'MetadataResult'] = None,
            downscale_factor: Optional[int] = None,
            title: str = '',
            xlim: tuple = (None, None),
            ylim: tuple = (None, None),
            scale_factor: float = 1.0,
            marker_size: Optional[float] = 0.1,
            latex: bool = False,
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
            downscale_factor=downscale_factor,
            title=title,
            xlim=xlim,
            ylim=ylim,
            scale_factor=scale_factor,
            marker_size=marker_size,
            latex=latex,
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
        downscale_factor: Optional[int] = None,
        title: str = '',
        xlim: tuple = (None, None),
        ylim: tuple = (None, None),
        scale_factor: float = 1.0,
        marker_size: Optional[float] = 0.1,
        latex: bool = False,
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
            if hasattr(self._db, 'md') and self._db.md:
                metadata = self._db.md
            else:
                print('Please provide metadata of the files you want to generate the data from...')
                return

        total_traces = len(metadata['flight_id'])

        if latex:
            from matplotlib import rc
            rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
            rc('text', usetex=True)

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
        if True: # FIXME
            gl = pd.read_pickle(self.gl_path)
            ax.plot(gl.x/1000, gl.y/1000, linewidth=1, color='k')

        # --- Plot Data ---
        colors = {}
        scatter = None
        values = None
        label = None
        extend = None
        vmin = 0
        vmax = 10

        if color_by == 'dataset':
            if downscale_factor == None:
                downscale_factor = 1
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
            for dataset in tqdm(datasets, desc="Plotting", total=len(datasets), unit='dataset'):
                metadata_impl = self._db.query(dataset=dataset, flight_id=flight_ids, retain_query=False)
                file_paths = self._db._get_file_paths_from_metadata(metadata_impl)
                file_paths = np.unique(file_paths)
                zorder = 0 if dataset in ['BEDMAP1', 'BEDMAP2', 'BEDMAP3'] else 1

                all_x, all_y = [], []
                for f in file_paths:
                    full_path = os.path.join(self._db.db_dir, f)
                    with h5py.File(full_path, 'r') as ds:
                        all_x.append(ds['PSX'][::downscale_factor])
                        all_y.append(ds['PSY'][::downscale_factor])
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
            if downscale_factor == None:
                downscale_factor = 1
            institutes = list(metadata['institute'])

            color_indices = np.linspace(0.1, 0.9, len(institutes))
            if cmap is None:
                cmap = self._custom_cmap()
            colors = {dataset: cmap(i) for i, dataset in zip(color_indices, institutes)}

            flight_ids = metadata['flight_id']
            for institute in tqdm(institutes, desc="Plotting", total=len(institutes), unit='institute'):
                metadata_impl = self._db.query(institute=institute, flight_id=flight_ids, retain_query=False)
                file_paths = self._db._get_file_paths_from_metadata(metadata_impl)
                file_paths = np.unique(file_paths)

                all_x, all_y = [], []
                for f in file_paths:
                    full_path = os.path.join(self._db.db_dir, f)
                    with h5py.File(full_path, 'r') as ds:
                        all_x.append(ds['PSX'][::downscale_factor])
                        all_y.append(ds['PSY'][::downscale_factor])
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

            if var == 'IRH_NUM':
                levels = np.linspace(1, 10, 10)
                if cmap == None:
                    cmap = self._custom_cmap_density()

                norm = BoundaryNorm(levels, ncolors=256)
                values = np.arange(1, 11)
                colors = cmap(np.linspace(0, 1, len(values)))
                discrete_cmap = ListedColormap(colors)
                bounds = np.arange(0.5, 11)
                norm = BoundaryNorm(bounds, ncolors=discrete_cmap.N)
                label = f'{var} [N]'
                extend = 'max'
                all_dfs = []
                for ds, _ in tqdm(self._db.data_generator(metadata, downscale_factor=downscale_factor), desc="Plotting", total=total_traces, unit='trace'):
                    all_dfs.append(ds)

                df = pd.concat(all_dfs)
                df[var] = df[var].fillna(0)
                df = df.sort_values(by=var)
                unique_values = df[var].unique()
                for i, val in enumerate(unique_values):
                    subset = df[df[var] == val]
                    scatter = ax.scatter(
                        subset.x / 1000,
                        subset.y / 1000,
                        c=subset[var],
                        cmap=discrete_cmap,
                        s=marker_size,
                        norm=norm,
                        linewidths=0,
                        zorder=i
                    )

            elif var in ['ICE_THK', 'SURF_ELEV', 'BED_ELEV', 'BASAL_UNIT', 'IRH_DEPTH', 'IRH_FRAC_DEPTH']:
                label = f'{var} [m]'
                if var == 'BED_ELEV':
                    if cmap == None:
                        cmap = cmaps.bukavu
                    extend = 'both'
                    vmin = -1000
                    vmax = 1000
                elif var == 'ICE_THK':
                    if cmap == None:
                        cmap = cmaps.torch_r
                    extend = 'max'
                    vmin = 0
                    vmax = 4000
                elif var == 'SURF_ELEV':
                    if cmap == None:
                        cmap = cmaps.torch_r
                    extend = 'max'
                    vmin = 2000
                    vmax = 4000
                elif var == 'BASAL_UNIT':
                    if cmap == None:
                        cmap = cmaps.torch_r
                    extend = 'both'
                    vmin = 2000
                    vmax = 4000
                elif var == 'IRH_DEPTH':
                    if cmap == None:
                        cmap = cmaps.torch_r
                    extend = 'both'
                    vmin = None
                    vmax = None
                    age = list(metadata['age'])
                    if len(age) > 1:
                        print('WARNING: Multiple layers provided: ', age,
                            '\nSelect a unique age for better results')
                    elif len(age) == 0:
                        print('No layer provided, please provide a valid age.')
                        return
                elif var == 'IRH_FRAC_DEPTH':
                    if cmap == None:
                        cmap = cmaps.torch_r
                    extend = 'both'
                    vmin = None
                    vmax = None
                    age = list(metadata['age'])
                    if len(age) > 1:
                        print('WARNING: Multiple layers provided: ', age,
                            '\nSelect a unique age for better results')
                    elif len(age) == 0:
                        print('No layer provided, please provide a valid age.')
                        return
                    age = age[0]
                    label = r'IRH Fractional Depth [\%]'

                all_dfs = []
                for ds, _ in tqdm(self._db.data_generator(metadata, downscale_factor=downscale_factor), desc="Plotting", total=total_traces, unit='trace'):
                    all_dfs.append(ds)

                df = pd.concat(all_dfs)
                scatter = plt.scatter(df.x/1000, df.y/1000, c=df[var], cmap=cmap, s=marker_size, vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)

                #     all_dfs.append(df)
                # combined_df = pd.concat(all_dfs, ignore_index=True)

        elif color_by == 'flight_id':
            flight_ids = list(metadata['flight_id'])
            color_indices = np.linspace(0.1, 0.9, len(flight_ids))
            if cmap == None:
                cmap = self._custom_cmap()
            colors = {tid: cmap(i) for i, tid in zip(color_indices, flight_ids)}

            datasets = list(metadata['dataset'])
            flight_ids = metadata['flight_id']
            for flight_id in tqdm(flight_ids, desc="Plotting", total=len(flight_ids), unit='flight_id'):
                metadata_impl = self._db.query(flight_id=flight_id, retain_query=False)
                file_paths = self._db._get_file_paths_from_metadata(metadata_impl)
                file_paths = np.unique(file_paths)
                all_x, all_y = [], []
                for f in file_paths:
                    full_path = os.path.join(self._db.db_dir, f)
                    with h5py.File(full_path, 'r') as ds:
                        all_x.append(ds['PSX'][:])
                        all_y.append(ds['PSY'][:])
                df = pd.DataFrame({'PSX': np.concatenate(all_x),
                                'PSY': np.concatenate(all_y)})
                plt.scatter(df['PSX'][::downscale_factor]/1000, df['PSY'][::downscale_factor]/1000, color=colors[flight_id], s=marker_size, linewidths=0)

                plt.plot([], [], color=colors[flight_id], label=flight_id, linewidth=3)
            ncol = 2 if len(flight_ids) > 40 else 1

        # --- Format Figure ---
        print('Formatting ...')
        x0, x1 = ax.get_xlim() if xlim == (None, None) else xlim
        y0, y1 = ax.get_ylim() if ylim == (None, None) else ylim
        x_extent = x1 - x0
        y_extent = y1 - y0
        aspect_ratio = y_extent / x_extent
        plt.gcf().set_size_inches(10 * scale_factor, 10 * aspect_ratio * scale_factor)
        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
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

        plt.tight_layout()

        # --- Plot IMBIE basins ---
        if basins:
            basins = gpd.read_file(self.imbie_path)
            basins.geometry = basins.geometry.scale(xfact=0.001, yfact=0.001, origin=(0, 0))
            basins.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)
            center_coords = gpd.read_file(self.center_coords)
            center_coords.geometry = center_coords.geometry.scale(xfact=0.001, yfact=0.001, origin=(0, 0))
            center_coords['x'] = center_coords['geometry'].x
            center_coords['y'] = center_coords['geometry'].y
            for x, y, sub in zip(center_coords['x'], center_coords['y'], center_coords['Subregion']):
                ax.text(x, y, sub, fontsize=12, color='k', ha='center',
                        path_effects=[path_effects.withStroke(linewidth=5, foreground=(1,1,1,0.7))]
                        )
        # --- Plot ice core sites ---
        if stations:
            site_coords = pd.read_pickle(self.site_coords_path)
            for i in site_coords.index:
                site = site_coords.loc[i]
                ax.scatter(site['x']/1000, site['y']/1000, color='red', s=50, marker='^', edgecolor='black', linewidth=1.5)
        # --- Save/Show ---
        with self._plot_context():
            if save:
                plt.savefig(save, dpi=200)
                print('Figure saved as', save)
            else:
                plt.show()
