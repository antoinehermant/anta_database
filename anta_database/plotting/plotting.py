import os
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
        imbie_path = files('anta_database.data').joinpath('ANT_Basins_IMBIE2_v1.6.shp')
        self.basins = gpd.read_file(imbie_path)
        center_coords = files('anta_database.data').joinpath('centeroid_coords_basins.shp')
        self.center_coords = gpd.read_file(center_coords)

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
            marker_size: Optional[float] = None,
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

    def flight_id(
            self,
            metadata: Union[None, Dict, 'MetadataResult'] = None,
            downscale_factor: Optional[int] = None,
            title: str = '',
            xlim: tuple = (None, None),
            ylim: tuple = (None, None),
            scale_factor: float = 1.0,
            marker_size: Optional[float] = None,
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
            marker_size: Optional[float] = None,
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
        marker_size: Optional[float] = None,
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
                downscale_fator = 1
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

            if marker_size == None:
                marker_size = 0.01
            flight_ids = metadata['flight_id']
            for dataset in tqdm(datasets, desc="Plotting", total=len(datasets), unit='dataset'):
                metadata_impl = self._db.query(dataset=dataset, flight_id=flight_ids)
                file_paths = self._db._get_file_paths_from_metadata(metadata_impl)
                unique_directories = {Path(fp).parent for fp in file_paths}
                zorder = 0 if dataset in ['BEDMAP1', 'BEDMAP2', 'BEDMAP3'] else 1
                files = [
                    str(self._db.db_dir / unique_dir / "TOTAL_PSXPSY.pkl")
                    for unique_dir in unique_directories
                ]

                data_frames = (pd.read_pickle(f) for f in files)
                combined_df = pd.concat(data_frames, ignore_index=True)
                plt.scatter(combined_df['PSX'][::downscale_factor]/1000, combined_df['PSY'][::downscale_factor]/1000, color=colors[dataset], s=marker_size, zorder=zorder)

            for dataset in datasets:
                citation = self._db.query(dataset=dataset)['reference']
                plt.plot([], [], color=colors[dataset], label=citation, linewidth=3)
            if ncol == None:
                if len(datasets) > 7:
                    ncol = 2
                if len(datasets) > 15:
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

            if var == 'IRH_DENS':
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
                if marker_size == None:
                    marker_size = 1.
                all_dfs = []
                for df, _ in tqdm(self._db.data_generator(metadata, downscale_factor=downscale_factor), desc="Plotting", total=total_traces, unit='trace'):
                    all_dfs.append(df)
                combined_df = pd.concat(all_dfs, ignore_index=True)
                scatter = plt.scatter(combined_df['PSX']/1000, combined_df['PSY']/1000, c=combined_df[var], cmap=discrete_cmap, s=marker_size, norm=norm)

            elif var in ['ICE_THCK', 'SURF_ELEV', 'BED_ELEV', 'BASAL_UNIT', 'IRH_DEPTH', 'IRH_FRAC_DEPTH']:
                label = f'{var} [m]'
                if marker_size == None:
                    marker_size = 1.
                if var == 'BED_ELEV':
                    if cmap == None:
                        cmap = cmaps.bukavu
                    extend = 'both'
                    vmin = -1000
                    vmax = 1000
                elif var == 'ICE_THCK':
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
                for df, _ in tqdm(self._db.data_generator(metadata, downscale_factor=downscale_factor), desc="Plotting", total=total_traces, unit='trace'):
                    all_dfs.append(df)
                combined_df = pd.concat(all_dfs, ignore_index=True)
                scatter = plt.scatter(combined_df['PSX']/1000, combined_df['PSY']/1000, c=combined_df[var], cmap=cmap, s=marker_size, vmin=vmin, vmax=vmax, rasterized=True)

        elif color_by == 'flight_id':
            flight_ids = list(metadata['flight_id'])
            color_indices = np.linspace(0.1, 0.9, len(flight_ids))
            if cmap == None:
                cmap = self._custom_cmap()
            colors = {tid: cmap(i) for i, tid in zip(color_indices, flight_ids)}
            if marker_size == None:
                marker_size = 1.

            datasets = list(metadata['dataset'])
            flight_ids = metadata['flight_id']
            for dataset in tqdm(datasets, desc="Plotting", total=len(datasets), unit='dataset'):
                metadata_impl = self._db.query(dataset=dataset, flight_id=flight_ids)
                file_paths = self._db._get_file_paths_from_metadata(metadata_impl)
                unique_directories = {Path(fp).parent for fp in file_paths}
                files = [
                    str(self._db.db_dir / unique_dir / "TOTAL_PSXPSY.pkl")
                    for unique_dir in unique_directories
                ]
                for unique_dir in unique_directories:
                    TOTAL_PSXPSY = pd.read_pickle(f'{self._db.db_dir}/{unique_dir}/TOTAL_PSXPSY.pkl')
                    flight_id = pd.read_csv(f'{self._db.db_dir}/{unique_dir}/metadata.csv').iloc[0]['flight_id']
                    plt.scatter(TOTAL_PSXPSY['PSX'][::downscale_factor]/1000, TOTAL_PSXPSY['PSY'][::downscale_factor]/1000, color=colors[flight_id], s=marker_size)

            for flight_id in flight_ids:
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
        if color_by == 'dataset':
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
            self.basins.geometry = self.basins.geometry.scale(xfact=0.001, yfact=0.001, origin=(0, 0))
            self.basins.plot(ax=ax, color='none', edgecolor='black', linewidth=0.5)
            self.center_coords.geometry = self.center_coords.geometry.scale(xfact=0.001, yfact=0.001, origin=(0, 0))
            self.center_coords['x'] = self.center_coords['geometry'].x
            self.center_coords['y'] = self.center_coords['geometry'].y
            for x, y, sub in zip(self.center_coords['x'], self.center_coords['y'], self.center_coords['Subregion']):
                ax.text(x, y, sub, fontsize=12, color='k', ha='center',
                    path_effects=[path_effects.withStroke(linewidth=5, foreground='white')]
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
