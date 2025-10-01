import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap
import colormaps as cmaps
from contextlib import contextmanager
from typing import Union, Dict, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from anta_database.database.database import Database, MetadataResult

class Plotting:
    def __init__(self, database_instance: 'Database') -> None:
        self._db = database_instance
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.gl = pd.read_pickle(os.path.join(module_dir, 'GL.pkl'))


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

    def author(
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
            save: Optional[str] = None,
    ) -> None:
        """
        Plot the data points on a Antarctic map with color-coded dataset authors
        """
        self._base_plot(
            color_by='author',
            metadata=metadata,
            downscale_factor=downscale_factor,
            marker_size=marker_size,
            title=title,
            xlim=xlim,
            ylim=ylim,
            scale_factor=scale_factor,
            latex=latex,
            cmap=cmap,
            save=save,
        )

    def trace_id(
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
            save: Optional[str] = None,
    ) -> None:
        """
        Plot the data points on a Antarctic map with color-coded trace IDs
        """
        self._base_plot(
            color_by='trace_id',
            metadata=metadata,
            downscale_factor=downscale_factor,
            scale_factor=scale_factor,
            marker_size=marker_size,
            title=title,
            xlim=xlim,
            ylim=ylim,
            latex=latex,
            cmap=cmap,
            save=save,
        )

    def depth(
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
            save: Optional[str] = None,
    ) -> None:
        """
        Plot the color-coded layer depth on Antarctic map
        """
        self._base_plot(
            color_by='depth',
            metadata=metadata,
            downscale_factor=downscale_factor,
            title=title,
            xlim=xlim,
            ylim=ylim,
            scale_factor=scale_factor,
            marker_size=marker_size,
            latex=latex,
            cmap=cmap,
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
        color_by: str = 'author',  # 'author', 'trace_id', 'depth', 'density'
        cmap: Optional['LinearSegmentedColormap'] = None,
        ncol: Optional[int] = None,
    ) -> None:
        # --- Setup ---
        if metadata is None:
            if hasattr(self._db, 'md') and self._db.md:
                metadata = self._db.md
            else:
                print('Please provide metadata of the files you want to generate the data from...')
                return

        if latex:
            from matplotlib import rc
            rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
            rc('text', usetex=True)

        if not self._pre_plot_check(metadata):
            return

        fig, ax = plt.subplots()

        # --- Plot Grounding Line ---
        plt.plot(self.gl.x/1000, self.gl.y/1000, linewidth=1, color='k')

        # --- Plot Data ---
        colors = {}
        scatter = None
        values = None
        label = None
        extend = None

        if color_by == 'author':
            authors = list(metadata['author'])
            color_indices = np.linspace(0.1, 0.9, len(authors))
            if cmap == None:
                cmap = self._custom_cmap()
            colors = {author: cmap(i) for i, author in zip(color_indices, authors)}
            if marker_size == None:
                marker_size = 0.3
            for df, md in self._db.data_generator(metadata, downscale_factor=downscale_factor):
                author = md['author']
                plt.scatter(df.x/1000, df.y/1000, color=colors[author], s=marker_size)
            for author in authors:
                citation = self._db.query(author=author)['reference']
                plt.plot([], [], color=colors[author], label=citation, linewidth=3)
            if ncol == None:
                if len(authors) > 7:
                    ncol = 2
                if len(authors) > 15:
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

            if var == 'IRHDensity':
                levels = np.linspace(1, 20, 20)
                if cmap == None:
                    cmap = self._custom_cmap_density()
                norm = BoundaryNorm(levels, ncolors=256)
                values = np.arange(1, 21)
                colors = cmap(np.linspace(0, 1, len(values)))
                discrete_cmap = ListedColormap(colors)
                bounds = np.arange(0.5, 21)
                norm = BoundaryNorm(bounds, ncolors=discrete_cmap.N)
                label = f'{var} [N]'
                if marker_size == None:
                    marker_size = 1.
                for df, md in self._db.data_generator(metadata, downscale_factor=downscale_factor):
                    scatter = plt.scatter(df['x']/1000, df['y']/1000, c=df[var], cmap=discrete_cmap, s=marker_size, norm=norm)

            elif var in ['IceThk', 'SurfElev', 'BedElev', 'BasalUnit']:
                label = f'{var} [m]'
                if marker_size == None:
                    marker_size = 1.
                if cmap == None:
                    if var == 'BedElev':
                        cmap = cmaps.bukavu
                        extend='both'
                        for df, md in self._db.data_generator(metadata, downscale_factor=downscale_factor):
                            scatter = plt.scatter(df['x']/1000, df['y']/1000, c=df[var], cmap=cmap, s=marker_size, vmin=-1000, vmax=1000)
                    else:
                        cmap = cmaps.torch_r
                        for df, md in self._db.data_generator(metadata, downscale_factor=downscale_factor):
                            scatter = plt.scatter(df['x']/1000, df['y']/1000, c=df[var], cmap=cmap, s=marker_size)


        elif color_by == 'trace_id':
            trace_ids = list(metadata['trace_id'])
            color_indices = np.linspace(0.1, 0.9, len(trace_ids))
            if cmap == None:
                cmap = self._custom_cmap()
            colors = {tid: cmap(i) for i, tid in zip(color_indices, trace_ids)}
            if marker_size == None:
                marker_size = 1.
            for df, md in self._db.data_generator(metadata, downscale_factor=downscale_factor):
                trace_id = md['trace_id']
                plt.scatter(df.x/1000, df.y/1000, linewidth=0.8, color=colors[trace_id], s=marker_size)
            for trace_id in trace_ids:
                plt.plot([], [], color=colors[trace_id], label=trace_id, linewidth=3)
            ncol = 2 if len(trace_ids) > 40 else 1

        elif color_by == 'depth':
            if marker_size == None:
                marker_size = 1.
            if cmap == None:
                cmap = cmaps.torch_r
            age = list(metadata['age'])
            if len(age) > 1:
                print('WARNING: Multiple layers provided: ', age,
                      '\nSelect a unique age for better results')
            elif len(age) == 0:
                print('No layer provided, please provide a valid age.')
                return
            age = age[0]
            label = 'Depth [m]'
            for df, md in self._db.data_generator(metadata, downscale_factor=downscale_factor):
                if not md.get('age'):
                    continue
                scatter = plt.scatter(df['x']/1000, df['y']/1000, c=df['IRHDepth'], cmap=cmap, s=marker_size)

        # --- Format Figure ---
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
        if color_by == 'author':
            plt.legend(ncols=ncol, loc='lower left', fontsize=12*scale_factor)
        elif color_by == 'trace_id':
            ax.legend(ncols=ncol, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.gcf().set_size_inches(10 * scale_factor * ncol/1.15, 10 * aspect_ratio * scale_factor)
        elif color_by in ('depth', 'var') and scatter is not None:
            if color_by == 'var' and values is not None:
                cbar = fig.colorbar(scatter, ax=ax, ticks=values, orientation='horizontal', pad=0.1, fraction=0.04)
            else:
                cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.1, fraction=0.04, extend=extend)
            cbar.ax.xaxis.set_ticks_position('bottom')
            if label:
                cbar.set_label(label)

        plt.tight_layout()

        # --- Save/Show ---
        with self._plot_context():
            if save:
                plt.savefig(save, dpi=200)
                print('Figure saved as', save)
            else:
                plt.show()
