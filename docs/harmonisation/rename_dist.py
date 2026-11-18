"""Rename the along-track distance Distance to DIST in the files of a build, in place.

The variable is renamed inside each file (an HDF5 link, no rewrite), so the values, the
compression and the other variables stay as they are. Files that already have DIST are skipped.
The package reads both names (anta_database.netcdf.with_dist), for the files of v1.

    python docs/harmonisation/rename_dist.py ~/data/isochrones/AntADatabase-v2-build [...]
"""
import glob
import os
import sys

import h5py

for root in sys.argv[1:]:
    renamed = skipped = 0
    for path in glob.glob(os.path.join(os.path.expanduser(root), "**", "*.nc"), recursive=True):
        with h5py.File(path, "a") as f:
            if "Distance" in f and "DIST" not in f:
                f.move("Distance", "DIST")
                renamed += 1
            else:
                skipped += 1
    print(f"{root}: {renamed} files renamed, {skipped} without Distance or with DIST already")
