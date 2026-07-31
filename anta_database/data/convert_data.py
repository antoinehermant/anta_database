#!/usr/bin/env python3

import pandas as pd

gl = pd.read_pickle("./GL.pkl")

gl.to_parquet("./GL.parquet")

site_coords = pd.read_pickle("site-coords.pkl")

site_coords.to_parquet("site-coords.parquet")

# %%
from anta_database import Database

db = Database("/home/anthe/data/isochrones/cloud_database/AntADatabase/")

db.query(IMBIE_basin="G-H")
db.plot.dataset()
