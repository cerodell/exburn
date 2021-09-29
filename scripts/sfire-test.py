import context
import wrf
import json
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from context import root_dir, vol_dir, data_dir, save_dir, gog_dir
import matplotlib.pylab as pylab


fireshape_path = str(gog_dir) + "/all_units/mygeodata_merged"
wrfin = str(vol_dir) + "/sfire/unit5/moist_false/"
aqsin = str(vol_dir) + "/sfire/unit5/obs/aq/"


aq_id = list(pd.read_csv(aqsin + "u5_air1.csv", skiprows=2))[1]
df = pd.read_csv(aqsin + "u5_air1.csv", skiprows=3).iloc[1:]
df["year"], df["month"] = "20" + df["year"], "0" + df["month"]
df["datetime"] = pd.to_datetime(
    df[["year", "month", "day", "hour", "minute", "second"]], format="%y%m%d"
)
P1Channels = df.filter(regex="PM2.5").mean(axis=1)


# with open(str(data_dir) + "/json/config.json") as f:
#     config = json.load(f)
# aqs = config['unit5']['obs']['aq']

# ## open smoke nc file
# smoke_ds = xr.open_dataset(str(wrfin) + "/firesmoke.nc")
# LAT, LONG = smoke_ds.LAT.values, smoke_ds.LONG.values


# ## create dataframe with columns of all lat/long
# wrf_locs = pd.DataFrame({"LAT": LAT.ravel(), "LONG": LONG.ravel()})
# ## build kdtree
# wrf_tree = KDTree(wrf_locs)
# print("WRF Domain KDTree built")

# def find_index(aq):
#     aq = np.array([aqs[aq]['lat'],aqs[aq]['lon']]).reshape(1, -1)
#     aq_dist, aq_ind = wrf_tree.query(aq, k=1)
#     aq_loc = list(np.unravel_index(int(aq_ind), LAT.shape))
#     return aq_loc

# aqs_loc = np.stack([find_index(aq) for aq in aqs])
# y = xr.DataArray(np.array(aqs_loc[:,0]), dims= 'aqs', coords= dict(aqs = list(aqs)))
# x = xr.DataArray(np.array(aqs_loc[:,1]), dims= 'aqs', coords= dict(aqs = list(aqs)))


# aq_ds = smoke_ds.TRACER.sel(y = y, x = x)
# # aq2m_ds = aq_ds.isel(z = 0)
