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
from utils.sfire import makeLL

fueltype = 6

ds = xr.open_dataset(str(data_dir) + f"/fuel{fueltype}/wrfout_d01_2019-05-11_17:49:11")

pm_ef = 21.05  # boreal wildfire emission factor Urbanski (2014)
fueltype = 10
fireshape_path = str(data_dir) + "/all_units/mygeodata_merged"
wrfin = str(data_dir) + f"/fuel{fueltype}/interp-unit5-tr17_1.nc"
aqsin = str(data_dir) + "/obs/aq/"
aqsin = sorted(Path(aqsin).glob(f"*"))

# ## open smoke nc file
wrf_ds = xr.open_dataset(wrfin)
times = wrf_ds.Time.values
XLAT, XLONG = makeLL("met")
## create dataframe with columns of all XLAT/XLONG
wrf_locs = pd.DataFrame({"XLAT": XLAT.values.ravel(), "XLONG": XLONG.values.ravel()})
## build kdtree
wrf_tree = KDTree(wrf_locs)
print("WRF Domain KDTree built")


def prep_aqs(aqin):
    print(str(aqin))
    df = pd.read_csv(str(aqin), skiprows=3).iloc[1:]
    str_list = ["year", "month", "day", "hour", "minute", "second"]
    float_list = [x for x in list(df) if x not in str_list]
    df[float_list] = df[float_list].astype(float)
    try:
        df["datetime"] = pd.to_datetime(df[str_list], format="%y%m%d")
    except:
        df["year"], df["month"] = "20" + df["year"], "0" + df["month"]
        df["datetime"] = pd.to_datetime(df[str_list], format="%y%m%d")
    # df = df.set_index('datetime').resample('10S').mean()
    df = df.set_index("datetime")[str(times[0]) : str(times[-1])]
    pm25 = df.filter(regex="PM2.5").mean(axis=1)
    df["pm25"] = pm25
    # df = df.resample('10S')
    return df


def get_ids(aqin):
    return list(pd.read_csv(str(aqin), skiprows=2))[1]


aq_dfs = [prep_aqs(aqin) for aqin in aqsin]
aq_ids = [get_ids(aqin) for aqin in aqsin]

with open(str(data_dir) + "/json/config.json") as f:
    config = json.load(f)
aqs = config["unit5"]["obs"]["aq"]
blah = list(aqs)


def find_index(aq):
    print(aq)
    aq = np.array([aqs[aq]["lat"], aqs[aq]["lon"]]).reshape(1, -1)
    aq_dist, aq_ind = wrf_tree.query(aq, k=1)
    aq_loc = list(np.unravel_index(int(aq_ind), XLAT.shape))
    return aq_loc


aqs_loc = np.stack([find_index(aq) for aq in aq_ids])
y = xr.DataArray(np.array(aqs_loc[:, 0]), dims="aqs", coords=dict(aqs=aq_ids))
x = xr.DataArray(np.array(aqs_loc[:, 1]), dims="aqs", coords=dict(aqs=aq_ids))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


aq_ds = wrf_ds.tr17_1.sel(south_north=y, west_east=x).isel(interp_level=1)


fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 1, 1)  # top and bottom left
for i in range(len(aq_ids)):
    modeld_aq = aq_ds.isel(aqs=i)
    ax.plot(modeld_aq.Time, modeld_aq, color=colors[i], label=aq_ids[i])
    ax.plot(
        aq_dfs[i].index.values, aq_dfs[i].pm25.values, color=colors[i], linestyle="--"
    )
ax.scatter(
    pd.Timestamp(2019, 5, 11, 17, 49, 3),
    0,
    marker="*",
    color="red",
    zorder=10,
    label="Ignition Time",
    s=100,
)

plt.legend()

# # aq2m_ds = aq_ds.isel(z = 0)
