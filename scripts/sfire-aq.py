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

modelrun = "F6V51M08R24"
configid = modelrun[:-6]

save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)
ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")

# pm_ef = 21.05  # boreal wildfire emission factor Urbanski (2014)
pm_ef = 10.400
fireshape_path = str(data_dir) + "/all_units/mygeodata_merged"
wrfin = str(data_dir) + f"/{modelrun}/interp-unit5-tr17_1.nc"
aqsin = str(data_dir) + "/obs/aq/"
aqsin = sorted(Path(aqsin).glob(f"*"))

# ## open smoke nc file
# wrf_ds = xr.open_dataset(wrfin)
wrf_ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")

# times = wrf_ds.Time.values
times = wrf_ds.XTIME.values

XLAT, XLONG = makeLL("met", configid)
## create dataframe with columns of all XLAT/XLONG
wrf_locs = pd.DataFrame({"XLAT": XLAT.values.ravel(), "XLONG": XLONG.values.ravel()})
## build kdtree
wrf_tree = KDTree(wrf_locs)
print("WRF Domain KDTree built")


def prep_aqs(aqin):
    print(str(aqin))
    aq_id = list(pd.read_csv(str(aqin), skiprows=2))[1]
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
    df = df.set_index("datetime")
    pm25 = df.filter(regex="PM2.5").mean(axis=1)
    df["pm25"] = pm25
    pm_max = np.max(pm25)
    pm_min = np.min(pm25)
    # df[f"pm25-{aq_id}"] = pm25
    # df = df[[f"pm25-{aq_id}"]]
    # df = df[~df.index.duplicated(keep="first")]
    # upsampled = df.resample("10S")
    # df = upsampled.interpolate(method="spline", order=5)
    df = df[str(times[0]) : str(times[-1])]
    # df = df.resample('10S')
    return df, aq_id, pm_max, pm_min


aq_dfs, aq_ids, pm_maxs, pm_mins = [], [], [], []
for aqin in aqsin:
    df, aq_id, pm_max, pm_min = prep_aqs(aqin)
    aq_dfs.append(df)
    aq_ids.append(aq_id)
    pm_maxs.append(pm_max)
    pm_mins.append(pm_min)

# aq_dfs = [prep_aqs(aqin) for aqin in aqsin]
# aq_dfs   = pd.concat(aq_dfs, axis=1)
# aq_dfs = aq_dfs[~aq_dfs.index.duplicated(keep="first")]
# norm_dfs =(aq_dfs-aq_dfs.min())/(aq_dfs.max()-aq_dfs.min())
# aq_ids = [get_ids(aqin) for aqin in aqsin]

with open(str(root_dir) + "/json/config.json") as f:
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


# aq_ds = wrf_ds.tr17_1.sel(south_north=y, west_east=x).isel(interp_level=1)
aq_ds = wrf_ds.tr17_1.sel(south_north=y, west_east=x).isel(bottom_top=1)

# def normalize(y):
#     x = y / np.linalg.norm(y)
#     return x
# norm_dfs =(aq_dfs-aq_dfs.min())/(aq_dfs.max()-aq_dfs.min())


def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


# def normalize_model(array):
#     return (array - np.min(aq_ds))/ (np.max(aq_ds)- np.min(aq_ds))


fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 1, 1)  # top and bottom left
for i in range(len(aq_ids)):
    modeld_aq = aq_ds.isel(aqs=i)
    ax.plot(modeld_aq.XTIME, modeld_aq, color=colors[i], label=aq_ids[i])
    # ax.plot(modeld_aq.XTIME, modeld_aq, color=colors[i], label=aq_ids[i])
    ax.plot(
        aq_dfs[i].index.values,
        aq_dfs[i]["pm25"].values,
        color=colors[i],
        linestyle="--",
    )
    # ax.plot(
    #     norm_dfs.index.values, norm_dfs[f"pm25-{aq_ids[i]}"].values, color=colors[i], linestyle="--"
    # )
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
plt.savefig(str(save_dir) + f"/smoke-aq-comparison.png", dpi=250)

# # aq2m_ds = aq_ds.isel(z = 0)
