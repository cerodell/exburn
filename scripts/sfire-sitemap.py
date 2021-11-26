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
from sklearn.preprocessing import normalize
from scipy.interpolate import griddata
import matplotlib.colors as colors
from matplotlib.pyplot import cm

from matplotlib.dates import DateFormatter

import glob

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap
from context import root_dir, data_dir, save_dir
from utils.sfire import makeLL
import matplotlib
import warnings
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({"font.size": 10})
warnings.filterwarnings("ignore", category=RuntimeWarning)


domain = "fire"
unit = "unit5"
modelrun = "F6V51M08Z22"
from pylab import *

levels_ros = np.arange(10, 500, 20)
cmap = cm.get_cmap("tab20c", 20)  # PiYG

hex = []
for i in range(cmap.N):
    rgba = cmap(i)
    hex.append(matplotlib.colors.rgb2hex(rgba))
cmap_ros = LinearSegmentedColormap.from_list("tab20c", hex, N=len(levels_ros))


configid = "F6V51"
title = "Time of Arrival"
var = "FGRNHFX"
ig_start = [55.7177497, -113.5713062]
ig_end = [55.7177507, -113.5751922]
v = np.arange(0, 201, 1)
Cnorm = colors.Normalize(vmin=0, vmax=200)
ros_filein = str(data_dir) + "/obs/ros/"
fireshape_path = str(root_dir) + "/data/unit_5/unit_5"
headers = ["day", "hour", "minute", "second", "temp"]
save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)

with open(str(root_dir) + "/json/config.json") as f:
    config = json.load(f)
ros = config["unit5"]["obs"]["ros"]
ros_ids = list(ros)
mets = config["unit5"]["obs"]["met"]
aqs = config["unit5"]["obs"]["aq"]
hfxs = config["unit5"]["obs"]["hfx"]

bounds = config["unit5"]["sfire"][configid]
south_north_subgrid = slice(540, 760, None)
west_east_subgrid = slice(220, 500, None)
fs = bounds["namelist"]["dxy"] / bounds["namelist"]["fs"]

ds = xr.open_zarr(str(data_dir) + "/wrfinput/SINGLELINE/fuel6.zarr")
wrf_ds = xr.open_dataset(
    str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11", chunks="auto"
)

var_da = wrf_ds[var]
var_da = var_da.sel(
    south_north_subgrid=south_north_subgrid,
    west_east_subgrid=west_east_subgrid,
    Time=slice(3, 50),
)

FUELS = ds.sel(XLAT=south_north_subgrid, XLONG=west_east_subgrid)
times = var_da.XTIME.values

XLAT, XLONG = makeLL(domain, configid)
XLAT = XLAT.sel(
    south_north_subgrid=south_north_subgrid, west_east_subgrid=west_east_subgrid
)
XLONG = XLONG.sel(
    south_north_subgrid=south_north_subgrid, west_east_subgrid=west_east_subgrid
)


def prepare_df(rosin):
    df = pd.read_csv(
        glob.glob(ros_filein + f"{rosin}*.txt")[0],
        sep="\t",
        index_col=False,
        skiprows=16,
        names=headers,
    )
    df["year"], df["month"] = "2019", "05"
    df = df[:-1]
    df["DateTime"] = pd.to_datetime(
        df[["year", "month"] + headers[:-1]], infer_datetime_format=True
    )
    df.drop(["year", "month"] + headers[:-1], axis=1, inplace=True)
    df = df.set_index("DateTime")
    df = df[~df.index.duplicated(keep="first")]
    upsampled = df.resample("1S")
    df = upsampled.interpolate(method="linear")
    df = df[str(times[0]) : str(times[-1])]
    df["DateTime"] = pd.to_datetime(df.index)
    return df


ros_dfs = [prepare_df(s) for s in ros]
dimT = len(ros_dfs[0])


def normalize(y):
    x = y / np.linalg.norm(y)
    return x


cols = ["C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
fig = plt.figure(figsize=(8, 6))
ax_map = fig.add_subplot(1, 1, 1)
bm = Basemap(
    llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1],
    epsg=4326,
    ax=ax_map,
)
polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=1)
shape = XLAT.shape
ax_map.set_xticks(np.linspace(bm.lonmin, bm.lonmax, 10))
labels = [item.get_text() for item in ax_map.get_xticklabels()]
xlabels = np.arange(0, shape[1] * fs, shape[1] * fs / len(labels)).astype(int)
ax_map.set_xticklabels(xlabels, fontsize=11)

ax_map.set_yticks(np.linspace(bm.latmin, bm.latmax, 5))
labels = [item.get_text() for item in ax_map.get_yticklabels()]
ylabels = np.arange(0, shape[0] * fs, shape[0] * fs / len(labels)).astype(int)
ax_map.set_yticklabels(ylabels, fontsize=11)
ax_map.set_xlabel("West-East (m)", fontsize=12)
ax_map.set_ylabel("South-North (m)", fontsize=12)
# ax_map.xaxis.grid(True, which='major')
# ax_map.yaxis.grid(True, which='major')
ax_map.grid(True, linestyle="--", lw=0.2, zorder=1)


# ax_map.text(
# 0.015, 1.1, 'A)', size=20, color="k", weight='bold', zorder=10, transform=plt.gca().transAxes
# )

for h in hfxs:
    if h == list(hfxs)[-1]:
        ax_map.scatter(
            hfxs[h]["lon"],
            hfxs[h]["lat"],
            zorder=9,
            s=30,
            color="tab:orange",
            marker=".",
            label="Heatflux",
        )
    else:
        ax_map.scatter(
            hfxs[h]["lon"],
            hfxs[h]["lat"],
            zorder=9,
            s=30,
            color="tab:orange",
            marker=".",
        )

for r in ros:
    if r == list(ros)[-1]:
        ax_map.scatter(
            ros[r]["lon"],
            ros[r]["lat"],
            zorder=9,
            s=10,
            color="tab:green",
            marker=".",
            label="Thermocouples",
        )
    else:
        ax_map.scatter(
            ros[r]["lon"],
            ros[r]["lat"],
            zorder=9,
            s=10,
            color="tab:green",
            marker=".",
        )

for aq in aqs:
    if aq == list(aqs)[-1]:
        ax_map.scatter(
            aqs[aq]["lon"],
            aqs[aq]["lat"],
            zorder=10,
            # label=aq_ids[i],
            color="tab:red",
            edgecolors="black",
            marker="^",
            s=100,
            label="Air Quality Monitor",
        )
    else:
        ax_map.scatter(
            aqs[aq]["lon"],
            aqs[aq]["lat"],
            zorder=10,
            # label=aq_ids[i],
            color="tab:red",
            edgecolors="black",
            marker="^",
            s=100,
        )


ax_map.scatter(
    mets["south_met"]["lon"],
    mets["south_met"]["lat"],
    zorder=10,
    edgecolors="black",
    color="tab:blue",
    marker="D",
    s=80,
    label="Met Tower",
)
legend = ax_map.legend(
    loc="upper right",
    ncol=1,
    fancybox=True,
    shadow=True,
)
ax_map.set_title(f"Pelican Mountain Unit 5 Sensor Location", fontsize=16)

fig.tight_layout()

plt.savefig(str(save_dir) + f"/site-map.png", dpi=250, bbox_inches="tight")


# contour = ax_map.contourf(
#     XLONG, XLAT, FUELS['fuel'], zorder=1, cmap = 'Greens_r'
# )
