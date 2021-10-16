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

matplotlib.rcParams.update({"font.size": 10})
warnings.filterwarnings("ignore", category=RuntimeWarning)


domain = "fire"
unit = "unit5"
modelrun = "F6V51M08R24"
title = "Time of Arrival"
# units = "degrees C"
var = "FGRNHFX"
v = np.arange(0, 201, 1)
Cnorm = colors.Normalize(vmin=0, vmax=200)
ros_filein = str(data_dir) + "/obs/ros/"
fireshape_path = str(data_dir) + "/all_units/mygeodata_merged"
headers = ["day", "hour", "minute", "second", "temp"]


with open(str(root_dir) + "/json/config.json") as f:
    config = json.load(f)
ros = config["unit5"]["obs"]["ros"]
ros_ids = list(ros)
bounds = config["unit5"]["sfire"][modelrun]
south_north_subgrid = slice(bounds["fire"]["sn"][0], bounds["fire"]["sn"][1])
west_east_subgrid = slice(bounds["fire"]["we"][0], bounds["fire"]["we"][1])
fs = bounds["namelist"]["dxy"] / bounds["namelist"]["fs"]

# wrf_ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/interp-unit5-temp.nc", chunks = 'auto')
wrf_ds = xr.open_dataset(
    str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11", chunks="auto"
)
var_da = wrf_ds[var]
var_da = var_da.sel(
    south_north_subgrid=south_north_subgrid,
    west_east_subgrid=west_east_subgrid,
    Time=slice(0, 50),
)
times = var_da.XTIME.values

XLAT, XLONG = makeLL(domain, modelrun)
XLAT = XLAT.sel(
    south_north_subgrid=south_north_subgrid, west_east_subgrid=west_east_subgrid
)
XLONG = XLONG.sel(
    south_north_subgrid=south_north_subgrid, west_east_subgrid=west_east_subgrid
)
## create dataframe with columns of all XLAT/XLONG
locs = pd.DataFrame({"XLAT": XLAT.values.ravel(), "XLONG": XLONG.values.ravel()})
## build kdtree
tree = KDTree(locs)
print("Domain KDTree built")


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
    df = df.rename(columns={"temp": f"temp-{rosin}"})
    # df["DateTime"] = pd.to_datetime(df.index)
    return df


ros_dfs = [prepare_df(s) for s in ros]
ros_dfs = pd.concat(ros_dfs, axis=1)

dimT = len(ros_dfs)


def find_index(s):
    q = np.array([ros[s]["lat"], ros[s]["lon"]]).reshape(1, -1)
    dist, ind = tree.query(q, k=1)
    loc = list(np.unravel_index(int(ind), XLAT.shape))
    return loc


ros_locs = np.stack([find_index(s) for s in ros_ids])
y = xr.DataArray(np.array(ros_locs[:, 0]), dims="ros", coords=dict(ros=ros_ids))
x = xr.DataArray(np.array(ros_locs[:, 1]), dims="ros", coords=dict(ros=ros_ids))
if domain == "fire":
    ros_da = var_da.sel(south_north_subgrid=y, west_east_subgrid=x)
elif domain == "met":
    ros_da = var_da.sel(south_north=y, west_east=x)
else:
    raise ValueError("Not a valied domain option")


def normalize(y):
    x = y / np.linalg.norm(y)
    return x


cols = ["C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
fig = plt.figure(figsize=(14, 8))
ax_map = fig.add_subplot(2, 2, 1)
bm = Basemap(
    llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1],
    epsg=4326,
    ax=ax_map,
)
polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=1)
ax_map.scatter(
    -113.571244, 55.71776, c="red", s=80, marker="*", label="ignition start", zorder=10
)
ax_map.scatter(
    -113.575172, 55.7177788, c="red", marker="H", s=60, label="ignition end", zorder=10
)
shape = XLAT.shape
ax_map.set_xticks(np.linspace(bm.lonmin, bm.lonmax, 15))
labels = [item.get_text() for item in ax_map.get_xticklabels()]
xlabels = np.arange(0, shape[1] * fs, shape[1] * fs / len(labels)).astype(int)
ax_map.set_xticklabels(xlabels)

ax_map.set_yticks(np.linspace(bm.latmin, bm.latmax, 10))
labels = [item.get_text() for item in ax_map.get_yticklabels()]
ylabels = np.arange(0, shape[0] * fs, shape[0] * fs / len(labels)).astype(int)
ax_map.set_yticklabels(ylabels)
ax_map.set_xlabel("meters", fontsize=14)
ax_map.set_ylabel("meters", fontsize=14)


def plotstuff(i):
    col = cols[i]
    # print(col)
    ax = fig.add_subplot(9, 2, (i + 1) * 2)
    indices = [ii for ii, s in enumerate(ros_ids) if col in s]
    n = len(indices)
    colors_default = iter(cm.tab20(np.linspace(0, 1, n)))
    for i in indices:
        # print(ros_ids[i])
        i = i
        c = next(colors_default)
        modeld_ros = ros_da.isel(ros=i)
        ax.plot(
            modeld_ros.XTIME, normalize(modeld_ros), color=c, label=ros_ids[i], zorder=8
        )
        ax.plot(
            ros_dfs[i].index.values,
            normalize(ros_dfs[i].temp.values / 4),
            color=c,
            linestyle="--",
            zorder=9,
        )
    ax.text(
        0.02, 0.65, col, size=14, color="k", zorder=10, transform=plt.gca().transAxes
    )
    ax.scatter(
        pd.Timestamp(2019, 5, 11, 17, 49, 3),
        0.03,
        marker="*",
        color="red",
        zorder=10,
        label="Ignition Time",
        s=100,
    )
    if col == "C9":
        myFmt = DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(myFmt)
        ax.tick_params(axis="x", labelrotation=45)
    else:
        ax.set_xticks([])
    colors_default = iter(cm.tab20(np.linspace(0, 1, n)))
    for ind in indices:
        c = next(colors_default)
        ax_map.scatter(
            ros[ros_ids[ind]]["lon"],
            ros[ros_ids[ind]]["lat"],
            zorder=9,
            s=100,
            color=c,
        )
    ax_map.set_title(f"{title}", fontsize=18)
    ax_map.annotate(
        col,
        xy=(ros[ros_ids[indices[0]]]["lon"], 55.71745),
        color="w",
        bbox=dict(boxstyle="circle", fc="black", ec="k", alpha=0.8),
        ha="center",
    )


[plotstuff(i) for i in range(len(cols))]
plt.savefig(str(save_dir) + f"/{modelrun}/ros-timeseries.png", dpi=300)


# fig = plt.figure(figsize=(6, 4))
# ax = fig.add_subplot(1, 1, 1)
# bm = Basemap(
#     llcrnrlon=XLONG[0, 0],
#     llcrnrlat=XLAT[0, 0],
#     urcrnrlon=XLONG[-1, -1],
#     urcrnrlat=XLAT[-1, -1],
#     epsg=4326,
#     ax=ax,
# )
# lats = np.array([ros[ros_ids[s]]["lat"] for s in range(len(ros_ids))])
# lons = np.array([ros[ros_ids[s]]["lon"] for s in range(len(ros_ids))])
# temps = np.array([ros_dfs[s].temp.iloc[0]/4 for s in range(len(ros_ids))])

# # generate grid data
# numcols, numrows = 240, 240
# xi = np.linspace(lons.min(), lons.max(), numcols)
# yi = np.linspace(lats.min(), lats.max(), numrows)
# xi, yi = np.meshgrid(xi, yi)

# zi = griddata((lons,lats),temps,(xi,yi),method='linear')
# polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=1)
# contour = ax.contourf(xi,yi,zi, norm = Cnorm,cmap ='YlOrRd', levels= v, extend = 'both')
# # scatter = ax.scatter(lons, lats, c= temps, zorder=10, s =40, vmin=0, vmax=200)
# cbar = plt.colorbar(contour, ax=ax, pad=0.05)
# cbar.ax.tick_params(labelsize=12)
# cbar.set_label(units, rotation=270, fontsize=16, labelpad=15)
# ax.set_title(f"{title} \n" + ros_dfs[0]['DateTime'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'), fontsize=18)

# def update_plot(i):
#     global contour
#     for c in contour.collections:
#         c.remove()
#     temps = [ros_dfs[s].temp.iloc[i]/4 for s in range(len(ros_ids))]
#     zi = griddata((lons,lats),temps,(xi,yi),method='linear')
#     contour = ax.contourf(xi,yi,zi, norm = Cnorm,cmap ='YlOrRd', levels= v, extend = 'both')
#     # scatter = ax.scatter(lons, lats, c= temps, zorder=10, s =40, vmin=0, vmax=200)
#     ax.set_title(f"{title} \n" + ros_dfs[0]['DateTime'].iloc[i].strftime('%Y-%m-%d %H:%M:%S'), fontsize=18)
#     return contour

# fig.tight_layout()
# ani = animation.FuncAnimation(fig, update_plot, dimT, interval=3)
# ani.save(str(save_dir) + f"/{modelrun}/{var}-ros.mp4", writer="ffmpeg", fps=10, dpi=250)
# plt.close()
