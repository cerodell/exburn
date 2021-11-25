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
bounds = config["unit5"]["sfire"][configid]
south_north_subgrid = slice(bounds["fire"]["sn"][0], bounds["fire"]["sn"][1])
west_east_subgrid = slice(bounds["fire"]["we"][0], bounds["fire"]["we"][1])
south_north_subgrid = slice(555, 610, None)
west_east_subgrid = slice(335, 400, None)
fs = bounds["namelist"]["dxy"] / bounds["namelist"]["fs"]

wrf_ds = xr.open_dataset(
    str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11", chunks="auto"
)
var_da = wrf_ds[var]
var_da = var_da.sel(
    south_north_subgrid=south_north_subgrid,
    west_east_subgrid=west_east_subgrid,
    Time=slice(3, 50),
)
times = var_da.XTIME.values

XLAT, XLONG = makeLL(domain, configid)
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
    df["DateTime"] = pd.to_datetime(df.index)
    return df


ros_dfs = [prepare_df(s) for s in ros]
dimT = len(ros_dfs[0])


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
# fig.suptitle(modelrun)
ax_map = fig.add_subplot(1, 2, 1)
# ax_map = fig.add_subplot(2, 1, 1)

bm = Basemap(
    llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1],
    epsg=4326,
    ax=ax_map,
)
time_of_arrival = var_da.argmax(dim="Time").values * 10

contour = ax_map.contourf(
    XLONG, XLAT, time_of_arrival + 8, levels=levels_ros, cmap=cmap_ros
)
cbar = plt.colorbar(contour, ax=ax_map, pad=0.004, location="bottom")
cbar.ax.tick_params(labelsize=12)
time_tile = times[0].astype(str)[:-11] + "8"
# cbar.set_label(f"Seconds from First Ignition {time_tile}", fontsize=13, labelpad=15)
cbar.set_label(f"Seconds from Ignition {time_tile}", fontsize=13, labelpad=15)

polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=1)


def ignite(ig_start, ig_end, line, color):
    ax_map.scatter(
        ig_start[1],
        ig_start[0],
        c=color,
        s=200,
        marker="*",
        # alpha=0.6,
        label=f"{line} ignition start",
        zorder=10,
        edgecolors="black",
    )
    ax_map.scatter(
        ig_end[1],
        ig_end[0],
        c=color,
        marker="X",
        s=150,
        # alpha=0.6,
        label=f"{line} ignition end",
        zorder=10,
        edgecolors="black",
    )
    ax_map.plot(
        [ig_start[1], ig_end[1]],
        [ig_start[0], ig_end[0]],
        linestyle="--",
        lw=2,
        marker="",
        zorder=9,
        color="k",
        # alpha=0.6,
    )


if ("I04" in modelrun) == True:
    print("Multi line ignition")
    ignite(
        ig_start=[55.7177529, -113.5713107],
        ig_end=[55.71773480, -113.57183453],
        line="1",
        color="tab:red",
    )
    ignite(
        ig_start=[55.7177109, -113.5721005],
        ig_end=[55.7177124, -113.5725656],
        line="2",
        color="tab:blue",
    )
    ignite(
        ig_start=[55.7177293, -113.5734885],
        ig_end=[55.7177437, -113.5744894],
        line="3",
        color="tab:green",
    )
    ignite(
        ig_start=[55.7177775603, -113.5747705233],
        ig_end=[55.717752429, -113.575192125],
        line="4",
        color="tab:grey",
    )

    # ignite(ig_start= [55.7177529, -113.5713107], ig_end = [55.71773480, -113.57183453], line = '1', color = 'red')
    # ignite(ig_start= [55.71774808, -113.57232778], ig_end = [55.71771973, -113.57299677], line = '2', color = 'blue')
    # ignite(ig_start= [55.71771900, -113.57341997], ig_end =[55.7177473680, -113.5742683254], line = '3', color = 'green')
    # ignite(ig_start= [55.7177775603, -113.5747705233], ig_end = [55.717752429, -113.575192125], line = '4', color = 'black')
else:
    print("Single line ignition")
    ignite(ig_start, ig_end, line="1", color="tab:red")


shape = XLAT.shape
ax_map.set_xticks(np.linspace(bm.lonmin, bm.lonmax, 12))
labels = [item.get_text() for item in ax_map.get_xticklabels()]
xlabels = np.arange(0, shape[1] * fs, shape[1] * fs / len(labels)).astype(int)
ax_map.set_xticklabels(xlabels, fontsize=11)

ax_map.set_yticks(np.linspace(bm.latmin, bm.latmax, 6))
labels = [item.get_text() for item in ax_map.get_yticklabels()]
ylabels = np.arange(0, shape[0] * fs, shape[0] * fs / len(labels)).astype(int)
ax_map.set_yticklabels(ylabels, fontsize=11)
ax_map.set_xlabel("West-East (m)", fontsize=12)
ax_map.set_ylabel("South-North (m)", fontsize=12)
ax_map.text(
    0.015,
    1.1,
    "A)",
    size=20,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)


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
    ax.set_ylim(0, 0.49)
    if ("I" in modelrun) == True:
        if col == "C2" or col == "C3":
            print(col)
            ax.scatter(
                pd.Timestamp(2019, 5, 11, 17, 49, 48),
                0.06,
                marker="*",
                # alpha=0.6,
                color="tab:red",
                zorder=10,
                label="Ignition Start Time",
                edgecolors="black",
                s=120,
            )
            ax.scatter(
                pd.Timestamp(2019, 5, 11, 17, 49, 56),
                0.06,
                marker="X",
                # alpha=0.6,
                color="tab:red",
                zorder=10,
                label="Ignition End Time",
                edgecolors="black",
                s=100,
            )
        elif col == "C4" or col == "C5":
            ax.scatter(
                pd.Timestamp(2019, 5, 11, 17, 50, 7),
                0.06,
                marker="*",
                # alpha=0.6,
                color="tab:blue",
                zorder=10,
                label="Ignition Start Time",
                edgecolors="black",
                s=120,
            )
            ax.scatter(
                pd.Timestamp(2019, 5, 11, 17, 50, 17),
                0.06,
                marker="X",
                # alpha=0.6,
                color="tab:blue",
                zorder=10,
                label="Ignition End Time",
                edgecolors="black",
                s=100,
            )

        elif col == "C6" or col == "C7":
            ax.scatter(
                pd.Timestamp(2019, 5, 11, 17, 50, 42),
                0.06,
                marker="*",
                # alpha=0.6,
                color="tab:green",
                zorder=10,
                label="Ignition Start Time",
                edgecolors="black",
                s=120,
            )
            ax.scatter(
                pd.Timestamp(2019, 5, 11, 17, 50, 57),
                0.06,
                marker="X",
                # alpha=0.6,
                color="tab:green",
                zorder=10,
                label="Ignition End Time",
                edgecolors="black",
                s=100,
            )
        elif col == "C8" or col == "C9":
            ax.scatter(
                pd.Timestamp(2019, 5, 11, 17, 52, 49),
                0.06,
                marker="*",
                # alpha=0.6,
                color="tab:grey",
                zorder=10,
                label="Ignition Start Time",
                edgecolors="black",
                s=120,
            )
            ax.scatter(
                pd.Timestamp(2019, 5, 11, 17, 52, 55),
                0.06,
                marker="X",
                # alpha=0.6,
                color="tab:grey",
                zorder=10,
                label="Ignition End Time",
                edgecolors="black",
                s=100,
            )
        else:
            pass

    else:
        ax.scatter(
            pd.Timestamp(2019, 5, 11, 17, 49, 48),
            0.06,
            marker="*",
            # alpha=0.6,
            color="tab:red",
            zorder=10,
            label="Ignition Start Time",
            edgecolors="black",
            s=120,
        )
        ax.scatter(
            pd.Timestamp(2019, 5, 11, 17, 51, 28),
            0.06,
            marker="X",
            # alpha=0.6,
            color="tab:red",
            zorder=10,
            label="Ignition End Time",
            edgecolors="black",
            s=100,
        )
    if col == "C9":
        myFmt = DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(myFmt)
        ax.tick_params(axis="x", labelrotation=20)
        ax.set_xlabel("Datetime (HH:MM:SS)")
    elif col == "C2":
        ax.set_title(
            f"Time Series of normalized \n observed temperature (dashed lines) and  \n normalized modeled heatflux (solid lines)",
            fontsize=12,
        )
        ax.set_xticks([])
        ax.text(
            0.02,
            1.2,
            "B)",
            size=20,
            color="k",
            weight="bold",
            zorder=10,
            transform=plt.gca().transAxes,
        )
    else:
        ax.set_xticks([])

    ax.tick_params(axis="both", which="major", labelsize=11)
    colors_default = iter(cm.tab20(np.linspace(0, 1, n)))
    for ind in indices:
        c = next(colors_default)
        ax_map.scatter(
            ros[ros_ids[ind]]["lon"],
            ros[ros_ids[ind]]["lat"],
            zorder=9,
            s=100,
            color=c,
            edgecolors="black",
        )
    ax_map.set_title(f"{title}", fontsize=18)
    ax_map.annotate(
        col,
        xy=(ros[ros_ids[indices[0]]]["lon"], 55.71740),
        color="w",
        bbox=dict(boxstyle="circle", fc="black", ec="k", alpha=0.8),
        ha="center",
    )


[plotstuff(i) for i in range(len(cols))]

# fig.tight_layout()
plt.savefig(str(save_dir) + f"/ros-timeseries.png", dpi=300)


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
# cbar = plt.colorbar(contour, ax=ax, pad=0.06)
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
