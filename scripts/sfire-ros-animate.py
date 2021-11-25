import context
import wrf
import json
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pylab as pylab

import dask
from datetime import datetime
from utils.sfire import compressor, sovle_pbl
from mpl_toolkits.basemap import Basemap
import pyproj as pyproj
import matplotlib.pyplot as plt
from matplotlib import animation
from context import root_dir, data_dir, save_dir
import matplotlib.pylab as pylab
import matplotlib.colors as colors
from matplotlib.pyplot import cm
from utils.sfire import makeLL
import matplotlib
import warnings

matplotlib.rcParams.update({"font.size": 10})
warnings.filterwarnings("ignore", category=RuntimeWarning)


domain = "fire"
unit = "unit5"
modelrun = "F6V51M08Z22I04B10"
configid = "F6V51"
title = "Time of Arrival"
var = "FGRNHFX"
ig_start = [55.7177497, -113.5713062]
ig_end = [55.7177507, -113.5751922]
levels = np.arange(0, 201, 1)
cmap = "Reds"
alpha = 0.8
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
fs = bounds["namelist"]["dxy"] / bounds["namelist"]["fs"]

wrf_ds = xr.open_dataset(
    str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:48", chunks="auto"
)
var_da = wrf_ds[var] / 1000
var_da = var_da.sel(
    south_north_subgrid=south_north_subgrid,
    west_east_subgrid=west_east_subgrid,
    Time=slice(0, 100),
)
times = var_da.XTIME.values

XLAT, XLONG = makeLL(domain, configid)
XLAT = XLAT.sel(
    south_north_subgrid=south_north_subgrid, west_east_subgrid=west_east_subgrid
)
XLONG = XLONG.sel(
    south_north_subgrid=south_north_subgrid, west_east_subgrid=west_east_subgrid
)
dimT = len(times)


cols = ["C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
bm = Basemap(
    llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1],
    epsg=4326,
    ax=ax,
)
polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=9)
da = var_da.isel(Time=0)
ax.set_title(f"{title} \n" + da.XTIME.values.astype(str)[:-10], fontsize=18)
contour = ax.contourf(
    XLONG, XLAT, da, zorder=2, cmap=cmap, levels=levels, extend="max", alpha=alpha
)
ax.set_xlabel("Longitude (degres)", fontsize=16)
ax.set_ylabel("Latitude (degres)", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=14)
cbar = plt.colorbar(contour, ax=ax, pad=0.05)
cbar.ax.tick_params(labelsize=12)


def plotstuff(i):
    col = cols[i]
    indices = [ii for ii, s in enumerate(ros_ids) if col in s]
    n = len(indices)
    colors_default = iter(cm.tab20(np.linspace(0, 1, n)))
    for ind in indices:
        c = next(colors_default)
        ax.scatter(
            ros[ros_ids[ind]]["lon"],
            ros[ros_ids[ind]]["lat"],
            zorder=9,
            s=100,
            color=c,
        )
    ax.set_title(f"{title}", fontsize=18)
    ax.annotate(
        col,
        xy=(ros[ros_ids[indices[0]]]["lon"], 55.71740),
        color="w",
        bbox=dict(boxstyle="circle", fc="black", ec="k", alpha=0.8),
        ha="center",
    )


[plotstuff(i) for i in range(len(cols))]


def update_plot(i):
    global contour
    for c in contour.collections:
        c.remove()
    print(i)
    da = var_da.isel(Time=i)
    ax.set_title(f"{title} \n" + da.XTIME.values.astype(str)[:-10], fontsize=18)
    contour = ax.contourf(
        XLONG,
        XLAT,
        da,
        zorder=2,
        cmap=cmap,
        levels=levels,
        extend="both",
        alpha=alpha,
    )
    return contour


fig.tight_layout()
ani = animation.FuncAnimation(fig, update_plot, dimT, interval=3)
ani.save(str(save_dir) + f"/{var}-topview.mp4", writer="ffmpeg", fps=10, dpi=250)
plt.close()
