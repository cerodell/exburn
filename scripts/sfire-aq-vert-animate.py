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
from utils.sfire import makeLL


##################### Define Inputs and File Directories ###################
modelrun = "F6V51M08Z22B10"
configid = "F6V51"
domain = "met"
# pm_ef = 21.05  # boreal wildfire emission factor Urbanski (2014)
pm_ef = 10.400
fireshape_path = str(data_dir) + "/unit_5/unit_5"
aqsin = str(data_dir) + "/obs/aq/"
aqsin = sorted(Path(aqsin).glob(f"*"))
save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)

var = "tr17_1"
title = "Tracer"
units = "Concentration g kg^-1"
cmap = "cubehelix_r"
levels = np.arange(0, 10000.0, 100)


with open(str(root_dir) + "/json/config.json") as f:
    config = json.load(f)
aqs = config["unit5"]["obs"]["aq"]
ros = config["unit5"]["obs"]["ros"]


wrf_ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:48")
wrf_ds = wrf_ds.sum("bottom_top")
XLAT, XLONG = makeLL(domain, configid)
times = wrf_ds.XTIME.values
dimT = len(times)


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
[ax.scatter(aqs[s]["lon"], aqs[s]["lat"], zorder=10) for s in aqs]
ds = wrf_ds.isel(Time=0)
ax.set_title(f"{title} \n" + ds.XTIME.values.astype(str)[:-10], fontsize=18)
contour = ax.contourf(
    XLONG, XLAT, ds[var], zorder=2, cmap=cmap, levels=levels, extend="max"
)

ax.set_xlabel("Longitude (degres)", fontsize=16)
ax.set_ylabel("Latitude (degres)", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=14)

cbar = plt.colorbar(contour, ax=ax, pad=0.05)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(units, rotation=270, fontsize=16, labelpad=15)


def update_plot(i):
    global contour, fire
    for c in contour.collections:
        c.remove()

    print(i)
    ds = wrf_ds.isel(Time=i)
    ax.set_title(f"{title} \n" + ds.XTIME.values.astype(str)[:-10], fontsize=18)
    contour = ax.contourf(
        XLONG,
        XLAT,
        ds[var],
        zorder=2,
        cmap=cmap,
        levels=levels,
        extend="both",
    )
    return contour


fig.tight_layout()
ani = animation.FuncAnimation(fig, update_plot, dimT, interval=3)
ani.save(str(save_dir) + f"/{var}-topview.mp4", writer="ffmpeg", fps=10, dpi=250)
plt.close()
