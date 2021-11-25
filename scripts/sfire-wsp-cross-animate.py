import context
import wrf
import json
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pylab as pylab
from netCDF4 import Dataset

import dask
from datetime import datetime
from utils.sfire import compressor, sovle_pbl
from mpl_toolkits.basemap import Basemap
import pyproj as pyproj
import matplotlib.pyplot as plt
from matplotlib import animation
from context import root_dir, data_dir, save_dir
import matplotlib.pylab as pylab


pm_ef = 21.05  # boreal wildfire emission factor Urbanski (2014)
modelrun = "F6V81"  #'F6V41' 'F6V81'
# if modelrun ==
ds = 25
# var = "tr17_1"
# title = "Tracer"
# units = "Concentration g kg^-1"
# cmap = "cubehelix_r"
# levels = np.arange(0, 10000, 10)

var = "temp"
title = "Temperature"
units = "K"
cmap = "jet"

fireshape_path = str(data_dir) + "/all_units/mygeodata_merged"
save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)

# var_ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/interp-unit5-{var}.nc")
# var_ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
ncfile = Dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
# var_ds = wrf.getvar(ncfile, 'theta_e', wrf.ALL_TIMES)
height = wrf.getvar(ncfile, "height")
height = np.round(height.values[:, 0, 0]).astype(int)

temp = wrf.getvar(ncfile, "temp", wrf.ALL_TIMES)
var_ds = temp.to_dataset()
var_ds["XTIME"] = var_ds["Time"]

levels = np.arange(float(temp.min()), float(temp.max()), 1)

# interp_level = var_ds.interp_level * 1000
interp_level = height
south_north = var_ds.south_north * ds
var_ds = var_ds.isel(west_east=71)
var_ds = var_ds.isel(Time=slice(0, 200))
# var_ds = var_ds.sum(dim = 'west_east')

dimT = len(var_ds.Time)


fig = plt.figure(figsize=(14, 4))
ax = fig.add_subplot(1, 1, 1)  # top and bottom left
ds = var_ds.isel(Time=0)
# ax.set_title(f"{title} \n" + ds.Time.values.astype(str)[:-10], fontsize=18)
ax.set_title(f"{title} \n" + ds.XTIME.values.astype(str)[:-10], fontsize=18)
contour = ax.contourf(
    south_north,
    interp_level,
    ds[var],
    zorder=1,
    cmap=cmap,
    levels=levels,
    extend="both",
)
ax.set_ylabel("Vertical (m)", fontsize=16)
ax.set_xlabel("Horizontal (m)", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=14)

cbar = plt.colorbar(contour, ax=ax, pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(units, rotation=270, fontsize=16, labelpad=15)


def update_plot(i):
    global contour
    for c in contour.collections:
        c.remove()
    print(i)
    ds = var_ds.isel(Time=i)
    # ax.set_title(f"{title} \n" + ds.Time.values.astype(str)[:-10], fontsize=18)
    ax.set_title(f"{title} \n" + ds.XTIME.values.astype(str)[:-10], fontsize=18)
    contour = ax.contourf(
        south_north,
        interp_level,
        ds[var],
        zorder=1,
        cmap=cmap,
        levels=levels,
        extend="both",
    )
    return contour


fig.tight_layout()
ani = animation.FuncAnimation(fig, update_plot, dimT, interval=3)
ani.save(str(save_dir) + f"/{var}-cross.mp4", writer="ffmpeg", fps=10, dpi=250)
plt.close()
