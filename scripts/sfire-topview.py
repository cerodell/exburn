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
from utils.sfire import makeLL


pm_ef = 21.05  # boreal wildfire emission factor Urbanski (2014)
fueltype = 6
var = "tr17_1"
title = "Tracer"
units = "Concentration g kg^-1"
cmap = "cubehelix_r"
levels = np.arange(0, 48600.0, 100)
fire_XLAT, fire_XLONG = makeLL("fire")
alpha = 0.7
TimeSlice = slice(0, 100)
fireshape_path = str(data_dir) + "/all_units/mygeodata_merged"
save_dir = Path(str(save_dir) + f"/fuel{fueltype}/")
save_dir.mkdir(parents=True, exist_ok=True)


with open(str(data_dir) + "/json/config.json") as f:
    config = json.load(f)
aqs = config["unit5"]["obs"]["aq"]

wrf_ds = xr.open_dataset(
    str(data_dir) + f"/fuel{fueltype}/wrfout_d01_2019-05-11_17:49:11", chunks="auto"
)
fire_XLAT = fire_XLAT.isel(
    south_north_subgrid=slice(550, 620), west_east_subgrid=slice(330, 405)
)
fire_XLONG = fire_XLONG.isel(
    south_north_subgrid=slice(550, 620), west_east_subgrid=slice(330, 405)
)
wrf_ds = wrf_ds.isel(
    south_north_subgrid=slice(550, 620),
    west_east_subgrid=slice(330, 405),
    south_north=slice(110, 180),
    west_east=slice(40, 100),
    Time=TimeSlice,
)

var_ds = xr.open_dataset(str(data_dir) + f"/fuel{fueltype}/interp-unit5-{var}.nc")
var_ds = var_ds.isel(interp_level=0)
var_ds = var_ds.isel(
    south_north=slice(110, 180), west_east=slice(40, 100), Time=TimeSlice
)

XLAT, XLONG = var_ds.XLAT.values, var_ds.XLONG.values
dimT = len(var_ds.Time)


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)  # top and bottom left
bm = Basemap(
    llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1],
    epsg=4326,
    ax=ax,
)
polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=9)
ax.scatter(aqs["303-100"]["lon"], aqs["303-100"]["lat"], zorder=10)
ax.scatter(aqs["303-200"]["lon"], aqs["303-200"]["lat"], zorder=10)
ax.scatter(aqs["303-300"]["lon"], aqs["303-300"]["lat"], zorder=10)
ax.scatter(aqs["401-100"]["lon"], aqs["401-100"]["lat"], zorder=10)
ax.scatter(aqs["401-200"]["lon"], aqs["401-200"]["lat"], zorder=10)
ds = var_ds.isel(Time=0)
ax.set_title(f"{title} \n" + ds.Time.values.astype(str)[:-10], fontsize=18)
contour = ax.contourf(
    XLONG, XLAT, ds[var], zorder=2, cmap=cmap, levels=levels, extend="max", alpha=alpha
)
fire = ax.contourf(
    fire_XLONG,
    fire_XLAT,
    wrf_ds["FGRNHFX"].isel(Time=0),
    zorder=1,
    cmap="Reds",
    levels=np.arange(100, 70000.0, 100),
    extend="max",
    alpha=alpha,
)
ax.set_xlabel("Longitude (degres)", fontsize=16)
ax.set_ylabel("Latitude (degres)", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=14)

cbar = plt.colorbar(contour, ax=ax, pad=0.05)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(units, rotation=270, fontsize=16, labelpad=15)

cbar_fire = plt.colorbar(fire, ax=ax, pad=0.01)
cbar_fire.ax.tick_params(labelsize=12)
cbar_fire.set_label("W m^-2", rotation=270, fontsize=16, labelpad=15)


def update_plot(i):
    global contour, fire
    for c in contour.collections:
        c.remove()
    for c in fire.collections:
        c.remove()
    print(i)
    ds = var_ds.isel(Time=i)
    ax.set_title(f"{title} \n" + ds.Time.values.astype(str)[:-10], fontsize=18)
    contour = ax.contourf(
        XLONG,
        XLAT,
        ds[var],
        zorder=2,
        cmap=cmap,
        levels=levels,
        extend="both",
        alpha=alpha,
    )
    fire = ax.contourf(
        fire_XLONG,
        fire_XLAT,
        wrf_ds["FGRNHFX"].isel(Time=i),
        zorder=1,
        cmap="Reds",
        levels=np.arange(100, 70000.0, 100),
        extend="max",
        alpha=alpha,
    )
    return contour, fire


fig.tight_layout()
ani = animation.FuncAnimation(fig, update_plot, dimT, interval=3)
ani.save(str(save_dir) + f"/{var}-topview.mp4", writer="ffmpeg", fps=10, dpi=250)
plt.close()
