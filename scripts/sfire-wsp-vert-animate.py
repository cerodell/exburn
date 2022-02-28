import context
import wrf
import json
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from netCDF4 import Dataset
import matplotlib.pylab as pylab

import dask
from datetime import datetime
from utils.sfire import compressor, sovle_pbl
from mpl_toolkits.basemap import Basemap
import pyproj as pyproj
import matplotlib.pyplot as plt
from matplotlib import animation
from context import root_dir, data_dir, save_dir
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pylab as pylab
import matplotlib.colors as colors
from utils.sfire import makeLL


##################### Define Inputs and File Directories ###################
modelrun = "F6V51M08Z22"
configid = "F6V51"
domain = "met"
fireshape_path = str(data_dir) + "/unit_5/unit_5"
aqsin = str(data_dir) + "/obs/aq/"
aqsin = sorted(Path(aqsin).glob(f"*"))
save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)

var = "wsp"
title = "Wind Speed 10m"
units = "m s^-1"
levels = np.arange(0, 10.0, 0.1)
colors = [
    "#FFFFFF",
    "#BBBBBB",
    "#646464",
    "#1563D3",
    "#2883F1",
    "#50A5F5",
    "#97D3FB",
    "#0CA10D",
    "#37D33C",
    "#97F58D",
    "#B5FBAB",
    "#FFE978",
    "#FFC03D",
    "#FFA100",
    "#FF3300",
    "#C10000",
    "#960007",
    "#643C32",
]
cmap = LinearSegmentedColormap.from_list("meteoblue", colors, N=len(levels))

# with open(str(root_dir) + "/json/config.json") as f:
#     config = json.load(f)
# aqs = config["unit5"]["obs"]["aq"]
# ros = config["unit5"]["obs"]["ros"]
south_north = slice(100, 140, None)
west_east = slice(60, 89, None)
time_slice = slice(0, 100)
# south_north = slice(50, 300, None)
# west_east = slice(10, 140, None)


## get heights from wrf-simulation
ncfile = Dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
wspd_wdir = wrf.getvar(ncfile, "uvmet10_wspd_wdir", wrf.ALL_TIMES)
uvmet10 = wrf.getvar(ncfile, "uvmet10", wrf.ALL_TIMES)
u10 = uvmet10.sel(u_v="u").isel(
    south_north=south_north, west_east=west_east, Time=time_slice
)
v10 = uvmet10.sel(u_v="v").isel(
    south_north=south_north, west_east=west_east, Time=time_slice
)
wdir = wspd_wdir.sel(wspd_wdir="wdir").isel(
    south_north=south_north, west_east=west_east, Time=time_slice
)
wsp = wspd_wdir.sel(wspd_wdir="wspd").isel(
    south_north=south_north, west_east=west_east, Time=time_slice
)


XLAT, XLONG = makeLL(domain, configid)
XLAT = XLAT.sel(south_north=south_north, west_east=west_east).values
XLONG = XLONG.sel(south_north=south_north, west_east=west_east).values
times = wsp.XTIME.values
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
# [ax.scatter(aqs[s]["lon"], aqs[s]["lat"], zorder=10) for s in aqs]
wsp_i = wsp.isel(Time=0)
u10_i = u10.isel(Time=0)
v10_i = v10.isel(Time=0)

ax.set_title(f"{title} \n" + wsp_i.Time.values.astype(str)[:-10], fontsize=18)
contour = ax.contourf(
    XLONG, XLAT, wsp_i, zorder=2, cmap=cmap, levels=levels, extend="max"
)
wind = ax.quiver(XLONG, XLAT, u10_i, v10_i, zorder=10, scale=120)
# wind = ax.quiver(XLONG[::4,::4], XLAT[::4,::4], u10_i[::4,::4], v10_i[::4,::4], zorder = 10, scale=120)
ax.set_xlabel("Longitude (degres)", fontsize=16)
ax.set_ylabel("Latitude (degres)", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=14)

cbar = plt.colorbar(contour, ax=ax, pad=0.05)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(units, rotation=270, fontsize=16, labelpad=15)


def update_plot(i):
    global contour, wind
    for c in contour.collections:
        c.remove()
    print(i)
    wsp_i = wsp.isel(Time=i)
    u10_i = u10.isel(Time=i)
    v10_i = v10.isel(Time=i)
    ax.set_title(f"{title} \n" + wsp_i.Time.values.astype(str)[:-10], fontsize=18)
    # wind.set_UVC(u10_i[::4,::4],v10_i[::4,::4])
    wind.set_UVC(u10_i, v10_i)

    contour = ax.contourf(
        XLONG,
        XLAT,
        wsp_i,
        zorder=2,
        cmap=cmap,
        levels=levels,
        extend="both",
    )
    return contour


# fig.tight_layout()
ani = animation.FuncAnimation(fig, update_plot, dimT, interval=3)
ani.save(str(save_dir) + f"/{var}-topview.mp4", writer="ffmpeg", fps=6, dpi=250)
plt.close()
