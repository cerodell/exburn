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
from utils.sfire import makeLL

import dask
from datetime import datetime
from utils.sfire import compressor, sovle_pbl
from mpl_toolkits.basemap import Basemap
import pyproj as pyproj
import matplotlib.pyplot as plt
from matplotlib import animation
from context import root_dir, data_dir, save_dir
import matplotlib.pylab as pylab
import matplotlib as mpl


sn_m = 2000
ew_m = 1200
pm_ef = 10.400
modelrun = "F6V51M08Z22"  #'F6V41' 'F6V81'
configid = "F6V51"
domain = "met"
ds = 25
# levels = np.arange(0, 2000.0, 10)
a = np.arange(1, 10)
b = 10 ** np.arange(4)
levels = (b[:, np.newaxis] * a).flatten()
# levels = levels[:-6]
cmap = mpl.cm.cubehelix_r
norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend="both")

var = "tr17_1"
units = "PM2.5 " + r"($μg m^{-3}$)"
title = "Estimated Ballon Launch Location"
cmap = "cubehelix_r"


fireshape_path = str(data_dir) + "/all_units/mygeodata_merged"
save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)
south_north = slice(110, 280, None)
west_east = slice(40, 125, None)
bottom_top = slice(0, 46, None)
wrf_ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
ncfile = Dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
height = wrf.getvar(ncfile, "height")
height = height.isel(south_north=132, west_east=74, bottom_top=bottom_top)

XLAT, XLONG = makeLL(domain, configid)
XLAT = XLAT.sel(south_north=south_north, west_east=west_east)
XLONG = XLONG.sel(south_north=south_north, west_east=west_east)

hfx = wrf_ds["GRNQFX"]
hfx = hfx.sel(
    south_north=south_north,
    # west_east=west_east,
    Time=slice(0, 100),
).sum(dim="west_east")

wrf_ds = wrf_ds.sel(
    south_north=south_north,
    west_east=west_east,
    bottom_top=bottom_top,
    Time=slice(0, 100),
)
vert = wrf_ds[var].sum(dim="west_east").to_dataset()
hort = wrf_ds[var].sum(dim="bottom_top").to_dataset()

sn = wrf_ds.south_north * 25
ew = wrf_ds.west_east * 25
dimT = len(wrf_ds.XTIME)


sn_idx = int(np.where(sn == sn_m)[0])
ew_idx = int(np.where(ew == ew_m)[0])


fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 4)

ax = fig.add_subplot(gs[0, 0])
bm = Basemap(
    llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1],
    epsg=4326,
    ax=ax,
)
## add unit boundary and aq station location to map
polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=10)

# ds = hort.mean(dim = 'Time')
ds = hort.isel(Time=hfx.sum(dim="south_north").argmax(dim="Time") + 40)

contour = ax.contourf(
    XLONG,
    XLAT,
    ds[var] / pm_ef,
    zorder=1,
    levels=levels,
    cmap=cmap,
    norm=norm,
    extend="max",  # cubehelix_r
)

long = float(XLONG.isel(south_north=sn_idx, west_east=ew_idx))
lat = float(XLAT.isel(south_north=sn_idx, west_east=ew_idx))
ax.scatter(
    long,
    lat,
    c="tab:grey",
    s=250,
    marker="*",
    # alpha=0.6,
    label=f"Launch Location",
    zorder=10,
    edgecolors="black",
)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.14),
    ncol=4,
    fancybox=True,
    shadow=True,
    # prop={'size': 7}
)
shape = XLAT.shape
dxy = 25
ax.set_xticks(np.linspace(bm.lonmin, bm.lonmax, 5))
labels = [item.get_text() for item in ax.get_xticklabels()]
xlabels = np.arange(0, shape[1] * dxy, shape[1] * dxy / len(labels)).astype(int)
ax.set_xticklabels(xlabels)

ax.set_yticks(np.linspace(bm.latmin, bm.latmax, 8))
labels = [item.get_text() for item in ax.get_yticklabels()]
ylabels = np.arange(0, shape[0] * dxy, shape[0] * dxy / len(labels)).astype(int)
ax.set_yticklabels(ylabels)
# ax.yaxis.tick_right()
# ax.yaxis.set_label_position("right")
ax.set_xlabel("West-East (m)", fontsize=10)
ax.set_ylabel("South-North (m)", fontsize=10)
ax.text(
    0.005,
    1.05,
    "A)",
    size=14,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)

ax = fig.add_subplot(gs[0, 1:])
ds = vert.isel(Time=hfx.sum(dim="south_north").argmax(dim="Time") + 40)
# ax.set_title(f"{title} \n" + ds.XTIME.values.astype(str)[:-10], fontsize=18)
contour = ax.contourf(
    sn,
    height,
    ds[var] / pm_ef,
    zorder=1,
    cmap=cmap,
    levels=levels,
    extend="both",
    norm=norm,
)
ax.axvline(sn[sn_idx], color="k", linestyle="dotted", lw=0.8)
ax.set_ylabel("Vertical (m)", fontsize=14, labelpad=10)
ax.set_xlabel("South-North (m)", fontsize=14, labelpad=10)
ax.tick_params(axis="both", which="major", labelsize=12)
# hfx_i = hfx.isel(Time = hfx.sum(dim= 'south_north').argmax(dim = 'Time')+40)
# ax_hx = ax.twinx()
# ax_hx.plot(sn, hfx_i/1000, color = 'tab:red', lw = 0.8, zorder =10)
# ax_hx.set_ylabel("Heatflux " + r"$(kW m^{-2})$" + "\n", fontsize=14, color = 'tab:red', rotation=-90, labelpad=30)
# ax_hx.tick_params(axis='y', colors='tab:red')
# ax_hx.set_ylim(0,400)
# ax_hx.tick_params(axis="both", which="major", labelsize=12)

# cbar = plt.colorbar(contour, ax=ax_hx, pad=0.1)
cbar = plt.colorbar(contour, ax=ax, pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(units, rotation=270, fontsize=14, labelpad=20)
ax.text(
    0.005,
    1.05,
    "B)",
    size=14,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)


time_idx = int(hfx.sum(dim="south_north").argmax(dim="Time") + 40)


def index_var(var, units):
    array = wrf.getvar(ncfile, var, timeidx=time_idx, units=units)
    array = array.isel(
        south_north=south_north, west_east=west_east, bottom_top=bottom_top
    )
    array = array.isel(south_north=sn_idx, west_east=ew_idx)
    return array


ax = fig.add_subplot(gs[1, 0])
temp = index_var("temp", units="C")
# ax.set_yticklabels([])
ax.plot(temp, height, color="red", linewidth=4)
ax.set_ylabel("Height (m)", fontsize=14)
ax.set_xlabel("Temperature (C)", fontsize=14)
ax.xaxis.grid(color="gray", linestyle="dashed")
ax.yaxis.grid(color="gray", linestyle="dashed")
ax.text(
    0.005,
    1.05,
    "C)",
    size=14,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)

ax = fig.add_subplot(gs[1, 1])
td = index_var("theta_e", units="C")
ax.set_yticklabels([])
ax.plot(td, height, color="purple", linewidth=4)
ax.set_xlabel("Potential Temperature (C)", fontsize=14)
ax.set_yticklabels([])
ax.xaxis.grid(color="gray", linestyle="dashed")
ax.yaxis.grid(color="gray", linestyle="dashed")
ax.text(
    0.005,
    1.05,
    "D)",
    size=14,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)


ax = fig.add_subplot(gs[1, 2])
# smoke = index_var('tr17_1', units = 'Dimensionless')
smoke = wrf_ds["tr17_1"].isel(south_north=sn_idx, west_east=ew_idx, Time=time_idx)
ax.plot(smoke / pm_ef, height, color="green", linewidth=4)
ax.set_yticklabels([])
ax.set_xlabel(r"PM2.5 Concenration $(\frac{μg}{m^{3}})$", fontsize=14)
ax.set_yticklabels([])
ax.xaxis.grid(color="gray", linestyle="dashed")
ax.yaxis.grid(color="gray", linestyle="dashed")
ax.text(
    0.005,
    1.05,
    "E)",
    size=14,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)

ax = fig.add_subplot(gs[1, 3])
U = index_var("ua", units="km h-1")
V = index_var("va", units="km h-1")
wspd_wdir = index_var("uvmet_wspd_wdir", units="km h-1")
wsp = wspd_wdir.sel(wspd_wdir="wspd")
ax.plot(wsp, height, color="black", linewidth=4, zorder=1)
ax.set_xlim(0, 30)
ax.set_yticklabels([])

one = np.ones_like(height) * 28
ax.barbs(one[0::4], height[0::4], U[0::4], V[0::4], color="black", zorder=10, length=5)
ax.set_xlabel(r"Wind Speed $(\frac{km}{hr})$", fontsize=14)
ax.set_yticklabels([])
ax.xaxis.grid(color="gray", linestyle="dashed")
ax.yaxis.grid(color="gray", linestyle="dashed")
ax.text(
    0.005,
    1.05,
    "F)",
    size=14,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)

#
fig.suptitle(
    f"{title} \n \n Latitude {str(lat)[:6]}, Longitude {str(long)[:8]}  \n"
    + ds.XTIME.values.astype(str)[:-10],
    fontsize=18,
)

fig.tight_layout()

plt.savefig(str(save_dir) + f"/simulated-sounding.png", dpi=250, bbox_inches="tight")
