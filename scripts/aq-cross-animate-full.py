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
import matplotlib as mpl


#################### Define Inputs #########################
pm_ef = 10.400  ## black spruce emissions factor prichard_wildland_2020
modelrun = "F6V51M08Z22"  ## choose model run F(fueltype)V(vertical levels)M(moisture precentage)Z(z grad stretch)
dxy = 25  ## horizontal resolution
## used to make plotting levels
a = np.arange(1, 10)
b = 10 ** np.arange(4)
levels = (b[:, np.newaxis] * a).flatten()
levels = levels[:-8]
cmap = mpl.cm.cubehelix_r  ## define colormap
norm = mpl.colors.BoundaryNorm(
    levels, cmap.N, extend="both"
)  ## normalize colormap with levels
var = "tr17_1"  ## name variable to plot
title = "Tracer"  ## give plot a title
units = "PM2.5 " + r"($μg m^{-3}$)"  ## define units
fireshape_path = (
    str(data_dir) + "/all_units/mygeodata_merged"
)  ## file dir to burn unit polygones
## define save directory adn create if none existent
save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)
## narrow domain bounds
south_north = slice(110, None, None)
west_east = slice(30, 129, None)
bottom_top = slice(0, None, None)
###########################################################################


## open wrf model output as xarray
wrf_ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
## use wrf python to get model heights
ncfile = Dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
height = wrf.getvar(ncfile, "height")
height = height.isel(south_north=132, west_east=74, bottom_top=bottom_top)


## get heatflux for plotting
hfx = wrf_ds["GRNQFX"]
hfx = hfx.sel(
    south_north=south_north,
).sum(dim="west_east")

## slice and dice domain to be smaller
wrf_ds = wrf_ds.sel(
    south_north=south_north,
    west_east=west_east,
    bottom_top=bottom_top,
)
## crosswind intigrate smoke
var_da = wrf_ds[var].sum(dim="west_east").to_dataset()
## get array of north south in meters
sn = wrf_ds.south_north * dxy
## get time lenght for animating
dimT = len(wrf_ds.XTIME)


## create a figure for animmation
fig = plt.figure(figsize=(14, 5))
## make a subplot in the fig
ax = fig.add_subplot(1, 1, 1)
## get first time index and plot
ds = var_da.isel(Time=0)
ax.set_title(f"{title} \n" + ds.XTIME.values.astype(str)[:-10], fontsize=18)
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
ax.set_ylabel("Vertical (m)", fontsize=14, labelpad=10)
ax.set_xlabel("Horizontal (m)", fontsize=14, labelpad=10)
ax.tick_params(axis="both", which="major", labelsize=12)

colors_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
hfx_i = hfx.isel(Time=0)
ax_hx = ax.twinx()
ax_hx.plot(sn, hfx_i / 1000, color="tab:red", lw=0.8, zorder=10)
ax_hx.set_ylabel(
    "Heatflux " + r"$(kW m^{-2})$" + "\n",
    fontsize=12,
    color="tab:red",
    rotation=-90,
    labelpad=30,
)
ax_hx.tick_params(axis="y", colors="tab:red")
ax_hx.set_ylim(0, 400)
ax_hx.tick_params(axis="both", which="major", labelsize=12)

# add colorbar
cbar = plt.colorbar(contour, ax=ax_hx, pad=0.1)
cbar.ax.tick_params(labelsize=12)
cbar.set_label(units, rotation=270, fontsize=14, labelpad=20)


def update_plot(i):
    """
    functions to animate
    input: (i) is time index
    """
    ax_hx.clear()
    global contour
    for c in contour.collections:
        c.remove()
    print(i)
    ds = var_da.isel(Time=i)
    # ax.set_title(f"{title} \n" + ds.Time.values.astype(str)[:-10], fontsize=18)
    ax.set_title(f"{title} \n" + ds.XTIME.values.astype(str)[:-10], fontsize=18)
    contour = ax.contourf(
        sn,
        height,
        ds[var] / pm_ef,
        zorder=1,
        cmap=cmap,
        levels=levels,
        extend="both",
    )
    hfx_i = hfx.isel(Time=i)
    ax_hx.plot(sn, hfx_i / 1000, color="tab:red", lw=0.8, zorder=10)
    ax_hx.set_ylabel(
        "Heatflux " + r"$(kW m^{-2})$" + "\n",
        fontsize=12,
        color="tab:red",
        rotation=-90,
        labelpad=30,
    )
    ax_hx.tick_params(axis="y", colors="tab:red")
    ax_hx.set_ylim(0, 400)
    ax_hx.tick_params(axis="both", which="major", labelsize=12)

    return contour


## make that fig tight
fig.tight_layout()
## animate and store plot
ani = animation.FuncAnimation(fig, update_plot, dimT, interval=3)
## save animation and close
ani.save(str(save_dir) + f"/{var}-cross-full.mp4", writer="ffmpeg", fps=10, dpi=250)
plt.close()
