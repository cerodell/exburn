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


modelrun = "F6V51M08Z22"
ds = 25


# WAF

save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)


# ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/wspd_wdir.nc")


ncfile = Dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
height = wrf.getvar(ncfile, "height")
height = height.isel(south_north=123, west_east=74).values
# height = height[:24]
height = np.insert(height, 0, 0)
x = np.arange(0, 4025, 25)
zz, xx = np.meshgrid(height, x)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.contourf(x, height, zz.T, levels=height, extend="min", cmap="terrain")
# ax.set_ylim(0,500)
# ax.set_xlim(0,50)

for i in range(len(height) - 1):
    ax.plot(x, zz[:, i + 1], color="tab:grey")

x = x[::10]
for i in range(len(x)):
    ax.axvline(x[i], color="tab:grey")

plt.savefig(str(save_dir) + f"/hyperbolic-full.png", dpi=250, bbox_inches="tight")
