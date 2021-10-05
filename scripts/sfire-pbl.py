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
from context import root_dir,  data_dir, save_dir
import matplotlib.pylab as pylab


fireshape_path = str(data_dir) + "/all_units/mygeodata_merged"


try:
  print('Found PBL dataset')
  pbhl_ds = xr.open_dataset(str(data_dir) + '/PBLH.nc')
except:
  print('Cant Find PBL dataset, solving...')
  pbhl_ds = sovle_pbl()

XLONG = pbhl_ds.XLONG.values
XLAT  = pbhl_ds.XLAT.values
# pbhl_ds = pbhl_ds.isel(Time =slice(0,4))
dimT = len(pbhl_ds.Time)


fig = plt.figure(figsize=(8, 8))
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0.05)
ax = fig.add_subplot(1, 1, 1)   #top and bottom left

bm = Basemap(
    llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1],
    epsg=4326,
    ax = ax
)
polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True)

ds = pbhl_ds.isel(Time=0)
ax.set_title(ds.Time.values.astype(str))
smoke = bm.contourf(
    XLONG,
    XLAT,
    ds.PBLH,
    zorder=1,
    cmap="coolwarm",
    levels=np.arange(800, 3410, 10),
    extend="both",
)
# fig.colorbar(smoke, cax=cax)
cb = bm.colorbar(smoke, "right", size="5%", pad="1%")



def update_plot(i):
    global smoke
    for c in smoke.collections: c.remove()
    print(i)
    ds = pbhl_ds.isel(Time = i)
    ax.set_title(ds.Time.values.astype(str))
    smoke = bm.contourf(XLONG, XLAT, ds.PBLH, zorder = 1,  cmap="coolwarm", levels=np.arange(1000, 3010, 10), extend="both")
    return smoke


ani=animation.FuncAnimation(fig, update_plot, dimT, interval=3)
# plt.show()
ani.save(str(save_dir) + '/PBLH.mp4', writer='ffmpeg',fps=10, dpi=250)
plt.close()


