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


pm_ef = 21.05  # boreal wildfire emission factor Urbanski (2014)
fueltype = 6
var = 'tr17_1'
title = 'Tracer'
units = 'Concentration g kg^-1'
cmap="cubehelix_r"
levels=np.arange(0, 30100.0, 100)

fireshape_path = str(data_dir) + "/all_units/mygeodata_merged"
save_dir = Path(str(save_dir) + f'/fuel{fueltype}/')
save_dir.mkdir(parents=True, exist_ok=True)

var_ds = xr.open_dataset(str(data_dir) + f'/fuel{fueltype}/interp-unit5-{var}.nc')
interp_level = var_ds.interp_level * 1000 
south_north = var_ds.south_north  * 25
# var_ds = var_ds.sum(dim = 'west_east')
var_ds = var_ds.isel(west_east = 71)

# var_ds = var_ds.isel(Time =slice(0,4))
dimT = len(var_ds.Time)


fig = plt.figure(figsize=(14, 4))
ax = fig.add_subplot(1, 1, 1)   #top and bottom left
ds = var_ds.isel(Time=100)
ax.set_title(f'{title} \n' + ds.Time.values.astype(str)[:-10], fontsize = 18)
contour = ax.contourf(
    south_north,
    interp_level,
    ds[var],
    zorder=1,
    cmap=cmap,
    levels=levels,
    extend="both",
)
ax.set_xlabel('Vertical (m)', fontsize = 16)
ax.set_ylabel('Horizontal (m)', fontsize = 16)
ax.tick_params(axis='both', which='major', labelsize=14)

cbar = plt.colorbar(contour, ax=ax, pad=0.01)
cbar.ax.tick_params(labelsize=12) 
cbar.set_label(units, rotation=270, fontsize = 16, labelpad=15)



def update_plot(i):
    global contour
    for c in contour.collections: c.remove()
    print(i)
    ds = var_ds.isel(Time = i)
    ax.set_title('Tracer \n' + ds.Time.values.astype(str)[:-10], fontsize = 18)
    contour = ax.contourf(
        south_north,
        interp_level,
        ds.tr17_1,
        zorder=1,
        cmap=cmap,
        levels=levels,
        extend="both",
    )
    return contour

fig.tight_layout()
ani=animation.FuncAnimation(fig, update_plot, dimT, interval=3)
ani.save(str(save_dir) + f'/{var}-cross.mp4', writer='ffmpeg',fps=10, dpi=250)
plt.close()


































# ## open combine interpolated dataset
# # ds = xr.open_zarr(str(data_dir) + "/interp_unit5.zarr")
# # ds = xr.open_dataset(str(data_dir) + "/interp-unit5-tr17_1.nc", chunks = 'auto')

# # ## get vertical levels and coverert to meters
# # interp_level = ds.interp_level.values * 1000
# # zstep = 20
# # levelup = 20
# # ## solve for PBL height
# # end = len(interp_level) - levelup
# # # tr17_1 = ds.tr17_1.isel(interp_level = slice(levelup,None), Time = slice(20,24), south_north = slice(10,15), west_east= slice(10,15))
# # tr17_1 = ds.tr17_1.isel(interp_level = slice(levelup,None))
# # XLAT = tr17_1.XLAT.values
# # chunk = tr17_1.chunks

# # print("Solve Temp Gradient: ", datetime.now())
# # statTIME = datetime.now()
# # levels = tr17_1.interp_level.values * 1000
# # del tr17_1['interp_level']
# # zi = tr17_1.isel(interp_level = slice(1,end))
# # zii = tr17_1.isel(interp_level = slice(0,end-1))

# # # dask.config.set({"array.slicing.split_large_chunks": False})
# # gradTLES = (zi - zii)/zstep
# # end = len(gradTLES.interp_level)
# # gradT2 = (gradTLES.isel(interp_level = slice(1,end)) - gradTLES.isel(interp_level = slice(0,end-1))) 
# # print("Temp Gradient Solved: ", datetime.now() - statTIME)

# # print("Computing: ", datetime.now())
# # statTIME = datetime.now()
# # # gradT2 = gradT2.compute()
# # print("Computed: ", datetime.now() - statTIME)

# # print("Build Height: ", datetime.now())
# # statTIME = datetime.now()
# # height = xr.DataArray(
# #     np.stack([[np.full_like(XLAT, level) for level in levels]]* len(tr17_1.Time)),
# #     name='PBLH',
# #     dims=('Time', 'interp_level', "south_north", "west_east"),
# # ).chunk(chunk)
# # print("Height Built: ", datetime.now() - statTIME)

# # print("Index Height: ", datetime.now())
# # statTIME = datetime.now()
# # PBLH = height.isel(gradT2.argmax(dim=['interp_level']))
# # print("Height Indexed: ", datetime.now() - statTIME)

# # print("Write PBLH: ", datetime.now())
# # statTIME = datetime.now()
# # PBLH, encoding = compressor(PBLH.to_dataset())
# # PBLH.to_netcdf(str(data_dir) + '/PBLH.nc',
# #             encoding = encoding,
# #             mode = 'w')
# # print("Write Time: ", datetime.now() - statTIME)







