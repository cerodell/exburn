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


theta_ds = xr.open_dataset(str(data_dir) + '/interp-unit5-theta_e.nc')
interp_level = theta_ds.interp_level * 1000 
south_north = theta_ds.south_north  * 25
theta_ds = theta_ds.isel(west_east = 71)

# theta_ds = theta_ds.isel(Time =slice(0,4))
dimT = len(theta_ds.Time)


fig = plt.figure(figsize=(14, 4))
ax = fig.add_subplot(1, 1, 1)   #top and bottom left
ds = theta_ds.isel(Time=0)
ax.set_title('Equivalent Potentail Temperature \n' + ds.Time.values.astype(str)[:-10], fontsize = 18)
theta_e = ax.contourf(
    south_north,
    interp_level,
    ds.theta_e,
    zorder=1,
    cmap="jet",
    levels=np.arange(295, 312, 0.1),
    extend="both",
)
ax.set_xlabel('Vertical (m)', fontsize = 16)
ax.set_ylabel('Horizontal (m)', fontsize = 16)
ax.tick_params(axis='both', which='major', labelsize=14)

cbar = plt.colorbar(theta_e, ax=ax, pad=0.01)
cbar.ax.tick_params(labelsize=12) 
cbar.set_label('Kelvin', rotation=270, fontsize = 16, labelpad=15)



def update_plot(i):
    global theta_e
    for c in theta_e.collections: c.remove()
    print(i)
    ds = theta_ds.isel(Time = i)
    ax.set_title('Equivalent Potentail Temperature \n' + ds.Time.values.astype(str)[:-10], fontsize = 18)
    theta_e = ax.contourf(
        south_north,
        interp_level,
        ds.theta_e,
        zorder=1,
        cmap="jet",
        levels=np.arange(295, 312, 0.1),
        extend="both",
    )
    return theta_e

fig.tight_layout()
ani=animation.FuncAnimation(fig, update_plot, dimT, interval=3)
# plt.show()
ani.save(str(save_dir) + '/theta_e-cross.mp4', writer='ffmpeg',fps=10, dpi=250)
plt.close()


































# ## open combine interpolated dataset
# # ds = xr.open_zarr(str(data_dir) + "/interp_unit5.zarr")
# # ds = xr.open_dataset(str(data_dir) + "/interp-unit5-theta_e.nc", chunks = 'auto')

# # ## get vertical levels and coverert to meters
# # interp_level = ds.interp_level.values * 1000
# # zstep = 20
# # levelup = 20
# # ## solve for PBL height
# # end = len(interp_level) - levelup
# # # theta_e = ds.theta_e.isel(interp_level = slice(levelup,None), Time = slice(20,24), south_north = slice(10,15), west_east= slice(10,15))
# # theta_e = ds.theta_e.isel(interp_level = slice(levelup,None))
# # XLAT = theta_e.XLAT.values
# # chunk = theta_e.chunks

# # print("Solve Temp Gradient: ", datetime.now())
# # statTIME = datetime.now()
# # levels = theta_e.interp_level.values * 1000
# # del theta_e['interp_level']
# # zi = theta_e.isel(interp_level = slice(1,end))
# # zii = theta_e.isel(interp_level = slice(0,end-1))

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
# #     np.stack([[np.full_like(XLAT, level) for level in levels]]* len(theta_e.Time)),
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







