import context
import wrf
import json
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from context import root_dir, vol_dir, data_dir, save_dir, gog_dir
import matplotlib.pylab as pylab


g = 9.81  # gravity
pm_ef = 21.05  # boreal wildfire emission factor Urbanski (2014)
zstep = 20.0  # vertical step to interpolate to
BLfrac = 0.75  # fraction of BL height to set zs at
interpz = np.arange(0, 4000, zstep)
interpz[0] = 2
interpz = np.insert(interpz, 1, 10)

with open(str(data_dir) + "/json/config.json") as f:
    config = json.load(f)

fireshape_path = str(gog_dir) + "/all_units/mygeodata_merged"

wrfrun = "/sfire/unit5/moist_false/"
filein = str(vol_dir) + wrfrun
save_dir = str(save_dir) + wrfrun
int_ds = xr.open_zarr(str(filein) + "/wrfout_unit5.zarr")

aqs = config["unit5"]["obs"]["aq"]

## open smoke nc file
smoke_ds = xr.open_dataset(str(filein) + "/firesmoke.nc")

smoke_ds.TRACER.isel(time=200, z=-1).plot()
XLAT, XLONG = smoke_ds.XLAT.values, smoke_ds.XLONG.values


# ## create dataframe with columns of all lat/long
# wrf_locs = pd.DataFrame({"XLAT": XLAT.ravel(), "XLONG": XLONG.ravel()})
# ## build kdtree
# wrf_tree = KDTree(wrf_locs)
# print("WRF Domain KDTree built")

# def find_index(aq):
#     aq = np.array([aqs[aq]['lat'],aqs[aq]['lon']]).reshape(1, -1)
#     aq_dist, aq_ind = wrf_tree.query(aq, k=1)
#     aq_loc = list(np.unravel_index(int(aq_ind), XLAT.shape))
#     return aq_loc

# aqs_loc = np.stack([find_index(aq) for aq in aqs])
# south_north = xr.DataArray(np.array(aqs_loc[:,0]), dims= 'aqs', coords= dict(aqs = list(aqs)))
# west_east = xr.DataArray(np.array(aqs_loc[:,1]), dims= 'aqs', coords= dict(aqs = list(aqs)))

# def smoke2m(ds):
#     zstag = (ds['PHB'] + ds['PH'])/ g
#     z = wrf.destagger(zstag,0)
#     tr17_1 = ds.tr17_1
#     tracer = wrf.interplevel(z, tr17_1, interpz)
#     time = ds.XTIME.values.astype("datetime64")
#     tracer = tracer.assign_coords({"Time": time}).expand_dims({"Time": 1}).rename({'dim_0': 'vertical_level','dim_1': 'south_north','dim_2': 'west_east'})
#     print(time)
#     return tracer

# # len(wrf_ds.Time)
# smoke = xr.concat([smoke2m(wrf_ds.isel(Time = i)) for i in range(len(wrf_ds.Time))], 'Time')
# # smoke = xr.concat([smoke2m(wrf_ds.isel(Time = i)) for i in range(3)], 'Time')
# smoke = smoke.to_dataset().rename_vars({'field3d_interp': 'tracer'})
# smoke = smoke.assign_coords({"vertical_level": interpz})
# XLAT = xr.DataArray(
#     name="XLAT",
#     data= wrf_ds['XLAT'].values,
#     dims=["south_north", "west_east"],
# )
# XLONG = xr.DataArray(
#     name="XLONG",
#     data= wrf_ds['XLONG'].values,
#     dims=["south_north", "west_east"],
# )
# smoke = smoke.assign_coords({"XLAT": XLAT})
# smoke = smoke.assign_coords({"XLONG": XLONG})

# # smoke = smoke.assign_coords({"X": X})
# smoke.to_zarr(str(filein) + "/smoke.zarr", mode = 'w')

# smoke_aq = smoke.sel(south_north = south_north, west_east = west_east)
# smoke_aq = smoke_aq.rename_vars({'tracer': 'tracer_aq'})
# balh = xr.concat([smoke,smoke_aq], 'Time')

# smoke_aq.to_zarr(str(filein) + "/smoke_at_aq.zarr", mode = 'w')
# smoke_aq.tracer.plot.line(x="Time")
# plt.show()
