import context
import salem
import numpy as np
import xarray as xr
import pyproj as pyproj

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

from context import root_dir

# # ============ INPUTS==============
# ds = 25           #LES grid spacing
# fs = 25            #fire mesh ratio
# ndx = 160-1         #EW number of grids
# ndy = 400-1         #NS number of grids
# ll_utm = [336524,6174820]           #lower left corner of the domain in UTM coordinates (meters)
# # ============ end of INPUTS==============

# #create a spatial grid using salem
# grid = salem.Grid(nxny=(ds*ndx/fs, ds*ndy/fs), dxdy=(int(ds/fs), int(ds/fs)), ll_corner=(ll_utm[0], ll_utm[1]), proj='EPSG:26912')
# ## get lat and long of grid
# WLONG, WLAT = grid.ll_coordinates
# ds_fuel = xr.open_zarr(str(root_dir) + '/data/zarr/fuels.zarr')
# fuel = ds_fuel.fuel.values

wrf_in = str(root_dir) + "/data/wrf/wrfout_d01_2019-05-11_17:05:00"

wrf_ds = xr.open_dataset(wrf_in)

wrf_ds = wrf_ds.isel(south_north=slice(104, 124), west_east=slice(60, 90))
wrf_ds.GRNHFX.plot(x="west_east", y="south_north", col="Time", col_wrap=3)
plt.show()
