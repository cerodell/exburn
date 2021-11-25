#!/Users/crodell/miniconda3/cr/bin/python3
import context
import wrf
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from netCDF4 import Dataset
from datetime import datetime
from utils.sfire import makeLL


from context import root_dir, data_dir
from utils.sfire import compressor
from wrf import omp_set_num_threads, omp_get_max_threads

# omp_set_num_threads(6)
# print(f"read files with {omp_get_max_threads()} threads")


startTime = datetime.now()
modelrun = "F6V51M08Z22I04"
configid = "F6V51"
zstep = 20.0
interpz = np.arange(0, 4000, zstep)
interpz[0] = 2
interpz = np.insert(interpz, 1, 10)
XLAT, XLONG = makeLL("met", configid)

interpz = np.array([1.65, 6.15, 10])


get_vars = ["wspd_wdir", "ua", "va", "wa"]
ncfile = Dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
height_agl = wrf.getvar(ncfile, "height_agl")
height_agl.values[:, 0, 0]
# wrf_ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/wspd_wdir.nc")
# wsp = wrf_ds['wspd']
# wdir = wrf_ds['wdir']

uvmet10_wspd_wdir = wrf.getvar(ncfile, "uvmet10_wspd_wdir", wrf.ALL_TIMES)
wdir = uvmet10_wspd_wdir.sel(wspd_wdir="wdir").rename("wdir").drop("wspd_wdir")
wsp = uvmet10_wspd_wdir.sel(wspd_wdir="wspd").rename("wspd").drop("wspd_wdir")

# wspd = wrf.vinterp(
#     ncfile,
#     field=wsp,
#     vert_coord="ght_agl",
#     interp_levels=interpz,
#     log_p=True,
#     timeidx=wrf.ALL_TIMES,
# )

# wdird = wrf.vinterp(
#     ncfile,
#     field=wdir,
#     vert_coord="ght_agl",
#     interp_levels=interpz,
#     log_p=True,
#     timeidx=wrf.ALL_TIMES,
# )

ds = xr.merge([wdir, wsp])

ds["XLAT"] = XLAT
ds["XLONG"] = XLONG
ds.attrs = {
    "description": "WRF SFIRE UNIT 5 MOISTURE OFF",
    "dx": "25 m",
    "dy": "25 m",
    "dz": "20 m",
}
for var in ds.data_vars:
    ds[var].attrs["projection"] = str(ds[var].attrs["projection"])
startWrite = datetime.now()
print("Writeing...")
ds, encoding = compressor(ds)
ds.to_netcdf(
    str(data_dir) + f"/{modelrun}/wspd_wdir_10m.nc",
    encoding=encoding,
    mode="w",
)
