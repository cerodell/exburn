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
fueltype = 6
# define vertical levels to interpolate to
zstep = 20.0
interpz = np.arange(0, 4000, zstep)
interpz[0] = 2
interpz = np.insert(interpz, 1, 10)
XLAT, XLONG = makeLL("met")


## list of variables to interpolate.
# get_vars   = ["tr17_1", 'QVAPOR', "temp", "td", "theta_e", "rh", "ua", "va", "wa", "pressure"]

get_vars = ["theta_e"]

ncfile = Dataset(str(data_dir) + f"/fuel{fueltype}/wrfout_d01_2019-05-11_17:49:11")


height_agl = wrf.getvar(ncfile, "height_agl")
height_agl.values[:, 0, 0]
height = wrf.getvar(ncfile, "zstag")
height.values[:, 0, 0]
#
startCache = datetime.now()
my_cache = wrf.extract_vars(
    ncfile, wrf.ALL_TIMES, ("P", "PSFC", "PB", "PH", "PHB", "T", "QVAPOR", "HGT")
)
print("Cache time: ", datetime.now() - startCache)


def interpolate(var):
    startVar = datetime.now()
    interp = wrf.vinterp(
        ncfile,
        field=wrf.getvar(ncfile, var, wrf.ALL_TIMES, cache=my_cache),
        vert_coord="ght_msl",
        interp_levels=interpz / 1000,
        extrapolate=True,
        log_p=True,
        timeidx=wrf.ALL_TIMES,
        cache=my_cache,
    )
    ds = interp.to_dataset()
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
    print(f"Interpolate {var} time: ", datetime.now() - startVar)
    startWrite = datetime.now()
    print("Writeing...")
    ds, encoding = compressor(ds)
    ds.to_netcdf(
        str(data_dir) + f"/fuel{fueltype}/interp-unit5-{var.lower()}.nc",
        encoding=encoding,
        mode="w",
    )
    print("Write time: ", datetime.now() - startWrite)
    return


[interpolate(var) for var in get_vars]


# print('Open netcdf')
# interp_ds = xr.merge([xr.open_dataset(str(data_dir) + f"/interp-unit5-{var.lower()}.nc", chunks='auto') for var in get_vars])
# interp_ds.attrs = {
#     "description": "WRF SFIRE UNIT 5 MOISTURE OFF",
#     "dx": "25 m",
#     "dy": "25 m",
#     "dz": "20 m",
# }
# print('netcdf merged')
# for var in interp_ds.data_vars:
#     interp_ds[var].attrs["projection"] = str(interp_ds[var].attrs["projection"])
# startWrite = datetime.now()

# # def compressor(ds):
# #     """
# #     this function comresses datasets
# #     """
# #     comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
# #     encoding = {{var: {"compressor": comp} for var in ds.data_vars}}
# #     return ds, encoding

# print("Writeing...")
# interp_ds.to_zarr(
#     str(data_dir) + "/interp_unit5.zarr",
#     mode="w",
#     # encoding=encoding
# )
# print("Write time: ", datetime.now() - startWrite)
# print("Total run time: ", datetime.now() - startTime)
