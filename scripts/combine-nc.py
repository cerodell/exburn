import context
import numpy as np
import xarray as xr
from pathlib import Path


import matplotlib.pyplot as plt
from context import root_dir, vol_dir, data_dir

filein = str(vol_dir) + "/sfire/unit5/moist_false/"

# ds = xr.open_zarr(str(filein) + '/wrfout_unit5.zarr')

# ds = test.chunk(chunks="auto")
# test2 = ds.load()

keep_vars = [
    "U",
    "V",
    "W",
    "PH",
    "PBH",
    "PHB",
    "T",
    "P",
    "PB",
    "QVAPOR",
    "GRNHFX",
    "GRNQFX",
    "CANHFX",
    "CANQFX",
    "UAH",
    "VAH",
    "FMC_GC",
    "tr17_1",
]

# paths = sorted(Path(filein).glob(f"wrfout_*"))
# datasets = [xr.open_dataset(p, chunks={"Time": 10}) for p in paths[:2]]
# combined = xr.concat(datasets, 'Time')


def read_netcdfs(filein, dim, grid):
    # def process_one_path(path, grid):
    #     ds = xr.open_dataset(path, chunks={"Time": 10})
    #     if grid == 'wrf':
    #       ds = ds.drop([var for var in list(ds) if var not in keep_vars])
    #     elif grid == 'fire':
    #       ds = ds.drop([var for var in list(ds) if var in keep_vars])
    #     else:
    #       pass
    #     return ds

    paths = sorted(Path(filein).glob(f"wrfout_*"))
    print("opening...")
    # datasets = [process_one_path(p, grid) for p in paths]
    datasets = [xr.open_dataset(p, chunks={"Time": 10}) for p in paths]
    print("opened")
    print("combine...")
    combined = xr.concat(datasets, dim)
    print("combined")
    return combined


wrf_ds = read_netcdfs(filein, dim="Time", grid="wrf")
# fire_ds = read_netcdfs(filein, dim='Time', grid = 'fire')

# def compressor(ds):
#     """
#     this function comresses datasets
#     """
#     # print('loading')
#     # ds = ds.load()
#     print('zipping')
#     comp = dict(zlib=True, complevel=9)
#     print('encoding')
#     encoding = {var: comp for var in ds.data_vars}
#     return ds, encoding

# wrf_ds, encoding = compressor(wrf_ds)
print(wrf_ds)
print("writing")
wrf_ds.to_zarr(str(data_dir) + "/wrfout_unit5.zarr")
# wrf_compressed_ds.to_netcdf(str(data_dir) + '/wrfout_unit5.nc', compute=False, mode ='w')
print("done :)")
