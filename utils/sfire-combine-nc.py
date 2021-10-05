import context
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pylab as pylab
from mpl_toolkits.basemap import Basemap
import pyproj as pyproj

import matplotlib.pyplot as plt
from context import root_dir, vol_dir, data_dir

filein = str(vol_dir) + "/sfire/unit5/moist_false/"

ds = 25  # LES grid spacing
fs = 5  # fire mesh ratio
ndx = 160  # EW number of grids
ndy = 400  # NS number of grids
t2 = 290  # surface temperature
buff = 30  # buffer size (ie firebreaks) around units
anderson = 10  # anderson fuels type
ig_start = [55.7177788, -113.571244]
ig_end = [55.7177788, -113.575172]
sw = [55.717153, -113.57668]
ne = [55.720270, -113.569517]
ll_utm = [
    336524,
    6174820,
]  # lower left corner of the domain in UTM coordinates (meters)
target_fuel = 10  # fuel type within the burn plot
rxloc = [55, -113]  # lat/lon location of the burn
rxtime = 14  # anticipated burn hour
utm = -8  # utm offset
# ============ end of INPUTS==============


## create grid WRF LAT/LONG with defined inputs
gridx, gridy = np.meshgrid(
    np.arange(0, ds * ndx, int(ds)), np.arange(0, ds * ndy, int(ds))
)
## stagger gird to fit on wrf_out
gridx, gridy = gridx - ds / 2, gridy - ds / 2
## drop first row/colum to match size
gridx, gridy = gridx[1:, 1:], gridy[1:, 1:]
# now adding a georefernce we have UTM grid (technically UTM_12N, or EPSG:26912
UTMx = gridx + ll_utm[0]
UTMy = gridy + ll_utm[1]
## transform into the same projection (for shapefiles and for basemaps)
wgs84 = pyproj.Proj("+init=EPSG:4326")
epsg26912 = pyproj.Proj("+init=EPSG:26912")
# reproject from UTM to WGS84
WGSx, WGSy = pyproj.transform(epsg26912, wgs84, UTMx.ravel(), UTMy.ravel())
XLONG, XLAT = np.reshape(WGSx, np.shape(UTMx)), np.reshape(WGSy, np.shape(UTMy))
XLAT = xr.DataArray(
    name="XLAT",
    data=XLAT,
    dims=["south_north", "west_east"],
)
XLONG = xr.DataArray(
    name="XLONG",
    data=XLONG,
    dims=["south_north", "west_east"],
)


## create grid FIRE LAT/LONG with defined inputs
fire_gridx, fire_gridy = np.meshgrid(
    np.arange(0, ds * ndx, int(ds / fs)), np.arange(0, ds * ndy, int(ds / fs))
)
fireX = xr.DataArray(
    name="fireX",
    data=fire_gridx,
    dims=["south_north_subgrid", "west_east_subgrid"],
)
fireY = xr.DataArray(
    name="fireY",
    data=fire_gridy,
    dims=["south_north_subgrid", "west_east_subgrid"],
)
# now adding a georefernce we have UTM grid (technically UTM_12N, or EPSG:26912
fire_UTMx = fire_gridx + ll_utm[0]
fire_UTMy = fire_gridy + ll_utm[1]
## transform into the same projection (for shapefiles and for basemaps)
wgs84 = pyproj.Proj("+init=EPSG:4326")
epsg26912 = pyproj.Proj("+init=EPSG:26912")
# reproject from UTM to WGS84
fire_WGSx, fire_WGSy = pyproj.transform(
    epsg26912, wgs84, fire_UTMx.ravel(), fire_UTMy.ravel()
)
fire_XLONG, fire_XLAT = np.reshape(fire_WGSx, np.shape(fire_UTMx)), np.reshape(
    fire_WGSy, np.shape(fire_UTMy)
)
fire_XLAT = xr.DataArray(
    name="FIRE_XLAT",
    data=fire_XLAT,
    dims=["south_north_subgrid", "west_east_subgrid"],
)
fire_XLONG = xr.DataArray(
    name="FIRE_XLONG",
    data=fire_XLONG,
    dims=["south_north_subgrid", "west_east_subgrid"],
)


# paths = sorted(Path(filein).glob(f"wrfout_*"))
# test = xr.open_dataset(paths[0], chunks={"Time": 10})

# Y = xr.DataArray(
#     name="Y",
#     data= test['XLAT'].sel(Time=0, drop=True).values,
#     dims=["south_north", "west_east"],
# )
# X = xr.DataArray(
#     name="X",
#     data= test['XLONG'].sel(Time=0, drop=True).values,
#     dims=["south_north", "west_east"],
# )
# test = test.assign_coords({"Y": Y})
# test = test.assign_coords({"X": X})
# test = test.assign_coords({"fireY": fireY})
# test = test.assign_coords({"fireX": fireX})
# test = test.assign_coords({"fire_XLONG": fire_XLONG})
# test = test.assign_coords({"fire_XLAT": fire_XLAT})

# test['XLONG'] = test['XLONG'].sel(Time=0, drop=True)
# test['XLAT'] = test['XLAT'].sel(Time=0, drop=True)
# test['XLONG'] =  XLONG
# test['XLAT'] = XLAT
# datasets = [xr.open_dataset(p, chunks={"Time": 10}) for p in paths[:2]]
# combined = xr.concat(datasets, 'Time')


def read_netcdfs(filein, dim, grid):

    paths = sorted(Path(filein).glob(f"wrfout_*"))
    print("opening...")
    datasets = [xr.open_dataset(p, chunks={"Time": 10}) for p in paths]
    print("opened")
    print("combine...")
    combined = xr.concat(datasets, dim)
    print("combined")
    return combined


wrf_ds = read_netcdfs(filein, dim="Time", grid="wrf")

Y = xr.DataArray(
    name="Y",
    data=wrf_ds["XLAT"].sel(Time=0, drop=True).values,
    dims=["south_north", "west_east"],
)
X = xr.DataArray(
    name="X",
    data=wrf_ds["XLONG"].sel(Time=0, drop=True).values,
    dims=["south_north", "west_east"],
)
wrf_ds = wrf_ds.assign_coords({"Y": Y})
wrf_ds = wrf_ds.assign_coords({"X": X})
wrf_ds = wrf_ds.assign_coords({"fireY": fireY})
wrf_ds = wrf_ds.assign_coords({"fireX": fireX})
wrf_ds = wrf_ds.assign_coords({"fire_XLONG": fire_XLONG})
wrf_ds = wrf_ds.assign_coords({"fire_XLAT": fire_XLAT})

wrf_ds["XLONG"] = wrf_ds["XLONG"].sel(Time=0, drop=True)
wrf_ds["XLAT"] = wrf_ds["XLAT"].sel(Time=0, drop=True)
wrf_ds["XLONG"] = XLONG
wrf_ds["XLAT"] = XLAT

print(wrf_ds)
print("writing")
wrf_ds.to_zarr(str(data_dir) + "/wrfout_unit5_ll.zarr", mode="w")
print("done :)")


# keep_vars = [
#     "U",
#     "V",
#     "W",
#     "PH",
#     "PBH",
#     "PHB",
#     "T",
#     "P",
#     "PB",
#     "QVAPOR",
#     "GRNHFX",
#     "GRNQFX",
#     "CANHFX",
#     "CANQFX",
#     "UAH",
#     "VAH",
#     "FMC_GC",
#     "tr17_1",
# ]

# def read_netcdfs(filein, dim, grid):
#     # def process_one_path(path, grid):
#     #     ds = xr.open_dataset(path, chunks={"Time": 10})
#     #     if grid == 'wrf':
#     #       ds = ds.drop([var for var in list(ds) if var not in keep_vars])
#     #     elif grid == 'fire':
#     #       ds = ds.drop([var for var in list(ds) if var in keep_vars])
#     #     else:
#     #       pass
#     #     return ds

#     paths = sorted(Path(filein).glob(f"wrfout_*"))
#     print("opening...")
#     # datasets = [process_one_path(p, grid) for p in paths]
#     datasets = [xr.open_dataset(p, chunks={"Time": 10}) for p in paths]
#     print("opened")
#     print("combine...")
#     combined = xr.concat(datasets, dim)
#     print("combined")
#     return combined


# wrf_ds = read_netcdfs(filein, dim="Time", grid="wrf")
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
