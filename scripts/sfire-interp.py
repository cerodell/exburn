import context
import wrf
import json
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from netCDF4 import Dataset
from datetime import datetime
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from context import root_dir, vol_dir, data_dir, save_dir, gog_dir
import matplotlib.pylab as pylab

from scipy.interpolate import interp1d, interp2d
from wrf import omp_set_num_threads, omp_get_max_threads

omp_set_num_threads(6)
print(f"read files with {omp_get_max_threads()} threads")


startTime = datetime.now()

g = 9.81  # gravity
pm_ef = 21.05  # boreal wildfire emission factor Urbanski (2014)
zstep = 20.0  # vertical step to interpolate to
BLfrac = 0.75  # fraction of BL height to set zs at
interpz = np.arange(0, 4000, zstep)
interpz[0] = 2
interpz = np.insert(interpz, 1, 10)


ncfile = Dataset(str(vol_dir) + "/sfire/unit5/wrf/wrfout_d01_2019-05-11_17:49:11")
# ncfile = Dataset(str(vol_dir) + "/sfire/unit5/wrfout_d01_0001-01-01_00:00:00")
save_dir = str(vol_dir) + "/sfire/unit5/wrf/"


startCache = datetime.now()
my_cache = wrf.extract_vars(
    ncfile, wrf.ALL_TIMES, ("P", "PSFC", "PB", "PH", "PHB", "T", "QVAPOR", "HGT")
)
print("Cache time: ", datetime.now() - startCache)

startInterp = datetime.now()

startVar = datetime.now()
tr17_1 = wrf.getvar(ncfile, "tr17_1", wrf.ALL_TIMES, cache=my_cache)
interp_tr17_1 = wrf.vinterp(
    ncfile,
    field=tr17_1,
    vert_coord="ght_msl",
    interp_levels=interpz / 1000,
    extrapolate=True,
    log_p=True,
    timeidx=wrf.ALL_TIMES,
    cache=my_cache,
)
print("Interpolate tracer time: ", datetime.now() - startVar)

startVar = datetime.now()
qvapor = wrf.getvar(ncfile, "QVAPOR", wrf.ALL_TIMES, cache=my_cache)
interp_qvapor = wrf.vinterp(
    ncfile,
    field=qvapor,
    vert_coord="ght_msl",
    interp_levels=interpz / 1000,
    extrapolate=True,
    log_p=True,
    timeidx=wrf.ALL_TIMES,
    cache=my_cache,
)
print("Interpolate qvapor time: ", datetime.now() - startVar)

startVar = datetime.now()
temp = wrf.getvar(ncfile, "temp", wrf.ALL_TIMES, cache=my_cache)
interp_temp = wrf.vinterp(
    ncfile,
    field=temp,
    vert_coord="ght_msl",
    interp_levels=interpz / 1000,
    extrapolate=True,
    log_p=True,
    timeidx=wrf.ALL_TIMES,
    field_type="tk",
    cache=my_cache,
)
print("Interpolate temp time: ", datetime.now() - startVar)

startVar = datetime.now()
td = wrf.getvar(ncfile, "td", wrf.ALL_TIMES, cache=my_cache)
interp_td = wrf.vinterp(
    ncfile,
    field=td,
    vert_coord="ght_msl",
    interp_levels=interpz / 1000,
    extrapolate=True,
    log_p=True,
    timeidx=wrf.ALL_TIMES,
    field_type="tk",
    cache=my_cache,
)
print("Interpolate td time: ", datetime.now() - startVar)

startVar = datetime.now()
theta_e = wrf.getvar(ncfile, "theta_e", wrf.ALL_TIMES, cache=my_cache)
interp_theta_e = wrf.vinterp(
    ncfile,
    field=theta_e,
    vert_coord="ght_msl",
    interp_levels=interpz / 1000,
    extrapolate=True,
    log_p=True,
    timeidx=wrf.ALL_TIMES,
    field_type="tk",
    cache=my_cache,
)
print("Interpolate theta_e time: ", datetime.now() - startVar)

startVar = datetime.now()
rh = wrf.getvar(ncfile, "rh", wrf.ALL_TIMES, cache=my_cache)
interp_rh = wrf.vinterp(
    ncfile,
    field=rh,
    vert_coord="ght_msl",
    interp_levels=interpz / 1000,
    extrapolate=True,
    log_p=True,
    timeidx=wrf.ALL_TIMES,
    cache=my_cache,
)
print("Interpolate rh time: ", datetime.now() - startVar)

startVar = datetime.now()
U = wrf.getvar(ncfile, "ua", wrf.ALL_TIMES, cache=my_cache)
interp_U = wrf.vinterp(
    ncfile,
    field=U,
    vert_coord="ght_msl",
    interp_levels=interpz / 1000,
    extrapolate=True,
    log_p=True,
    timeidx=wrf.ALL_TIMES,
    cache=my_cache,
)
print("Interpolate U time: ", datetime.now() - startVar)


startVar = datetime.now()
V = wrf.getvar(ncfile, "va", wrf.ALL_TIMES, cache=my_cache)
interp_V = wrf.vinterp(
    ncfile,
    field=V,
    vert_coord="ght_msl",
    interp_levels=interpz / 1000,
    extrapolate=True,
    log_p=True,
    timeidx=wrf.ALL_TIMES,
    cache=my_cache,
)
print("Interpolate V time: ", datetime.now() - startVar)


startVar = datetime.now()
W = wrf.getvar(ncfile, "wa", wrf.ALL_TIMES, cache=my_cache)
interp_W = wrf.vinterp(
    ncfile,
    field=W,
    vert_coord="ght_msl",
    interp_levels=interpz / 1000,
    extrapolate=True,
    log_p=True,
    timeidx=wrf.ALL_TIMES,
    cache=my_cache,
)
print("Interpolate W time: ", datetime.now() - startVar)


startVar = datetime.now()
pressure = wrf.getvar(ncfile, "pressure", wrf.ALL_TIMES, cache=my_cache)
interp_pressure = wrf.vinterp(
    ncfile,
    field=pressure,
    vert_coord="ght_msl",
    interp_levels=interpz / 1000,
    extrapolate=True,
    log_p=True,
    timeidx=wrf.ALL_TIMES,
    cache=my_cache,
)
print("Interpolate pressure time: ", datetime.now() - startVar)


def compressor(ds):
    """
    this function comresses datasets
    """
    # print('loading')
    # ds = ds.load()
    print("zipping")
    comp = dict(zlib=True, complevel=9)
    print("encoding")
    encoding = {var: comp for var in ds.data_vars}
    return ds, encoding


interp_ds = xr.merge(
    [
        interp_tr17_1,
        interp_qvapor,
        interp_temp,
        interp_td,
        interp_theta_e,
        interp_rh,
        interp_U,
        interp_V,
        interp_W,
        interp_pressure,
    ]
)
interp_ds.attrs = {
    "description": "WRF SFIRE UNIT 5 MOISTURE OFF",
    "dx": "25 m",
    "dy": "25 m",
    "dz": "20 m",
}
for var in interp_ds.data_vars:
    interp_ds[var].attrs["projection"] = str(interp_ds[var].attrs["projection"])
print("Interpolate time: ", datetime.now() - startInterp)

# startCompress= datetime.now()
# comp_ds, encoding = compressor(interp_ds)
# print("Compress time: ", datetime.now() - startCompress)

startWrite = datetime.now()
## write the new dataset
interp_ds.to_netcdf(
    str(save_dir) + "/interp_unit5.nc",
    # encoding=encoding,
    mode="w",
)
print("Write time: ", datetime.now() - startWrite)

print("Total run time: ", datetime.now() - startTime)


# def smoke2levels(ds):
#     zstag = (ds['PHB'] + ds['PH'])/ g
#     z = wrf.destagger(zstag,0)
#     z0 = np.mean(z, (1, 2))
#     ## interpolate tracer to levels
#     tr17_1 = ds.tr17_1
#     tracer = wrf.interplevel(z, tr17_1, interpz)
#     time = ds.XTIME.values.astype("datetime64")
#     tracer = tracer.assign_coords({"Time": time}).expand_dims({"Time": 1}).rename({'dim_0': 'vertical_level','dim_1': 'south_north','dim_2': 'west_east'})
#     tracer = tracer.to_dataset().rename_vars({'field3d_interp': 'tracer'})
#     # print(tracer)

#     ## interpolate temperature to levels
#     # t = ds.T + 300
#     # def something(i, j):
#     #     interpfLES = interp1d(z0, t[:, i, j], fill_value="extrapolate")
#     #     soundingLES = interpfLES(interpz)
#     #     return soundingLES
#     # temp = np.stack([[something(i, j) for i in range(t.shape[1])]for j in range(t.shape[2])])
#     # temp = xr.DataArray(temp, name="temp", dims=('west_east', "south_north", "vertical_level")).assign_coords({"Time": time}).expand_dims({"Time": 1})
#     # print(temp)

#     # ## interpolate U to levels
#     # U = wrf.destagger(wrf_ds.U, 2)
#     # U = xr.DataArray(U, name="U", dims=('vertical_level', "south_north", "west_east"))
#     # U = wrf.interplevel(z, U, interpz)
#     # dsi  = xr.merge([temp, tracer])

#     print(time)
#     return tracer

# da_smoke = xr.concat([smoke2levels(wrf_ds.isel(Time = i)) for i in range(2)], 'Time')

# print(da_smoke)


# da_smoke = xr.concat([smoke2levels(wrf_ds.isel(Time = i)) for i in range(len(wrf_ds.Time))], 'Time')
# ds_smoke = da_smoke.to_dataset().rename_vars({'field3d_interp': 'tracer'})
# ds_smoke = ds_smoke.assign_coords({"vertical_level": interpz})
# ds_smoke = ds_smoke.fillna(0)
# tracer = ds_smoke.tracer.values.astype("float32")


# Times = ds_smoke['Time'].values.astype("datetime64[s]")
# time = [t.decode('utf-8') for t in Times.astype("S29").tolist()]
# x = np.arange(0, ds_smoke.dims["west_east"])
# y = np.arange(0, ds_smoke.dims["south_north"])
# z = np.arange(0, ds_smoke.dims["vertical_level"])

# aqs = config['unit5']['obs']['aq']

# south_north = [110, 400]
# west_east = [10, 150]
# south_north_subgrid = [550, 620]
# west_east_subgrid = [330, 405]


# wrf_ds = int_ds.isel(south_north = slice(south_north[0],south_north[1]),
#                      west_east = slice(west_east[0],west_east[1]),
#                      south_north_subgrid = slice(south_north_subgrid[0],south_north_subgrid[1]),
#                      west_east_subgrid = slice(west_east_subgrid[0],west_east_subgrid[1]))
#                     #  Time= slice(10,14,2))


# XLAT, XLONG = wrf_ds.XLAT.values, wrf_ds.XLONG.values
# GRNHFX = wrf_ds.GRNHFX.values

# ds = wrf_ds.isel(Time = 0)
# temp = ds.T.values+ 300
# zstag = (ds['PHB'] + ds['PH'])/ g
# z = wrf.destagger(zstag,0)
# z0 = np.mean(z, (1, 2))


# interpfLES = interp1d(z0, temp[:, i, j], fill_value="extrapolate")


# U = wrf.destagger(ds.U, 2)
# U = xr.DataArray(U, name="U", dims=('vertical_level', "south_north", "west_east"))


# ds_cf = xr.Dataset(
#     data_vars=dict(
#         TRACER=(["time", "z", "y", "x"], tracer),
#         GRNHFX=(["time", "y", "x"], GRNHFX),
#         x=(["x"], x.astype("int32")),
#         y=(["y"], y.astype("int32")),
#         z=(["z"], z.astype("int32")),
#         TIMES=(["time"], time),
#     ),
#     coords=dict(
#         LONG=(["y", "x"], XLONG.astype("float32")),
#         LAT=(["y", "x"], XLAT.astype("float32")),
#         HEIGHT=(["z"], ds_smoke.vertical_level.values.astype("float32")),
#         time=time,
#     ),
#     attrs=dict(description="WRF SFIRE"),
# )

# ## add axis attributes from cf compliance
# ds_cf["time"].attrs["axis"] = "Time"
# ds_cf["x"].attrs["axis"] = "X"
# ds_cf["y"].attrs["axis"] = "Y"
# ds_cf["z"].attrs["axis"] = "Z"

# ## add units attributes from cf compliance
# ds_cf["HEIGHT"].attrs["units"] = "m"
# ds_cf["LONG"].attrs["units"] = "degree_east"
# ds_cf["LAT"].attrs["units"] = "degree_north"
# ds_cf["TRACER"].attrs["units"] = "k kg^-1"
# ds_cf["GRNHFX"].attrs["units"] = "W m^-2"


# def compressor(ds):
#     """
#     Compresses datasets
#     """
#     ## load ds to memory
#     ds = ds.load()
#     ## use zlib to compress to level 9
#     comp = dict(zlib=True, complevel=9)
#     ## create endcoding for each variable in dataset
#     encoding = {var: comp for var in ds.data_vars}

#     return ds, encoding


# ds_cf, encoding = compressor(ds_cf)


# ## write the new dataset
# ds_cf.to_netcdf(
#     str(filein) + "/firesmoke.nc",
#     encoding=encoding,
#     mode="w",
# )


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
