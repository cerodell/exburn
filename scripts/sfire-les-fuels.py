import context
import json
import salem
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from pathlib import Path
from netCDF4 import Dataset
from matplotlib import path

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pyproj as pyproj

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from context import data_dir, gog_dir, vol_dir, root_dir
from datetime import datetime, date, timedelta

startTime = datetime.now()
print("RUN STARTED AT: ", str(startTime))

# ============ INPUTS==============
ds = 25  # LES grid spacing
fs = 5  # fire mesh ratio
ndx = 160  # EW number of grids
ndy = 400  # NS number of grids
t2 = 290  # surface temperature
fireshape_path = str(gog_dir) + "/all_units/mygeodata_merged.shp"


ll_utm = [
    336524,
    6174820,
]  # lower left corner of the domain in UTM coordinates (meters)
target_fuel = 10  # fuel type within the burn plot
rxloc = [55, -113]  # lat/lon location of the burn
rxtime = 14  # anticipated burn hour
utm = -8  # utm offset
# ============ end of INPUTS==============

# Part I: create a perturbed surface temperature to start off convection-------------
surface_T = (
    (np.random.rand(ndx, ndy) - 0.5) * 1.5
) + t2  # perturbes surface by half a degree +/-
dim_header = ",".join(
    map(str, np.shape(surface_T))
)  # creates a header required by wrf-sfire, which is just dimensions

# save output file
np.savetxt(
    "input_tsk", surface_T, header=dim_header, comments="", fmt="%1.2f"
)  # saves as text file, with some formatting

# Part II: create a fuel mask and locate ignition -----------------------------------
# create a spatial grid using salem
grid = salem.Grid(
    nxny=(ds * ndx / fs, ds * ndy / fs),
    dxdy=(int(ds / fs), int(ds / fs)),
    ll_corner=(ll_utm[0], ll_utm[1]),
    proj="EPSG:26912",
)
## get lat and long of grid
WLONG, WLAT = grid.ll_coordinates
## convert to dataset
ds_grid = grid.to_dataset()

## open dataset of fuels types as defined in the fbp system
# cwfis_fbp = str(vol_dir) + f"/fireweather/data/fuels/can_fuels2019b/can_fuels2019b.tif"
# fbp2019_tiff = salem.open_xr_dataset(cwfis_fbp)


# ## reproject fuel type to the les domain
# ds_fuels = ds_grid.salem.transform(fbp2019_tiff, interp='nearest')


# ## open fuel converter spreadsheet to change from fbp to anderson fuel catagories
# fuel_converter = str(root_dir) + "/data/fuel_converter.csv"
# fc_df = pd.read_csv(fuel_converter)


# ## Using nearest neighbor transformation tif to wrf domain
# # fbp2019_ds = wrf_ds.salem.transform(fbp2019_tiff)


# def getunique(array):
#     unique, count = np.unique(array[~np.isnan(array)], return_counts=True)
#     unique = np.unique(array[~np.isnan(array)]).astype(int)
#     return unique, count

# df = fc_df[fc_df.LF_16 != -99]
# levels = range(0, len(df.FWF_Code.values))

# unique, count = np.unique(ds_fuels.data.values, return_counts=True)
# fuels = ds_fuels.data.values
# zero_array = np.zeros_like(fuels)


# ## with converter fbp feuls to anderson 13
# for index, row in df.iterrows():
#     zero_array[fuels == row['National_FBP_Fueltypes_2019']] = row['LF_16']
#     print(f"{row['National_FBP_Fueltypes_2019']}  == {row['LF_16']}")

# new_fuels = zero_array
# new_fuels[new_fuels==0] = 10 ## default the remainder to 10


# %%


gdf = gpd.read_file(fireshape_path)
gdf = gdf.to_crs(epsg=26912)
gdf["geometry"] = gdf.geometry.buffer(30)
gdf.to_file(str(root_dir) + "/data/shp/unit_buffer.shp", driver="ESRI Shapefile")
gdf = gpd.read_file(str(root_dir) + "/data/shp/unit_buffer.shp")
gdf = gdf.to_crs(epsg=4326)
gdf.to_file(str(root_dir) + "/data/shp/unit_buffer.shp", driver="ESRI Shapefile")


bm = Basemap(
    llcrnrlon=WLONG[0, 0],
    llcrnrlat=WLAT[0, 0],
    urcrnrlon=WLONG[-1, -1],
    urcrnrlat=WLAT[-1, -1],
    resolution="f",
    epsg=4326,
)
# bm = Basemap(llcrnrlon=WLONG[0,0], llcrnrlat=WLAT[0,0],\
# 					 urcrnrlon=WLONG[-1,-1], urcrnrlat=WLAT[-1,-1],  epsg=4326)

polygons = bm.readshapefile(fireshape_path[:-4], name="units", drawbounds=True)
polygons_buff = bm.readshapefile(
    str(root_dir) + "/data/shp/unit_buffer", name="buff", drawbounds=True, color="pink"
)


fuel = np.full_like(WLONG, 10)

for i in range(len(bm.units)):
    unit = path.Path(bm.units[i])
    unit_mask = unit.contains_points(np.array(list(zip(WLONG.ravel(), WLAT.ravel()))))
    unit_mask = np.reshape(unit_mask, np.shape(WLONG))
    buff = path.Path(bm.buff[i])
    buffer_mask = buff.contains_points(np.array(list(zip(WLONG.ravel(), WLAT.ravel()))))
    buffer_mask = np.reshape(buffer_mask, np.shape(WLONG))
    fuel[buffer_mask != unit_mask] = 14
for i in range(len(bm.units)):
    unit = path.Path(bm.units[i])
    unit_mask = unit.contains_points(np.array(list(zip(WLONG.ravel(), WLAT.ravel()))))
    unit_mask = np.reshape(unit_mask, np.shape(WLONG))
    fuel[unit_mask] = target_fuel


# %%
# sanity-check plot
plt.figure(figsize=(20, 12))
bm.contourf(WLONG, WLAT, fuel)
# polygons = bm.readshapefile(fireshape_path[:-4],name='units',drawbounds=True, color='red')
bm.contourf(WLONG, WLAT, fuel)
# polygons_buff = bm.readshapefile(str(root_dir) + "/data/shp/unit_buffer",name='buff',drawbounds=True, color='pink')
plt.colorbar(orientation="horizontal")
plt.title("ENTIRE LES DOMAIN WITH FIRE PLOT")
plt.show()


# %%

# UNIT 4 SOUTHERLY
plt.figure(figsize=(20, 12))
plt.title("CLOSEUP OF THE FIRE PLOT")
bmX = Basemap(
    llcrnrlon=WLONG[300, 350],
    llcrnrlat=WLAT[300, 350],
    urcrnrlon=WLONG[600, 500],
    urcrnrlat=WLAT[500, 500],
    epsg=4326,
)
polygonsX = bmX.readshapefile(
    fireshape_path[:-4], name="units", drawbounds=True, color="red"
)
bmX.contourf(WLONG[300:600, 350:500], WLAT[300:600, 350:500], fuel[300:600, 350:500])
plt.colorbar(orientation="horizontal", label="fuel category")
plt.scatter(
    WLONG[402, 436], WLAT[406, 404], c="red", marker="*", label="ignition start"
)
plt.scatter(WLONG[402, 404], WLAT[402, 404], c="red", label="ignition end")
plt.legend()
plt.show()


# #sanity-check plot
# contourf = bm.contourf(WLONG, WLAT,new_fuels+0.5)
# polygons = bm.readshapefile(fireshape_path,name='units',drawbounds=True, color='red')
# cbar = plt.colorbar()

# tick_levels = list(np.unique(new_fuels) - 0.5)
# cbar = plt.colorbar(ticks=tick_levels)
# cbar.set_ticklabels(ticks)  # set ticks of your format
# # cbar.bm.axes.tick_params(length=0)

# plt.show()

# fbp2019[fbp2019 == 0] = np.nan
# fbp2019[fbp2019 == 65535] = np.nan


# ## take US Anderson 13 fuel valsue and convert to CFFDRS fuel types
# us_array = us_ds.values
# us_array_og = us_array

# us_unique, us_count = getunique(us_array_og)
# ## Ensure ponderosa fuel types carry over
# mask = np.where((XLONG < -100) & (us_array_og == 9))

# us_count[us_unique == 9]
# for i in range(len(fc_df.LF_16.values)):
#     if fc_df.LF_16[i] == -99:
#         pass
#     else:
#         us_array[us_array == float(fc_df.LF_16[i])] = fc_df.FWF_Code[i]
#         # if fc_df.LF_16[i] == 6:
#         #     us_array[us_array == fc_df.LF_16[i]] = fc_df.FWF_Code[i]
#         #     us_array = np.where((XLONG < -96) & (us_array == fc_df.FWF_Code[i]), 3, us_array)
#         # else:
#         #     us_array[us_array == fc_df.LF_16[i]] = fc_df.FWF_Code[i]

# us_array = np.where((XLONG > -120) & (us_array == 5), 3, us_array)
# us_array = np.where((XLONG < -96) & (us_array == 11), 3, us_array)
# us_array[mask] = 7


# # XLAT = XLAT.ravel()
# # XLONG = XLONG.ravel()
# # us_array = us_array.ravel()
# # mask = list(np.where((XLAT< 49.0) & (XLONG < -100)  & (us_array == 3))[0])
# # us_array[mask] = np.random.choice([3,3,3,3,3,3,3,3,7],len(mask))
# # us_array = np.reshape(us_array, us_array_og.shape)

# us_unique_new, us_count_new = getunique(us_array)

# ind = np.isnan(fbp2019)
# fbp2019[ind] = us_array[ind]


# ## loop all tiffs of AK to gridded adn mask fuels type tag to be the same as CFFDRS
# folders = ["%.2d" % i for i in range(1, 21)]
# print(folders)
# for folder in folders:
#     ak_filein = str(vol_dir) + f"/fuels/resampled/{domain}/ak_{folder}.tif"
#     ak_tiff = salem.open_xr_dataset(ak_filein)
#     ak_ds = wrf_ds.salem.transform(ak_tiff.data, ks=1)
#     ak_array = ak_ds.values
#     for i in range(len(fc_df.AK_Fuels.values)):
#         if fc_df.AK_Fuels[i] == -99:
#             pass
#         else:
#             ak_array[ak_array == fc_df.AK_Fuels[i]] = fc_df.FWF_Code[i]

#     ind = np.isnan(fbp2019)
#     fbp2019[ind] = ak_array[ind]

# fbp_unique, fbp_count = getunique(fbp2019)

# ## concidered remianing missing data to be water
# ind = np.isnan(fbp2019)
# fbp2019[np.isnan(fbp2019)] = 17


# ## make dataset add coordinates and write to zarr file
# fuels_ds = xr.DataArray(fbp2019, name="fuels", dims=("south_north", "west_east"))
# T2 = wrf_ds.T2
# fuels_ds = xr.merge([fuels_ds, T2])
# fuels_ds = fuels_ds.drop_vars("T2")
# fuels_final = fuels_ds.isel(Time=0)
# fuels_final.to_zarr(save_zarr, mode="w")

# ### Timer
# print("Total Run Time: ", datetime.now() - startTime)

# %%
