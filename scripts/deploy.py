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
from sklearn.neighbors import KDTree
from scipy.interpolate import interp1d


import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.colors as colors
from matplotlib.dates import DateFormatter
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from context import root_dir, vol_dir, data_dir, save_dir, gog_dir
import matplotlib.pylab as pylab
from utils.sfire import makeLL_new
import matplotlib as mpl
import cartopy.crs as crs

from wrf import (
    to_np,
    getvar,
    smooth2d,
    get_cartopy,
    cartopy_xlim,
    cartopy_ylim,
    latlon_coords,
)

##################### Define Inputs and File Directories ###################
unit = "unit4"
domain = "fire"
var = "tr17_1"

# pm_ef = 21.05  # boreal wildfire emission factor Urbanski (2014)
pm_ef = 10.400
# levels = np.arange(0, 1200.0, 10)
a = np.arange(1, 10)
b = 10 ** np.arange(4)
levels = (b[:, np.newaxis] * a).flatten()
levels = levels[:-5]
cmap = mpl.cm.cubehelix_r
norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend="both")

filein = str(root_dir) + "/sat/response.tiff"
fireshape_path = str(data_dir) + "/all_units/mygeodata_merged"
aqsin = str(data_dir) + "/obs/aq/"
aqsin = sorted(Path(aqsin).glob(f"*"))
save_dir = Path(str(save_dir) + f"/{unit}/")
save_dir.mkdir(parents=True, exist_ok=True)

# a = np.arange(1,14)
# b = 10**np.arange(4)
# levels = (b[:, np.newaxis] * a).flatten()


################### Open Datsets ###################
## Open COnfig File and Get Relevant Paramters
with open(str(root_dir) + "/json/config-new.json") as f:
    config = json.load(f)
namelist = config[unit]["sfire"]

# south_north = namelist['south_north']
# west_east = namelist['west_east']

# south_north_subgrid = namelist['south_north_subgrid']
# west_east_subgrid = namelist['west_east_subgrid']
## create Lat and Long array based on wrf-sfire configuration

XLAT, XLONG = makeLL_new("met", unit)
# fire_XLAT, fire_XLONG = makeLL_new('fire', unit)

# create dataframe with columns of all XLAT/XLONG
wrf_locs = pd.DataFrame({"XLAT": XLAT.values.ravel(), "XLONG": XLONG.values.ravel()})
wrf_tree = KDTree(wrf_locs)
print("WRF Domain KDTree built")


def get_loc(lat, lon):
    dist, ind = wrf_tree.query(np.array([lat, lon]).reshape(1, -1), k=1)
    loc = list(np.unravel_index(int(ind), XLAT.shape))
    return loc


met_nsew = namelist["met-nsew"]
ll = get_loc(met_nsew[1], met_nsew[3])
ur = get_loc(met_nsew[0], met_nsew[2])
south_north_met = slice(ll[0], ur[0])
west_east_met = slice(ll[1], ur[1])

fire_nsew = namelist["fire-nsew"]
ll = get_loc(fire_nsew[1], fire_nsew[3])
ur = get_loc(fire_nsew[0], fire_nsew[2])
south_north_fire = slice(ll[0], ur[0])
west_east_fire = slice(ll[1], ur[1])


# Open the NetCDF file
ncfile = Dataset(str(data_dir) + f"/{unit}/wrfout_d01_2019-05-11_17:49:11")

# Get the sea level pressure
slp = getvar(ncfile, "T2", timeidx=10)

# Smooth the sea level pressure since it tends to be noisy near the
# mountains
smooth_slp = smooth2d(slp, 3, cenweight=4)

# Get the latitude and longitude points
lats, lons = latlon_coords(slp)

# Get the cartopy mapping object
cart_proj = get_cartopy(slp)

## open wrf-sfire simulation
wrf_ds = xr.open_dataset(str(data_dir) + f"/{unit}/wrfout_d01_2019-05-11_17:49:11")
wrf_ds[var] = wrf_ds[var] / pm_ef

met_XLAT = XLAT.sel(south_north=south_north_met, west_east=west_east_met)
met_XLONG = XLONG.sel(south_north=south_north_met, west_east=west_east_met)

fire_XLAT = XLAT.sel(south_north=south_north_fire, west_east=west_east_fire)
fire_XLONG = XLONG.sel(south_north=south_north_fire, west_east=west_east_fire)

met_ds = wrf_ds.sel(
    south_north=south_north_met,
    west_east=west_east_met,
    Time=slice(0, 124, 6),
)
met_ds = met_ds.sum(dim="bottom_top")
met_ds["Time"] = met_ds.XTIME.values.astype("datetime64[s]")


# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(1, 1, 1)
# bm = Basemap(
#     # llcrnrlon=XLONG[0, 0] + 0.0014,
#     llcrnrlon=XLONG[0, 0],
#     llcrnrlat=XLAT[0, 0],
#     urcrnrlon=XLONG[-1, -1],
#     # urcrnrlat=XLAT[-1, -1] + 0.001,
#     urcrnrlat=XLAT[-1, -1],
#     epsg=4326,
#     ax=ax,
# )
# wesn = [XLONG[0, 0], XLONG[-1, -1], XLAT[0, 0], XLAT[-1, -1]]
# # ax = fig.add_subplot(1,1,1, projection=ccrs.UTM(zone=12))
# factor = 3.5 / 255
# clip_range = (0, 1)
# real = np.clip(plt.imread(filein) * factor, *clip_range)
# real = real[::-1, :, :]
# bm.imshow(real, zorder=1, extent=wesn)

# tr17_1 = met_ds['tr17_1'].isel(Time = 20)
# contour = ax.contourf(
#     met_XLONG,
#     met_XLAT,
#     tr17_1,
#     zorder=1,
#     levels=levels,
#     cmap=cmap,
#     norm=norm,
#     extend="max",  # cubehelix_r
#     alpha = 0.6
# )
# # ax.set_xlim(met_nsew[3], met_nsew[2]-0.004)
# ax.set_xlim(met_nsew[3], met_nsew[2])
# ax.set_ylim(met_nsew[1], met_nsew[0])


fire_ds = wrf_ds.sel(
    south_north=south_north_fire,
    west_east=west_east_fire,
    Time=slice(0, 124, 6),
)

# %%

# Create a figure
fig = plt.figure(figsize=(12, 6))
# Set the GeoAxes to the projection used by WRF
# ax = plt.axes(projection=cart_proj)
ax = plt.axes()
wesn = [XLONG[0, 0], XLONG[-1, -1], XLAT[0, 0], XLAT[-1, -1]]
factor = 3.5 / 255
clip_range = (0, 1)
real = np.clip(plt.imread(filein) * factor, *clip_range)
# real = real[::-1, :, :]
ax.imshow(real, zorder=1, extent=wesn)
tr17_1 = wrf_ds["GRNHFX"].isel(Time=10)
# contour = plt.contourf(
#     XLONG,
#     XLAT,
#     tr17_1,
#     zorder=10,
#     # transform=crs.Mercator(),
#     # cmap=cmap,
#     # norm=norm,
#     # extend="max",  # cubehelix_r
#     alpha = 0.6
# )
# ax.set_xlim(fire_nsew[3]-0.004, fire_nsew[2])
# ax.set_ylim(fire_nsew[1], fire_nsew[0])


# %%
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
bm = Basemap(
    projection="tmerc",
    llcrnrlon=XLONG[0, 0] + 0.0045,
    # llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1] + 0.0015,
    # urcrnrlat=XLAT[-1, -1],
    epsg=4326,
    ax=ax,
)

# read shape file
# gdf = gpd.read_file(fireshape_path + ".shp")
# gdf = gdf.to_crs("EPSG:3395")
# gdf.to_file(str(root_dir) + "/data/shp/unit_mer.shp", driver="ESRI Shapefile")
# polygons = bm.readshapefile(str(root_dir) + "/data/shp/unit_mer", name="units", drawbounds=True, zorder =10)
polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=10)


wesn = [XLONG[0, 0], XLONG[-1, -1], XLAT[0, 0], XLAT[-1, -1]]
factor = 3.5 / 255
clip_range = (0, 1)
real = np.clip(plt.imread(filein) * factor, *clip_range)
real = real[::-1, :, :]
bm.imshow(real, zorder=1, extent=wesn)

# tr17_1 = fire_ds["GRNHFX"].isel(Time=1)
# contour = ax.contourf(
#     fire_XLONG,
#     fire_XLAT,
#     tr17_1,
#     zorder=1,
#     # transform=crs.PlateCarree(),
#     # cmap=cmap,
#     # norm=norm,
#     extend="max",  # cubehelix_r
#     alpha=0.6,
# )
ax.set_xlim(fire_nsew[3] - 0.002, fire_nsew[2])
ax.set_ylim(fire_nsew[1], fire_nsew[0])


# met_ds.tr17_1.plot(
#     col="Time",
#     col_wrap=4,
#     levels=levels,
#     cmap=cmap,
#     extend="max",
# )


# fire_ds = wrf_ds.sel(
#     south_north=south_north_fire,
#     west_east=south_north_fire,
# )


# fire_ds.GRNHFX.plot(
#     col="Time",
#     col_wrap=4,
#     # cmap="cubehelix_r",
#     # levels=np.arange(0, 30100, 100),
#     extend="max",
# )
# plt.savefig(str(save_dir) + "/tracer-vert-int.png")
# %%
