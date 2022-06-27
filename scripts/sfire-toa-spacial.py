import context
import glob
from pathlib import Path
import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split("lib")[0]
proj_lib = os.path.join(os.path.join(conda_dir, "share"), "proj")
os.environ["PROJ_LIB"] = proj_lib

from mpl_toolkits.basemap import Basemap
import json
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from osgeo import gdal, osr
from pathlib import Path
from sklearn.neighbors import KDTree
import matplotlib.colors as colors
from matplotlib.pyplot import cm
from wrf import getvar, get_cartopy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

from context import root_dir, data_dir, save_dir, img_dir
from utils.sfire import makeLL_new, prepare_df, ignition_line

import warnings
from pylab import *


warnings.filterwarnings("ignore", category=RuntimeWarning)


################################ INPUTS ################################
domain = "fire"
unit = "unit5"
modelrun = "F6V51M08Z22"
sim_date = "2019-05-11_17:49:11"

title = "Time of Arrival"
var = "FGRNHFX"

ros_tiff = str(data_dir) + "/obs/tiff/arrival_time_seconds.tiff"
unit_shapefile = str(data_dir) + "/unit_5/unit_5"
# unit_shapefile = str(data_dir) + "/all_units/mygeodata_merged"

levels_ros = np.arange(10, 500, 20)
# cmap = cm.get_cmap("jet", len(levels_ros))  # PiYG
cmap = cm.get_cmap("turbo", 20)  # PiYG
# norm = colors.BoundaryNorm(levels_ros, cmap.N)


fire_nsew = [55.719627, 55.717018, -113.570663, -113.576227]

## create img dir for unit amd simualtion
img_dir = Path(str(img_dir) + f"/{unit}-{sim_date[:10]}")
img_dir.mkdir(parents=True, exist_ok=True)
############################# END INPUTS ##############################

################################ OPEN CONFIG ################################
with open(str(root_dir) + "/json/config-new.json") as f:
    config = json.load(f)
ros = config["unit5"]["obs"]["ros"]
cmap_obs = cm.get_cmap("viridis", len(ros))
ros_ids = list(ros)
sfire = config["unit5"]["sfire"]
south_north_subgrid = slice(sfire["fire"]["sn"][0], sfire["fire"]["sn"][1])
west_east_subgrid = slice(sfire["fire"]["we"][0], sfire["fire"]["we"][1])
fs = sfire["namelist"]["dxy"] / sfire["namelist"]["fs"]

ig_start = sfire["namelist"]["ig_start"]
ig_end = sfire["namelist"]["ig_end"]

save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)


############################################################################


############################ WRF SIFIRE SETUP ################################
## Open the NetCDF file
ncfile = Dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_{sim_date}")
slp = getvar(ncfile, "T2")

cart_proj = get_cartopy(slp)
# cart_proj = '+proj=ob_tran +a=6370000.0 +b=6370000.0 +nadgrids=@null +o_proj=latlon +o_lon_p=180.0 +o_lat_p=-0.0 +lon_0=180.0 +to_meter=111177.4733520388 +no_defs +type=crs'

## Open wrf out single line
wrf_ds = xr.open_dataset(
    str(data_dir) + f"/{modelrun}/wrfout_d01_{sim_date}", chunks="auto"
)

XLAT, XLONG = makeLL_new(domain, unit)
wrf_ds["XLAT"], wrf_ds["XLONG"] = XLAT, XLONG
wrf_ds = wrf_ds.sel(
    south_north_subgrid=south_north_subgrid,
    west_east_subgrid=west_east_subgrid,
    Time=slice(3, 54),
)
var_da = wrf_ds[var]
times = var_da.XTIME.values

## Open wrf out multi line
wrf_ds_I04 = xr.open_dataset(
    str(data_dir) + f"/{modelrun+'I04'}/wrfout_d01_{sim_date}", chunks="auto"
)
wrf_ds_I04["XLAT"], wrf_ds_I04["XLONG"] = XLAT, XLONG
wrf_ds_I04 = wrf_ds_I04.sel(
    south_north_subgrid=south_north_subgrid,
    west_east_subgrid=west_east_subgrid,
    Time=slice(3, 54),
)
var_da_I04 = wrf_ds_I04[var]


XLAT, XLONG = var_da_I04["XLAT"], var_da_I04["XLONG"]
times = var_da_I04.XTIME.values
############################################################################


# model_time = ros_da.XTIME.values
fulltime = (times[-1] - times[0]).astype("timedelta64[s]")


######################## Plotting set up ##########################
# Read shape file of all unit pelican mnt
reader = shpreader.Reader(f"{unit_shapefile}.shp")

## open geo tiff file of sat image and get useful projection/transform info for plotting
ds = gdal.Open(ros_tiff)
data = ds.ReadAsArray()
gt = ds.GetGeoTransform()
proj = ds.GetProjection()
inproj = osr.SpatialReference()
inproj.ImportFromWkt(proj)
projcs = inproj.GetAuthorityCode("PROJCS")
projection = ccrs.epsg(projcs)
# print(projection)

extent = (
    gt[0],
    gt[0] + ds.RasterXSize * gt[1],
    gt[3] + ds.RasterYSize * gt[5],
    gt[3],
)
factor = 3.5 / 255
clip_range = (0, 1)
real = np.clip(plt.imread(ros_tiff) * factor, *clip_range)

subplot_kw = dict(projection=projection)
###########################################################################


########################### Plot Time of Arrival ############################


mpl.rcParams.update({"font.size": 10})

# %%
fig = plt.figure(figsize=(18, 6))
fig.suptitle("Fire Front Arrival Time", fontsize=20)


# ax = fig.add_subplot(1, 2, 2,projection = cart_proj )
ax = fig.add_subplot(1, 3, 2, projection=projection)
ax.set_title(
    "Observed",
    weight="bold",
)
ignition_line("F6V51M08Z22I04", ax, ig_start, ig_end)
data = np.where(data != -99999.0, data, np.nan)
## plot satalite tiff image
im = ax.imshow(
    data - np.nanmin(data),
    zorder=1,
    extent=extent,
    origin="upper",
    interpolation="bilinear",
    cmap=cmap,
    vmax=levels_ros.max(),
    vmin=levels_ros.min(),
)

## plot shapefile polygons of units
ax.add_geometries(
    reader.geometries(),
    crs=ccrs.Geodetic(),
    edgecolor="#1E3C47",
    alpha=0.8,
    facecolor="none",
    lw=2,
)

## set map extent
ax.set_extent(
    [fire_nsew[3], fire_nsew[2], fire_nsew[1], fire_nsew[0]], crs=ccrs.PlateCarree()
)  ##  (x0, x1, y0, y1)


##############################################################################


ax = fig.add_subplot(1, 3, 1, projection=projection)
ax.set_title(
    "Singles",
    weight="bold",
)
ignition_line("F6V51M08Z22", ax, ig_start, ig_end)

time_of_arrival = var_da.argmax(dim="Time").values * 10
contour = ax.contourf(
    XLONG,
    XLAT,
    time_of_arrival,
    zorder=1,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    levels=levels_ros,
    extend="max",
)

## plot shapefile polygons of units
ax.add_geometries(
    reader.geometries(),
    crs=ccrs.Geodetic(),
    edgecolor="#1E3C47",
    alpha=0.8,
    facecolor="none",
    lw=2,
)

## set map extent
ax.set_extent(
    [fire_nsew[3], fire_nsew[2], fire_nsew[1], fire_nsew[0]], crs=ccrs.PlateCarree()
)  ##  (x0, x1, y0, y1)


ax = fig.add_subplot(1, 3, 3, projection=projection)
ax.set_title(
    "Multiple",
    weight="bold",
)
ignition_line("F6V51M08Z22I04", ax, ig_start, ig_end)

time_of_arrival = var_da_I04.argmax(dim="Time").values * 10
contour = ax.contourf(
    XLONG,
    XLAT,
    time_of_arrival,
    zorder=1,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    levels=levels_ros,
    extend="max",
)

## plot shapefile polygons of units
ax.add_geometries(
    reader.geometries(),
    crs=ccrs.Geodetic(),
    edgecolor="#1E3C47",
    alpha=0.8,
    facecolor="none",
    lw=2,
)

## set map extent
ax.set_extent(
    [fire_nsew[3], fire_nsew[2], fire_nsew[1], fire_nsew[0]], crs=ccrs.PlateCarree()
)  ##  (x0, x1, y0, y1)


fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.165, 0.1, 0.7, 0.05])  # [left, bottom, width, height]
cbar = fig.colorbar(im, orientation="horizontal", cax=cbar_ax)
time_tile = times[0].astype(str)[:-11] + "8"
cbar.set_label(f"Seconds from Ignition {time_tile}", fontsize=13, labelpad=15)


plt.show()

plt.savefig(str(save_dir) + f"/toa-modeled-v-obs.png", dpi=300, bbox_inches="tight")
