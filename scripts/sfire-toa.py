import context
import glob
from pathlib import Path

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

# from wrf import getvar, get_cartopy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

from context import root_dir, data_dir, save_dir, img_dir
from utils.sfire import makeLL_new, prepare_df, ignite

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

ros_filein = str(data_dir) + "/obs/ros/"
sat_tiff = str(root_dir) + "/sat/response.tiff"
unit_shapefile = str(data_dir) + "/unit_5/unit_5"
# unit_shapefile = str(data_dir) + "/all_units/mygeodata_merged"

levels_ros = np.arange(10, 500, 20)
camp = cm.get_cmap("tab20c", 20)  # PiYG

drop_ids = ["C2R1", "C2R2", "C2R3", "C4R4", "C4R5", "C9R2", "C9R3"]
cols = ["C3", "C4", "C5", "C6", "C7", "C8"]

drop_ids = [
    "C2R1",
    "C2R2",
    "C2R3",
    "C4R4",
    "C4R5",
    "C9R2",
    "C9R3",
    "C5R1",
    "C5R1",
    "C5R2",
    "C5R3",
    "C5R4",
    "C5R5",
    "C5R6",
    "C8R1",
    "C8R2",
    "C8R3",
    "C8R4",
    "C7R2",
    "C7R3",
    "C7R4",
    "C7R5",
]
cols = ["C3", "C4", "C6"]

drop_ids = [
    "C2R1",
    "C2R2",
    "C2R3",
    "C4R4",
    "C4R5",
    "C9R2",
    "C9R3",
    "C3R1",
    "C3R2",
    "C4R1",
    "C4R3",
    "C6R1",
    "C6R2",
    "C6R3",
    "C6R4",
    "C6R5",
]
cols = ["C5", "C7", "C8"]

fire_nsew = [55.719627, 55.717018, -113.570663, -113.576227]

## create img dir for unit amd simualtion
img_dir = Path(str(img_dir) + f"/{unit}-{sim_date[:10]}")
img_dir.mkdir(parents=True, exist_ok=True)
############################# END INPUTS ##############################

################################ OPEN CONFIG ################################
with open(str(root_dir) + "/json/config-new.json") as f:
    config = json.load(f)
ros = config["unit5"]["obs"]["ros"]
ros = {key: ros[key] for key in ros if key not in drop_ids}
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
# ncfile = Dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_{sim_date}")
# slp = getvar(ncfile, "T2")

# cart_proj = get_cartopy(slp)

## Open wrf out
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
XLAT, XLONG = var_da["XLAT"], var_da["XLONG"]
times = var_da.XTIME.values

############################################################################


## create dataframe with columns of all XLAT/XLONG
locs = pd.DataFrame({"XLAT": XLAT.values.ravel(), "XLONG": XLONG.values.ravel()})
## build kdtree
tree = KDTree(locs)
print("Domain KDTree built")

ros_dfs = [prepare_df(s, ros_filein, times) for s in ros]
dimT = len(ros_dfs[0])


def find_index(s):
    q = np.array([ros[s]["lat"], ros[s]["lon"]]).reshape(1, -1)
    dist, ind = tree.query(q, k=1)
    loc = list(np.unravel_index(int(ind), XLAT.shape))
    return loc


ros_locs = np.stack([find_index(s) for s in ros_ids])
y = xr.DataArray(np.array(ros_locs[:, 0]), dims="ros", coords=dict(ros=ros_ids))
x = xr.DataArray(np.array(ros_locs[:, 1]), dims="ros", coords=dict(ros=ros_ids))
if domain == "fire":
    ros_da = var_da.sel(south_north_subgrid=y, west_east_subgrid=x)
elif domain == "met":
    ros_da = var_da.sel(south_north=y, west_east=x)
else:
    raise ValueError("Not a valid domain option")


# model_time = ros_da.XTIME.values
fulltime = (times[-1] - times[0]).astype("timedelta64[s]")


def first_nonzero_index(array):
    """Return the index of the first non-zero element of array. If all elements are zero, return -1."""

    fnzi = -1  # first non-zero index
    indices = np.flatnonzero(array)

    if len(indices) > 0:
        fnzi = indices[0]

    return fnzi


arr = ros_da.differentiate(coord="Time").values
first_ind = np.apply_along_axis(first_nonzero_index, axis=0, arr=arr)

# max_ind = ros_da.argmax(dim="Time").values
all_offset = []
for i in range(len(ros_dfs)):
    # obs_time = ros_dfs[i]["temp"].idxmax()
    obs_time = pd.to_datetime("2019-05-11T" + ros[ros_ids[i]]["top"])

    condtion = ros_da.isel(ros=i).ros.values == ros_ids[i]
    if condtion == True:
        pass
    else:
        raise ValueError("Missmatch in ROS IDs")
    off_set = (obs_time - times[first_ind[i]]).total_seconds()
    all_offset.append(off_set)
    print(ros_ids[i])
    print(f"Obs time:    {obs_time}")
    print(f"WRF time:    {times[first_ind[i]]}")
    print(f"Offset time: {np.round(off_set,2)}")
    print(f"Offset percentage:  {np.round((off_set / 499) * 100,1)} %")
    print("---------------------------------")
ros_ids = np.array(ros_ids)
all_offset = np.array(all_offset)

print(f"Mean offsite time:       {np.round(np.mean(np.abs(all_offset)),2)} secs")
print(
    f"Mean offset percentage:  {np.round((np.mean(np.abs(all_offset)) / 499) * 100,1)} %"
)


######################## Plotting set up ##########################

# Read shape file of all unit pelican mnt
# reader = shpreader.Reader(f"{unit_shapefile}.shp")

# ## open geo tiff file of sat image and get useful projection/transform info for plotting
# ds = gdal.Open(sat_tiff)
# data = ds.ReadAsArray()
# gt = ds.GetGeoTransform()
# proj = ds.GetProjection()
# inproj = osr.SpatialReference()
# inproj.ImportFromWkt(proj)
# projcs = inproj.GetAuthorityCode("PROJCS")
# projection = ccrs.epsg(projcs)
# # print(projection)

# extent = (
#     gt[0],
#     gt[0] + ds.RasterXSize * gt[1],
#     gt[3] + ds.RasterYSize * gt[5],
#     gt[3],
# )
# factor = 3.5 / 255
# clip_range = (0, 1)
# real = np.clip(plt.imread(sat_tiff) * factor, *clip_range)

# subplot_kw = dict(projection=projection)
###########################################################################


########################### Plot Time of Arrival ############################
hex = []
for i in range(camp.N):
    rgba = camp(i)
    hex.append(matplotlib.colors.rgb2hex(rgba))
cmap_ros = LinearSegmentedColormap.from_list("tab20c", hex, N=len(levels_ros))

hex_obs = []
for i in range(len(ros)):
    rgba = cmap_obs(i)
    hex_obs.append(matplotlib.colors.rgb2hex(rgba))


mpl.rcParams.update({"font.size": 10})

# %%
fig = plt.figure(figsize=(12, 4))
fig.suptitle(
    "Offset of Fire Front Arrival Time \n Observed minus Model", y=1.1, fontsize=16
)
ax = fig.add_subplot(1, 2, 2)
ax.bar(ros_ids[::-1], all_offset[::-1], color=hex_obs[::-1])

ax.text(
    0.02,
    1.05,
    "B)",
    size=20,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)
ax.tick_params(axis="x", labelrotation=90)
ax.tick_params(axis="both", which="major", labelsize=11)
ax.set_xlabel("Sensor", fontsize=12)
ax.set_ylabel("Seconds", fontsize=12)


ax = fig.add_subplot(1, 2, 1)
ds_i = wrf_ds.isel(Time=i)

bm = Basemap(
    llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1],
    epsg=4326,
    ax=ax,
)

## plot satalite tiff image
# ax.imshow(real, zorder=1, extent=extent, origin="upper")

## plot shapefile polygons of units
polygons = bm.readshapefile(unit_shapefile, name="units", drawbounds=True, zorder=1)

time_of_arrival = var_da.argmax(dim="Time").values * 10

contour = ax.contourf(
    XLONG,
    XLAT,
    time_of_arrival + 8,
    levels=levels_ros,
    cmap=cmap_ros,
)


cbar = plt.colorbar(contour, ax=ax, location="bottom", pad=0.2)
cbar.ax.tick_params(labelsize=12)
time_tile = times[0].astype(str)[:-11] + "8"
cbar.set_label(f"Seconds from Ignition {time_tile}", fontsize=13, labelpad=15)


if ("I04" in modelrun) == True:
    print("Multi line ignition")
    ignite(
        ax,
        ig_start=[55.7177529, -113.5713107],
        ig_end=[55.71773480, -113.57183453],
        line="1",
        color="tab:red",
    )
    ignite(
        ax,
        ig_start=[55.7177109, -113.5721005],
        ig_end=[55.7177124, -113.5725656],
        line="2",
        color="tab:blue",
    )
    ignite(
        ax,
        ig_start=[55.7177293, -113.5734885],
        ig_end=[55.7177437, -113.5744894],
        line="3",
        color="tab:green",
    )
    ignite(
        ax,
        ig_start=[55.7177775603, -113.5747705233],
        ig_end=[55.717752429, -113.575192125],
        line="4",
        color="tab:grey",
    )
else:
    print("Single line ignition")
    ignite(ax, ig_start, ig_end, line="1", color="tab:red")


for col in cols:
    indices = [ii for ii, s in enumerate(ros_ids) if col in s]
    ax.annotate(
        col,
        xy=(ros[ros_ids[indices[0]]]["lon"], 55.71730),
        color="w",
        bbox=dict(boxstyle="circle", fc="black", ec="k", alpha=0.8),
        ha="center",
    )


# ## set map extent
# ax.set_extent(
#     [fire_nsew[3], fire_nsew[2], fire_nsew[1], fire_nsew[0]], crs=ccrs.PlateCarree()
# )  ##  (x0, x1, y0, y1)


# ros_ids[index_by_ros]
for i in range(len(ros)):
    ax.scatter(
        ros[ros_ids[i]]["lon"],
        ros[ros_ids[i]]["lat"],
        zorder=9,
        s=100,
        color=hex_obs[i],
        edgecolors="black",
    )

shape = XLAT.shape
ax.set_xticks(np.linspace(bm.lonmin, bm.lonmax, 12))
labels = [item.get_text() for item in ax.get_xticklabels()]
xlabels = np.arange(0, shape[1] * fs, shape[1] * fs / len(labels)).astype(int)
ax.set_xticklabels(xlabels, fontsize=11)

ax.set_yticks(np.linspace(bm.latmin, bm.latmax, 6))
labels = [item.get_text() for item in ax.get_yticklabels()]
ylabels = np.arange(0, shape[0] * fs, shape[0] * fs / len(labels)).astype(int)
ax.set_yticklabels(ylabels, fontsize=11)
ax.set_xlabel("West-East (m)", fontsize=12)
ax.set_ylabel("South-North (m)", fontsize=12)
ax.text(
    0.015,
    1.05,
    "A)",
    size=20,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)

# plt.savefig(str(save_dir) + f"/ros-offsets.png", dpi=300, bbox_inches="tight")


# fig = plt.figure(figsize=(9, 9))  # (Width, height) in inches.
# ax = fig.add_subplot(111, projection = projection )
# # fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=subplot_kw)
# ds_i = wrf_ds.isel(Time=i)

# ## plot satalite tiff image
# ax.imshow(real, zorder=1, extent=extent, origin="upper")

# ## plot shapefile polygons of units
# ax.add_geometries(
#     reader.geometries(),
#     crs=ccrs.Geodetic(),
#     edgecolor="#1E3C47",
#     alpha=0.8,
#     facecolor="none",
#     lw=2,
# )


# ## set map extent
# ax.set_extent(
#     [fire_nsew[3], fire_nsew[2], fire_nsew[1], fire_nsew[0]], crs=ccrs.PlateCarree()
# )  ##  (x0, x1, y0, y1)

# def normalize(y):
#     x = y / np.linalg.norm(y)
#     return x


# cols = ["C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
# fig = plt.figure(figsize=(14, 8))
# # fig.suptitle(modelrun)
# ax = fig.add_subplot(1, 2, 1)
# # ax = fig.add_subplot(2, 1, 1)

# bm = Basemap(
#     llcrnrlon=XLONG[0, 0],
#     llcrnrlat=XLAT[0, 0],
#     urcrnrlon=XLONG[-1, -1],
#     urcrnrlat=XLAT[-1, -1],
#     epsg=4326,
#     ax=ax,
# )
# time_of_arrival = var_da.argmax(dim="Time").values * 10

# contour = ax.contourf(
#     XLONG, XLAT, time_of_arrival + 8, levels=levels_ros, cmap=cmap_ros
# )
# cbar = plt.colorbar(contour, ax=ax, pad=0.004, location="bottom")
# cbar.ax.tick_params(labelsize=12)
# time_tile = times[0].astype(str)[:-11] + "8"
# # cbar.set_label(f"Seconds from First Ignition {time_tile}", fontsize=13, labelpad=15)
# cbar.set_label(f"Seconds from Ignition {time_tile}", fontsize=13, labelpad=15)

# polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=1)


# def ignite(ig_start, ig_end, line, color):
#     ax.scatter(
#         ig_start[1],
#         ig_start[0],
#         c=color,
#         s=200,
#         marker="*",
#         # alpha=0.6,
#         label=f"{line} ignition start",
#         zorder=10,
#         edgecolors="black",
#     )
#     ax.scatter(
#         ig_end[1],
#         ig_end[0],
#         c=color,
#         marker="X",
#         s=150,
#         # alpha=0.6,
#         label=f"{line} ignition end",
#         zorder=10,
#         edgecolors="black",
#     )
#     ax.plot(
#         [ig_start[1], ig_end[1]],
#         [ig_start[0], ig_end[0]],
#         linestyle="--",
#         lw=2,
#         marker="",
#         zorder=9,
#         color="k",
#         # alpha=0.6,
#     )


# if ("I04" in modelrun) == True:
#     print("Multi line ignition")
#     ignite(
#         ig_start=[55.7177529, -113.5713107],
#         ig_end=[55.71773480, -113.57183453],
#         line="1",
#         color="tab:red",
#     )
#     ignite(
#         ig_start=[55.7177109, -113.5721005],
#         ig_end=[55.7177124, -113.5725656],
#         line="2",
#         color="tab:blue",
#     )
#     ignite(
#         ig_start=[55.7177293, -113.5734885],
#         ig_end=[55.7177437, -113.5744894],
#         line="3",
#         color="tab:green",
#     )
#     ignite(
#         ig_start=[55.7177775603, -113.5747705233],
#         ig_end=[55.717752429, -113.575192125],
#         line="4",
#         color="tab:grey",
#     )

#     # ignite(ig_start= [55.7177529, -113.5713107], ig_end = [55.71773480, -113.57183453], line = '1', color = 'red')
#     # ignite(ig_start= [55.71774808, -113.57232778], ig_end = [55.71771973, -113.57299677], line = '2', color = 'blue')
#     # ignite(ig_start= [55.71771900, -113.57341997], ig_end =[55.7177473680, -113.5742683254], line = '3', color = 'green')
#     # ignite(ig_start= [55.7177775603, -113.5747705233], ig_end = [55.717752429, -113.575192125], line = '4', color = 'black')
# else:
#     print("Single line ignition")
#     ignite(ig_start, ig_end, line="1", color="tab:red")


# shape = XLAT.shape
# ax.set_xticks(np.linspace(bm.lonmin, bm.lonmax, 12))
# labels = [item.get_text() for item in ax.get_xticklabels()]
# xlabels = np.arange(0, shape[1] * fs, shape[1] * fs / len(labels)).astype(int)
# ax.set_xticklabels(xlabels, fontsize=11)

# ax.set_yticks(np.linspace(bm.latmin, bm.latmax, 6))
# labels = [item.get_text() for item in ax.get_yticklabels()]
# ylabels = np.arange(0, shape[0] * fs, shape[0] * fs / len(labels)).astype(int)
# ax.set_yticklabels(ylabels, fontsize=11)
# ax.set_xlabel("West-East (m)", fontsize=12)
# ax.set_ylabel("South-North (m)", fontsize=12)
# ax.text(
#     0.015,
#     1.1,
#     "A)",
#     size=20,
#     color="k",
#     weight="bold",
#     zorder=10,
#     transform=plt.gca().transAxes,
# )


# def plotstuff(i):
#     col = cols[i]
#     # print(col)
#     ax = fig.add_subplot(9, 2, (i + 1) * 2)
#     indices = [ii for ii, s in enumerate(ros_ids) if col in s]
#     n = len(indices)
#     colors_default = iter(cm.tab20(np.linspace(0, 1, n)))
#     for i in indices:
#         # print(ros_ids[i])
#         i = i
#         c = next(colors_default)
#         modeld_ros = ros_da.isel(ros=i)
#         ax.plot(
#             modeld_ros.XTIME, normalize(modeld_ros), color=c, label=ros_ids[i], zorder=8
#         )
#         ax.plot(
#             ros_dfs[i].index.values,
#             normalize(ros_dfs[i].temp.values / 4),
#             color=c,
#             linestyle="--",
#             zorder=9,
#         )
#     ax.text(
#         0.02, 0.65, col, size=14, color="k", zorder=10, transform=plt.gca().transAxes
#     )
#     ax.set_ylim(0, 0.49)
#     if ("I" in modelrun) == True:
#         if col == "C2" or col == "C3":
#             print(col)
#             ax.scatter(
#                 pd.Timestamp(2019, 5, 11, 17, 49, 48),
#                 0.06,
#                 marker="*",
#                 # alpha=0.6,
#                 color="tab:red",
#                 zorder=10,
#                 label="Ignition Start Time",
#                 edgecolors="black",
#                 s=120,
#             )
#             ax.scatter(
#                 pd.Timestamp(2019, 5, 11, 17, 49, 56),
#                 0.06,
#                 marker="X",
#                 # alpha=0.6,
#                 color="tab:red",
#                 zorder=10,
#                 label="Ignition End Time",
#                 edgecolors="black",
#                 s=100,
#             )
#         elif col == "C4" or col == "C5":
#             ax.scatter(
#                 pd.Timestamp(2019, 5, 11, 17, 50, 7),
#                 0.06,
#                 marker="*",
#                 # alpha=0.6,
#                 color="tab:blue",
#                 zorder=10,
#                 label="Ignition Start Time",
#                 edgecolors="black",
#                 s=120,
#             )
#             ax.scatter(
#                 pd.Timestamp(2019, 5, 11, 17, 50, 17),
#                 0.06,
#                 marker="X",
#                 # alpha=0.6,
#                 color="tab:blue",
#                 zorder=10,
#                 label="Ignition End Time",
#                 edgecolors="black",
#                 s=100,
#             )

#         elif col == "C6" or col == "C7":
#             ax.scatter(
#                 pd.Timestamp(2019, 5, 11, 17, 50, 42),
#                 0.06,
#                 marker="*",
#                 # alpha=0.6,
#                 color="tab:green",
#                 zorder=10,
#                 label="Ignition Start Time",
#                 edgecolors="black",
#                 s=120,
#             )
#             ax.scatter(
#                 pd.Timestamp(2019, 5, 11, 17, 50, 57),
#                 0.06,
#                 marker="X",
#                 # alpha=0.6,
#                 color="tab:green",
#                 zorder=10,
#                 label="Ignition End Time",
#                 edgecolors="black",
#                 s=100,
#             )
#         elif col == "C8" or col == "C9":
#             ax.scatter(
#                 pd.Timestamp(2019, 5, 11, 17, 52, 49),
#                 0.06,
#                 marker="*",
#                 # alpha=0.6,
#                 color="tab:grey",
#                 zorder=10,
#                 label="Ignition Start Time",
#                 edgecolors="black",
#                 s=120,
#             )
#             ax.scatter(
#                 pd.Timestamp(2019, 5, 11, 17, 52, 55),
#                 0.06,
#                 marker="X",
#                 # alpha=0.6,
#                 color="tab:grey",
#                 zorder=10,
#                 label="Ignition End Time",
#                 edgecolors="black",
#                 s=100,
#             )
#         else:
#             pass

#     else:
#         ax.scatter(
#             pd.Timestamp(2019, 5, 11, 17, 49, 48),
#             0.06,
#             marker="*",
#             # alpha=0.6,
#             color="tab:red",
#             zorder=10,
#             label="Ignition Start Time",
#             edgecolors="black",
#             s=120,
#         )
#         ax.scatter(
#             pd.Timestamp(2019, 5, 11, 17, 51, 28),
#             0.06,
#             marker="X",
#             # alpha=0.6,
#             color="tab:red",
#             zorder=10,
#             label="Ignition End Time",
#             edgecolors="black",
#             s=100,
#         )
#     if col == "C9":
#         myFmt = DateFormatter("%H:%M:%S")
#         ax.xaxis.set_major_formatter(myFmt)
#         ax.tick_params(axis="x", labelrotation=20)
#         ax.set_xlabel("Datetime (HH:MM:SS)")
#     elif col == "C2":
#         ax.set_title(
#             f"Time Series of normalized \n observed temperature (dashed lines) and  \n normalized modeled heatflux (solid lines)",
#             fontsize=12,
#         )
#         ax.set_xticks([])
#         ax.text(
#             0.02,
#             1.2,
#             "B)",
#             size=20,
#             color="k",
#             weight="bold",
#             zorder=10,
#             transform=plt.gca().transAxes,
#         )
#     else:
#         ax.set_xticks([])

#     ax.tick_params(axis="both", which="major", labelsize=11)
#     colors_default = iter(cm.tab20(np.linspace(0, 1, n)))
#     for ind in indices:
#         c = next(colors_default)
#         ax.scatter(
#             ros[ros_ids[ind]]["lon"],
#             ros[ros_ids[ind]]["lat"],
#             zorder=9,
#             s=100,
#             color=c,
#             edgecolors="black",
#         )
#     ax.set_title(f"{title}", fontsize=18)
#     ax.annotate(
#         col,
#         xy=(ros[ros_ids[indices[0]]]["lon"], 55.71740),
#         color="w",
#         bbox=dict(boxstyle="circle", fc="black", ec="k", alpha=0.8),
#         ha="center",
#     )


# [plotstuff(i) for i in range(len(cols))]

# # # fig.tight_layout()
# # # plt.savefig(str(save_dir) + f"/ros-timeseries.png", dpi=300, bbox_inches="tight")


# # # fig = plt.figure(figsize=(6, 4))
# # # ax = fig.add_subplot(1, 1, 1)
# # # bm = Basemap(
# # #     llcrnrlon=XLONG[0, 0],
# # #     llcrnrlat=XLAT[0, 0],
# # #     urcrnrlon=XLONG[-1, -1],
# # #     urcrnrlat=XLAT[-1, -1],
# # #     epsg=4326,
# # #     ax=ax,
# # # )
# # # lats = np.array([ros[ros_ids[s]]["lat"] for s in range(len(ros_ids))])
# # # lons = np.array([ros[ros_ids[s]]["lon"] for s in range(len(ros_ids))])
# # # temps = np.array([ros_dfs[s].temp.iloc[0]/4 for s in range(len(ros_ids))])

# # # # generate grid data
# # # numcols, numrows = 240, 240
# # # xi = np.linspace(lons.min(), lons.max(), numcols)
# # # yi = np.linspace(lats.min(), lats.max(), numrows)
# # # xi, yi = np.meshgrid(xi, yi)

# # # zi = griddata((lons,lats),temps,(xi,yi),method='linear')
# # # polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=1)
# # # contour = ax.contourf(xi,yi,zi, norm = Cnorm,cmap ='YlOrRd', levels= v, extend = 'both')
# # # # scatter = ax.scatter(lons, lats, c= temps, zorder=10, s =40, vmin=0, vmax=200)
# # # cbar = plt.colorbar(contour, ax=ax, pad=0.06)
# # # cbar.ax.tick_params(labelsize=12)
# # # cbar.set_label(units, rotation=270, fontsize=16, labelpad=15)
# # # ax.set_title(f"{title} \n" + ros_dfs[0]['DateTime'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'), fontsize=18)

# # # def update_plot(i):
# # #     global contour
# # #     for c in contour.collections:
# # #         c.remove()
# # #     temps = [ros_dfs[s].temp.iloc[i]/4 for s in range(len(ros_ids))]
# # #     zi = griddata((lons,lats),temps,(xi,yi),method='linear')
# # #     contour = ax.contourf(xi,yi,zi, norm = Cnorm,cmap ='YlOrRd', levels= v, extend = 'both')
# # #     # scatter = ax.scatter(lons, lats, c= temps, zorder=10, s =40, vmin=0, vmax=200)
# # #     ax.set_title(f"{title} \n" + ros_dfs[0]['DateTime'].iloc[i].strftime('%Y-%m-%d %H:%M:%S'), fontsize=18)
# # #     return contour

# # # fig.tight_layout()
# # # ani = animation.FuncAnimation(fig, update_plot, dimT, interval=3)
# # # ani.save(str(save_dir) + f"/{modelrun}/{var}-ros.mp4", writer="ffmpeg", fps=10, dpi=300)
# # # plt.close()

# # %%
