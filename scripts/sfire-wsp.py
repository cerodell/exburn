import context

# import wrf
import json
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap
from context import root_dir, data_dir, save_dir
from utils.sfire import makeLL

##################### Define Inputs and File Directories ###################
modelrun = "F6V51M08Z22"
configid = "F6V51"
domain = "met"
levels = np.arange(0, 10.0, 0.1)
colors = [
    "#FFFFFF",
    "#BBBBBB",
    "#646464",
    "#1563D3",
    "#2883F1",
    "#50A5F5",
    "#97D3FB",
    "#0CA10D",
    "#37D33C",
    "#97F58D",
    "#B5FBAB",
    "#FFE978",
    "#FFC03D",
    "#FFA100",
    "#FF3300",
    "#C10000",
    "#960007",
    "#643C32",
]
cmap = LinearSegmentedColormap.from_list("meteoblue", colors, N=len(levels))

fireshape_path = str(data_dir) + "/unit_5/unit_5"
met_filein = str(data_dir) + "/obs/met/"
save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)


################### Open Datsets ###################
## Open COnfig File and Get Relevant Paramters
with open(str(root_dir) + "/json/config.json") as f:
    config = json.load(f)
mets = config["unit5"]["obs"]["met"]
met_ids = list(mets)
bounds = config["unit5"]["sfire"][configid]
south_north = slice(bounds[domain]["sn"][0], bounds[domain]["sn"][1])
west_east = slice(bounds[domain]["we"][0], bounds[domain]["we"][1])
# south_north = slice(100, 300, None)
# west_east = slice(30, 129, None)
south_north = slice(100, 140, None)
west_east = slice(60, 90, None)
south_north_subgrid = slice(555, 610, None)
west_east_subgrid = slice(335, 400, None)


wrf_ds = xr.open_dataset(
    str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11", chunks="auto"
)
var_da = wrf_ds["GRNHFX"]
var_da = var_da.sel(
    # south_north_subgrid=south_north_subgrid,
    # west_east_subgrid=west_east_subgrid,
    # south_north=south_north,
    # west_east=west_east,
    Time=slice(3, 50),
)

# var_da = var_da.sum(dim='west_east_subgrid').max('south_north_subgrid')
# var_da = var_da.sum(dim='west_east').max('south_north')
# var_da = var_da.mean(dim=['west_east_subgrid','south_north_subgrid'])
# var_da = var_da.max(dim=['west_east','south_north'])
var_da = var_da.sum(dim=["west_east", "south_north"])
# var_da = var_da.sum(dim='west_east').max('south_north')


wrf_ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/wspd_wdir.nc")
wrf_ds = wrf_ds.sel(
    south_north=south_north,
    west_east=west_east,
).isel(Time=slice(3, 50))


times = wrf_ds.Time.values

## get heights from wrf-simulation
ncfile = Dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
height_array = wrf.getvar(ncfile, "height")
height_array = height_array.sel(south_north=south_north, west_east=west_east)
height = height_array.values[:, 0, 0]
# print(height)
H = 0.762 * 3.28084
WAH = 1.83 / np.log((20 + 0.36 * H) / (0.13 * H))
MFH = WAH * 6.1


## get Lat and Long array wrf-sfire configuration
XLAT, XLONG = wrf_ds.XLAT.values, wrf_ds.XLONG.values


## open 2d sonic tower data and kestral ops met data
def prepare_df(i):
    df = pd.read_csv(str(met_filein) + f"{'south_met'}.csv")
    df["DateTime"] = pd.to_datetime(df["TIMESTAMP"], infer_datetime_format=True)
    df = df.set_index("DateTime")[str(times[0]) : str(times[-1])]
    return df


met_dfs = [prepare_df(i) for i in range(len(met_ids))]


############## Find location of met tower within model domain ##############
loc = pd.DataFrame({"XLAT": XLAT.ravel(), "XLONG": XLONG.ravel()})
tree = KDTree(loc)
print("KDTree built")


def find_index(met):
    print(met)
    met = np.array([mets[met]["lat"], mets[met]["lon"]]).reshape(1, -1)
    met_dist, met_ind = tree.query(met, k=1)
    # print(met_dist)
    met_loc = list(np.unravel_index(int(met_ind), XLAT.shape))
    return met_loc


mets_loc = np.stack([find_index(met) for met in met_ids])


y = xr.DataArray(np.array(mets_loc[:, 0]), dims="mets", coords=dict(mets=met_ids))
x = xr.DataArray(np.array(mets_loc[:, 1]), dims="mets", coords=dict(mets=met_ids))
if domain == "fire":
    met_ds = wrf_ds.sel(south_north_subgrid=y, west_east_subgrid=x)
elif domain == "met":
    met_ds = wrf_ds.sel(south_north=y, west_east=x)
else:
    raise ValueError("Not a valied domain option")


# find boundary layer height
interpz = np.arange(0, 4000, 6.15)

try:
    wsp = np.loadtxt(str(data_dir) + f"/{modelrun}/wsp.txt", dtype=float)
    wdir = np.loadtxt(str(data_dir) + f"/{modelrun}/wdir.txt", dtype=float)
except:
    wsp, wdir = [], []
    for i in range(len(times)):
        wsp_i = met_ds["wspd"].isel(mets=0, Time=i)
        interpfLES = interp1d(height, wsp_i, fill_value="extrapolate")
        soundingLES = interpfLES(interpz)
        wsp.append(soundingLES[1])
        wdir_i = met_ds["wdir"].isel(mets=0, Time=i)
        interpfLES = interp1d(height, wdir_i, fill_value="extrapolate")
        soundingLES = interpfLES(interpz)
        wdir.append(soundingLES[1])
    np.savetxt(str(data_dir) + f"/{modelrun}/wsp.txt", wsp, fmt="%1.8f")
    np.savetxt(str(data_dir) + f"/{modelrun}/wdir.txt", wdir, fmt="%1.2f")

timeofint = np.argmax(wsp)
wsp3d = wrf_ds["wspd"].isel(Time=timeofint)
ua3d = wrf_ds["ua"].isel(Time=timeofint)
va3d = wrf_ds["va"].isel(Time=timeofint)

wsp615cm = wrf.interplevel(wsp3d, height_array, 6.15)
ua615cm = wrf.interplevel(ua3d, height_array, 6.15)
va615cm = wrf.interplevel(va3d, height_array, 6.15)
################### Plot met Simulation to Observed Values ###################

fig = plt.figure(figsize=(10, 4))  # (Width, height) in inches.
# fig.suptitle(modelrun)
## make subplot for timeseries of smoke at each met station compared to modeled smoke values
# ax = fig.add_subplot(1, 2, 2)
ax = fig.add_subplot(2, 2, 2)
ax_hx = ax.twinx()
ax_hx.plot(var_da.XTIME, var_da / 1000, color="tab:red", lw=0.8, zorder=8)
ax_hx.set_ylabel(
    "Fire Integrated  \n Heat Flux  \n " + r"$(\mathrm{~kW} \mathrm{~m}^{-2})$" + "\n",
    fontsize=9,
    color="tab:red",
    rotation=-90,
    labelpad=40,
)
ax_hx.tick_params(axis="y", colors="tab:red")

modeld_met = met_ds.isel(mets=0)
df_met = met_dfs[0]
ax.plot(modeld_met.Time, wdir, label="Modeled", color="tab:red", lw=1.2)
ax.plot(modeld_met.Time, wdir, label="Modeled", color="tab:blue", lw=1.5)
ax.plot(
    df_met.index,
    df_met["WindDir"],
    label="Observed",
    color="tab:blue",
    lw=1.5,
    linestyle="--",
)
ax.set_ylabel("Wind Direcation \n" + r"($degs$)", fontsize=9, color="tab:blue")

ax.tick_params(axis="y", colors="tab:blue")
ax.set_xticklabels([])
ax.axvline(times[timeofint], color="k", linestyle="--", lw=0.4)
ax.scatter(
    pd.Timestamp(2019, 5, 11, 17, 49, 48),
    64,
    marker="*",
    color="tab:red",
    zorder=10,
    label="Ignition Time",
    edgecolors="black",
    s=100,
)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.34),
    ncol=4,
    fancybox=True,
    shadow=True,
    prop={"size": 7},
)
ax.text(
    0.015,
    1.35,
    "B)",
    size=12,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)
ax.set_title(
    f"Time Series of Wind Speed and Direction \n at Met Tower 6.15 m Above Ground Level \n \n",
    fontsize=10,
)
ax = fig.add_subplot(2, 2, 4)


modeld_met = met_ds.isel(mets=0)
df_met = met_dfs[0]
ax.plot(
    modeld_met.Time,
    wsp,
    color="tab:blue",
    lw=1.5,
)
ax.plot(df_met.index, df_met[" WS_ms"], color="tab:blue", lw=1.5, linestyle="--")
ax.set_ylabel(
    "Wind Speed \n" + r"($\mathrm{~m} \mathrm{~s}^{-1}$)" + "\n",
    fontsize=9,
    color="tab:blue",
)
ax.tick_params(axis="y", colors="tab:blue")
ax.set_xlabel("DateTime (HH:MM:SS)", fontsize=10)
myFmt = DateFormatter("%H:%M:%S")
ax.xaxis.set_major_formatter(myFmt)
ax.tick_params(axis="x", labelrotation=20)
ax.axvline(times[timeofint], color="k", linestyle="--", lw=0.4)
ax.scatter(
    pd.Timestamp(2019, 5, 11, 17, 49, 48),
    0.4,
    marker="*",
    color="tab:red",
    zorder=10,
    edgecolors="black",
    s=100,
)

ax_hx = ax.twinx()
ax_hx.plot(var_da.XTIME, var_da / 1000, color="tab:red", lw=0.8, zorder=8)
ax_hx.set_ylabel(
    "Fire Integrated \n Heat Flux \n" + r"$(\mathrm{~kW} \mathrm{~m}^{-2})$" + "\n",
    fontsize=9,
    color="tab:red",
    rotation=-90,
    labelpad=40,
)
ax_hx.tick_params(axis="y", colors="tab:red")

## make subplot for top view map of smoke at lowest model grid levels
ax_map = fig.add_subplot(1, 2, 1)
bm = Basemap(
    llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1],
    epsg=4326,
    ax=ax_map,
)
## add unit boundary and met station location to map
polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=10)
tiletime = str(times[timeofint])[:-10]
ax_map.set_title(
    f"Wind Speed and Direction at \n 6.15 m Above Ground Level \n at {tiletime}",
    fontsize=10,
)
colors_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
ax_map.scatter(
    mets["south_met"]["lon"],
    mets["south_met"]["lat"],
    zorder=10,
    label="Met Tower",
    edgecolors="black",
    color="grey",
    s=100,
)

contour = ax_map.contourf(
    XLONG, XLAT, wsp615cm, zorder=1, extend="max", levels=levels, cmap=cmap
)
ax_map.streamplot(
    XLONG,
    XLAT,
    ua615cm.values,
    va615cm.values,
    zorder=9,
    color="k",
    linewidth=0.3,
    arrowsize=0.5,
    density=1.4,
)

cbar = plt.colorbar(contour, ax=ax_map, pad=0.04, location="left")
cbar.ax.tick_params(labelsize=10)
cbar.set_label(
    "Wind Speed \n" + r"($\mathrm{~m} \mathrm{~s}^{-1}$)",
    rotation=90,
    fontsize=10,
    labelpad=15,
)

shape = XLAT.shape
dxy = 25
ax_map.set_xticks(np.linspace(bm.lonmin, bm.lonmax, int(len(wsp615cm.west_east) / 3)))
labels = [item.get_text() for item in ax_map.get_xticklabels()]
xlabels = np.arange(0, shape[1] * dxy, shape[1] * dxy / len(labels)).astype(int)
# xlabels = [0, 100, 200, 300, 400, 500, 600, 700]
ax_map.set_xticklabels(xlabels)

ax_map.set_yticks(np.linspace(bm.latmin, bm.latmax, int(len(wsp615cm.south_north) / 4)))
labels = [item.get_text() for item in ax_map.get_yticklabels()]
ylabels = np.arange(0, shape[0] * dxy, shape[0] * dxy / len(labels)).astype(int)
ax_map.set_yticklabels(ylabels)
ax_map.yaxis.tick_right()
ax_map.yaxis.set_label_position("right")
ax_map.set_xlabel("West-East (m)", fontsize=10)
ax_map.set_ylabel("South-North (m)", fontsize=10)
legend = ax_map.legend(
    loc="upper right",
    ncol=1,
    fancybox=True,
    shadow=True,
)
ax_map.text(
    0.005,
    1.05,
    "A)",
    size=12,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)
legend.set_zorder(102)
legend.get_frame().set_facecolor("w")
fig.tight_layout()
plt.savefig(str(save_dir) + f"/wsp_wdir-comparison.png", dpi=250, bbox_inches="tight")
