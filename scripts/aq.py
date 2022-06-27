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
from utils.sfire import makeLL
import matplotlib as mpl


##################### Define Inputs and File Directories ###################
modelrun = "F6V51M08Z22FIRE"
configid = "F6V51"
domain = "met"
var = "tr17_1"
var = "fire_smoke"

# pm_ef = 21.05  # boreal wildfire emission factor Urbanski (2014)
pm_ef = 10.400
# levels = np.arange(0, 1200.0, 10)
a = np.arange(1, 10)
b = 10 ** np.arange(4)
levels = (b[:, np.newaxis] * a).flatten()
print(levels)
cmap = mpl.cm.cubehelix_r
norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend="both")

fireshape_path = str(data_dir) + "/unit_5/unit_5"
aqsin = str(data_dir) + "/obs/aq/"
aqsin = sorted(Path(aqsin).glob(f"*"))
save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)

# a = np.arange(1,14)
# b = 10**np.arange(4)
# levels = (b[:, np.newaxis] * a).flatten()


################### Open Datsets ###################
## Open COnfig File and Get Relevant Paramters
with open(str(root_dir) + "/json/config.json") as f:
    config = json.load(f)
aqs = config["unit5"]["obs"]["aq"]
bounds = config["unit5"]["sfire"][configid]
south_north = slice(bounds[domain]["sn"][0], bounds[domain]["sn"][1])
west_east = slice(bounds[domain]["we"][0], bounds[domain]["we"][1])
south_north = slice(110, 300, None)
west_east = slice(30, 129, None)

## open wrf-sfire simulation
wrf_ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
wrf_ds[var] = wrf_ds[var] / pm_ef
wrf_ds = wrf_ds.sel(
    south_north=south_north,
    west_east=west_east,
    # Time=slice(3, 50),
)
## create Lat and Long array based on wrf-sfire configuration
XLAT, XLONG = makeLL(domain, configid)
XLAT = XLAT.sel(south_north=south_north, west_east=west_east)
XLONG = XLONG.sel(south_north=south_north, west_east=west_east)


## get heights from wrf-simulation
ncfile = Dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
height = wrf.getvar(ncfile, "height")
height = height.values[:, 0, 0]
print(np.round(height))

## make datatime array
times = wrf_ds.XTIME.values

## finally open AQ observations Datasets to compare to wrf-sfire simulation
def prep_aqs(aqin):
    """
    function open/prepares aq obs csv files to be
    compatable with wrf-sfire simulation by reformating datetime
    """
    # print(str(aqin))
    aq_id = list(pd.read_csv(str(aqin), skiprows=2))[1]
    df = pd.read_csv(str(aqin), skiprows=3).iloc[1:]
    str_list = ["year", "month", "day", "hour", "minute", "second"]
    float_list = [x for x in list(df) if x not in str_list]
    df[float_list] = df[float_list].astype(float)
    try:
        df["datetime"] = pd.to_datetime(df[str_list], format="%y%m%d")
    except:
        df["year"], df["month"] = "20" + df["year"], "0" + df["month"]
        df["datetime"] = pd.to_datetime(df[str_list], format="%y%m%d")
    # df = df.set_index('datetime').resample('10S').mean()
    df = df.set_index("datetime")
    pm25 = df.filter(regex="PM2.5").mean(axis=1)
    df["pm25"] = pm25
    pm_max = np.max(pm25)
    pm_min = np.min(pm25)
    df = df[str(times[0]) : str(times[-1])]
    arg_max = np.argmax(df["pm25"])

    return df, aq_id, pm_max, pm_min, arg_max


## loop all csv file open prepare and store sensor ID and max min obs values
aq_ddxy, aq_ids, pm_maxs, pm_mins, arg_maxs = [], [], [], [], []
for aqin in aqsin:
    df, aq_id, pm_max, pm_min, arg_max = prep_aqs(aqin)
    aq_ddxy.append(df)
    aq_ids.append(aq_id)
    pm_maxs.append(pm_max)
    pm_mins.append(pm_min)
    arg_maxs.append(arg_max)


################### Find Nearest Model Grid to AQ Observations ###################
## create dataframe with columns of all XLAT/XLONG
wrf_locs = pd.DataFrame({"XLAT": XLAT.values.ravel(), "XLONG": XLONG.values.ravel()})
## build kdtree
wrf_tree = KDTree(wrf_locs)
print("WRF Domain KDTree built")


def find_index(aq):
    print(aq)
    aq = np.array([aqs[aq]["lat"], aqs[aq]["lon"]]).reshape(1, -1)
    aq_dist, aq_ind = wrf_tree.query(aq, k=1)
    aq_loc = list(np.unravel_index(int(aq_ind), XLAT.shape))
    return aq_loc


aqs_loc = np.stack([find_index(aq) for aq in aq_ids])
y = xr.DataArray(np.array(aqs_loc[:, 0]), dims="aqs", coords=dict(aqs=aq_ids))
x = xr.DataArray(np.array(aqs_loc[:, 1]), dims="aqs", coords=dict(aqs=aq_ids))
colors_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# colors = iter(cm.tab20(np.linspace(0, 1, len(aq_ids))))

## Index wrf-sfire simulation for nearest model grid to aq sensors
aq_ds = wrf_ds[var].sel(south_north=y, west_east=x).isel(bottom_top=0)


## Find boundary layer height
# zstep = 10.
# interpz =np.arange(0,2000,zstep)
# temp = wrf_ds['T'].values[:,:,:,:] + 300
# T0 = np.mean(temp,(0,2,3))
# interpfLES= interp1d(height, T0,fill_value='extrapolate')
# soundingLES = interpfLES(interpz)

# gradTLES = (soundingLES[1:] - soundingLES[:-1])/zstep
# gradT2 = gradTLES[1:] - gradTLES[:-1]
# drop = 10
# ziidx = np.argmax(gradT2[drop:]) + drop
# zi = interpz[ziidx]
# print(zi)
zi = 2400.0


################### Plot AQ Simulation to Observed Values ###################
# %%
fig = plt.figure(figsize=(10, 5.8))

# fig.suptitle(modelrun)
## make subplot for timeseries of smoke at each AQ station compared to modeled smoke values
ax = fig.add_subplot(2, 2, 4)
modeld_aq = aq_ds.isel(aqs=2)
time_max_tr17_1 = np.argmax(modeld_aq.values)


idx_max, value_max = [], []
for i in range(len(aq_ids)):
    # c = next(colors)
    modeld_aq = aq_ds.isel(aqs=i)
    idx_max.append(np.argmax(modeld_aq.values))
    value_max.append(np.max(modeld_aq.values))
    ax.plot(modeld_aq.XTIME, modeld_aq, color=colors_list[i], label=aq_ids[i])
    ax.plot(
        aq_ddxy[i].index.values,
        aq_ddxy[i]["pm25"].values,
        color=colors_list[i],
        linestyle="--",
    )
ax.scatter(
    pd.Timestamp(2019, 5, 11, 17, 49, 48),
    0,
    marker="*",
    color="tab:red",
    zorder=10,
    label="Ignition Time",
    edgecolors="black",
    s=100,
)
ax.set_title(
    f"Time Series of PM2.5 Concentration \n Dash Line: Observed and Solid Line: Modeled",
    fontsize=10,
)
ax.set_ylabel("PM2.5 \n" + r"($\mathrm{~μg} \mathrm{~m}^{-3}$)", fontsize=10)
ax.set_xlabel("DateTime (HH:MM:SS)", fontsize=10)
myFmt = DateFormatter("%H:%M:%S")
ax.xaxis.set_major_formatter(myFmt)
ax.tick_params(axis="x", labelrotation=20)
ax.text(
    0.015,
    1.15,
    "C)",
    size=16,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)

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
## add unit boundary and aq station location to map
polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=10)

colors = iter(cm.cool(np.linspace(0, 1, len(aq_ids))))
for i in range(len(aq_ids)):
    # c = next(colors)
    ax_map.scatter(
        aqs[aq_ids[i]]["lon"],
        aqs[aq_ids[i]]["lat"],
        zorder=10,
        label=aq_ids[i],
        color=colors_list[i],
        edgecolors="black",
        s=100,
    )

tr17_1 = wrf_ds[var].sum(dim=["bottom_top"]).isel(Time=idx_max[np.argmax(value_max)])
tiletime = str(times[idx_max[np.argmax(value_max)]])[:-10]

off_time = (
    times[idx_max[np.argmax(value_max)]]
    - aq_ddxy[np.argmax(value_max)].index[arg_maxs[np.argmax(value_max)]]
).total_seconds()
off_mag = value_max[np.argmax(value_max)] - pm_maxs[np.argmax(value_max)]
print(
    f"Time difference of Modeled to Observed max PM2.5 concentrations {off_time} seconds"
)
print(f"Values difference of Modeled to Observed max PM2.5 concentrations {off_mag}")


ax_map.set_title(f"Vertically Integrated Smoke \n at {tiletime}", fontsize=10)
contour = ax_map.contourf(
    XLONG,
    XLAT,
    tr17_1,
    zorder=1,
    levels=levels,
    cmap=cmap,
    norm=norm,
    extend="max",  # cubehelix_r
)

cbar = plt.colorbar(contour, ax=ax_map, pad=0.04, location="left")
cbar.ax.tick_params(labelsize=10)
cbar.set_label(
    "PM2.5 Concentration  \n" + r"($\mathrm{~μg} \mathrm{~m}^{-3}$)",
    rotation=90,
    fontsize=10,
    labelpad=15,
)

shape = XLAT.shape
dxy = 25
test = wrf_ds.west_east * 25
# ax_map.set_xticks(test)
ax_map.set_xticks(np.linspace(bm.lonmin, bm.lonmax, 5))
labels = [item.get_text() for item in ax_map.get_xticklabels()]
xlabels = np.arange(0, shape[1] * dxy, shape[1] * dxy / len(labels)).astype(int)
xlabels = [0, 500, 1000, 1500, 2000]
ax_map.set_xticklabels(xlabels)

ax_map.set_yticks(np.linspace(bm.latmin, bm.latmax, 5))
labels = [item.get_text() for item in ax_map.get_yticklabels()]
ylabels = np.arange(0, shape[0] * dxy, shape[0] * dxy / len(labels)).astype(int)
ylabels = [0, 1000, 2000, 3000, 4000]
ax_map.set_yticklabels(ylabels)
ax_map.yaxis.tick_right()
ax_map.yaxis.set_label_position("right")
ax_map.set_xlabel("West-East (m)", fontsize=10)
ax_map.set_ylabel("South-North (m)", fontsize=10)
ax_map.text(
    0.015,
    1.05,
    "A)",
    size=16,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)
plt.legend()


# make subplot for cross section of smoke
ax = fig.add_subplot(2, 2, 2)
tr17_1 = wrf_ds[var].sum(dim=["west_east"]).isel(Time=idx_max[np.argmax(value_max)])
sn = tr17_1.south_north * 25
ax.set_title(f"Cross Wind Integrated Smoke \n at {tiletime}", fontsize=10)
contour = ax.contourf(
    sn, height, tr17_1, zorder=1, levels=levels, cmap=cmap, norm=norm, extend="max"
)
# ax.axhline(y=zi, color='grey', linestyle='--', lw= 0.5, alpha = 0.6)
# for i in range(len(aq_ids)):
yy = y.values
# ax.axvline(x=sn[yy[0]], color='grey', linestyle='--', lw= 0.5, alpha = 0.6)

ax.set_ylabel("Vertical Height \n (m)", fontsize=10)
ax.set_xlabel("Horizontal (m)", fontsize=10)
ax.tick_params(axis="both", which="major", labelsize=8)
ax.text(
    0.015,
    1.1,
    "B)",
    size=16,
    color="k",
    weight="bold",
    zorder=10,
    transform=plt.gca().transAxes,
)
# ax.set_ylim(0, 100)
ax.set_xlim(0, 4000)
# cbar = plt.colorbar(contour, ax=ax, pad=0.01)
# cbar.ax.tick_params(labelsize=10)
# cbar.set_label("g kg^-1", rotation=270, fontsize=8, labelpad=15)

fig.tight_layout()
plt.savefig(str(save_dir) + f"/smoke-aq-comparison.png", dpi=300, bbox_inches="tight")

# # aq2m_ds = aq_ds.isel(z = 0)
