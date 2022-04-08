import context
import wrf
import json
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import geopandas as gpd
from pathlib import Path
from netCDF4 import Dataset
from scipy.interpolate import interp1d
from sklearn.neighbors import KDTree
from matplotlib.pyplot import cm

import glob

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
hfx_filein = str(data_dir) + "/obs/heat_flux/"
save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)


################### Open Datsets ###################
## Open COnfig File and Get Relevant Paramters
with open(str(root_dir) + "/json/config.json") as f:
    config = json.load(f)
mets = config["unit5"]["obs"]["met"]
hfxs = config["unit5"]["obs"]["hfx"]
met_ids = list(mets)
hfx_ids = list(hfxs)

south_north = slice(100, 140, None)
west_east = slice(60, 89, None)
south_north_subgrid = slice(555, 610, None)
west_east_subgrid = slice(335, 400, None)
time_slice = slice(14, 60)

wrf_ds = xr.open_dataset(
    str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11", chunks="auto"
)
hfx_ds = wrf_ds["GRNHFX"]
hfx_ds = hfx_ds.sel(
    south_north=south_north,
    west_east=west_east,
    Time=time_slice,
)

hfx_da = hfx_ds.sum(dim="west_east").max("south_north")

wrf_ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/wspd_wdir.nc")
wrf_ds = wrf_ds.sel(
    south_north=south_north,
    west_east=west_east,
).isel(Time=time_slice)


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


def prepare_df(hfx):
    print(hfx)
    df = pd.read_csv(hfx_filein + f"{hfx.lower()}.csv")
    df["DateTime"] = [x[:-4] for x in df["DateTime"].values]
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["HFX"] = df["HeatFlux (kW/m^2)"].astype(float)
    df = df.set_index("DateTime")
    df = df[~df.index.duplicated(keep="first")]
    upsampled = df.resample("1S")
    df = upsampled.interpolate(method="linear")
    df = df[str(times[0]) : str(times[-1])]
    df["DateTime"] = pd.to_datetime(df.index)
    return df


hfx_dfs = [prepare_df(s) for s in hfxs]


met_df = pd.read_csv(str(met_filein) + "south_met.csv")
met_df["DateTime"] = pd.to_datetime(met_df["TIMESTAMP"], infer_datetime_format=True)
met_df = met_df.set_index("DateTime")[str(times[0]) : str(times[-1])]


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


############# Find locations of heatflux sensors within model domain ##############
def find_index(hfx):
    print(hfx)
    hfx = np.array([hfxs[hfx]["lat"], hfxs[hfx]["lon"]]).reshape(1, -1)
    hfx_dist, hfx_ind = tree.query(hfx, k=1)
    # print(met_dist)
    hfx_loc = list(np.unravel_index(int(hfx_ind), XLAT.shape))
    return hfx_loc


hfxs_loc = np.stack([find_index(hfx) for hfx in hfxs])

y = xr.DataArray(np.array(hfxs_loc[:, 0]), dims="hfxs", coords=dict(hfxs=hfx_ids))
x = xr.DataArray(np.array(hfxs_loc[:, 1]), dims="hfxs", coords=dict(hfxs=hfx_ids))
if domain == "fire":
    hfx_ds = hfx_ds.sel(south_north_subgrid=y, west_east_subgrid=x)
elif domain == "met":
    hfx_ds = hfx_ds.sel(south_north=y, west_east=x)
else:
    raise ValueError("Not a valied domain option")

fig = plt.figure(figsize=(14, 4))  # (Width, height) in inches.
colors_default = iter(cm.Set1(np.linspace(0, 1, len(hfx_dfs) + 5)))
ax = fig.add_subplot(1, 1, 1)
hfx_list = hfx_ds.hfxs.values
for i in range(len(hfx_dfs)):
    c = next(colors_default)
    ax.plot(hfx_dfs[i].index, np.abs(hfx_dfs[i]["HFX"]), color=c, linestyle="dotted")
    ax.plot(
        hfx_ds.isel(hfxs=i).XTIME,
        hfx_ds.isel(hfxs=i) / 1000,
        color=c,
        label=hfx_list[i],
    )


colors_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
hfx_list = hfx_ds.hfxs.values

obs_max, mod_max = [], []
for i in range(len(hfx_dfs)):
    obs_max.append(np.max(hfx_dfs[i]["HFX"].values))
    mod_max.append(np.max(hfx_ds.isel(hfxs=i).values / 1000))

obs_max = np.array(obs_max).flatten()
obs_max_str = np.full(obs_max.shape, "Observation")
mod_max = np.array(mod_max).flatten()
mod_max_str = np.full(mod_max.shape, "Modeled")
max_hfx = np.concatenate((obs_max, mod_max), axis=None)
max_hfx_str = np.concatenate((obs_max_str, mod_max_str), axis=None)
max_hue = np.full(max_hfx.shape, "Max")


obs, mod = [], []
for i in range(len(hfx_dfs)):
    obs.append(np.abs(hfx_dfs[i]["HFX"].values))
    mod.append(hfx_ds.isel(hfxs=i).values / 1000)
    # obs.append(np.mean(np.abs(hfx_dfs[i]['HFX'].values)))
    # mod.append(np.mean(hfx_ds.isel(hfxs= i).values/1000))


obs = np.array(obs).flatten()
# obs = np.mean(obs)
obs_str = np.full(obs.shape, "Observation")
mod = np.array(mod).flatten()
# mod = np.mean(mod)
mod_str = np.full(mod.shape, "Modeled")

mean_hfx = np.concatenate((obs, mod), axis=None)
mean_hfx_str = np.concatenate((obs_str, mod_str), axis=None)
mean_hue = np.full(mean_hfx.shape, "Average")

all_hfx = np.concatenate((max_hfx, mean_hfx), axis=None)
all_hfx_str = np.concatenate((max_hfx_str, mean_hfx_str), axis=None)
all_hue = np.concatenate((max_hue, mean_hue), axis=None)

# d = dict( HeatFlux = all_hfx, Method = all_hfx_str, Type= all_hue)
# plot_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))

d = {
    "Heat Flux Mean": mean_hfx,
    "Mean": mean_hfx_str,
    "Heat Flux Max": max_hfx,
    "Max": max_hfx_str,
}
plot_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))

sns.set_theme(style="ticks", palette="pastel")
# fig, ax = plt.subplots(1, 2, figsize=(12,4))
# # Draw a nested boxplot to show bills by day and time
# g1 =sns.boxplot(x="Heat Flux Mean", y="Mean",
#             palette=["tab:green", "tab:blue"],
#             width=0.25,
#             ax=ax[0],
#             data=plot_df)
# g1.set(yticklabels=[])  # remove the tick labels
# g1.set(title='Mean')  # add a title
# g1.set(ylabel=None)  # remove the axis label
# g1.tick_params(left=False)  # remove the ticks
# ax[0].set_xlabel('Heatflux')

# g2 = sns.boxplot(x="Heat Flux Max", y="Max",
#             palette=["tab:green", "tab:blue"],
#             width=0.5,
#             hue='Max',
#             ax=ax[1],
#             data=plot_df)

# g2.set(yticklabels=[])
# g2.set(title='Max')
# g2.set(ylabel=None)
# g2.tick_params(left=False)  # remove the ticks
# ax[1].set_xlabel('Heatflux')
# ax[1].legend(
#     loc="upper center",
#     bbox_to_anchor=(-0.1, 1.24),
#     ncol=4,
#     fancybox=True,
#     shadow=True,
#     # prop={'size': 7}
# )
# # sns.despine(offset=0, trim=True)


fig = plt.figure(figsize=(6, 4))  # (Width, height) in inches.
ax = fig.add_subplot(111)
g2 = sns.boxplot(
    y="Heat Flux Max",
    x="Max",
    palette=["tab:green", "tab:blue"],
    width=0.5,
    # hue='Max',
    data=plot_df,
)

g2.set(title="Maximum Heat Flux")
ax.set_ylabel("Heat Flux \n" + r"$(\mathrm{~kW} \mathrm{~m}^{-2})$")
ax.set_xlabel("")
plt.savefig(str(save_dir) + f"/hfx-comparison.png", dpi=300, bbox_inches="tight")

# ax.legend(
#     loc="upper center",
#     bbox_to_anchor=(0.5, 1.24),
#     ncol=4,
#     fancybox=True,
#     shadow=True,
#     # prop={'size': 7}
# )
# sns.despine(offset=0, trim=True)


# plt.xlim(0, 200)
# plt.legend(
#     loc="upper center",
#     bbox_to_anchor=(0.45, 1.15),
#     ncol=4,
#     fancybox=True,
#     shadow=True,
#     # prop={'size': 7}
# )

# fig, ax = plt.subplots()
# pos = np.arange(len(heatflux)) + 1
# bp = ax.boxplot(heatflux, sym='k+', positions=pos,
#                 notch=1, bootstrap=5000
#                 )
