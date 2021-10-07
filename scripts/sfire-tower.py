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

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from context import root_dir, data_dir, save_dir
from utils.sfire import makeLL


############## Define inputs ##############
domain = "met"
unit = "unit5"
fueltype = 6
var = "temp"
flux_filein = str(data_dir) + "/obs/met/"


############## Open datafile ##############
## open config file
with open(str(data_dir) + "/json/config.json") as f:
    config = json.load(f)
mets = config["unit5"]["obs"]["met"]
met_ids = list(mets)

## open wrf-sfire
# wrf_ds = xr.open_dataset(str(data_dir) + f"/fuel{fueltype}/wrfout_d01_2019-05-11_17:49:11", chunks = 'auto')
# times = wrf_ds.XTIME.values
# wrf_ds = xr.open_dataset(str(data_dir) + f"/fuel{fueltype}/interp-unit5-temp.nc", chunks = 'auto')

uvmet10 = wrf.getvar(
    Dataset(str(data_dir) + f"/fuel{fueltype}/wrfout_d01_2019-05-11_17:49:11"),
    "uvmet10_wspd_wdir",
)

# times = uvmet10.Time.values
var_da = wrf_ds[var].isel(interp_level=0) - 273.15
XLAT, XLONG = makeLL(domain)

## open 2d sonic tower data and kestral ops met data
def prepare_df(i):
    df = pd.read_csv(str(flux_filein) + f"{met_ids[i]}.csv")
    df["DateTime"] = pd.to_datetime(df["TIMESTAMP"], infer_datetime_format=True)
    df = df.set_index("DateTime")[str(times[0]) : str(times[-1])]
    return df


met_dfs = [prepare_df(i) for i in range(len(met_ids))]


############## Find location of met tower within model domain ##############
## open or create kdtree of domain if not found
try:
    ## try and open kdtree for domain
    tree, loc = pickle.load(open(str(data_dir) + f"/tree/{unit}-{domain}.p", "rb"))
    print("Found KDTree")
except:
    print("Could not find KDTree \n building....")
    loc = pd.DataFrame({"XLAT": XLAT.values.ravel(), "XLONG": XLONG.values.ravel()})
    tree = KDTree(loc)
    ## save tree
    pickle.dump([tree, loc], open(str(data_dir) + f"/tree/{unit}-{domain}.p", "wb"))
    print("KDTree built")


def find_index(met):
    print(met)
    met = np.array([mets[met]["lat"], mets[met]["lon"]]).reshape(1, -1)
    met_dist, met_ind = tree.query(met, k=1)
    met_loc = list(np.unravel_index(int(met_ind), XLAT.shape))
    return met_loc


mets_loc = np.stack([find_index(met) for met in met_ids])


y = xr.DataArray(np.array(mets_loc[:, 0]), dims="mets", coords=dict(mets=met_ids))
x = xr.DataArray(np.array(mets_loc[:, 1]), dims="mets", coords=dict(mets=met_ids))
if domain == "fire":
    met_ds = var_da.sel(south_north_subgrid=y, west_east_subgrid=x)
elif domain == "met":
    met_ds = var_da.sel(south_north=y, west_east=x)
else:
    raise ValueError("Not a valied domain option")


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 1, 1)  # top and bottom left
for i in range(len(met_ids)):
    modeld_met = met_ds.isel(mets=i)
    ax.plot(modeld_met.Time, modeld_met, color=colors[i], label=met_ids[i])
    ax.plot(
        met_dfs[i].index.values,
        met_dfs[i].Temp_C.values / 4,
        color=colors[i],
        linestyle="--",
    )
ax.scatter(
    pd.Timestamp(2019, 5, 11, 17, 49, 3),
    0,
    marker="*",
    color="red",
    zorder=10,
    label="Ignition Time",
    s=100,
)

plt.legend()
