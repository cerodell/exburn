import context
import wrf
import json
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from context import root_dir, data_dir, save_dir
from utils.sfire import makeLL
import matplotlib

matplotlib.rcParams.update({"font.size": 10})


domain = "met"
unit = "unit5"
fueltype = 6
var = "T2"
flux_filein = str(data_dir) + "/obs/heat_flux/"
flux_filein = sorted(Path(flux_filein).glob(f"*.xlsx"))

with open(str(data_dir) + "/json/config.json") as f:
    config = json.load(f)
hfs = config["unit5"]["obs"]["hf"]
hf_ids = list(hfs)

# wrf_ds = xr.open_dataset(str(data_dir) + f"/fuel{fueltype}/interp-unit5-temp.nc", chunks = 'auto')
wrf_ds = xr.open_dataset(
    str(data_dir) + f"/fuel{fueltype}/wrfout_d01_2019-05-11_17:49:11", chunks="auto"
)
times = wrf_ds.XTIME.values
var_da = wrf_ds[var] - 273.15
XLAT, XLONG = makeLL(domain)


def prepare_df(i):
    try:
        df = pd.read_csv(str(data_dir) + f"/obs/heat_flux/{hf_ids[i]}.csv")
        df["DateTime"] = pd.to_datetime(df["DateTime"], infer_datetime_format=True)
        df = df.set_index("DateTime")[str(times[0]) : str(times[-1])]
    except:
        flux_name = str(flux_filein[i]).rsplit("/", 1)[1][2:6]
        names = ["DateTime", "Times", "Temp_C", "Heat_Flux_kW_mA2"]
        df = pd.read_excel(
            str(flux_filein[i]),
            sheet_name="Raw Data",
            skiprows=16,
            engine="openpyxl",
            names=names,
            usecols=range(4),
        )
        df = df.dropna()
        df["DateTime"] = df["DateTime"].str[:-4]
        df["DateTime"] = df["DateTime"].replace(" ", "T", regex=True)
        df["DateTime"] = pd.to_datetime(df["DateTime"], infer_datetime_format=True)
        df.to_csv(
            str(data_dir) + f"/obs/heat_flux/{flux_name.lower()}.csv", index=False
        )
    return df


hf_dfs = [prepare_df(i) for i in range(len(hf_ids))]


try:
    ## try and open kdtree for domain
    fire_tree, fire_locs = pickle.load(
        open(str(data_dir) + f"/tree/fire-{unit}-{domain}.p", "rb")
    )
    print("Found Fire Tree")
except:
    print("Could not find Fire KDTree building....")
    fire_locs = pd.DataFrame(
        {"XLAT": XLAT.values.ravel(), "XLONG": XLONG.values.ravel()}
    )
    fire_tree = KDTree(fire_locs)
    ## save tree
    pickle.dump(
        [fire_tree, fire_locs],
        open(str(data_dir) + f"/tree/fire-{unit}-{domain}.p", "wb"),
    )
    print("KDTree built")


def find_index(hf):
    print(hf)
    hf = np.array([hfs[hf]["lat"], hfs[hf]["lon"]]).reshape(1, -1)
    hf_dist, hf_ind = fire_tree.query(hf, k=1)
    print(hf_dist)
    hf_loc = list(np.unravel_index(int(hf_ind), XLAT.shape))
    return hf_loc


hfs_loc = np.stack([find_index(hf) for hf in hf_ids])
y = xr.DataArray(np.array(hfs_loc[:, 0]), dims="hfs", coords=dict(hfs=hf_ids))
x = xr.DataArray(np.array(hfs_loc[:, 1]), dims="hfs", coords=dict(hfs=hf_ids))
if domain == "fire":
    hf_ds = var_da.sel(south_north_subgrid=y, west_east_subgrid=x)
elif domain == "met":
    hf_ds = var_da.sel(south_north=y, west_east=x)
else:
    raise ValueError("Not a valied domain option")


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 1, 1)  # top and bottom left
for i in range(len(hf_ids)):
    modeld_hf = hf_ds.isel(hfs=i)
    ax.plot(modeld_hf.XTIME, modeld_hf, color=colors[i], label=hf_ids[i])
    ax.plot(
        hf_dfs[i].index.values,
        hf_dfs[i].Temp_C.values / 4,
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
