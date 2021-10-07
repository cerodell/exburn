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


df = pd.read_csv(str(data_dir) + f"/obs/ros/ROS_GPSLocations_Time.csv")

conig = {}
for i in range(len(df)):
    top = df["Time"][i]
    if top == "error":
        top == "99"
    else:
        pass
    conig.update(
        {
            df["ident"][i]: {
                "lat": df["lat"][i],
                "lon": df["long"][i],
                "top": df["Time"][i],
            }
        }
    )


# domain = 'fire'
# unit = 'unit5'
# fueltype = 6
# var = 'FGRNHFX'
# flux_filein = str(data_dir) + '/obs/heat_flux/'
# flux_filein = sorted(Path(flux_filein).glob(f"*.xlsx"))

# with open(str(data_dir) + "/json/config.json") as f:
#     config = json.load(f)
# hfs = config["unit5"]["obs"]["hf"]
# hf_ids = list(hfs)

# # wrf_ds = xr.open_dataset(str(data_dir) + f"/fuel{fueltype}/interp-unit5-temp.nc", chunks = 'auto')
# wrf_ds = xr.open_dataset(str(data_dir) + f"/fuel{fueltype}/wrfout_d01_2019-05-11_17:49:11", chunks = 'auto')
# times = wrf_ds.XTIME.values
# var_da = wrf_ds[var] -273.15
# XLAT, XLONG = makeLL(domain)
