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
from matplotlib import animation
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from context import root_dir, vol_dir, data_dir, save_dir, gog_dir
import matplotlib.pylab as pylab
from utils.sfire import makeLL


ds_6 = xr.open_dataset(str(data_dir) + f"/fuel{6}/wrfout_d01_2019-05-11_17:49:11")
ds_10 = xr.open_dataset(str(data_dir) + f"/fuel{10}/wrfout_d01_2019-05-11_17:49:11")


FGRNHFX6 = ds_6.FGRNHFX.isel(Time = 100).values
FGRNHFX10 = ds_10.FGRNHFX.isel(Time = 100).values