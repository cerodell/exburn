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


wrfrun = "/sfire/unit5/moist_true/"
filein = str(vol_dir) + wrfrun
save_dir = str(save_dir) + wrfrun
int_ds = xr.open_dataset(str(filein) + "/wrfinput_d01")
FMC_GC = int_ds.FMC_GC
FMC_G = int_ds.FMC_G

wrfrun = "/sfire/unit5/moist_false/"
filein = str(vol_dir) + wrfrun
save_dir = str(save_dir) + wrfrun
old_ds = xr.open_zarr(str(filein) + "/wrfout_unit5.zarr")
