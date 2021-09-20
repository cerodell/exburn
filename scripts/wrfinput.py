import context
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

from context import root_dir

wrf_in = str(root_dir) + "/data/wrfinput/wrfinput_d01"

ds = xr.open_dataset(wrf_in)


ds.NFUEL_CAT.max()

plt.imshow(ds.FMC_G.values[0, :, :])

FMC_GC = ds.FMC_GC.isel(Time=0, fuel_moisture_classes_stag=0).values
plt.imshow(FMC_GC)


ds.FMC_G.min()
