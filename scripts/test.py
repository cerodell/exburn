import context
import wrf
import numpy as np
import pandas as pd
import xarray as xr


filein = "/Volumes/Scratch/FWF-WAN00CG/d02/202106/fwf-hourly-d02-2021063006.nc"
ds = xr.open_dataset(filein)

ds.F.isel(time=50).plot()
