# %% [markdown]
# Stull

# %%
import context
import numpy as np
import xarray as xr
from pathlib import Path


import matplotlib.pyplot as plt
from context import root_dir, vol_dir, data_dir, save_dir
import matplotlib.pylab as pylab


params = {
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.labelsize": 14,
}

pylab.rcParams.update(params)


# %% [markdown]
# Open dataset
# %%
wrfrun = "/sfire/unit5/moist_false/"
filein = str(vol_dir) + wrfrun
save_dir = str(save_dir) + wrfrun
ds = xr.open_zarr(str(filein) + "/wrfout_unit5.zarr")

# %% [markdown]
# Plot Heat flux from ground during ignition
# %%
dsi = ds.isel(
    south_north_subgrid=slice(550, 620),
    west_east_subgrid=slice(330, 405),
    Time=slice(0, 16, 3),
)
dsi["Time"] = dsi.XTIME.values.astype("datetime64[s]")
dsi.FGRNHFX.plot(col="Time", col_wrap=3, cmap="Reds", extend="max")
plt.savefig(str(save_dir) + "/FGRNHFX_Ignition.png")


# %% [markdown]
# Plot Heat flux from ground fire every min for first 20 mins
# %%
dsi = ds.isel(
    south_north_subgrid=slice(550, 620),
    west_east_subgrid=slice(330, 405),
    Time=slice(0, 168, 6),
)
dsi["Time"] = dsi.XTIME.values.astype("datetime64[s]")
dsi.FGRNHFX.plot(col="Time", col_wrap=4, cmap="Reds", extend="max")
plt.savefig(str(save_dir) + "/FGRNHFX.png")


# %% [markdown]
# Plot PM25 concentrations at ground level every min for full 40 mins
# %%
dsi = ds.isel(
    south_north=slice(110, 160),
    west_east=slice(60, 85),
    south_north_subgrid=slice(550, 620),
    west_east_subgrid=slice(330, 405),
    bottom_top=0,
    Time=slice(0, 252, 12),
)

dsi["Time"] = dsi.XTIME.values.astype("datetime64[s]")
dsi.tr17_1.plot(
    col="Time",
    col_wrap=4,
    cmap="cubehelix_r",
    levels=np.arange(0, 30100, 100),
    extend="max",
)
plt.savefig(str(save_dir) + "/PM25_ground.png")


# %% [markdown]
# Plot Vertically integrated PM25 concentrations every min for full 40 mins
# %%
dsi = ds.isel(
    south_north=slice(110, 399),
    west_east=slice(30, 200),
    south_north_subgrid=slice(550, 620),
    west_east_subgrid=slice(330, 405),
    Time=slice(0, 252, 12),
)

dsi["Time"] = dsi.XTIME.values.astype("datetime64[s]")
dsi = dsi.sum(dim="bottom_top")
dsi.tr17_1.plot(
    col="Time",
    col_wrap=4,
    cmap="cubehelix_r",
    levels=np.arange(0, 30100, 100),
    extend="max",
)
plt.savefig(str(save_dir) + "/PM25_vert_int.png")


# %%
# dsi = ds.isel(south_north = slice(110,160), west_east = slice(60,85),
#               south_north_subgrid = slice(550,620), west_east_subgrid = slice(330,405),
#               bottom_top = 0,
#               Time = 239)

# # dsi['Time'] = dsi.XTIME.values.astype('datetime64[s]')
# dsi.tr17_1.plot(cmap="cubehelix_r", levels=np.arange(0,30100,100), extend= 'max')
# plt.savefig(str(save_dir)+"/PM25_ground.png")


# dsi = ds.isel(south_north = slice(104,124), west_east = slice(60,90), Time = 2)
# dsi.GRNHFX.plot()
# plt.show()
# dsii = ds.isel(south_north_subgrid = slice(550,620), west_east_subgrid = slice(330,405), Time = 80)
# dsii.FGRNHFX.plot(cmap="Reds")
#
