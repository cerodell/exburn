# %% [markdown]
# Stull

# %%
import context
import wrf
import numpy as np
import xarray as xr
from pathlib import Path
from netCDF4 import Dataset


import matplotlib.pyplot as plt
from context import root_dir, vol_dir, data_dir, save_dir
import matplotlib.pylab as pylab
import matplotlib

matplotlib.rcParams.update({"font.size": 10})

modelrun = "F6V51Z20"

params = {
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.labelsize": 14,
}

pylab.rcParams.update(params)


# %% [markdown]
# Open dataset
# %%

save_dir = Path(str(save_dir) + f"/{modelrun}")
save_dir.mkdir(parents=True, exist_ok=True)
ds = xr.open_dataset(
    str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_16:49:01", chunks="auto"
)


# %% [markdown]
# Plot Gound Heat flux

ncfile = Dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_16:49:01")

height = wrf.getvar(ncfile, "height")
height.values[:, 0, 0]

temp = wrf.getvar(ncfile, "theta_e", wrf.ALL_TIMES)
dsi = temp.isel(Time=60, south_north=slice(110, 160), west_east=71)
dsi.plot()


# %% [markdown]
# Plot Gound Heat flux
# %%
dsi = ds.isel(
    south_north_subgrid=slice(550, 620),
    west_east_subgrid=slice(330, 405),
    Time=slice(0, 44, 1),
)
dsi["Time"] = dsi.XTIME.values.astype("datetime64[s]")
dsi.FGRNHFX.plot(col="Time", col_wrap=3, cmap="Reds", extend="max", aspect=2, size=3)
plt.savefig(str(save_dir) + "/FGRNHFX.png")

# %% [markdown]
# Plot Fire Area
# %%
dsi.FIRE_AREA.plot(
    col="Time", col_wrap=3, cmap="gist_heat_r", extend="max", aspect=2, size=3
)
plt.savefig(str(save_dir) + "/FIRE_AREA.png")

# # %% [markdown]
# # Plot Observed Fire Heat Flux
# # %%
# dsi.FIRE_HFX.plot(col="Time", col_wrap=3, cmap="Reds", extend="max", aspect=2, size=3)
# plt.savefig(str(save_dir) + "/FIRE_HFX.png")

# %% [markdown]
# Plot Fuel Fraction
# %%
dsi.FUEL_FRAC.plot(
    col="Time", col_wrap=3, cmap="summer_r", extend="max", aspect=2, size=3
)
plt.savefig(str(save_dir) + "/FUEL_FRAC.png")


# %% [markdown]
# Plot Rate of Spread
# %%
dsi.ROS.plot(col="Time", col_wrap=3, cmap="Purples", extend="max", aspect=2, size=3)
plt.savefig(str(save_dir) + "/ROS.png")

# %% [markdown]
# Plot Fireline Intensity
# %%
dsi.FLINEINT.plot(col="Time", col_wrap=3, cmap="Reds", extend="max", aspect=2, size=3)
plt.savefig(str(save_dir) + "/FLINEINT.png")

# %% [markdown]
# Plot Heat flux from ground fire every min for first 20 mins
# %%
dsi = ds.isel(
    south_north=slice(110, 160),
    west_east=slice(60, 85),
    Time=slice(0, 168, 6),
)
dsi["Time"] = dsi.XTIME.values.astype("datetime64[s]")
dsi.GRNHFX.plot(col="Time", col_wrap=4, cmap="Reds", extend="max")
plt.savefig(str(save_dir) + "/GRNHFX.png")


# %% [markdown]
# Plot PM25 concentrations at ground level every min for full 40 mins
# %%
dsi = ds.isel(
    south_north=slice(110, 160),
    west_east=slice(60, 85),
    south_north_subgrid=slice(550, 620),
    west_east_subgrid=slice(330, 405),
    bottom_top=0,
    # Time=slice(0, 252, 12),
    Time=slice(0, 235, 12),
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
    # Time=slice(0, 252, 12),
    Time=slice(0, 235, 12),
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


# %% [markdown]
# Plot Surface Temp over fire
# %%
dsi = ds.isel(
    south_north=slice(110, 160),
    west_east=slice(60, 85),
    south_north_subgrid=slice(550, 620),
    west_east_subgrid=slice(330, 405),
    # Time=slice(0, 252, 12),
    Time=slice(0, 235, 12),
)

dsi["Time"] = dsi.XTIME.values.astype("datetime64[s]")
dsi = dsi.sum(dim="bottom_top")
dsi.T2.plot(
    col="Time",
    col_wrap=4,
    cmap="coolwarm",
    # levels=np.arange(0, 30100, 100),
    extend="both",
)
plt.savefig(str(save_dir) + "/T2.png")
