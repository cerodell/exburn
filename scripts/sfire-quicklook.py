# %% [markdown]
# Stull

# %%
import context
import wrf
import json
import numpy as np
import xarray as xr
from pathlib import Path
from netCDF4 import Dataset


import matplotlib.pyplot as plt
from context import root_dir, vol_dir, data_dir, save_dir
import matplotlib.pylab as pylab
import matplotlib

matplotlib.rcParams.update({"font.size": 10})

modelrun = "F6V51M08Z22"
# configid = modelrun[:-6]
configid = "F6V51"

ncfile = Dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
height = wrf.getvar(ncfile, "height")
height = height.values[:, 0, 0]
print(height)

with open(str(root_dir) + "/json/config.json") as f:
    config = json.load(f)
bounds = config["unit5"]["sfire"][configid]
south_north = slice(bounds["met"]["sn"][0], bounds["met"]["sn"][1])
west_east = slice(bounds["met"]["we"][0], bounds["met"]["we"][1])
south_north_subgrid = slice(bounds["fire"]["sn"][0], bounds["fire"]["sn"][1])
west_east_subgrid = slice(bounds["fire"]["we"][0], bounds["fire"]["we"][1])

params = {
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.labelsize": 14,
}
dx = 25
pylab.rcParams.update(params)


# %% [markdown]
# Open dataset
# %%

save_dir = Path(str(save_dir) + f"/{modelrun}")
save_dir.mkdir(parents=True, exist_ok=True)
ds = xr.open_dataset(
    str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11", chunks="auto"
)

# %% [markdown]
# Plot Fuel Moisture

dsi = ds.isel(
    south_north_subgrid=south_north_subgrid,
    west_east_subgrid=west_east_subgrid,
    south_north=south_north,
    west_east=west_east,
    Time=slice(0, 54, 6),
    # fuel_moisture_classes_stag = 0
)
dsi["Time"] = dsi.XTIME.values.astype("datetime64[s]")
U = dsi["U"].isel(bottom_top=1)
U.plot(col="Time", col_wrap=3, cmap="coolwarm", extend="max", aspect=2, size=3)

# %% [markdown]
# Plot Gound Heat flux
# %%
dsi = ds.isel(
    south_north_subgrid=south_north_subgrid,
    west_east_subgrid=west_east_subgrid,
    Time=slice(0, 54, 6),
)
dsi["Time"] = dsi.XTIME.values.astype("datetime64[s]")
dsi.FCANHFX.plot(col="Time", col_wrap=3, cmap="Reds", extend="max", aspect=2, size=3)
plt.savefig(str(save_dir) + "/FCANHFX.png")

# %% [markdown]
# Plot Gound Heat flux
# %%
dsi = ds.isel(
    south_north_subgrid=south_north_subgrid,
    west_east_subgrid=west_east_subgrid,
    Time=slice(
        0,
        60,
    ),
)

dsi["Time"] = dsi.XTIME.values.astype("datetime64[s]")
dsi.FGRNHFX.plot(col="Time", col_wrap=6, cmap="Reds", extend="max", aspect=2, size=3)
plt.savefig(str(save_dir) + "/FGRNHFX.png")

# %% [markdown]
# Plot Ground Moisture flux
# %%
dsi = ds.isel(
    south_north_subgrid=south_north_subgrid,
    west_east_subgrid=west_east_subgrid,
    Time=slice(0, 54, 6),
)
dsi["Time"] = dsi.XTIME.values.astype("datetime64[s]")
dsi.FGRNQFX.plot(col="Time", col_wrap=3, cmap="Greens", extend="max", aspect=2, size=3)
plt.savefig(str(save_dir) + "/FGRNQFX.png")


# %% [markdown]
# Plot Fire Area
# %%
dsi.FIRE_AREA.plot(
    col="Time", col_wrap=3, cmap="gist_heat_r", extend="max", aspect=2, size=3
)
plt.savefig(str(save_dir) + "/FIRE_AREA.png")


# %% [markdown]
# Plot Vertically integrated PM25 concentrations every min for full 40 mins
# %%
dsi = ds.isel(
    south_north=south_north,
    west_east=west_east,
    south_north_subgrid=south_north_subgrid,
    west_east_subgrid=west_east_subgrid,
    # Time=slice(0, 252, 12),
    Time=slice(0, 235, 10),
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
plt.savefig(str(save_dir) + "/tracer-vert-int.png")


# %% [markdown]
# Plot Max Temp crosssection
# %%
# interp_level = height
# temp = wrf.getvar(ncfile, "temp", timeidx=wrf.ALL_TIMES)
# dsi = temp.isel(south_north=south_north).max(dim=["west_east", "Time"])


# sn = dsi.south_north * dx
# levels = np.arange(260, 320, 1)
# fig = plt.figure(figsize=(14, 4))
# ax = fig.add_subplot(1, 1, 1)  # top and bottom left
# # ax.set_title(f"Temperature \n" + dsi.Time.values.astype(str)[:-10], fontsize=18)
# ax.set_title(f"Temperature Max {modelrun}", fontsize=18)
# contour = ax.contourf(
#     sn,
#     interp_level,
#     dsi,
#     zorder=1,
#     levels=levels,
#     cmap="jet",
#     extend="both",
# )
# ax.set_ylabel("Vertical (m)", fontsize=16)
# ax.set_xlabel("Horizontal (m)", fontsize=16)
# ax.tick_params(axis="both", which="major", labelsize=14)
# ax.set_ylim(0, 3200)
# ax.set_xlim(0, 3000)

# cbar = plt.colorbar(contour, ax=ax, pad=0.01)
# cbar.ax.tick_params(labelsize=12)
# cbar.set_label("K", rotation=270, fontsize=16, labelpad=15)
# plt.savefig(str(save_dir) + "/temp-cross.png")


# %% [markdown]
# Plot Max Equivalent Potential Temp crosssection
# %%
# theta_e = wrf.getvar(ncfile, 'theta_e', timeidx = wrf.ALL_TIMES)
# dsi = theta_e.isel(south_north=south_north).max(dim = ['west_east', 'Time'])
# south_north = dsi.south_north * dx
# levels = np.arange(294, 340, 0.5)

# fig = plt.figure(figsize=(14, 4))
# ax = fig.add_subplot(1, 1, 1)  # top and bottom left
# # ax.set_title(f"Equivalent Potential Temperature \n" + dsi.Time.values.astype(str)[:-10], fontsize=18)
# ax.set_title(f"Equivalent Potential Temperature Max {modelrun}", fontsize=18)
# contour = ax.contourf(
#     south_north,
#     interp_level,
#     dsi,
#     zorder=1,
#     levels = levels,
#     cmap='jet',
#     extend="both",
# )
# ax.set_ylabel("Vertical (m)", fontsize=16)
# ax.set_xlabel("Horizontal (m)", fontsize=16)
# ax.tick_params(axis="both", which="major", labelsize=14)
# ax.set_ylim(0,3200)

# cbar = plt.colorbar(contour, ax=ax, pad=0.01)
# cbar.ax.tick_params(labelsize=12)
# cbar.set_label('K', rotation=270, fontsize=16, labelpad=15)
# plt.savefig(str(save_dir) + "/theta_e-cross.png")


# %% [markdown]
# Plot Max Tracer crosssection

# %%

tr17_1 = ds["tr17_1"].isel(south_north=south_north).max(dim=["west_east", "Time"])
sn = tr17_1.south_north * dx
fig = plt.figure(figsize=(14, 4))
ax = fig.add_subplot(1, 1, 1)  # top and bottom left
ax.set_title(f"Tracer Max {modelrun}", fontsize=18)
contour = ax.pcolormesh(
    sn, height, tr17_1, zorder=1, vmin=0, vmax=10000, cmap="cubehelix_r", shading="auto"
)
ax.set_ylabel("Vertical (m)", fontsize=16)
ax.set_xlabel("Horizontal (m)", fontsize=16)
ax.tick_params(axis="both", which="major", labelsize=14)
ax.set_ylim(0, 3200)
ax.set_xlim(0, 3000)
cbar = plt.colorbar(contour, ax=ax, pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label("g kg^-1", rotation=270, fontsize=16, labelpad=15)
plt.savefig(str(save_dir) + "/tracer-cross.png")


# %%
