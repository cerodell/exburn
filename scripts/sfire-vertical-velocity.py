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

##################### Define Inputs and File Directories ###################
modelrun = "F6V51M08Z22I04"
configid = "F6V51"
domain = "met"
levels = np.arange(-10, 10.0, 0.5)
cmap = "coolwarm"

fireshape_path = str(data_dir) + "/unit_5/unit_5"
flux_filein = str(data_dir) + "/obs/met/"
save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)


################### Open Datsets ###################
## Open COnfig File and Get Relevant Paramters
with open(str(root_dir) + "/json/config.json") as f:
    config = json.load(f)
mets = config["unit5"]["obs"]["met"]
met_ids = list(mets)
bounds = config["unit5"]["sfire"][configid]
south_north = slice(bounds[domain]["sn"][0], bounds[domain]["sn"][1])
west_east = slice(bounds[domain]["we"][0], bounds[domain]["we"][1])
south_north = slice(110, 150, None)
west_east = slice(60, 89, None)

## open wrf-sfire simulation
wrf_ds = xr.open_dataset(str(data_dir) + f"/{modelrun}/wspd_wdir.nc")
wrf_ds = wrf_ds.sel(
    south_north=south_north,
    west_east=west_east,
).isel(Time=slice(0, 50))
XLAT, XLONG = wrf_ds.XLAT, wrf_ds.XLONG


times = wrf_ds.Time.values

## get heights from wrf-simulation
ncfile = Dataset(str(data_dir) + f"/{modelrun}/wrfout_d01_2019-05-11_17:49:11")
height = wrf.getvar(ncfile, "height")
height = height.values[:, 0, 0]
# print(height)
H = 0.762 * 3.28084
WAH = 1.83 / np.log((20 + 0.36 * H) / (0.13 * H))
MFH = WAH * 6.1

# test = wrf_ds['ua'].differentiate("south_north")
# wrf_ds["south_north"] = wrf_ds["south_north"] * 25
# test2 = wrf_ds['U'].differentiate("south_north")


row, col = 8, 6
# row, col = 2, 2
fig = plt.figure(figsize=(col * 2, row * 2 + 1))
fig.suptitle(
    f"Vertical velocity and Horizontal wind streamlines \n at varied heights above ground level (AGL) \n \n {modelrun} \n \n"
)
xlist = np.arange((col * row) - col, (col * row))
for i in range(len(height[: (col * row)])):
    ax = fig.add_subplot(row, col, i + 1)
    bm = Basemap(
        llcrnrlon=XLONG[0, 0],
        llcrnrlat=XLAT[0, 0],
        urcrnrlon=XLONG[-1, -1],
        urcrnrlat=XLAT[-1, -1],
        epsg=4326,
        ax=ax,
    )
    ## add unit boundary and met station location to map
    polygons = bm.readshapefile(
        fireshape_path, name="units", drawbounds=True, zorder=10
    )

    colors_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    wrf_dsi = wrf_ds.isel(Time=28, bottom_top=i)
    contour = ax.contourf(
        XLONG,
        XLAT,
        wrf_dsi["wa"],
        zorder=1,
        levels=levels,
        cmap=cmap,
        extend="both",  # cubehelix_r
    )

    ax.streamplot(
        XLONG.values,
        XLAT.values,
        wrf_dsi["ua"].values,
        wrf_dsi["va"].values,
        zorder=10,
        color="k",
        linewidth=0.3,
        arrowsize=0.3,
        density=1.4,
    )
    ax.set_title(f"{round(height[i])} m AGL")

    shape = XLAT.shape
    dxy = 25
    if (i == 0) or (i % col == 0):
        ax.set_yticks(np.linspace(bm.latmin, bm.latmax, 5))
        labels = [item.get_text() for item in ax.get_yticklabels()]
        ylabels = np.arange(0, shape[0] * dxy, shape[0] * dxy / len(labels)).astype(int)
        ax.set_yticklabels(ylabels)
        ax.yaxis.tick_left()
        ax.set_ylabel("meters", fontsize=8)
    else:
        pass
    if i in xlist:
        ax.set_xticks(np.linspace(bm.lonmin, bm.lonmax, 5))
        labels = [item.get_text() for item in ax.get_xticklabels()]
        xlabels = np.arange(0, shape[1] * dxy, shape[1] * dxy / len(labels)).astype(int)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("meters", fontsize=8)
    else:
        pass


fig.tight_layout()
fig.subplots_adjust(bottom=0.08)
cax = fig.add_axes([0.06, 0.048, 0.9, 0.012])  # [left, bottom, width, height]
cbar = fig.colorbar(contour, cax=cax, orientation="horizontal")
cbar.ax.tick_params(labelsize=10)
cbar.set_label("Vertical Velocity (m/s)", fontsize=10)
# plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
plt.savefig(str(save_dir) + f"/Vertical-Velocity.png", dpi=250)

################### Plot met Simulation to Observed Values ###################

# fig = plt.figure(figsize=(14, 14))
# fig.suptitle(modelrun)
# for i in range(len(height[:32])):
#     print(i)
#     ax = fig.add_subplot(8, 4, i+1)

#     bm = Basemap(
#         llcrnrlon=XLONG[0, 0],
#         llcrnrlat=XLAT[0, 0],
#         urcrnrlon=XLONG[-1, -1],
#         urcrnrlat=XLAT[-1, -1],
#         epsg=4326,
#         ax=ax,
#     )
#     ## add unit boundary and met station location to map
#     polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=10)

#     colors_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     wrf_dsi = wrf_ds.isel(Time = 28, bottom_top = i)
#     contour = ax.contourf(
#         XLONG, XLAT, wrf_dsi['wa'], zorder=1, levels=levels, cmap=cmap, extend="max"  # cubehelix_r
#     )

#     # cbar = plt.colorbar(contour, ax=ax, pad=0.04, location="left")
#     # cbar.ax.tick_params(labelsize=10)
#     # cbar.set_label(
#     #     "Wind Speed", rotation=90, fontsize=10, labelpad=15
#     # )

#     ax.streamplot(XLONG.values, XLAT.values,wrf_dsi['ua'].values , wrf_dsi['va'].values,
#                      zorder = 10, color = 'k', linewidth =0.4, arrowsize =0.4, density= 1.5)
#     ax.set_title(f'AGL {round(height[i])}')

#     shape = XLAT.shape
#     dxy = 25
#     ax.set_xticks(np.linspace(bm.lonmin, bm.lonmax, 9))
#     labels = [item.get_text() for item in ax.get_xticklabels()]
#     xlabels = np.arange(0, shape[1] * dxy, shape[1] * dxy / len(labels)).astype(int)
#     ax.set_xticklabels(xlabels)
#     ax.set_yticks(np.linspace(bm.latmin, bm.latmax, 10))
#     labels = [item.get_text() for item in ax.get_yticklabels()]
#     ylabels = np.arange(0, shape[0] * dxy, shape[0] * dxy / len(labels)).astype(int)
#     ax.set_yticklabels(ylabels)
#     ax.yaxis.tick_right()
#     ax.yaxis.set_label_position("right")
#     ax.set_xlabel("East-West (m)", fontsize=10)
#     ax.set_ylabel("North-South (m)", fontsize=10)

# fig.tight_layout()
# plt.savefig(str(save_dir) + f"/wsp-height.png", dpi=250)


# # make subplot for cross section of smoke
# ax = fig.add_subplot(2, 2, 2)
# tr17_1 = (
#     wrf_ds["tr17_1"].sum(dim=["west_east"]).isel(Time=idx_max[np.argmax(value_max)])
# )
# sn = tr17_1.south_north * 25
# ax.set_title(f"Cross Wind Integrated Smoke at {tiletime}", fontsize=10)
# contour = ax.contourf(
#     sn, height, tr17_1, zorder=1, levels=levels, cmap=cmap, extend="max"
# )
# ax.set_ylabel("Vertical Height \n (m)", fontsize=10)
# ax.set_xlabel("East-West \n Horizontal (m)", fontsize=10)
# ax.tick_params(axis="both", which="major", labelsize=8)
# # ax.set_ylim(0, 3200)
# # ax.set_xlim(0, 3000)
# # cbar = plt.colorbar(contour, ax=ax, pad=0.01)
# # cbar.ax.tick_params(labelsize=10)
# # cbar.set_label("g kg^-1", rotation=270, fontsize=8, labelpad=15)

# fig.tight_layout()
# plt.savefig(str(save_dir) + f"/smoke-wsp-comparison.png", dpi=250)

# # # met2m_ds = met_ds.isel(z = 0)
