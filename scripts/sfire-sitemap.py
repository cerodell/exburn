import context
import json
import numpy as np
from pathlib import Path
from utils.sfire import makeLL
import pyproj as pyproj
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from context import root_dir, data_dir, save_dir
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============ INPUTS==============
domain = "fire"
unit = "unit5"
modelrun = "F6V51M08Z22"
configid = "F6V51"
ds = 25  # LES grid spacing
fs = 5  # fire mesh ratio
ndx = 160  # EW number of grids
ndy = 400  # NS number of grids
ll_utm = [
    336524,
    6174820,
]  # lower left corner of the domain in UTM coordinates (meters)
filein = str(save_dir) + "/SAT/test/1884739755f8400be5009eb79a98bb00/response.tiff"
fireshape_path = str(root_dir) + "/data/all_units/mygeodata_merged"

save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)
# ============ end of INPUTS==============


with open(str(root_dir) + "/json/config.json") as f:
    config = json.load(f)
ros = config["unit5"]["obs"]["ros"]
ros_ids = list(ros)
mets = config["unit5"]["obs"]["met"]
aqs = config["unit5"]["obs"]["aq"]
hfxs = config["unit5"]["obs"]["hfx"]

bounds = config["unit5"]["sfire"][configid]
south_north_subgrid = slice(540, 760, None)
west_east_subgrid = slice(220, 500, None)
fs = bounds["namelist"]["dxy"] / bounds["namelist"]["fs"]
XLAT, XLONG = makeLL(domain, configid)


def plot_image(image, factor=1.0, clip_range=None, **kwargs):
    """
    Utility function for plotting RGB images.
    """
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    return


# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(1, 1, 1)
# bm = Basemap(
#     llcrnrlon=XLONG[0, 0]+0.0014,
#     # llcrnrlon=XLONG[0, 0],
#     llcrnrlat=XLAT[0, 0],
#     urcrnrlon=XLONG[-1, -1],
#     urcrnrlat=XLAT[-1, -1]+0.001,
#     # urcrnrlat=XLAT[-1, -1],
#     epsg=4326,
#     ax=ax,
# )
# wesn = [XLONG[0, 0], XLONG[-1, -1],XLAT[0, 0], XLAT[-1, -1]]
# # ax = fig.add_subplot(1,1,1, projection=ccrs.UTM(zone=12))
# factor=3.5/255
# clip_range=(0,1)
# real = np.clip(plt.imread(filein) * factor, *clip_range)
# real = real[::-1,:,:]
# bm.imshow(real, zorder = 1, extent =wesn)

# # polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=10)
# # shape = XLAT.shape
# ax.set_xticks(np.round(np.linspace(bm.lonmin, bm.lonmax, 5),3))
# # labels = [item.get_text() for item in ax.get_xticklabels()]
# # xlabels = np.arange(0, shape[1] * fs, shape[1] * fs / len(labels)).astype(int)
# # ax.set_xticklabels(xlabels, fontsize=11)

# ax.set_yticks(np.round(np.linspace(bm.latmin, bm.latmax, 10),3))
# # labels = [item.get_text() for item in ax.get_yticklabels()]
# # ylabels = np.arange(0, shape[0] * fs, shape[0] * fs / len(labels)).astype(int)
# # ax.set_yticklabels(ylabels, fontsize=11)
# ax.set_xlabel("West-East (deg)", fontsize=12)
# ax.set_ylabel("South-North (deg)", fontsize=12)
# ax.grid(True, linestyle="--", lw=0.2, zorder=1)

# ax.set_title(f"WRF SFIRE Domain \n Pelican Mountain Unit 5", fontsize=16)

# fig.tight_layout()
# ax.ticklabel_format(useOffset=False)
# plt.savefig(str(save_dir) + f"/site-map-full.png", dpi=250, bbox_inches="tight")


# %%

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
bm = Basemap(
    llcrnrlon=XLONG[0, 0] + 0.0019,
    # llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1] + 0.001,
    # urcrnrlat=XLAT[-1, -1],
    epsg=4326,
    ax=ax,
)
wesn = [XLONG[0, 0], XLONG[-1, -1], XLAT[0, 0], XLAT[-1, -1]]
factor = 3.5 / 255
clip_range = (0, 1)
real = np.clip(plt.imread(filein) * factor, *clip_range)
real = real[::-1, :, :]
bm.imshow(real, zorder=1, extent=wesn)

polygons = bm.readshapefile(
    fireshape_path,
    name="units",
    drawbounds=True,
    zorder=10,
    color="tab:red",
    linewidth=2,
)
shape = XLAT.shape
ax.set_xticks(np.round(np.linspace(bm.lonmin, bm.lonmax, 32), 3))
# labels = [item.get_text() for item in ax.get_xticklabels()]
# xlabels = np.arange(0, shape[1] * fs, shape[1] * fs / len(labels)).astype(int)
# ax.set_xticklabels(xlabels-1000, fontsize=11)

ax.set_yticks(np.round(np.linspace(bm.latmin, bm.latmax, 50), 3))
# labels = [item.get_text() for item in ax.get_yticklabels()]
# ylabels = np.arange(0, shape[0] * fs, shape[0] * fs / len(labels)).astype(int)
# ax.set_yticklabels(ylabels-2600, fontsize=11)
ax.set_xlabel("West-East (m)", fontsize=12)
ax.set_ylabel("South-North (m)", fontsize=12)
ax.grid(True, linestyle="--", lw=0.2, zorder=1)

ax.set_title(f"Pelican Mountain", fontsize=16)
ax.set_xlim(-113.58093195, -113.56039335)
ax.set_ylim(55.70318713, 55.72648132)

fig.tight_layout()
ax.ticklabel_format(useOffset=False)
# plt.savefig(str(save_dir) + f"/site-map-close.png", dpi=250, bbox_inches="tight")


# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(1, 1, 1)
# bm = Basemap(
#     llcrnrlon=XLONG[0, 0]+0.0014,
#     # llcrnrlon=XLONG[0, 0],
#     llcrnrlat=XLAT[0, 0],
#     urcrnrlon=XLONG[-1, -1],
#     urcrnrlat=XLAT[-1, -1]+0.001,
#     # urcrnrlat=XLAT[-1, -1],
#     epsg=4326,
#     ax=ax,
# )
# wesn = [XLONG[0, 0], XLONG[-1, -1],XLAT[0, 0], XLAT[-1, -1]]
# # ax = fig.add_subplot(1,1,1, projection=ccrs.UTM(zone=12))
# factor=3.5/255
# clip_range=(0,1)
# real = np.clip(plt.imread(filein) * factor, *clip_range)
# real = real[::-1,:,:]
# bm.imshow(real, zorder = 1, extent =wesn)

# polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=10)
# shape = XLAT.shape
# ax.set_xticks(np.linspace(bm.lonmin, bm.lonmax, 32))
# labels = [item.get_text() for item in ax.get_xticklabels()]
# xlabels = np.arange(0, shape[1] * fs, shape[1] * fs / len(labels)).astype(int)
# ax.set_xticklabels(xlabels-1000, fontsize=11)

# ax.set_yticks(np.linspace(bm.latmin, bm.latmax, 50))
# labels = [item.get_text() for item in ax.get_yticklabels()]
# ylabels = np.arange(0, shape[0] * fs, shape[0] * fs / len(labels)).astype(int)
# ax.set_yticklabels(ylabels-2600, fontsize=11)
# ax.set_xlabel("West-East (m)", fontsize=12)
# ax.set_ylabel("South-North (m)", fontsize=12)
# ax.grid(True, linestyle="--", lw=0.2, zorder=1)


# for h in hfxs:
#     if h == list(hfxs)[-1]:
#         ax.scatter(
#             hfxs[h]["lon"],
#             hfxs[h]["lat"],
#             zorder=9,
#             s=30,
#             color="tab:orange",
#             marker=".",
#             label="Heatflux",
#         )
#     else:
#         ax.scatter(
#             hfxs[h]["lon"],
#             hfxs[h]["lat"],
#             zorder=9,
#             s=30,
#             color="tab:orange",
#             marker=".",
#         )

# for r in ros:
#     if r == list(ros)[-1]:
#         ax.scatter(
#             ros[r]["lon"],
#             ros[r]["lat"],
#             zorder=9,
#             s=10,
#             color="tab:green",
#             marker=".",
#             label="Thermocouples",
#         )
#     else:
#         ax.scatter(
#             ros[r]["lon"],
#             ros[r]["lat"],
#             zorder=9,
#             s=10,
#             color="tab:green",
#             marker=".",
#         )

# for aq in aqs:
#     if aq == list(aqs)[-1]:
#         ax.scatter(
#             aqs[aq]["lon"],
#             aqs[aq]["lat"],
#             zorder=10,
#             # label=aq_ids[i],
#             color="tab:red",
#             edgecolors="black",
#             marker="^",
#             s=100,
#             label="Air Quality Monitor",
#         )
#     else:
#         ax.scatter(
#             aqs[aq]["lon"],
#             aqs[aq]["lat"],
#             zorder=10,
#             # label=aq_ids[i],
#             color="tab:red",
#             edgecolors="black",
#             marker="^",
#             s=100,
#         )


# ax.scatter(
#     mets["south_met"]["lon"],
#     mets["south_met"]["lat"],
#     zorder=10,
#     edgecolors="black",
#     color="tab:blue",
#     marker="D",
#     s=80,
#     label="Met Tower",
# )
# legend = ax.legend(
#     loc="upper right",
#     ncol=1,
#     fancybox=True,
#     shadow=True,
# )
# ax.set_title(f"Pelican Mountain Unit 5 Sensor Location", fontsize=16)

# ax.set_xlim(-113.58493195, -113.56339335)
# ax.set_ylim(55.71618713, 55.72648132)

# fig.tight_layout()
# # ax.ticklabel_format(useOffset=False)
# plt.savefig(str(save_dir) + f"/site-map-close.png", dpi=250, bbox_inches="tight")


# %%
