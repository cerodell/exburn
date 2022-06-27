import context
import json
import numpy as np
from pathlib import Path
from utils.sfire import makeLL
import pyproj as pyproj
import matplotlib.pyplot as plt
from osgeo import gdal, osr
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader


from context import root_dir, data_dir, save_dir
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============ INPUTS==============
unit = "unit5"
modelrun = "F6V51M08Z22"

sat_tiff = str(save_dir) + "/SAT/test/1884739755f8400be5009eb79a98bb00/response.tiff"
unit_shapefile = str(root_dir) + "/data/unit_5/unit_5"

save_dir = Path(str(save_dir) + f"/{modelrun}/")
save_dir.mkdir(parents=True, exist_ok=True)
# ============ end of INPUTS==============


with open(str(root_dir) + "/json/config-new.json") as f:
    config = json.load(f)
ros = config[unit]["obs"]["ros"]
mets = config[unit]["obs"]["met"]
aqs = config[unit]["obs"]["aq"]
hfxs = config[unit]["obs"]["hfx"]


######################## Plotting set up ##########################

# Read shape file of all unit pelican mnt
reader = shpreader.Reader(f"{unit_shapefile}.shp")

## open geo tiff file of sat image and get useful projection/transform info for plotting
ds = gdal.Open(sat_tiff)
data = ds.ReadAsArray()
gt = ds.GetGeoTransform()
proj = ds.GetProjection()
inproj = osr.SpatialReference()
inproj.ImportFromWkt(proj)
projcs = inproj.GetAuthorityCode("PROJCS")
projection = ccrs.epsg(projcs)
# print(projection)

extent = (
    gt[0],
    gt[0] + ds.RasterXSize * gt[1],
    gt[3] + ds.RasterYSize * gt[5],
    gt[3],
)
factor = 3.5 / 255
clip_range = (0, 1)
real = np.clip(plt.imread(sat_tiff) * factor, *clip_range)

subplot_kw = dict(projection=projection)
###########################################################################


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection=projection)

ax.imshow(real, zorder=1, extent=extent, origin="upper")
ax.add_geometries(
    reader.geometries(),
    # crs=ccrs.Geodetic(),
    crs=ccrs.PlateCarree(),
    edgecolor="k",
    alpha=0.8,
    facecolor="none",
    lw=1.0,
    zorder=3,
)

gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=True,
    linewidth=1,
    color="gray",
    alpha=0.5,
    linestyle="--",
    zorder=2,
)
gl.xlabels_top = False
gl.ylabels_left = False
gl.xlabel_style = {"size": 14}
gl.ylabel_style = {"size": 14}


for h in hfxs:
    if h == list(hfxs)[-1]:
        ax.scatter(
            hfxs[h]["lon"],
            hfxs[h]["lat"],
            zorder=9,
            s=40,
            color="tab:orange",
            marker=".",
            label="Heatflux",
            transform=ccrs.PlateCarree(),
        )
    else:
        ax.scatter(
            hfxs[h]["lon"],
            hfxs[h]["lat"],
            zorder=9,
            s=40,
            color="tab:orange",
            marker=".",
            transform=ccrs.PlateCarree(),
        )

for r in ros:
    if r == list(ros)[-1]:
        ax.scatter(
            ros[r]["lon"],
            ros[r]["lat"],
            zorder=9,
            s=20,
            color="tab:green",
            marker=".",
            label="Thermocouples",
            transform=ccrs.PlateCarree(),
        )
    else:
        ax.scatter(
            ros[r]["lon"],
            ros[r]["lat"],
            zorder=9,
            s=20,
            color="tab:green",
            marker=".",
            transform=ccrs.PlateCarree(),
        )

for aq in aqs:
    if aq == list(aqs)[-1]:
        ax.scatter(
            aqs[aq]["lon"],
            aqs[aq]["lat"],
            zorder=10,
            # label=aq_ids[i],
            color="tab:red",
            edgecolors="black",
            marker="^",
            s=100,
            label="Air Quality Monitor",
            transform=ccrs.PlateCarree(),
        )
    else:
        ax.scatter(
            aqs[aq]["lon"],
            aqs[aq]["lat"],
            zorder=10,
            # label=aq_ids[i],
            color="tab:red",
            edgecolors="black",
            marker="^",
            s=100,
            transform=ccrs.PlateCarree(),
        )


ax.scatter(
    mets["south_met"]["lon"],
    mets["south_met"]["lat"],
    zorder=10,
    edgecolors="black",
    color="tab:blue",
    marker="D",
    s=80,
    label="Met Tower",
    transform=ccrs.PlateCarree(),
)
legend = ax.legend(loc="upper right", ncol=1, fancybox=True, shadow=True, fontsize=12)
ax.set_title(f"Pelican Mountain Unit 5 Sensor Location", fontsize=18)


# ## set map extent
ax.set_extent(
    [-113.58493195, -113.56339335, 55.71618713, 55.72608132], crs=ccrs.PlateCarree()
)  ##  (x0, x1, y0, y1)

fig.tight_layout()
plt.savefig(str(save_dir) + f"/site-map-close.png", dpi=300, bbox_inches="tight")


# %%
