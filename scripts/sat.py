import context
from sentinelhub import SHConfig
from PIL import Image
from io import BytesIO
from context import save_dir

config = SHConfig()
config.sh_client_id = "e3da4967-e08b-42fc-b292-b581199a743e"
config.sh_client_secret = "KGVMQ4#9gGLdI&in8]~6-e*8[W>ClGKvdTiejxAP"

config.save()

import os
import datetime
import numpy as np
import pyproj as pyproj
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from sentinelhub import (
    MimeType,
    CRS,
    BBox,
    SentinelHubRequest,
    SentinelHubDownloadClient,
    DataCollection,
    bbox_to_dimensions,
    DownloadRequest,
)


# %%
# # ============ INPUTS SOUTH WINDS==============
# name = 'south'
# ds = 25  # LES grid spacing
# fs = 5  # fire mesh ratio
# ndx = 160  # EW number of grids
# ndy = 400  # NS number of grids
# ll_utm = [
#     336524,
#     6174820,
# ]  # lower left corner of the domain in UTM coordinates (meters)
# # ============ end of INPUTS==============


# ============ INPUTS WEST WINDS==============
name = "west"
ds = 25  # LES grid spacing
fs = 5  # fire mesh ratio
ndx = 400  # EW number of grids
ndy = 160  # NS number of grids
ll_utm = [
    336176.37,
    6175835.73,
]  # lower left corner of the domain in UTM coordinates (meters)
# ============ end of INPUTS==============


# Part IIa: create a fuel mask -----------------------------------
# create a spatial grid

gridx, gridy = np.meshgrid(
    np.arange(0, ds * ndx, int(ds / fs)), np.arange(0, ds * ndy, int(ds / fs))
)
UTMx = (
    gridx + ll_utm[0]
)  # now adding a georefernce we have UTM grid (technically UTM_12N, or EPSG:26912)
UTMy = gridy + ll_utm[1]
## transform into the same projection (for shapefiles and for basemaps)
wgs84 = pyproj.Proj("+init=EPSG:4326")
epsg26912 = pyproj.Proj("+init=EPSG:26912")
WGSx, WGSy = pyproj.transform(
    epsg26912, wgs84, UTMx.ravel(), UTMy.ravel()
)  # reproject from UTM to WGS84
XLONG, XLAT = np.reshape(WGSx, np.shape(UTMx)), np.reshape(WGSy, np.shape(UTMy))


# print('.. configuring a map')
# bm = Basemap(llcrnrlon=XLONG[0,0], llcrnrlat=XLAT[0,0],\
# 					 urcrnrlon=X[-1,-1], urcrnrlat=XLAT[-1,-1], resolution='f', epsg=4326) #full resolution is slow
print(".. configuring a map")
bm = Basemap(
    llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1],
    epsg=4326,
)


evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

# betsiboka_coords_wgs84 = [46.16, -16.15, 46.51, -15.58]

resolution = 4
# betsiboka_bbox = BBox(bbox=betsiboka_coords_wgs84, crs=CRS.WGS84)
# betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)


bounding_coords = [UTMx[0, 0], UTMy[0, 0], UTMx[-1, -1], UTMy[-1, -1]]
betsiboka_bbox = BBox(bbox=bounding_coords, crs=CRS.UTM_12N)
betsiboka_size = bbox_to_dimensions(betsiboka_bbox, resolution=resolution)


print(f"Image shape at {resolution} m resolution: {betsiboka_size} pixels")


request_true_color = SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=("2020-06-12", "2020-06-13"),
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=betsiboka_bbox,
    size=betsiboka_size,
    config=config,
)

request_true_color = SentinelHubRequest(
    data_folder=str(save_dir) + "/SAT/test",
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L1C,
            time_interval=("2019-05-05", "2019-05-11"),
            mosaicking_order="leastCC",
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=betsiboka_bbox,
    size=betsiboka_size,
    config=config,
)

data_with_cloud_mask = request_true_color.get_data()


def plot_image(image, factor=1.0, clip_range=None, **kwargs):
    """
    Utility function for plotting RGB images.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.show()


plot_image(request_true_color.get_data()[0], factor=3.5 / 255, clip_range=(0, 1))
# plt.savefig(str(save_dir) + f"/{name}.png", dpi=250, bbox_inches="tight")

all_bands_img = request_true_color.get_data(save_data=True)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
bm = Basemap(
    llcrnrlon=XLONG[0, 0] + 0.0014,
    # llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1] + 0.001,
    # urcrnrlat=XLAT[-1, -1],
    epsg=4326,
    ax=ax,
)
wesn = [XLONG[0, 0], XLONG[-1, -1], XLAT[0, 0], XLAT[-1, -1]]
# ax = fig.add_subplot(1,1,1, projection=ccrs.UTM(zone=12))
factor = 3.5 / 255
clip_range = (0, 1)
real = np.clip(request_true_color.get_data()[0] * factor, *clip_range)
real = real[::-1, :, :]
bm.imshow(real, zorder=1, extent=wesn)

# polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True, zorder=10)
# shape = XLAT.shape
ax.set_xticks(np.round(np.linspace(bm.lonmin, bm.lonmax, 10), 3))


ax.set_yticks(np.round(np.linspace(bm.latmin, bm.latmax, 5), 3))

ax.set_xlabel("West-East (deg)", fontsize=12)
ax.set_ylabel("South-North (deg)", fontsize=12)
ax.grid(True, linestyle="--", lw=0.2, zorder=1)


ax.set_title(f"WRF SFIRE Domain \n Pelican Mountain", fontsize=16)

fig.tight_layout()
ax.ticklabel_format(useOffset=False)
plt.savefig(str(save_dir) + f"/{name}-site-map-full.png", dpi=250, bbox_inches="tight")
