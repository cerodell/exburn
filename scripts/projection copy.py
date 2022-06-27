import context
import json
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from netCDF4 import Dataset
from sklearn.neighbors import KDTree

from osgeo import gdal, osr
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.io.shapereader as shpreader
from utils.sfire import makeLL_new


from context import root_dir, data_dir, img_dir


from wrf import getvar, interplevel, destagger

import warnings

warnings.filterwarnings("ignore")
################################ INPUTS ################################
# define unit for creating the la/long grind and obtaning qucik slice info
unit = "unit4"
sim_date = "2019-05-11_17:49:11"

## define grid slcie to make arrays smaller and centered over loc of int
# south_north_fire = slice(76, 95, None)
# west_east_fire = slice(77, 92, None)
south_north_fire = slice(68, 102, None)
west_east_fire = slice(69, 97, None)

sat_tiff = str(root_dir) + "/sat/response.tiff"
unit_shapefile = str(data_dir) + "/all_units/mygeodata_merged"

## define contour levels
levels = np.arange(0, 10.0, 0.5)

## defiena map exten to show
fire_nsew = [55.71536, 55.70755, -113.56331, -113.57447]
# fire_nsew = [55.71384, 55.70928, -113.5655, -113.57115]

## create img dir for unit amd simualtion
img_dir = Path(str(img_dir) + f"/{unit}-{sim_date[:10]}")
img_dir.mkdir(parents=True, exist_ok=True)
############################# END INPUTS ##############################


############################ WRF SIFIRE SETUP ################################

## Open the NetCDF file
ncfile = Dataset(str(data_dir) + f"/{unit}/wrfout_d01_{sim_date}")

## Open wrf out
wrf_ds = xr.open_dataset(str(data_dir) + f"/{unit}/wrfout_d01_{sim_date}")

## get model heights and add to wrf_ds
ht = getvar(ncfile, "z", units="m")
del ht["XTIME"]
del ht["Time"]
wrf_ds["height"] = ht

## destagger u and v and replace on wrf_ds
wrf_ds["U"] = destagger(wrf_ds["U"], -1, True)
wrf_ds["V"] = destagger(wrf_ds["V"], -2, True)

## create lat/lomn grids for staggered and destaggered
XLAT, XLONG = makeLL_new("met", unit)
s_XLAT, s_XLONG = makeLL_new("met", unit, stagger=False)

## add grids to wrf_ds
wrf_ds["XLAT"], wrf_ds["XLONG"] = XLAT, XLONG
wrf_ds["s_XLAT"], wrf_ds["s_XLONG"] = s_XLAT, s_XLONG


## slice the eniter wrf_ds to be centered over loc of int
wrf_ds = wrf_ds.sel(
    south_north=south_north_fire,
    west_east=west_east_fire,
    Time=slice(0, 124, 6),
)

## now interpolate U and V to 6 meter heights and add to wrf_ds
wrf_ds["U6"] = interplevel(wrf_ds["U"], wrf_ds["height"], 6)
wrf_ds["V6"] = interplevel(wrf_ds["V"], wrf_ds["height"], 6)

## solve for wind speed and add to wrf_ds
wrf_ds["wsp6"] = (wrf_ds["U6"] ** 2 + wrf_ds["V6"] ** 2) ** (0.5)

##################################################################


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

##################################################################

####################### Plot wind feilds #########################

colors = [
    "#FFFFFF",
    "#BBBBBB",
    "#646464",
    "#1563D3",
    "#2883F1",
    "#50A5F5",
    "#97D3FB",
    "#0CA10D",
    "#37D33C",
    "#97F58D",
    "#B5FBAB",
    "#FFE978",
    "#FFC03D",
    "#FFA100",
    "#FF3300",
    "#C10000",
    "#960007",
    "#643C32",
]
cmap = LinearSegmentedColormap.from_list("meteoblue", colors, N=len(levels))

for i in range(len(wrf_ds.XTIME)):
    # for i in range(1):
    fig = plt.figure(figsize=(9, 9))  # (Width, height) in inches.
    ax = fig.add_subplot(111, projection=projection)
    # fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=subplot_kw)
    ds_i = wrf_ds.isel(Time=i)

    ## plot satalite tiff image
    ax.imshow(real, zorder=1, extent=extent, origin="upper")

    ## plot shapefile polygons of units
    ax.add_geometries(
        reader.geometries(),
        crs=ccrs.Geodetic(),
        edgecolor="#1E3C47",
        alpha=0.8,
        facecolor="none",
        lw=2,
    )

    ## set map extent
    ax.set_extent(
        [fire_nsew[3], fire_nsew[2], fire_nsew[1], fire_nsew[0]], crs=ccrs.PlateCarree()
    )  ##  (x0, x1, y0, y1)

    ## plot grid
    shape = ds_i["s_XLONG"].shape
    for i in range(shape[1]):
        ax.plot(
            ds_i["s_XLONG"][:, i],
            ds_i["s_XLAT"][:, i],
            color="tab:red",
            zorder=4,
            transform=ccrs.PlateCarree(),
            lw=0.2,
            alpha=0.5,
        )

    for i in range(shape[0]):
        ax.plot(
            ds_i["s_XLONG"][i, :],
            ds_i["s_XLAT"][i, :],
            color="tab:red",
            zorder=4,
            transform=ccrs.PlateCarree(),
            lw=0.2,
            alpha=0.5,
        )

    ## plot grid center points
    # ax.scatter(
    #     ds_i["XLONG"][1:, 1:],
    #     ds_i["XLAT"][1:, 1:],
    #     transform=ccrs.PlateCarree(),
    #     color="k",
    #     s=0.2,
    #     zorder=4,
    # )

    ## plot countoru fill of wind feild
    contour = ax.contourf(
        ds_i["XLONG"][1:, 1:],
        ds_i["XLAT"][1:, 1:],
        ds_i["wsp6"][1:, 1:],
        zorder=1,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap=cmap,
        # norm=norm,
        extend="max",  # cubehelix_r
        alpha=0.7,
    )
    ax.streamplot(
        ds_i["XLONG"][1:, 1:].values,
        ds_i["XLAT"][1:, 1:].values,
        ds_i["U6"][1:, 1:].values,
        ds_i["V6"][1:, 1:].values,
        transform=ccrs.PlateCarree(),
        zorder=10,
        color="k",
        linewidth=0.4,
        arrowsize=1.0,
        # density=1.4,
    )
    # contour = ax.pcolormesh(
    #     ds_i["XLONG"][1:, 1:],
    #     ds_i["XLAT"][1:, 1:],
    #     ds_i["wsp6"][1:, 1:],
    #     zorder=1,
    #     transform=ccrs.PlateCarree(),
    #     # levels=levels,
    #     cmap=cmap,
    #     norm=norm,
    #     # extend="max",  # cubehelix_r
    #     # alpha=0.6,
    # )

    ## add color bar and give title
    cbar = plt.colorbar(contour, ax=ax, pad=0.04, location="right")
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(
        "Wind Speed  " + r"($\frac{\mathrm{m}}{\mathrm{~s}}$)",
        rotation=270,
        fontsize=14,
        labelpad=20,
    )

    ## set figure title
    ax.set_title(
        f"Unit 4 Wind Speed at 6 meters \n  {str(ds_i.XTIME.values)[11:-10]}",
        fontsize=15,
    )

    ## save fig
    plt.savefig(
        str(img_dir) + f"/wsp-{str(ds_i.XTIME.values)[11:-10]}.png",
        dpi=300,
        bbox_inches="tight",
    )

# test = xr.merge([s_XLONG, s_XLAT, XLAT, XLONG])

# test.to_zarr(str(data_dir)+'/grid.zarr')
