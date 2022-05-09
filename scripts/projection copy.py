import context
import json
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from sklearn.neighbors import KDTree

from osgeo import gdal, osr
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from utils.sfire import makeLL_new


from context import root_dir, data_dir, img_dir

from wrf import getvar, interplevel, destagger

# define unit
unit = "unit4"

# # Open the NetCDF file
ncfile = Dataset(str(data_dir) + f"/{unit}/wrfout_d01_2019-05-11_17:49:11")

# uvmet = getvar(ncfile, "uvmet", timeidx = ALL_TIMES)
wrf_ds = xr.open_dataset(str(data_dir) + f"/{unit}/wrfout_d01_2019-05-11_17:49:11")
ht = getvar(ncfile, "z", units="m")
del ht["XTIME"]
del ht["Time"]
wrf_ds["height"] = ht

wrf_ds["U"] = destagger(wrf_ds["U"], -1, True)
wrf_ds["V"] = destagger(wrf_ds["V"], -2, True)

XLAT, XLONG = makeLL_new("met", unit)
s_XLAT, s_XLONG = makeLL_new("met", unit, stagger=False)
wrf_ds["XLAT"], wrf_ds["XLONG"] = XLAT, XLONG
wrf_ds["s_XLAT"], wrf_ds["s_XLONG"] = s_XLAT, s_XLONG

south_north_fire = slice(76, 95, None)
west_east_fire = slice(77, 92, None)

wrf_ds = wrf_ds.sel(
    south_north=south_north_fire,
    west_east=west_east_fire,
    Time=slice(0, 124, 6),
)


wrf_ds["U6"] = interplevel(wrf_ds["U"], wrf_ds["height"], 6)
wrf_ds["V6"] = interplevel(wrf_ds["V"], wrf_ds["height"], 6)
wrf_ds["wsp6"] = (wrf_ds["U6"] ** 2 + wrf_ds["V6"] ** 2) ** (0.5)


# gdal.UseExceptions()
filein = str(root_dir) + "/sat/response.tiff"
shapefile_in = str(data_dir) + "/all_units/mygeodata_merged"


# Read shape file
reader = shpreader.Reader(f"{shapefile_in}.shp")


ds = gdal.Open(filein)
data = ds.ReadAsArray()
gt = ds.GetGeoTransform()
proj = ds.GetProjection()

inproj = osr.SpatialReference()
inproj.ImportFromWkt(proj)
# print(inproj)


projcs = inproj.GetAuthorityCode("PROJCS")
projection = ccrs.epsg(projcs)
# print(projection)


subplot_kw = dict(projection=projection)
for i in range(len(wrf_ds.XTIME)):
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=subplot_kw)
    ds_i = wrf_ds.isel(Time=i)

    levels = np.arange(0, 10, 0.5)
    # cmap = mpl.cm.YlOrRd
    cmap = mpl.cm.jet
    norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend="max")
    extent = (
        gt[0],
        gt[0] + ds.RasterXSize * gt[1],
        gt[3] + ds.RasterYSize * gt[5],
        gt[3],
    )
    factor = 3.5 / 255
    clip_range = (0, 1)
    real = np.clip(plt.imread(filein) * factor, *clip_range)
    ax.imshow(real, zorder=1, extent=extent, origin="upper")
    ax.add_geometries(
        reader.geometries(),
        crs=ccrs.Geodetic(),
        edgecolor="black",
        alpha=0.8,
        facecolor="none",
        lw=2,
    )
    fire_nsew = [55.71384, 55.70928, -113.5655, -113.57115]
    # ax.set_extent([-113.57284, -113.5655, 55.70928, 55.71384], crs=ccrs.PlateCarree()) ##  (x0, x1, y0, y1)
    ax.set_extent(
        [fire_nsew[3], fire_nsew[2], fire_nsew[1], fire_nsew[0]], crs=ccrs.PlateCarree()
    )  ##  (x0, x1, y0, y1)

    shape = ds_i["s_XLONG"].shape
    for i in range(shape[1]):
        ax.plot(
            ds_i["s_XLONG"][:, i],
            ds_i["s_XLAT"][:, i],
            color="tab:red",
            zorder=10,
            transform=ccrs.PlateCarree(),
            lw=0.5,
        )

    for i in range(shape[0]):
        ax.plot(
            ds_i["s_XLONG"][i, :],
            ds_i["s_XLAT"][i, :],
            color="tab:red",
            zorder=10,
            transform=ccrs.PlateCarree(),
            lw=0.5,
        )

    ax.scatter(
        ds_i["XLONG"][1:, 1:],
        ds_i["XLAT"][1:, 1:],
        transform=ccrs.PlateCarree(),
        color="k",
        s=0.5,
        zorder=10,
    )
    # tr17_1= xr.where(tr17_1==0, np.nan, tr17_1)

    contour = ax.contourf(
        ds_i["XLONG"][1:, 1:],
        ds_i["XLAT"][1:, 1:],
        ds_i["wsp6"][1:, 1:],
        zorder=1,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="max",  # cubehelix_r
        alpha=0.6,
    )
    cbar = plt.colorbar(contour, ax=ax, pad=0.04, location="right")
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(
        "Wind Speed  " + r"($\frac{\mathrm{m}}{\mathrm{~s}}$)",
        rotation=270,
        fontsize=14,
        labelpad=20,
    )
    ax.set_title(
        f"Unit 4 Wind Speed at 6 meters \n  {str(ds_i.XTIME.values)[11:-10]}",
        fontsize=15,
    )

    plt.savefig(
        str(img_dir) + f"/wsp-{str(ds_i.XTIME.values)[11:-10]}.png",
        dpi=300,
        bbox_inches="tight",
    )

# test = xr.merge([s_XLONG, s_XLAT, XLAT, XLONG])

# test.to_zarr(str(data_dir)+'/grid.zarr')
