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

from wrf import getvar, interplevel

# define unit for creating the la/long grind and obtaning qucik slice info
unit = "unit4"

# # Open the NetCDF file
ncfile = Dataset(str(data_dir) + f"/{unit}/wrfout_d01_2019-05-11_17:49:11")

## et u and v met
uvmet = getvar(ncfile, "uvmet")
## get height
ht = getvar(ncfile, "z", units="m")

uvmet6 = interplevel(uvmet, ht, 6)

U6 = uvmet6.sel(u_v="u")
V6 = uvmet6.sel(u_v="v")


# gdal.UseExceptions()

filein = str(root_dir) + "/sat/response.tiff"
shapefile_in = str(data_dir) + "/all_units/mygeodata_merged"


## Open COnfig File and Get Relevant Paramters
with open(str(root_dir) + "/json/config-new.json") as f:
    config = json.load(f)
namelist = config[unit]["sfire"]


XLAT, XLONG = makeLL_new("met", unit)
s_XLAT, s_XLONG = makeLL_new("met", unit, stagger=False)


# create dataframe with columns of all XLAT/XLONG
wrf_locs = pd.DataFrame({"XLAT": XLAT.values.ravel(), "XLONG": XLONG.values.ravel()})
wrf_tree = KDTree(wrf_locs)
print("WRF Domain KDTree built")


def get_loc(lat, lon):
    dist, ind = wrf_tree.query(np.array([lat, lon]).reshape(1, -1), k=1)
    loc = list(np.unravel_index(int(ind), XLAT.shape))
    return loc


met_nsew = namelist["met-nsew"]
ll = get_loc(met_nsew[1], met_nsew[3])
ur = get_loc(met_nsew[0], met_nsew[2])
south_north_met = slice(ll[0], ur[0])
west_east_met = slice(ll[1], ur[1])

fire_nsew = namelist["fire-nsew"]
ll = get_loc(fire_nsew[1], fire_nsew[3])
ur = get_loc(fire_nsew[0], fire_nsew[2])
south_north_fire = slice(ll[0] - 5, ur[0] + 5)
west_east_fire = slice(ll[1] - 5, ur[1] + 5)

south_north_fire = slice(76, 95, None)
west_east_fire = slice(77, 92, None)

## open wrf-sfire simulation
wrf_ds = xr.open_dataset(str(data_dir) + f"/{unit}/wrfout_d01_2019-05-11_17:49:11")
# wrf_ds[var] = wrf_ds[var] / pm_ef

wrf_ds["XLAT"], wrf_ds["XLONG"] = XLAT, XLONG
wrf_ds["s_XLAT"], wrf_ds["s_XLONG"] = s_XLAT, s_XLONG


fire_ds = wrf_ds.sel(
    south_north=south_north_fire,
    west_east=west_east_fire,
    Time=slice(0, 124, 6),
)


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
for i in range(len(fire_ds.XTIME)):

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=subplot_kw)
    ds_i = fire_ds.isel(Time=i)

    # a = np.arange(1, 10)
    # b = 10 ** np.arange(4)
    # levels = (b[:, np.newaxis] * a).flatten()
    # levels = np.arange(5, 120, 5)
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

    shape = fire_ds["s_XLONG"].shape
    for i in range(shape[1]):
        ax.plot(
            fire_ds["s_XLONG"][:, i],
            fire_ds["s_XLAT"][:, i],
            color="tab:red",
            zorder=10,
            transform=ccrs.PlateCarree(),
            lw=0.5,
        )

    for i in range(shape[0]):
        ax.plot(
            fire_ds["s_XLONG"][i, :],
            fire_ds["s_XLAT"][i, :],
            color="tab:red",
            zorder=10,
            transform=ccrs.PlateCarree(),
            lw=0.5,
        )

    ax.scatter(
        fire_ds["XLONG"][1:, 1:],
        fire_ds["XLAT"][1:, 1:],
        transform=ccrs.PlateCarree(),
        color="k",
        s=0.5,
        zorder=10,
    )
    wsp = (ds_i["U10"] ** 2 + ds_i["V10"] ** 2) ** (0.5)
    # tr17_1= xr.where(tr17_1==0, np.nan, tr17_1)

    contour = ax.contourf(
        fire_ds["XLONG"][1:, 1:],
        fire_ds["XLAT"][1:, 1:],
        wsp[1:, 1:],
        zorder=1,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="max",  # cubehelix_r
        alpha=0.8,
    )
    cbar = plt.colorbar(contour, ax=ax, pad=0.04, location="right")
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(
        "Wind Speed  \n" + r"($\frac{\mathrm{m}}{\mathrm{~s}}$)",
        rotation=270,
        fontsize=14,
        labelpad=15,
    )
    ax.set_title(
        f"Unit 4 Heat Flux \n at {str(ds_i.XTIME.values)[11:-10]}", fontsize=10
    )

    plt.savefig(
        str(img_dir) + f"/wsp-{str(ds_i.XTIME.values)[11:-10]}.png",
        dpi=300,
        bbox_inches="tight",
    )

# test = xr.merge([s_XLONG, s_XLAT, XLAT, XLONG])

# test.to_zarr(str(data_dir)+'/grid.zarr')
