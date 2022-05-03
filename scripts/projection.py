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

from wrf import (
    to_np,
    getvar,
    smooth2d,
    get_cartopy,
    cartopy_xlim,
    cartopy_ylim,
    latlon_coords,
)


# # Open the NetCDF file
# ncfile = Dataset(str(data_dir) + f"/{unit}/wrfout_d01_2019-05-11_17:49:11")

# # Get the sea level pressure
# slp = getvar(ncfile, "T2", timeidx=10)

# # Smooth the sea level pressure since it tends to be noisy near the
# # mountains
# smooth_slp = smooth2d(slp, 3, cenweight=4)

# # Get the latitude and longitude points
# lats, lons = latlon_coords(slp)


gdal.UseExceptions()

unit = "unit4"
var = "tr17_1"
pm_ef = 10.400


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


## open wrf-sfire simulation
wrf_ds = xr.open_dataset(str(data_dir) + f"/{unit}/wrfout_d01_2019-05-11_17:49:11")
wrf_ds[var] = wrf_ds[var] / pm_ef

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
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=subplot_kw)
# a = np.arange(1, 10)
# b = 10 ** np.arange(4)
# levels = (b[:, np.newaxis] * a).flatten()
levels = np.arange(5, 120, 5)
cmap = mpl.cm.YlOrRd
norm = mpl.colors.BoundaryNorm(levels, cmap.N, extend="max")
extent = (gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3])
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
    fire_ds["XLONG"],
    fire_ds["XLAT"],
    transform=ccrs.PlateCarree(),
    color="k",
    s=0.5,
    zorder=10,
)
tr17_1 = fire_ds["GRNHFX"].isel(Time=1) / 1000
# tr17_1= xr.where(tr17_1==0, np.nan, tr17_1)

contour = ax.contourf(
    fire_ds["XLONG"],
    fire_ds["XLAT"],
    tr17_1,
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
    "Heat Flux  \n" + r"($\frac{\mathrm{kW}}{\mathrm{~m}^{-2}}$)",
    rotation=270,
    fontsize=14,
    labelpad=15,
)
ax.set_title(f"Unit 4 Heat Flux \n at {str(tr17_1.XTIME.values)[11:-10]}", fontsize=10)

plt.savefig(str(img_dir) + f"/near-grid-hfx.png", dpi=300, bbox_inches="tight")
