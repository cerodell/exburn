# %%
# April 19, 2021
# nmoisseeva@eoas.ubc.ca
# crodell@eoas.ubc.ca
# This script generates input fields for sfire for use with prescribed burn

# %%
from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from mpl_toolkits.basemap import Basemap
import pyproj as pyproj
from netCDF4 import Dataset
import pandas as pd
import xarray as xr
from pathlib import Path
import zarr
import geopandas as gpd
from sklearn.neighbors import KDTree
import wrf
from context import gog_dir, data_dir, met_dir, root_dir
import pickle
from wrf import getvar
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# %%
# ============ INPUTS==============
modelrun = "F6V51M08"
ds = 25  # LES grid spacing
fs = 5  # fire mesh ratio
ndx = 160  # EW number of grids
ndy = 400  # NS number of grids
skin = 290  # skin surface temperature
buff = 40  # buffer size (ie firebreaks) around units
fueltype = 6  # anderson fuels type
ig_start = [55.71765, -113.571112]
ig_end = [55.71765, -113.575172]
sw = [55.717153, -113.57668]
ne = [55.720270, -113.569517]
fireshape_path = str(data_dir) + "/all_units/mygeodata_merged"
ll_utm = [
    336524,
    6174820,
]  # lower left corner of the domain in UTM coordinates (meters)
rxloc = [55, -113]  # lat/lon location of the burn
rxtime = 14  # anticipated burn hour
utm = -8  # utm offset
# ============ end of INPUTS==============

save_dir = Path(str(data_dir) + f"/wrfinput/{modelrun}")
save_dir.mkdir(parents=True, exist_ok=True)

img_dir = Path(str(root_dir) + f"/img/{modelrun}")
img_dir.mkdir(parents=True, exist_ok=True)

# %%
# Part I: create a perturbed surface temperature to start off convection-------------
surface_T = (
    (np.random.rand(ndx, ndy) - 0.5) * 1.5
) + skin  # perturbes surface by half a degree +/-
dim_header = ",".join(
    map(str, np.shape(surface_T))
)  # creates a header required by wrf-sfire, which is just dimensions

# save output file
np.savetxt(
    str(save_dir) + "/input_tsk", surface_T, header=dim_header, comments="", fmt="%1.2f"
)  # saves as text file, with some formatting


# %%
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


print(".. configuring a map")
bm = Basemap(
    llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1],
    epsg=4326,
)

# try:
#     ds_fuel = xr.open_zarr(str(save_dir) + f"/fuel{fueltype}.zarr")
#     fuel = ds_fuel.fuel.values
#     print(f"found {modelrun} fuels dataset")
# except:
# %%

# gpkg_df = gpd.read_file(str(data_dir) + "/pel/Polygons_Unit5_zontations.gpkg")
# print("could not find fuels dataset, building.....")

## create buffer around each unit plot
gdf = gpd.read_file(str(data_dir) + "/unit_5/unit_5.shp")
gdf = gdf.to_crs(epsg=26912)
gdf["geometry"] = gdf.geometry.buffer(10)
gdf.to_file(str(root_dir) + "/data/shp/extend_5.shp", driver="ESRI Shapefile")
gdf = gpd.read_file(str(root_dir) + "/data/shp/extend_5.shp")
gdf = gdf.to_crs(epsg=4326)
gdf.to_file(str(root_dir) + "/data/shp/extend_5.shp", driver="ESRI Shapefile")

gdf = gpd.read_file(str(root_dir) + "/data/shp/extend_5.shp")
gdf = gdf.to_crs(epsg=26912)
gdf["geometry"] = gdf.geometry.buffer(10)
gdf.to_file(str(root_dir) + "/data/shp/buff_5.shp", driver="ESRI Shapefile")
gdf = gpd.read_file(str(root_dir) + "/data/shp/buff_5.shp")
gdf = gdf.to_crs(epsg=4326)
gdf.to_file(str(root_dir) + "/data/shp/buff_5.shp", driver="ESRI Shapefile")

gdf = gpd.read_file(fireshape_path + ".shp")
gdf = gdf.to_crs(epsg=26912)
gdf["geometry"] = gdf.geometry.buffer(buff)
gdf.to_file(str(root_dir) + "/data/shp/unit_buffer.shp", driver="ESRI Shapefile")
gdf = gpd.read_file(str(root_dir) + "/data/shp/unit_buffer.shp")
gdf = gdf.to_crs(epsg=4326)
gdf.to_file(str(root_dir) + "/data/shp/unit_buffer.shp", driver="ESRI Shapefile")

# read shape file
polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True)
polygons_buff = bm.readshapefile(
    str(root_dir) + "/data/shp/unit_buffer",
    name="buff",
    drawbounds=True,
    color="blue",
)

fuel = np.full_like(XLONG, fueltype)
ravel_array = np.array(list(zip(XLONG.ravel(), XLAT.ravel())))
for i in range(len(bm.units)):
    unit = path.Path(bm.units[i])
    unit_mask = unit.contains_points(ravel_array)
    unit_mask = np.reshape(unit_mask, np.shape(XLONG))
    buff = path.Path(bm.buff[i])
    buffer_mask = buff.contains_points(ravel_array)
    buffer_mask = np.reshape(buffer_mask, np.shape(XLONG))
    fuel[buffer_mask != unit_mask] = 14  ## define fire breaks as non-fuel


for i in range(len(bm.units)):
    unit = path.Path(bm.units[i])
    unit_mask = unit.contains_points(ravel_array)
    unit_mask = np.reshape(unit_mask, np.shape(XLONG))
    fuel[unit_mask] = fueltype


polygons_buff5 = bm.readshapefile(
    str(root_dir) + "/data/shp/buff_5",
    name="buff_5",
    drawbounds=True,
    color="pink",
)

buff5 = path.Path(bm.buff_5[0])
buff5_mask = buff5.contains_points(ravel_array)
buff5_mask = np.reshape(buff5_mask, np.shape(XLONG))
fuel[buff5_mask] = 14  ## define fire breaks as non-fuel

polygons_extend5 = bm.readshapefile(
    str(root_dir) + "/data/shp/extend_5",
    name="extend5",
    drawbounds=True,
    color="red",
)

extend5 = path.Path(bm.extend5[0])
extend5_mask = extend5.contains_points(ravel_array)
extend5_mask = np.reshape(extend5_mask, np.shape(XLONG))
fuel[extend5_mask] = fueltype  ## define fire breaks as non-fuel


da = xr.DataArray(
    name="fuel",
    data=fuel,
    dims=["XLAT", "XLONG"],
    coords=dict(
        lon=(["XLAT", "XLONG"], XLONG),
        lat=(["XLAT", "XLONG"], XLAT),
    ),
    attrs=dict(
        description="WRF-SFIRE FUELS MAP.",
    ),
)
ds_fuel = da.to_dataset()
ds_fuel.to_zarr(str(save_dir) + f"/fuel{fueltype}.zarr", mode="w")


dim_header_fire = ",".join(map(str, np.shape(fuel.T)))
np.savetxt(
    str(save_dir) + "/input_fc",
    fuel.T,
    header=dim_header_fire,
    comments="",
    fmt="%d",
)

# sanity-check plot
plt.figure(figsize=(10, 8))
contour = bm.contourf(XLONG, XLAT, fuel)
plt.colorbar(contour, orientation="horizontal")
plt.title("ENTIRE LES DOMAIN WITH FIRE PLOT")
plt.savefig(str(img_dir) + "/les-domain.png")
plt.show()

# %%
# Part IIb: locate ignition -----------------------------------
## find ignition location on wrf grid based on lat and lon
# try:
#     ## try and open kdtree for domain
#     dmtree, dmlocs = pickle.load(open(str(save_dir) + f"/dmtree.p", "rb"))
#     print("Found Domain Tree")
# except:
## build a kd-tree for fwf domain if not found
print("Could not find Domain KDTree building.")
## create dataframe with columns of all lat/long in the domianrows are cord pairs
dmlocs = pd.DataFrame({"XLAT": XLAT.ravel(), "XLONG": XLONG.ravel()})
## build kdtree
dmtree = KDTree(dmlocs)
## save tree
pickle.dump([dmtree, dmlocs], open(str(save_dir) + f"/dmtree.p", "wb"))
print("Domain KDTree built")

sw_bool = np.array(sw).reshape(1, -1)
ne_bool = np.array(ne).reshape(1, -1)
sw_dist, sw_ind = dmtree.query(sw_bool, k=1)
ne_dist, ne_ind = dmtree.query(ne_bool, k=1)
sw_ = np.unravel_index(int(sw_ind), XLAT.shape)
ne_ = np.unravel_index(int(ne_ind), XLAT.shape)
nsew = [ne_[0], sw_[0], ne_[1], sw_[1]]


igs_bool = np.array(ig_start).reshape(1, -1)
ige_bool = np.array(ig_end).reshape(1, -1)
igs_dist, igs_ind = dmtree.query(igs_bool, k=1)
ige_dist, ige_ind = dmtree.query(ige_bool, k=1)
igs = np.unravel_index(int(igs_ind), XLAT.shape)
ige = np.unravel_index(int(ige_ind), XLAT.shape)


# UNIT 4 WESTERLY
plt.figure(figsize=(10, 8))
plt.title("CLOSEUP OF THE FIRE PLOT")
bmX = Basemap(
    llcrnrlon=XLONG[nsew[1], nsew[3]],
    llcrnrlat=XLAT[nsew[1], nsew[3]],
    urcrnrlon=XLONG[nsew[0], nsew[2]],
    urcrnrlat=XLAT[nsew[0], nsew[2]],
    epsg=4326,
)
polygonsX = bmX.readshapefile(
    fireshape_path, name="units", drawbounds=True, color="red"
)

bmX.contourf(
    XLONG[nsew[1] : nsew[0], nsew[3] : nsew[2]],
    XLAT[nsew[1] : nsew[0], nsew[3] : nsew[2]],
    fuel[nsew[1] : nsew[0], nsew[3] : nsew[2]],
)
plt.colorbar(orientation="horizontal", label="fuel category")
plt.scatter(
    XLONG[igs[0], igs[1]], XLAT[igs[0], igs[1]], c="red", label="ignition start"
)
plt.scatter(
    XLONG[ige[0], ige[1]],
    XLAT[ige[0], ige[1]],
    c="red",
    marker="*",
    s=60,
    label="ignition end",
)
plt.legend()
plt.savefig(str(img_dir) + "/unit-domain.png")
plt.show()


# #UNIT 5 SOUTHERLY
print("fire_ignition_start_x1 = %s" % gridx[igs[0], igs[1]])
print("fire_ignition_start_y1 = %s" % gridy[igs[0], igs[1]])
print("fire_ignition_end_x1 = %s" % gridx[ige[0], ige[1]])
print("fire_ignition_end_y1 = %s" % gridy[ige[0], ige[1]])

# %%
# Part III: Generate sounding from forecast-----------------------------------

"""
Needed format:
P0(mb)    T0(K)    Q0(g/kg)
z1(m)     T1       Q1       U1(m/s)   V1(m/s)

zn(m)     Tn       Qn       Un(m/s)   Vn(m/s)
"""

# # pull correct forecast
# today = pd.Timestamp.today()
# # metpath = '/content/drive/Shareddrives/WAN00CG-01/' + today.strftime(format='%y%m%d') + \
# #       '00/wrfout_d02_' + today.strftime(format='%Y-%m-') + str(today.day + 1) + '_' + str(rxtime + utm) + ':00:00'
# metpath = str(met_dir) + "/WAN00CP-04/19051100/wrfout_d03_2019-05-11_23:00:00"
# ds_wrf = xr.open_dataset(metpath)


# # #find closest lat, lon (kd-tree) to the fire
# locs = pd.DataFrame(
#     {"XLAT": ds_wrf["XLAT"].values.ravel(), "XLONG": ds_wrf["XLONG"].values.ravel()}
# )
# fire_loc = np.array(rxloc).reshape(1, -1)
# tree = KDTree(locs)
# dist, ind = tree.query(fire_loc, k=1)
# iz, ilat, ilon = np.unravel_index(ind[0], shape=np.shape(ds_wrf.variables["XLAT"]))


# # #get surface vars
# surface = [
#     float(ds_wrf["PSFC"][0, ilat, ilon] / 100.0),
#     float(ds_wrf["T2"][0, ilat, ilon]),
#     float(ds_wrf["Q2"][0, ilat, ilon] / 1000.0),
# ]

# # get height vector from geopotential
# zstag = (ds_wrf["PHB"][0, :, ilat, ilon] + ds_wrf["PH"][0, :, ilat, ilon]) // 9.81
# Z = np.squeeze(wrf.destagger(zstag, 0))

# # get profiles
# T = np.squeeze(ds_wrf["T"][0, :, ilat, ilon] + 300)
# Q = np.squeeze(ds_wrf["QVAPOR"][0, :, ilat, ilon] / 1000.0)
# U = np.squeeze(ds_wrf["U"][0, :, ilat, ilon])
# V = np.squeeze(ds_wrf["V"][0, :, ilat, ilon])
# sounding = np.column_stack((Z, T, Q, U, V))

# P = np.squeeze(ds_wrf["P"][0, :, ilat, ilon] + ds_wrf["PB"][0, :, ilat, ilon]) / 1000

# TD = (
#     (1 / 273.15) - (1.844e-4) * np.log((Q * 1000 * P) / (0.6113 * (Q * 1000 + 0.622)))
# ) ** -1 - 273.15

# temp = getvar(Dataset(metpath, "r"), "temp", meta=True)
# temp = np.squeeze(temp[:, ilat, ilon])

# RH = getvar(Dataset(metpath, "r"), "rh", meta=True)
# RH = np.squeeze(RH[:, ilat, ilon])

# Rd = 287  # Units (Jkg^-1K^-1)
# cp = 1004.0  # Units (Jkg^-1K^-1)
# P0 = 1000.0  # Units (hPa)
# theta = T * (P0 / P * 10) ** (Rd / cp)


# # profile plot
# fig, ax = plt.subplots(1, 4, figsize=(16, 10))
# fig.suptitle("input sounding", fontsize=16, fontweight="bold")
# ax[0].set_ylabel("Height (m)", fontsize=14)
# ax[0].set_xlabel("$Temp$ (K)", fontsize=14)
# ax[0].plot(temp, Z, color="red", linewidth=4)
# ax[1].set_xlabel("$Potential Temp$ (K)", fontsize=14)
# ax[1].plot(T, Z, color="purple", linewidth=4)
# ax[2].plot(RH, Z, color="green", linewidth=4)
# ax[2].set_xlabel("RH (%)", fontsize=14)
# wsp = (U ** 2 + V ** 2) ** 0.5
# Xq, Yq = np.meshgrid(np.max(wsp) + 8, Z)
# ax[3].plot(wsp, Z, color="k", linewidth=4)
# ax[3].barbs(Xq, Yq, U, V, color="k", linewidth=2)
# ax[3].set_xlabel("Wsp/Dir (m s-1)", fontsize=14)
# plt.show()


# # profile plot
# inx = 10
# fig, ax = plt.subplots(1, 4, figsize=(16, 10))
# fig.suptitle("input sounding lower pbl", fontsize=16, fontweight="bold")
# ax[0].set_ylabel("Height (m)", fontsize=14)
# ax[0].set_xlabel("$Temp$ (C)", fontsize=14)
# ax[0].plot(temp[:inx] - 273.15, Z[:inx], color="red", linewidth=4)
# ax[1].plot(T[:inx] - 273.15, Z[:inx], color="purple", linewidth=4)
# ax[1].set_xlabel("$Potential Temp$ (C)", fontsize=14)
# ax[2].plot(RH[:inx], Z[:inx], color="green", linewidth=4)
# ax[2].set_xlabel("RH (%)", fontsize=14)
# wsp = (U[:inx] ** 2 + V[:inx] ** 2) ** 0.5
# Xq, Yq = np.meshgrid(np.max(wsp * (60 * 60 / 1000)) + 4, Z[:inx])
# ax[3].plot(wsp * (60 * 60 / 1000), Z[:inx], color="k", linewidth=4)
# ax[3].barbs(Xq, Yq, U[:inx], V[:inx], color="k", linewidth=2)
# ax[3].set_xlabel("Wsp/Dir (km hr-1)", fontsize=14)
# plt.show()


# # #save sounding data input field
# sounding_header = " ".join(map(str, surface))
# np.savetxt(
#     str(save_dir) + "/input_sounding",
#     sounding,
#     header=sounding_header,
#     comments="",
#     fmt="%d",
# )


# %%
# #Part IV: Visualization tools  -----------------------------------
# #EVERYTHNG BELOW THIS STEP IS OPTIONAL TOOLS FOR VISUALIZATION
# #This installs a some mad satellite data hub, cuz the old script for WMS landsat pulling is dead

# #This configures it to my personal account: I am on trial for 30 days (April 21, 2021) - it will likely die after
# from sentinelhub import SHConfig
# from PIL import Image
# from io import BytesIO


# config = SHConfig()
# config.instance_id = 'e0803795-1f6d-4887-bd64-e619f6d7a17a'
# config.sh_client_id = '22f4a73a-6cc5-4856-8e34-5e61196b496b'
# config.sh_client_secret = '8ebcd565-9d69-4bed-8885-8383e5200c6d'
# config.save()

# #Subset the landsat times to needed bounding box
# from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox, DataCollection
# bounding_coords = [UTMx[0,0], UTMy[0,0], UTMx[-1,-1], UTMy[-1,-1]]
# get_bbox = BBox(bbox=bounding_coords, crs=CRS.UTM_12N)

# layer = 'TRUE_COLOR'
# time='latest'
# width=512
# height=856

# #pull the landsat image
# wms_true_color_request = WmsRequest(
#     data_folder= str(root_dir) + '/img/test/',
#     data_collection=DataCollection.SENTINEL2_L1C,
#     layer= layer,
#     bbox=get_bbox,
#     time=time,
#     width=width,
#     height=height,
#     image_format=MimeType.TIFF,
#     config=config)
# wms_true_color_img = wms_true_color_request.get_data(save_data=True)


# #This plots the image you pull on the current basemap instance
# def plot_image(image, factor=1):
#     """
#     Utility function for plotting RGB images.
#     """
#     fig = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))

#     if np.issubdtype(image.dtype, np.floating):
#         bm.imshow(np.minimum(image * factor, 1))
#         print('1')
#     else:
#         bm.imshow(image, origin='upper')
#         print('2')


# plot_image(wms_true_color_img[-1])


# # fig = plt.subplots(nrows=1, ncols=1, frameon=False)
# bm.imshow(wms_true_color_img[-1], origin='upper')
# plt.savefig('/Users/rodell/Desktop/test2.tiff', bbox_inches='tight',pad_inches=0)
# # #This saves image to disk as geotiff
# # wms_true_color_request = WmsRequest(
# #     data_collection=DataCollection.SENTINEL2_L1C,
# #     data_folder= str(root_dir) + '/img/',
# #     layer= layer,
# #     bbox=get_bbox,
# #     time=time,
# #     width=width,
# #     height=height,
# #     image_format=MimeType.TIFF,
# #     config=config)
# # wms_true_color_request.save_data()
