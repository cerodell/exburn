import context
import wrf
import json
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from context import root_dir, vol_dir, data_dir, save_dir, gog_dir
import matplotlib.pylab as pylab


g = 9.81  # gravity
pm_ef = 21.05  # boreal wildfire emission factor Urbanski (2014)
zstep = 20.0  # vertical step to interpolate to
BLfrac = 0.75  # fraction of BL height to set zs at
interpz = np.arange(0, 4000, zstep)


with open(str(data_dir) + "/json/config.json") as f:
    config = json.load(f)

fireshape_path = str(gog_dir) + "/all_units/mygeodata_merged"

wrfrun = "/sfire/unit5/moist_false/"
filein = str(vol_dir) + wrfrun
save_dir = str(save_dir) + wrfrun
int_ds = xr.open_zarr(str(filein) + "/wrfout_unit5_ll.zarr")


aqs = config["unit5"]["obs"]["aq"]
# south_north = config['unit5']['sfire']['met']['south_north']
# west_east = config['unit5']['sfire']['met']['west_east']
# south_north_subgrid = config['unit5']['sfire']['fire']['south_north']
# west_east_subgrid = config['unit5']['sfire']['fire']['west_east']

south_north = [110, 400]
west_east = [10, 150]
south_north_subgrid = [550, 620]
west_east_subgrid = [330, 405]


wrf_ds = int_ds.isel(
    south_north=slice(south_north[0], south_north[1]),
    west_east=slice(west_east[0], west_east[1]),
    south_north_subgrid=slice(south_north_subgrid[0], south_north_subgrid[1]),
    west_east_subgrid=slice(west_east_subgrid[0], west_east_subgrid[1]),
)
# Time= slice(10,100,2))


XLAT, XLONG = wrf_ds.XLAT.values, wrf_ds.XLONG.values
dimT = len(wrf_ds.Times)


# ## create dataframe with columns of all lat/long
# wrf_locs = pd.DataFrame({"XLAT": XLAT.ravel(), "XLONG": XLONG.ravel()})
# ## build kdtree
# wrf_tree = KDTree(wrf_locs)
# print("WRF Domain KDTree built")

# def find_index(aq):
#     aq = np.array([aqs[aq]['lat'],aqs[aq]['lon']]).reshape(1, -1)
#     aq_dist, aq_ind = wrf_tree.query(aq, k=1)
#     aq_loc = list(np.unravel_index(int(aq_ind), XLAT.shape))
#     return aq_loc

# aqs_loc = np.stack([find_index(aq) for aq in aqs])
# south_north = xr.DataArray(np.array(aqs_loc[:,0]), dims= 'aqs', coords= dict(aqs = list(aqs)))
# west_east = xr.DataArray(np.array(aqs_loc[:,1]), dims= 'aqs', coords= dict(aqs = list(aqs)))

# def smoke2m(ds):
#     zstag = (ds['PHB'] + ds['PH'])/ g
#     z = wrf.destagger(zstag,0)
#     tr17_1 = ds.tr17_1
#     tracer = wrf.interplevel(z, tr17_1, 2.0) / pm_ef
#     tracer = tracer.assign_coords({"Time": ds.XTIME.values.astype("datetime64")}).expand_dims({"Time": 1}).rename({'dim_1': 'south_north','dim_2': 'west_east'})
#     return tracer

# # ds = wrf_ds.isel(Time = 0)
# # ds.XTIME.values.astype("datetime64[s]")

# smoke = xr.concat([smoke2m(wrf_ds.isel(Time = i)) for i in range(len(wrf_ds.Time))], 'Time')
# smoke = smoke.to_dataset().rename_vars({'field3d_interp': 'smoke'})
# smoke = smoke.sel(south_north = south_north, west_east = west_east)
# smoke.smoke.plot.line(x="Time")


# aq_smoke = [smoke[:,aq_loc[0],aq_loc[1]] for aq_loc in aqs_loc]

# def find_index(aq):
#     aq = np.array([aqs[aq]['lat'],aqs[aq]['lon']]).reshape(1, -1)
#     aq_dist, aq_ind = wrf_tree.query(aq, k=1)
#     aq_loc = np.unravel_index(int(aq_ind), XLAT.shape)
#     plt.plot()
#     return aq_loc


fig = plt.figure(figsize=(4, 8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0.05)
bm = Basemap(
    llcrnrlon=XLONG[0, 0],
    llcrnrlat=XLAT[0, 0],
    urcrnrlon=XLONG[-1, -1],
    urcrnrlat=XLAT[-1, -1],
    epsg=4326,
)
polygons = bm.readshapefile(fireshape_path, name="units", drawbounds=True)

bm.scatter(aqs["303–100"]["lon"], aqs["303–100"]["lat"], zorder=10)
bm.scatter(aqs["303–200"]["lon"], aqs["303–200"]["lat"], zorder=10)
bm.scatter(aqs["303–300"]["lon"], aqs["303–300"]["lat"], zorder=10)
bm.scatter(aqs["401–100"]["lon"], aqs["401–100"]["lat"], zorder=10)
bm.scatter(aqs["401–200"]["lon"], aqs["401–200"]["lat"], zorder=10)

ds = wrf_ds.isel(Time=44)
ax.set_title(ds.Times.values.astype(str))
zstag = (ds["PHB"] + ds["PH"]) / g
z = wrf.destagger(zstag, 0)
tr17_1 = ds.tr17_1.values
tracer = wrf.interplevel(z, tr17_1, 2.0) / pm_ef
# tracer = wrf.smooth2d(tracer, 3, cenweight=4)
smoke = bm.contourf(
    XLONG,
    XLAT,
    tracer,
    zorder=1,
    cmap="cubehelix_r",
    levels=np.arange(0, 200, 10),
    extend="max",
)
# fig.colorbar(smoke, cax=cax)
cb = bm.colorbar(smoke, "right", size="5%", pad="1%")

# def update_plot(i):
#     global smoke
#     for c in smoke.collections: c.remove()
#     print(i)
#     ds = wrf_ds.isel(Time = i)
#     ax.set_title(ds.Times.values.astype(str))
#     zstag = (ds['PHB'] + ds['PH'])/ g
#     z = wrf.destagger(zstag,0)
#     tr17_1 = ds.tr17_1.values
#     tracer = wrf.interplevel(z, tr17_1, 2.0) / pm_ef
#     # tracer = wrf.smooth2d(tracer, 3, cenweight=4)
#     smoke = bm.contourf(XLONG, XLAT, tracer, zorder = 1,  cmap="cubehelix_r", levels=np.arange(0, 200, 10), extend="max")
#     return smoke


# ani=animation.FuncAnimation(fig, update_plot, dimT, interval=3)
# # plt.show()
# ani.save(save_dir + '/smoke.mp4', writer='ffmpeg',fps=10, dpi=250)
# plt.close()

# #get geopotential array and convert to height
# zstag = (wrf_ds['PHB'] + wrf_ds['PH'])/ g
# z = wrf.destagger(zstag,0)
# z_ave = np.mean(z, (1,2))
# tr17_1 = wrf_ds.tr17_1.values
# tracer = wrf.interplevel(z, tr17_1, 2.0)

# bm.contourf(XLONG, XLAT, tracer, zorder = 1)
# bm.scatter(aqs['303–100']['lon'],aqs['303–100']['lat'])
# bm.scatter(aqs['303–200']['lon'],aqs['303–200']['lat'])
# bm.scatter(aqs['303–300']['lon'],aqs['303–300']['lat'])
# bm.scatter(aqs['401–100']['lon'],aqs['401–100']['lat'])
# bm.scatter(aqs['401–200']['lon'],aqs['401–200']['lat'])


# ## create dataframe with columns of all lat/long
# wrf_locs = pd.DataFrame({"XLAT": XLAT.ravel(), "XLONG": XLONG.ravel()})
# ## build kdtree
# wrf_tree = KDTree(wrf_locs)
# print("WRF Domain KDTree built")
# aq_303_100 = np.array([aqs['303–100']['lat'],aqs['303–100']['lon']]).reshape(1, -1)
# aq_303_100_dist, aq_303_100_ind = wrf_tree.query(aq_303_100, k=1)
# aq_303_100_loc = np.unravel_index(int(aq_303_100_ind), XLAT.shape)


# try:
#     ## try and open kdtree for domain
#     wrf_tree, wrf_locs = pickle.load(open(str(root_dir) + f"/data/tree/wrf-tree.p", "rb"))
#     print("Found WRF Domain Tree")
# except:
#     ## build a kd-tree for fwf domain if not found
#     print("Could not find Domain KDTree building.")
#     ## create dataframe with columns of all lat/long in the domianrows are cord pairs
#     wrf_locs = pd.DataFrame({"XLAT": XLAT.ravel(), "XLONG": XLONG.ravel()})
#     ## build kdtree
#     wrf_tree = KDTree(wrf_locs)
#     ## save tree
#     pickle.dump([wrf_tree, wrf_locs], open(str(root_dir) + f"/data/tree/wrf-tree.p", "wb"))
#     print("WRF Domain KDTree built")
