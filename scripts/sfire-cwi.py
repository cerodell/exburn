# %% [markdown]
# Stull

# %%
import context
import wrf
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.interpolate import interp1d, interp2d


import matplotlib.pyplot as plt
from context import root_dir, vol_dir, data_dir, save_dir

import matplotlib.pylab as pylab


# params = {
#          'xtick.labelsize':14,
#          'ytick.labelsize': 14,
#           'axes.labelsize':14,
#          }

# pylab.rcParams.update(params)

# %% [markdown]
# Open dataset
# %%

g = 9.81  # gravity
zstep = 20.0  # vertical step to interpolate to
BLfrac = 0.75  # fraction of BL height to set zs at

interpz = np.arange(0, 4000, zstep)

# x, y, z = np.meshgrid(x_, y_, interpz, indexing='ij')


wrfrun = "/sfire/unit5/moist_false/"
filein = str(vol_dir) + wrfrun
save_dir = str(save_dir) + wrfrun
ds = xr.open_zarr(str(filein) + "/wrfout_unit5.zarr")


# slice_ds = ds.isel(Time = slice(0,10), south_north = slice(100,399))

slice_ds = ds.isel(
    Time=slice(0, 168, 6), south_north=slice(100, 250), west_east=slice(40, 105)
)
slice_ds["Time"] = slice_ds.XTIME.values.astype("datetime64[s]")

smoke = slice_ds["tr17_1"].sum(dim="west_east").values
temp = slice_ds["T"].values + 300.0


# get height data
zstag = (slice_ds["PHB"].values + slice_ds["PH"].values) / g
zdestag = wrf.destagger(zstag, 1)

z0 = np.mean(zdestag, (0, 2, 3))
z0stag = np.mean(zstag, (0, 2, 3))


# find boundary layer height
T0 = np.mean(temp, (0, 2, 3))
interpfLES_nm = interp1d(z0, T0, fill_value="extrapolate")
soundingLES_nm = interpfLES_nm(interpz)

gradTLES_nm = (soundingLES_nm[1:] - soundingLES_nm[:-1]) / zstep
gradT2_nm = gradTLES_nm[1:] - gradTLES_nm[:-1]
ziidx_nm = np.argmax(gradT2_nm[10:]) + 10
zi_nm = interpz[ziidx_nm]

# plt.figure()
# plt.plot(T0,z0)
# plt.plot(soundingLES_nm,interpz)
# plt.show()
# plt.close()


def something(i, j, k):
    interpfLES = interp1d(z0, temp[k, :, j, i], fill_value="extrapolate")
    soundingLES = interpfLES(interpz)
    return soundingLES


soundingLES = np.stack(
    [
        [
            [something(i, j, k) for k in range(temp.shape[0])]
            for j in range(temp.shape[2])
        ]
        for i in range(temp.shape[3])
    ]
)


gradTLES = (soundingLES[:, :, :, 1:] - soundingLES[:, :, :, :-1]) / zstep
gradT2 = gradTLES[:, :, :, 1:] - gradTLES[:, :, :, :-1]
ziidx = np.argmax(gradT2[:, :, :, 10:], axis=3) + 10


zi = np.stack(
    [
        [
            [interpz[ziidx[i, j, k]] for i in range(ziidx.shape[0])]
            for j in range(ziidx.shape[1])
        ]
        for k in range(ziidx.shape[2])
    ]
)
PBLH = xr.DataArray(zi, name="PBLH", dims=("Time", "south_north", "west_east"))
slice_ds["PBLH"] = PBLH

zi_max = np.max(zi)
zi_min = np.min(zi)
zi_mid = (zi_max + zi_min) / 2
levels = np.arange(zi_min, zi_max + 20, 20)

slice_ds["PBLH"].plot(col="Time", col_wrap=4, cmap="coolwarm", levels=levels)
plt.savefig(str(save_dir) + "/PBLH.png")


# fig = plt.figure(figsize=(15, 3))
# # fig.suptitle(r"PM 2.5 ($\frac{\mu g}{m^3}$)", fontsize=16)
# ax = fig.add_subplot(1, 1, 1)
# ax.contourf(zi, cmap="Reds")
# zS = zi*BLfrac
# zsidx = np.argmin(abs(interpz - zS))
# plt.figure()
# plt.plot(T0,z0)
# plt.plot(soundingLES,interpz)
# plt.show()
# plt.close()
# cwi_ds.tr17_1.plot(cmap="cubehelix_r", levels=np.arange(0,30100,100), extend= 'max')


# fig = plt.figure(figsize=(15, 3))
# # fig.suptitle(r"PM 2.5 ($\frac{\mu g}{m^3}$)", fontsize=16)
# ax = fig.add_subplot(1, 1, 1)
# ax.contourf(smoke, cmap="cubehelix_r", levels=np.arange(0,20100,100),  extend= 'max')
