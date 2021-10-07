#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import context
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd

from metpy.calc import wind_components, mixing_ratio_from_relative_humidity

# from metpy.plots import Hodograph, SkewT
from metpy.units import units
from context import data_dir


user = "crodell"
file = "2019-05-11_1456"
# filein  = '/Users/'+user+'/Google Drive File Stream/Shared drives/Research/CRodell/Pelican_Mnt_Fire/Data/windsond/'+file+'.sounding.csv'
# save    = '/Users/'+user+'/Google Drive File Stream/Team Drives/research/CRodell/Pelican_Mnt_Fire/Images/'


ylabel = 14
fig_size = 20
tick_size = 12
title_size = 16


"######################  Read File ######################"
df_sonde = (
    pd.read_csv(str(data_dir) + "/sounding.csv", skiprows=[0])
    .replace(r"  ", np.NaN, regex=True)
    .astype(float)
)

# df_sonde = df_sonde.replace(r'  ', np.NaN, regex=True)
##Make a list of foats of each variable
height, press = list(df_sonde["Height (m AGL)"]), list(df_sonde[" Pressure (mb)"])
temp, rh = list(df_sonde[" Temperature (C)"]), list(df_sonde[" Relative humidity (%)"])
wsp_si, wdir = list(df_sonde[" Wind speed (m/s)"]), list(
    df_sonde[" Wind direction (true deg)"]
)

wsp = np.array(wsp_si) * 3.6

"#################Solve for Dew Point ###################"
##Solve for Dew Point (source: http://irtfweb.ifa.hawaii.edu/~tcs3/tcs3/Misc/Dewpoint_Calculation_Humidity_Sensor_E.pdf)
##constant varibles needed
beta = 17.62
lambda_ = 243.15  # deg C

dew = []  # initiate list for Dew-Point

for i in range(len(temp)):
    dew_i = round(
        (lambda_ * ((np.log(rh[i] / 100)) + ((beta * temp[i]) / (lambda_ + temp[i]))))
        / (beta - (np.log((rh[i] / 100) + ((beta * temp[i]) / (lambda_ + temp[i]))))),
        3,
    )
    dew.append(dew_i)


"#################Solve for u, and LCL ###################"
##Add units so Metpy can use data
index = 3
tempC, dew, the_press, rh = (
    temp * units.degC,
    dew * units.degC,
    press * units.hPa,
    rh * units.percent,
)
wsp, wdir = wsp * units(("kilometer/hour")), wdir * units.degrees
wsp_si = wsp_si * units(("meter/second"))

##Calcualte u and v
u, v = wind_components(wsp, wdir)
u_si, v_si = wind_components(wsp_si, wdir)

w = mixing_ratio_from_relative_humidity(the_press, tempC, rh)

"###################### Make Potential Temperature Profile ######################"

Rd = 287  # Units (Jkg^-1K^-1)
cp = 1004.0  # Units (Jkg^-1K^-1)
P0 = 1000.0  # Units (hPa)

height = np.array(height)
Z = height
tempK = np.array(temp) + 273.15
press = np.array(press)

theta = tempK * (P0 / press) ** (Rd / cp)

wsp_kh = wsp  # * 3.6 ## ms^-1 to kmh^-1

##Plot customization
label = 12
fig_size = 20
tick_size = 12
title_size = 16


## make wrrf les input_sounding
T = np.squeeze(np.array(temp) + 273.15)
Q = np.squeeze(np.array(w))
U = np.squeeze(np.array(u_si))
V = np.squeeze(np.array(v_si))
sounding = np.column_stack((Z, T, Q, U, V))

# #get surface vars
surface = [
    float(np.array(the_press)[0]),
    float(T[0]),
    float(Q[0]),
]

# #save sounding data input field
sounding_header = " ".join(map(str, surface))
np.savetxt(
    str(data_dir) + "/wrfinput/input_sounding",
    sounding,
    header=sounding_header,
    comments="",
    fmt="%d",
)


############################################


fig, ax = plt.subplots(1, 4, figsize=(12, 6))
# fig.suptitle('Atmospheric Profile', fontsize=16, fontweight="bold")

fig.suptitle(
    "Atmospheric Profile \n Pelican Mountain 11 May 2019 1500 MDT", fontsize=16
)

ax[0].plot(temp, height, color="red", linewidth=4)
ax[0].plot(temp, height, color="red", linewidth=4)
ax[0].axhline(y=2395, color="black", linestyle="-")

ax[0].set_ylabel("Height (meters)", fontsize=label)
ax[0].set_xlabel("Temperature (C)", fontsize=label)
ax[0].xaxis.grid(color="gray", linestyle="dashed")
ax[0].yaxis.grid(color="gray", linestyle="dashed")
# ax[0].set_facecolor('lightgrey')

ax[1].plot(theta - 273.15, height, color="purple", linewidth=4)
ax[1].set_xlabel("Potential Temperature (C)", fontsize=label)
ax[1].axhline(y=2395, color="black", linestyle="-")
ax[1].set_yticklabels([])
ax[1].xaxis.grid(color="gray", linestyle="dashed")
ax[1].yaxis.grid(color="gray", linestyle="dashed")
# ax[1].set_facecolor('lightgrey')

ax[2].plot(rh, height, color="green", linewidth=4)
ax[2].set_xlabel("Relative Humidity (%)", fontsize=label)
ax[2].axhline(y=2395, color="black", linestyle="-")
ax[2].set_yticklabels([])
ax[2].xaxis.grid(color="gray", linestyle="dashed")
ax[2].yaxis.grid(color="gray", linestyle="dashed")
# ax[2].set_facecolor('lightgrey')

one = np.ones_like(wsp_kh) * 52
ax[3].barbs(
    one[0::5], height[0::5], u[0::5], v[0::5], color="black", zorder=10, length=5
)
ax[3].plot(wsp_kh, height, color="black", linewidth=4, zorder=1)
ax[3].set_xlim(0, 56)
ax[3].axhline(y=2395, color="black", linestyle="-")
ax[3].set_xlabel(r"Wind Speed $(\frac{km}{hr})$", fontsize=label)
ax[3].set_yticklabels([])
ax[3].xaxis.grid(color="gray", linestyle="dashed")
ax[3].yaxis.grid(color="gray", linestyle="dashed")
# ax[3].set_facecolor('lightgrey')

# ax[3].plot(theta-273.15,height, color = 'purple', linewidth = 4)


# #ax.legend(loc ='upper left')
# ax.tick_params(axis='both', which='major', labelsize=tick_size)
# #ax.get_legend().remove()
# #ax.set_xticklabels([])
# ax.xaxis.grid(color='gray', linestyle='dashed')
# ax.yaxis.grid(color='gray', linestyle='dashed')
# #ax.set_ylim(0,65)
# ax.set_facecolor('lightgrey')
# xfmt = DateFormatter('%m/%d %H:%M')
# ax.xaxis.set_major_formatter(xfmt)
# ax.legend(loc='upper center', bbox_to_anchor=(.50,1.16), shadow=True, ncol=4)
#
plt.show()
