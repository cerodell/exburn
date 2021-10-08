#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:08:25 2019

@author: crodell
"""


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from scipy.interpolate import griddata
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable


"######################  Adjust for user/times of interest/plot customization ######################"
##User and file location
user = "crodell"
# file    = 'UBC10475163_0'
filein = (
    "/Users/"
    + user
    + "/Google Drive File Stream/Shared drives/Research/CRodell/Pelican_Mnt_Fire/Data/hobo_v2/"
)
save = (
    "/Users/"
    + user
    + "/Google Drive File Stream/Shared drives/Research/CRodell/Pelican_Mnt_Fire/Images/hobo_v2/array/"
)

##Plot customization
ylabel = 14
fig_size = 20
tick_size = 12
title_size = 16


##Time of Interst...Start and Stop
start = datetime(2019, 5, 11, 17, 00)
start_str = start.strftime("%Y-%m-%d %H:%M")

stop = datetime(2019, 5, 11, 20, 00)
stop_str = stop.strftime("%Y-%m-%d %H:%M")

wesn = [-113.58732, -113.56174, 55.71347, 55.73186]


locs = {
    "Hobo 01": [-113.57310, 55.72225],
    "Hobo 03": [-113.57010, 55.71806],
    "Hobo 07": [-113.57362, 55.72039],
    "Hobo 09": [-113.58120, 55.72463],
    "Hobo 11": [-113.57655, 55.71809],
    "Hobo 13": [-113.57414, 55.71719],
    "Hobo 14": [-113.57356, 55.72124],
    "Hobo 16": [-113.57845, 55.71871],
    "Hobo 17": [-113.57930, 55.71961],
    "Hobo 20": [-113.56644, 55.72111],
    "Hobo 21": [-113.57783, 55.72118],
}


"######################  Read File ######################"
temp_array, rh_array, time_array, hobo = [], [], [], []


hobo_list = [
    "HOBO_01",
    "HOBO_03",
    "HOBO_07",
    "HOBO_09",
    "HOBO_11",
    "HOBO_13",
    "HOBO_14",
    "HOBO_16",
    "HOBO_17",
    "HOBO_20",
    "HOBO_21",
]

# hobo_sorted = hobo.sorted()
for i in range(len(hobo_list)):
    path_list = filein + hobo_list[i]
    df_hobo = pd.read_csv(path_list + ".csv", skiprows=[0])
    hobo.append(path_list[-2:])

    ##Make a list of foats of each variable
    time = list(df_hobo[df_hobo.columns[1]])
    temp, rh = np.array(df_hobo[df_hobo.columns[2]]), np.array(
        df_hobo[df_hobo.columns[3]]
    )

    time, temp, rh = time[:-2], temp[:-2], rh[:-2]

    ##############################################
    #######Create a colum of times in MDT
    ##############################################
    mdt = []

    def datetime_range(start, end, delta):
        current = start
        while current < end:
            yield current
            current += delta

    dts = [
        dt.strftime("%Y-%m-%d %H:%M")
        for dt in datetime_range(
            datetime(2019, 5, 6, 16, 30),
            datetime(2019, 5, 21, 23, 55),
            timedelta(minutes=5),
        )
    ]

    for i in range(len(time)):
        mdt.append(dts[i])

    ##Find idex with start and stop time then make new list of each variable within that time interval
    time_start = mdt.index(start_str)
    time_stop = mdt.index(stop_str) + 1
    time = mdt[time_start:time_stop]
    temp, rh = temp[time_start:time_stop], rh[time_start:time_stop]
    temp_array.append(temp)
    rh_array.append(rh)
    time = [i[-5:] for i in time]
    time_array.append(time)


temp_array = np.array(temp_array)
rh_array = np.array(rh_array)

lat = [
    55.72225,
    55.71806,
    55.72039,
    55.72463,
    55.71809,
    55.71719,
    55.72124,
    55.71871,
    55.71961,
    55.72111,
    55.72118,
]

lon = [
    -113.57310,
    -113.57010,
    -113.57362,
    -113.58120,
    -113.57655,
    -113.57414,
    -113.57356,
    -113.57845,
    -113.57930,
    -113.56644,
    -113.57783,
]

# lat = np.append(lat, [wesn[2], wesn[3], wesn[3], wesn[2]])
# lon = np.append(lon, [wesn[0], wesn[1], wesn[0], wesn[1]])

lat = np.array(lat)
lon = np.array(lon)


# generate grid data
numcols, numrows = 240, 240
xi = np.linspace(lon.min(), lon.max(), numcols)
yi = np.linspace(lat.min(), lat.max(), numrows)
xi, yi = np.meshgrid(xi, yi)


img = mpimg.imread(filein[:-8] + "png/site_map.png")

wesn = [-113.58732, -113.56174, 55.71347, 55.73186]
time_save = [s.replace(":", "") for s in time]


v = np.linspace(20, 35, 31)
Cnorm = colors.Normalize(vmin=20, vmax=35)


for i in range(len(time)):
    # interpolate, there are better methods, especially if you have many datapoints
    zi = griddata((lon, lat), temp_array[:, i], (xi, yi), method="cubic")

    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(
        img[:, :, :], extent=wesn
    )  # overlay this image over the original background,

    for key in locs.keys():
        x = float(locs[key][0])
        y = float(locs[key][1])
        ax.scatter(x, y, c="k", s=2, zorder=10)
    #        ax.annotate(key, xy = (x,y), color='w',xytext = (x,y),\
    #                     bbox = dict(boxstyle="round", fc="black", ec="b",alpha=.4))
    C = ax.contourf(xi, yi, zi, alpha=0.7, norm=Cnorm, cmap="YlOrRd", levels=v)
    ax.set_title("May 11th  " + time_save[i] + " (MDT)", fontsize=16)
    ax.set_xlim(-113.582, -113.566)
    ax.set_ylim(55.7168, 55.7250)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = plt.colorbar(C, cax=cax)
    clb.set_label("Temperature (\N{DEGREE SIGN}C)", fontsize=ylabel)
    clb.ax.tick_params(labelsize=tick_size)
    clb.set_alpha(0.95)
    clb.draw_all()

    fig.savefig(save + "v2_Array_Temp" + time_save[i])


v = np.linspace(15, 40, 51)
Cnorm = colors.Normalize(vmin=15, vmax=40)

for i in range(len(time)):
    # interpolate, there are better methods, especially if you have many datapoints
    zi = griddata((lon, lat), rh_array[:, i], (xi, yi), method="cubic")

    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(
        img[:, :, :], extent=wesn
    )  # overlay this image over the original background,

    for key in locs.keys():
        x = float(locs[key][0])
        y = float(locs[key][1])
        ax.scatter(x, y, c="k", s=4, zorder=10)
    #        ax.annotate(key, xy = (x,y), color='w',xytext = (x,y),\
    #                     bbox = dict(boxstyle="round", fc="black", ec="b",alpha=.4))
    C = ax.contourf(xi, yi, zi, alpha=0.7, norm=Cnorm, cmap=plt.cm.BuPu, levels=v)
    ax.set_title("May 11th  " + time_save[i] + " (MDT)", fontsize=16)
    ax.set_xlim(-113.582, -113.566)
    ax.set_ylim(55.7168, 55.7250)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = plt.colorbar(C, cax=cax)
    clb.set_label("Relative Humidity (%)", fontsize=ylabel)
    clb.ax.tick_params(labelsize=tick_size)
    clb.set_alpha(0.95)
    clb.draw_all()

    fig.savefig(save + "v2_Array_RH" + time_save[i])
