import context
import glob
import dask
import json
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from pathlib import Path
import pyproj as pyproj
from datetime import datetime
import cartopy.crs as ccrs

from context import data_dir, root_dir
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

with open(str(root_dir) + "/json/config.json") as f:
    config = json.load(f)


def compressor(ds):
    """
    this function comresses datasets
    """
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    return ds, encoding


def sovle_pbl(fueltype):
    ds = xr.open_dataset(
        str(data_dir) + f"/fuel{fueltype}/interp-unit5-theta_e.nc", chunks="auto"
    )

    ## get vertical levels and coverert to meters
    interp_level = ds.interp_level.values * 1000
    zstep = 20
    levelup = 30
    ## solve for PBL height
    end = len(interp_level) - levelup
    print(f"Bottom level {interp_level[levelup]}")
    # theta_e = ds.theta_e.isel(interp_level = slice(levelup,None), Time = slice(20,24), south_north = slice(10,15), west_east= slice(10,15))
    theta_e = ds.theta_e.isel(interp_level=slice(levelup, None))
    XLAT = theta_e.XLAT.values
    chunk = theta_e.chunks

    print("Solve Temp Gradient: ", datetime.now())
    statTIME = datetime.now()
    levels = theta_e.interp_level.values * 1000
    del theta_e["interp_level"]
    zi = theta_e.isel(interp_level=slice(1, end))
    zii = theta_e.isel(interp_level=slice(0, end - 1))

    # dask.config.set({"array.slicing.split_large_chunks": False})
    gradTLES = (zi - zii) / zstep
    end = len(gradTLES.interp_level)
    gradT2 = gradTLES.isel(interp_level=slice(1, end)) - gradTLES.isel(
        interp_level=slice(0, end - 1)
    )
    print("Temp Gradient Solved: ", datetime.now() - statTIME)

    print("Computing: ", datetime.now())
    statTIME = datetime.now()
    gradT2 = gradT2.compute()
    print("Computed: ", datetime.now() - statTIME)

    print("Build Height: ", datetime.now())
    statTIME = datetime.now()
    try:
        height = xr.open_dataarray(str(data_dir) + "/height.nc", mode="w")
    except:
        height = xr.DataArray(
            np.stack(
                [[np.full_like(XLAT, level) for level in levels]] * len(theta_e.Time)
            ),
            name="PBLH",
            dims=("Time", "interp_level", "south_north", "west_east"),
        ).chunk(chunk)
        height.to_netcdf(str(data_dir) + "/height.nc", mode="w")

    print("Height Built: ", datetime.now() - statTIME)

    print("Index Height: ", datetime.now())
    statTIME = datetime.now()
    PBLH = height.isel(gradT2.argmax(dim=["interp_level"]))
    print("Height Indexed: ", datetime.now() - statTIME)

    XLAT, XLONG = makeLL()
    PBLH = PBLH.to_dataset()
    PBLH["XLAT"] = XLAT
    PBLH["XLONG"] = XLONG

    print("Write PBLH: ", datetime.now())
    statTIME = datetime.now()
    PBLH, encoding = compressor(PBLH)
    PBLH.to_netcdf(str(data_dir) + "/PBLH.nc", encoding=encoding, mode="w")
    print("Write Time: ", datetime.now() - statTIME)
    return PBLH


def makeLL_new(domain, unit, stagger=True):
    with open(str(root_dir) + "/json/config-new.json") as f:
        config = json.load(f)
    bounds = config[unit]["sfire"]["namelist"]
    ds = bounds["dxy"]  # LES grid spacing
    fs = bounds["fs"]  # fire mesh ratio
    ndx = bounds["ndx"]  # EW number of grids
    ndy = bounds["ndy"]  # NS number of grids
    ll_utm = [
        bounds["ll_utm"][0],
        bounds["ll_utm"][1],
    ]  # lower left corner of the domain in UTM coordinates (meters)
    # ============ end of INPUTS==============
    ## create grid WRF LAT/LONG with defined inputs
    if domain == "met":
        gridx, gridy = np.meshgrid(
            np.arange(0, ds * ndx, int(ds)), np.arange(0, ds * ndy, int(ds))
        )
        ## stagger gird to fit on wrf_out
        if stagger == True:
            gridx, gridy = gridx - ds / 2, gridy - ds / 2
            XLAT_name, XLONG_name = "XLAT", "XLONG"
        elif stagger == False:
            XLAT_name, XLONG_name = "s_XLAT", "s_XLONG"
        else:
            raise ValueError("Stagger must be True or False")

        ## drop first row/colum to match size
        gridx, gridy = gridx[1:, 1:], gridy[1:, 1:]
        # now adding a georefernce we have UTM grid (technically UTM_12N, or EPSG:26912
        UTMx = gridx + ll_utm[0]
        UTMy = gridy + ll_utm[1]
        ## transform into the same projection (for shapefiles and for basemaps)
        wgs84 = pyproj.Proj("+init=EPSG:4326")
        epsg26912 = pyproj.Proj("+init=EPSG:26912")
        # reproject from UTM to WGS84
        WGSx, WGSy = pyproj.transform(epsg26912, wgs84, UTMx.ravel(), UTMy.ravel())
        XLONG, XLAT = np.reshape(WGSx, np.shape(UTMx)), np.reshape(WGSy, np.shape(UTMy))
        XLAT = xr.DataArray(
            name=XLAT_name,
            data=XLAT,
            dims=["south_north", "west_east"],
        )
        XLONG = xr.DataArray(
            name=XLONG_name,
            data=XLONG,
            dims=["south_north", "west_east"],
        )
    elif domain == "fire":
        ## create grid FIRE LAT/LONG with defined inputs
        fire_gridx, fire_gridy = np.meshgrid(
            np.arange(0, ds * ndx, int(ds / fs)), np.arange(0, ds * ndy, int(ds / fs))
        )
        fireX = xr.DataArray(
            name="fireX",
            data=fire_gridx,
            dims=["south_north_subgrid", "west_east_subgrid"],
        )
        fireY = xr.DataArray(
            name="fireY",
            data=fire_gridy,
            dims=["south_north_subgrid", "west_east_subgrid"],
        )
        # now adding a georefernce we have UTM grid (technically UTM_12N, or EPSG:26912
        fire_UTMx = fire_gridx + ll_utm[0]
        fire_UTMy = fire_gridy + ll_utm[1]
        ## transform into the same projection (for shapefiles and for basemaps)
        wgs84 = pyproj.Proj("+init=EPSG:4326")
        epsg26912 = pyproj.Proj("+init=EPSG:26912")
        # reproject from UTM to WGS84
        fire_WGSx, fire_WGSy = pyproj.transform(
            epsg26912, wgs84, fire_UTMx.ravel(), fire_UTMy.ravel()
        )
        fire_XLONG, fire_XLAT = np.reshape(fire_WGSx, np.shape(fire_UTMx)), np.reshape(
            fire_WGSy, np.shape(fire_UTMy)
        )
        XLAT = xr.DataArray(
            name="XLAT",
            data=fire_XLAT,
            dims=["south_north_subgrid", "west_east_subgrid"],
        )
        XLONG = xr.DataArray(
            name="XLONG",
            data=fire_XLONG,
            dims=["south_north_subgrid", "west_east_subgrid"],
        )
    else:
        raise ValueError("Invalid domain option")
    return XLAT, XLONG


def makeLL(domain, modelrun):
    bounds = config["unit5"]["sfire"][modelrun]["namelist"]
    ds = bounds["dxy"]  # LES grid spacing
    fs = bounds["fs"]  # fire mesh ratio
    ndx = bounds["ndx"]  # EW number of grids
    ndy = bounds["ndy"]  # NS number of grids
    ll_utm = [
        bounds["ll_utm"][0],
        bounds["ll_utm"][1],
    ]  # lower left corner of the domain in UTM coordinates (meters)
    # ============ end of INPUTS==============
    ## create grid WRF LAT/LONG with defined inputs
    if domain == "met":
        gridx, gridy = np.meshgrid(
            np.arange(0, ds * ndx, int(ds)), np.arange(0, ds * ndy, int(ds))
        )
        ## stagger gird to fit on wrf_out
        gridx, gridy = gridx - ds / 2, gridy - ds / 2
        ## drop first row/colum to match size
        gridx, gridy = gridx[1:, 1:], gridy[1:, 1:]
        # now adding a georefernce we have UTM grid (technically UTM_12N, or EPSG:26912
        UTMx = gridx + ll_utm[0]
        UTMy = gridy + ll_utm[1]
        ## transform into the same projection (for shapefiles and for basemaps)
        wgs84 = pyproj.Proj("+init=EPSG:4326")
        epsg26912 = pyproj.Proj("+init=EPSG:26912")
        # reproject from UTM to WGS84
        WGSx, WGSy = pyproj.transform(epsg26912, wgs84, UTMx.ravel(), UTMy.ravel())
        XLONG, XLAT = np.reshape(WGSx, np.shape(UTMx)), np.reshape(WGSy, np.shape(UTMy))
        XLAT = xr.DataArray(
            name="XLAT",
            data=XLAT,
            dims=["south_north", "west_east"],
        )
        XLONG = xr.DataArray(
            name="XLONG",
            data=XLONG,
            dims=["south_north", "west_east"],
        )
    elif domain == "fire":
        ## create grid FIRE LAT/LONG with defined inputs
        fire_gridx, fire_gridy = np.meshgrid(
            np.arange(0, ds * ndx, int(ds / fs)), np.arange(0, ds * ndy, int(ds / fs))
        )
        fireX = xr.DataArray(
            name="fireX",
            data=fire_gridx,
            dims=["south_north_subgrid", "west_east_subgrid"],
        )
        fireY = xr.DataArray(
            name="fireY",
            data=fire_gridy,
            dims=["south_north_subgrid", "west_east_subgrid"],
        )
        # now adding a georefernce we have UTM grid (technically UTM_12N, or EPSG:26912
        fire_UTMx = fire_gridx + ll_utm[0]
        fire_UTMy = fire_gridy + ll_utm[1]
        ## transform into the same projection (for shapefiles and for basemaps)
        wgs84 = pyproj.Proj("+init=EPSG:4326")
        epsg26912 = pyproj.Proj("+init=EPSG:26912")
        # reproject from UTM to WGS84
        fire_WGSx, fire_WGSy = pyproj.transform(
            epsg26912, wgs84, fire_UTMx.ravel(), fire_UTMy.ravel()
        )
        fire_XLONG, fire_XLAT = np.reshape(fire_WGSx, np.shape(fire_UTMx)), np.reshape(
            fire_WGSy, np.shape(fire_UTMy)
        )
        XLAT = xr.DataArray(
            name="XLAT",
            data=fire_XLAT,
            dims=["south_north_subgrid", "west_east_subgrid"],
        )
        XLONG = xr.DataArray(
            name="XLONG",
            data=fire_XLONG,
            dims=["south_north_subgrid", "west_east_subgrid"],
        )
    else:
        raise ValueError("Invalid domain option")
    return XLAT, XLONG


def prepare_df(rosin, ros_filein, times):
    headers = ["day", "hour", "minute", "second", "temp"]
    df = pd.read_csv(
        glob.glob(ros_filein + f"{rosin}*.txt")[0],
        sep="\t",
        index_col=False,
        skiprows=16,
        names=headers,
    )
    df["year"], df["month"] = "2019", "05"
    df = df[:-1]
    df["DateTime"] = pd.to_datetime(
        df[["year", "month"] + headers[:-1]], infer_datetime_format=True
    )
    df.drop(["year", "month"] + headers[:-1], axis=1, inplace=True)
    df = df.set_index("DateTime")
    df = df[~df.index.duplicated(keep="first")]
    upsampled = df.resample("1S")
    df = upsampled.interpolate(method="linear")
    df = df[str(times[0]) : str(times[-1])]
    df["DateTime"] = pd.to_datetime(df.index)
    return df


def ignition_line(modelrun, ax, ig_start, ig_end):
    if ("I04" in modelrun) == True:
        print("Multi line ignition")
        ignite(
            ax,
            ig_start=[55.7177529, -113.5713107],
            ig_end=[55.71773480, -113.57183453],
            line="1",
            color="tab:red",
        )
        ignite(
            ax,
            ig_start=[55.7177109, -113.5721005],
            ig_end=[55.7177124, -113.5725656],
            line="2",
            color="tab:blue",
        )
        ignite(
            ax,
            ig_start=[55.7177293, -113.5734885],
            ig_end=[55.7177437, -113.5744894],
            line="3",
            color="tab:green",
        )
        ignite(
            ax,
            ig_start=[55.7177775603, -113.5747705233],
            ig_end=[55.717752429, -113.575192125],
            line="4",
            color="tab:grey",
        )
    else:
        print("Single line ignition")
        ignite(ax, ig_start, ig_end, line="1", color="tab:red")


def ignite(ax, ig_start, ig_end, line, color):
    ax.scatter(
        ig_start[1],
        ig_start[0],
        c=color,
        s=200,
        marker="*",
        # alpha=0.6,
        label=f"{line} ignition start",
        zorder=10,
        edgecolors="black",
        transform=ccrs.PlateCarree(),
    )
    ax.scatter(
        ig_end[1],
        ig_end[0],
        c=color,
        marker="X",
        s=150,
        # alpha=0.6,
        label=f"{line} ignition end",
        zorder=10,
        edgecolors="black",
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        [ig_start[1], ig_end[1]],
        [ig_start[0], ig_end[0]],
        linestyle="--",
        lw=2,
        marker="",
        zorder=9,
        color="k",
        transform=ccrs.PlateCarree(),
        # alpha=0.6,
    )
    return
