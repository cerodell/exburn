#!/bluesky/fireweather/miniconda3/envs/fwf/bin/python
import context
import wrf
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from netCDF4 import Dataset
from sklearn.neighbors import KDTree

from context import root_dir




def get_ibc(rxtime, utm, rxloc, wrf_run):
    '''
    Generate sounding from forecast

    Needed format:
    P0(mb)    T0(K)    Q0(g/kg)
    z1(m)     T1       Q1       U1(m/s)   V1(m/s)
    ...       ...      ...      ...       ...
    zn(m)     Tn       Qn       Un(m/s)   Vn(m/s)

    returns
        - writes inpit_sound.YYYYMMDDHH as text file
    '''
    #pull correct forecast
    today = pd.Timestamp.today()

    forecast_date = pd.Timestamp("today").strftime(f"%Y%m%d{wrf_run}")
    forecast_hour = pd.Timestamp("today") + pd.DateOffset(days=1)
    if rxtime + utm < 10:
        forecast_hour = forecast_hour.strftime(f"%Y-%m-%d_0{rxtime + utm}:00:00") 
    else:
        forecast_hour = forecast_hour.strftime(f"%Y-%m-%d_{rxtime + utm}:00:00") 

    metpath = f'/bluesky/working/wrf2arl/WAN{wrf_run}CG-01/{forecast_date}/wrfout_d02_{forecast_hour}'
    print(f"Forecast file for IBC: ~/WAN{wrf_run}CG-01/{forecast_date}/wrfout_d02_{forecast_hour}")

    # metpath = f'/bluesky/working/wrf2arl/WAN{wrf_run}CG-01/{forecast_date}/wrfout_d02_' + today.strftime(format='%Y-%m-') + str(today.day + 1) + '_' + str(rxtime + utm) + ':00:00'
    ds = xr.open_dataset(metpath)
    print(f"Time from netcdf {ds.XTIME.values}")

    #find closest lat, lon (kd-tree) to the fire
    locs = pd.DataFrame({'XLAT': ds['XLAT'].values.ravel(), 'XLONG': ds['XLONG'].values.ravel()})
    fire_loc = np.array(rxloc).reshape(1, -1)
    tree = KDTree(locs)
    dist, ind = tree.query(fire_loc, k=1) 
    iz,ilat,ilon =  np.unravel_index(ind[0],shape = np.shape(ds.variables['XLAT']))

    #get surface vars
    surface = [float(ds['PSFC'][0,ilat,ilon]/100.),float(ds['T2'][0,ilat,ilon]),float(ds['Q2'][0,ilat,ilon]/1000.)]

    #get height vector from geopotential
    zstag = (ds['PHB'][0,:,ilat,ilon] + ds['PH'][0,:,ilat,ilon])//9.81
    Z = np.squeeze(wrf.destagger(zstag,0))

    #get profiles
    T = np.squeeze(ds['T'][0,:,ilat,ilon] + 300)
    Q = np.squeeze(ds['QVAPOR'][0,:,ilat,ilon]/1000.)
    U = np.squeeze(ds['U'][0,:,ilat,ilon])
    V = np.squeeze(ds['V'][0,:,ilat,ilon])

    sounding = np.column_stack((Z,T,Q,U,V))

    # #save sounding data input field
    sounding_header = ' '.join(map(str, surface))  
    np.savetxt(str(root_dir) + f'/burns/inputs/input_sounding.{forecast_date}',sounding,header=sounding_header,comments='',fmt='%d')
