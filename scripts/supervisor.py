#!/bluesky/fireweather/miniconda3/envs/fwf/bin/python

"""

Used to get input_sounding as IBCs to run WRF-SFIRE on cedar for Pelican Mnt Forecast Simulations

"""

import context
import os
import sys
import time
import pandas as pd
from pathlib import Path

from utils.ibc import get_ibc
from context import wrf_dir, root_dir
from datetime import datetime, date, timedelta

startTime = datetime.now()


<<<<<<< HEAD

################## INPUTS #################
domain = "d02"
rxtime = 14       #anticipated burn hour
utm = 6          #utm offset
rxloc = [55, -113]   #lat/lon location of the burn
wrf_run = "12"     # which utc run

avg_wrf = 35
min_wrf = 32
=======
################## INPUTS #################
domain = "d02"
rxtime = 14  # anticipated burn hour
utm = 6  # utm offset
rxloc = [55, -113]  # lat/lon location of the burn
wrf_run = "12"  # which utc run

avg_wrf = 80
min_wrf = 40
>>>>>>> e32867a1b63578b24a1c3b7a1450abab8c2a528f
wait_max = 3  ## wait a max of hours then give up
wait_max = wait_max * 60  ## convert hours to mins
################## end of INPUTS #################


forecast_date = pd.Timestamp("today").strftime(f"%Y%m%d{wrf_run}")
filein = str(wrf_dir) + f"/WAN{wrf_run}CG-01/{forecast_date}/"
lenght = len(sorted(Path(filein).glob(f"wrfout_{domain}_*00")))
<<<<<<< HEAD
input_sounding = str(root_dir) + f'/burns/inputs/input_sounding.{forecast_date}'
command = f"{root_dir}/bin/mv_cedar.sh"
=======
input_sounding = str(root_dir) + f"/burns/inputs/input_sounding.{forecast_date}"
command = f"scp -r {input_sounding} rodell@cedar.computecanada.ca:/home/rodell/projects/def-rstull/shared_data/"
>>>>>>> e32867a1b63578b24a1c3b7a1450abab8c2a528f


if lenght >= avg_wrf:
    print(f"WRf Folder lenght of {lenght} mathces excepted lenght of {avg_wrf}")
    print(f"Running: get_ibc")
    get_ibc(rxtime, utm, rxloc, wrf_run)
    os.system(command)
else:
    elapsed = 0
    while lenght < avg_wrf:
        print(f"slepping for {elapsed} mins....WRf Folder lenght: {lenght}")
        time.sleep(60)
        elapsed += 1
        lenght = len(sorted(Path(filein).glob(f"wrfout_{domain}_*00")))
        if lenght >= avg_wrf:
            print(f"WRf Folder lenght of {lenght} mathces excepted lenght of {avg_wrf}")
            print(f"Running: get_ibc")
            get_ibc(rxtime, utm, rxloc, wrf_run)
            # os.system(command)

        elif elapsed >= wait_max:
            if lenght >= min_wrf:
                print(
                    f"WRf Folder lenght of {lenght} mathces minimum needed number of files to run FWF"
                )
                print(f"Running: get_ibc")
                get_ibc(rxtime, utm, rxloc, wrf_run)
                # os.system(command)
                sys.exit(0)
            else:
                print(
                    f"Quitting trying to run FWF...WRF folder lenght of {lenght} is not eneough"
                )
                sys.exit(0)
        else:
            pass
