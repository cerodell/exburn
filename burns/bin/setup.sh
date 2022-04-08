#!/bin/bash
#SBATCH -t 00:01:00
#SBATCH --mem-per-cpu=3000M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=crodell@eoas.ubc.ca
#SBATCH --account=rrg-rstull

ml StdEnv/2020  intel/2020.1.217  openmpi/4.0.3
module load wrf/4.2.1

CASE=burns/unit5/single-line
cd ~/exburn/$CASE

ln -sv ../../../WRF-SFIRE/main/wrf.exe
ln -sv ../../../WRF-SFIRE/main/ideal.exe
ln -sv ../../../WRF-SFIRE/run/GENPARM.TBL
ln -sv ../../../WRF-SFIRE/run/RRTMG_LW_DATA
ln -sv ../../../WRF-SFIRE/run/RRTMG_SW_DATA
ln -sv ../../../WRF-SFIRE/run/SOILPARM.TBL
ln -sv ../../../WRF-SFIRE/run/URBPARM.TBL
ln -sv ../../../WRF-SFIRE/run/VEGPARM.TBL
ln -sv ../../../WRF-SFIRE/run/ETAMPNEW_DATA

rm -r namelist.input
ln -sv namelist.input.spinup namelist.input

mpirun -np 1 ./ideal.exe
