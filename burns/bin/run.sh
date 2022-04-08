#!/bin/bash
#SBATCH -t 04:00:00
#SBATCH --mem-per-cpu=4000M
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=crodell@eoas.ubc.ca
#SBATCH --account=rrg-rstull

ml StdEnv/2020  intel/2020.1.217  openmpi/4.0.3
module load wrf/4.2.1

CASE=burns/unit5/single-line
cd ~/exburn/$CASE

rm -r namelist.input
ln -sv namelist.input.spinup namelist.input

srun ./wrf.exe 1>wrf.log 2>&1

mv rsl.* ~/exburn/$CASE/log/spinup
mv wrf.log ~/exburn/$CASE/log/spinup

cd ~/exburn/$CASE
ml nco
ml netcdf
ncatted -O -h -a WRF_ALARM_SECS_TIL_NEXT_RING_01,global,m,i,10 wrfrst_d01_2019-05-11_17:49:01

rm -r namelist.input
ln -sv namelist.input.restart namelist.input

srun ./wrf.exe 1>wrf.log 2>&1

mv rsl* ~/exburn/$CASE/log/restart
mv wrf.log ~/exburn/$CASE/log/restart
