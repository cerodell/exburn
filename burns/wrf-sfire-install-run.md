# Installing SFIRE on Cedar

1. Download the code within `exburn/`:
```
[~]$ wget https://github.com/openwfm/WRF-SFIRE/archive/a2c3118f08ce424885705e9155b127ea28879f8b.zip
```
2. Unpack the program:
```
[~]$ unzip 'a2c3118f08ce424885705e9155b127ea28879f8b.zip'
[~]$ mv WRF-SFIRE-a2c3118f08ce424885705e9155b127ea28879f8b ./WRF-SFIRE
[~]$ ls

MDPI/                                         data/
README.md                                     img/
WRF-SFIRE/                                    json/
a2c3118f08ce424885705e9155b127ea28879f8b.zip  logs/
burns/                                        scripts/
comps.code-workspace                          utils/
context.py
```

3. Remove and replace Registry folder from `~/exburn/WRF-SFIRE`.

NOTE this step insnt need but will make runtimes faster and wrfout files small. This is becasue its not writing nearly as much data to the wrfoutfiles.
```
cd ~/exburn/WRF-SFIRE
ls

arch   compile      configure.wrf         dyn_em    frame  LICENSE.txt  phys       README-SFIRE.md  share       tools
chem   compile.log  configure.wrf.backup  dyn_nmm   hydro  main         README     Registry         standalone  var
clean  configure    doc                   external  inc    Makefile     README.md  run              test        wrftladj

rm -rf Registry
cp -r ~/exburn/burns/Registry  ~/exburn/WRF-SFIRE/
```


3. Load the modules:
```
[~]$ ml StdEnv/2020  intel/2020.1.217  openmpi/4.0.3
[~]$ module load wrf/4.2.1

[~]$ module list

Currently Loaded Modules:
  1) CCconfig          4) gcccore/.9.3.0   (H)      7) ucx/1.8.0             10) jasper/2.0.16 (vis)  13) mpi4py/3.0.3     (t)   16) netcdf-fortran-mpi/4.5.2 (io)
  2) gentoo/2020 (S)   5) imkl/2020.1.217  (math)   8) libfabric/1.10.1      11) libffi/3.3           14) hdf5-mpi/1.10.6  (io)  17) wrf/4.2.1                (geo)
  3) StdEnv/2020 (S)   6) intel/2020.1.217 (t)      9) openmpi/4.0.3    (m)  12) python/3.8.10 (t)    15) netcdf-mpi/4.7.4 (io)

```
4. Set the environment variables and configure the program:
```
[~]$ cd SFIRE
[~]$ ls
arch  clean    configure  dyn_em   external  hydro  LICENSE.txt  Makefile  README     Registry  share  tools  wrftladj
chem  compile  doc        dyn_nmm  frame     inc    main         phys      README.md  run       test   var

export NETCDF=$EBROOTNETCDFMINFORTRAN
export NETCDF_classic=1
export WRFIO_NCD_LARGE_FILE_SUPPORT=1
```
5. Run: ./clean -a to delete the remains of any previous build and ensure start from a known state

```
[~] ./clean -a
```

6. Run: ./configure, choose nesting 1=basic.

```
[~] ./configure

checking for perl5... no
checking for perl... found /cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/perl/5.30.2/bin/perl (perl)
Will use NETCDF in dir: /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Compiler/gcc9/netcdf-fortran/4.5.2
HDF5 not set in environment. Will configure WRF for use without.
PHDF5 not set in environment. Will configure WRF for use without.
Will use 'time' to report timing information
$JASPERLIB or $JASPERINC not found in environment, configuring to build without grib2 I/O...
------------------------------------------------------------------------
Please select from among the following Linux x86_64 options:

  1. (serial)   2. (smpar)   3. (dmpar)   4. (dm+sm)   PGI (pgf90/gcc)
  5. (serial)   6. (smpar)   7. (dmpar)   8. (dm+sm)   PGI (pgf90/pgcc): SGI MPT
  9. (serial)  10. (smpar)  11. (dmpar)  12. (dm+sm)   PGI (pgf90/gcc): PGI accelerator
 13. (serial)  14. (smpar)  15. (dmpar)  16. (dm+sm)   INTEL (ifort/icc)
                                         17. (dm+sm)   INTEL (ifort/icc): Xeon Phi (MIC architecture)
 18. (serial)  19. (smpar)  20. (dmpar)  21. (dm+sm)   INTEL (ifort/icc): Xeon (SNB with AVX mods)
 22. (serial)  23. (smpar)  24. (dmpar)  25. (dm+sm)   INTEL (ifort/icc): SGI MPT
 26. (serial)  27. (smpar)  28. (dmpar)  29. (dm+sm)   INTEL (ifort/icc): IBM POE
 30. (serial)               31. (dmpar)                PATHSCALE (pathf90/pathcc)
 32. (serial)  33. (smpar)  34. (dmpar)  35. (dm+sm)   GNU (gfortran/gcc)
 36. (serial)  37. (smpar)  38. (dmpar)  39. (dm+sm)   IBM (xlf90_r/cc_r)
 40. (serial)  41. (smpar)  42. (dmpar)  43. (dm+sm)   PGI (ftn/gcc): Cray XC CLE
 44. (serial)  45. (smpar)  46. (dmpar)  47. (dm+sm)   CRAY CCE (ftn $(NOOMP)/cc): Cray XE and XC
 48. (serial)  49. (smpar)  50. (dmpar)  51. (dm+sm)   INTEL (ftn/icc): Cray XC
 52. (serial)  53. (smpar)  54. (dmpar)  55. (dm+sm)   PGI (pgf90/pgcc)
 56. (serial)  57. (smpar)  58. (dmpar)  59. (dm+sm)   PGI (pgf90/gcc): -f90=pgf90
 60. (serial)  61. (smpar)  62. (dmpar)  63. (dm+sm)   PGI (pgf90/pgcc): -f90=pgf90
 64. (serial)  65. (smpar)  66. (dmpar)  67. (dm+sm)   INTEL (ifort/icc): HSW/BDW
 68. (serial)  69. (smpar)  70. (dmpar)  71. (dm+sm)   INTEL (ifort/icc): KNL MIC
 72. (serial)  73. (smpar)  74. (dmpar)  75. (dm+sm)   FUJITSU (frtpx/fccpx): FX10/FX100 SPARC64 IXfx/Xlfx

Enter selection [1-75] : 15
------------------------------------------------------------------------
Compile for nesting? (1=basic, 2=preset moves, 3=vortex following) [default 1]: 1

Configuration successful!
...

```

7. Compile WRF-SFIRE:
```
[~]$ ./compile em_fire >& compile.log


```
Review the compile log to make sure there are no errors. Successful installation will produce an ideal.exe and wrf.exe files in ./main folder

----
# Running SFIRE on Cedar


1. Create folder directories for your simulations. This repo has the unit5 simulation under `~/burns/unit5/`


2. Submit `setup.sh` shell script from `~/bin`
This create symlinks to wrf.exe and other dependencies in your desired simulation folder. You'll need to edit the `CASE=` for your simulations.

```
[~] cd bin
[~] sbatch setup.sh
```

3. Submit `run.sh` shell script from `~/bin`
This runs your simulation
```
[~] cd bin
[~] sbatch run.sh
```
