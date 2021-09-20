# INSTRUCTIONS FOR RUNNING WRF-SFIRE FOR PELICAN MOUNTAIN BURNS

This example will walk you through performing a sample run for Unit 4.

1. Clone the pelican_les repo to a clean projects directory on Cedar (e.g. /home/rodell/projects/def-rstull or /home/rodell/projects/rrg-rstull)
2. Complete steps 1-4 in README_INSTALL_SFIRE.md to prepare for installing WRF-SFIRE in your user account.
3. In WRF-SFIRE folder, replace the default contents of Registry directory with those from Registry_NM_042021.tar
4. Complete the remaining installation steps in README_INSTALL_SFIRE.md. A successfull compile will produce wrf.exe and ideal.exe in `main` subdirectory.
5. Using this Colab notebook, prepare input files for your simulation, including:
    - input_tsk : a perturbed surface temperature field used to inialize convection (on atmospheric mesh)
    - input_fc : a fuel map for the domain (on fine fire mesh)
    - ignition start and end points
6. On Cedar in your local pelican_les repository, switch to spinup_run subdirectory. Edit the namelist.input file with disired settings.
   Things to keep in mind:
    - Your spinup time should be at least an hour long (preferably two), during which there should be no fire
    - Limit history outputs to avoid data storage issue (every 5-10 min is fine for spinup)
    - Generate a restart file at the end of the spinup
    - Use a "summer" half of the year for simulation date, to ensure surface setting are correct
    - keep domain consisten with those in INPUTS section of this notebook
    - fire model has to be turned on for spinup, even if there is no fire
    - include ignition settings from this notebook in the fire subsection of the namelist
7. Edit the input_sounding with a desired profile (see standard wrf tutorials for format). Keep simulations DRY for now (see "UNSOLVED ISSUES")
8. Copy the input_tsk and input_fc files you created with this notebook to spinup_run subdirectory on Cedar.
9. Symlink wrf.exe and ideal.exe from WRF-SFIRE/main folder into the current directory as well.
10. If necessary, modify other config files in the subdirecory as you see fit. Some general descriptions are availble in README_RUN.md
11. If necessary, modify the slurm script and submit to scheduler:
    sbatch submit_to_slurm_spinup.sh
12. Once the spinup is complete, switch to ../fire_run directory and symlink the wrfrst_* file.
13. Edit the namelist settings, if necessary.
   Things to keep in mind:
    - some setting must remain consistent with spinup simulation, while others can change
    - history output for the fire simulation should be more frequent, then for spinup. I typically use 15 seconds - 1 min.
    - fire ignition times are always calculated for them beginning of the *spinup* simulation
14. Submit to scheduler.
15. Use VAPOR TwoDData Renderer to plot GRNHFX to visualize the fire. Use Volume Renderer to plot QVAPOR variable to visualize smoke (see "UNSOLVED ISSUES")

UNRESOLVED ISSUES:
i) Currently, I cannot the get the tracers to work properly with the new WRF-SFIRE version. Keep your soundings completely dry for now.
This will allow you to use QVAPOR variable to visualize smoke (it will be passive, because microphysical are turned off).

ii) Standard WMS server for pulling landsat image for visualization is dead. The above python api is a trial and is a pain in the ass.
Need a better vizualiation solution. Also, it saves monochrome and vapor no longer accepts non-georeferences pngs/jpgs.

iii) Spinup should be longer

iv) The fire didn't burn through the entire lot in 20 min. But longer run might recirculate smoke. For demo this could work.
Alernatively, could drop windspeed and let it run longer.
