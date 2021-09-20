# Chris Rodell - PhD Comprehensive Exam assignment from Prof. Stull
## Topic: WRF-SFIRE simulations. Assigned: 28 Mar 2021
### Assignment:
- A. Do the readings listed below. Build a bibliography that includes and
expands beyond the citations listed below, on WRF-SFIRE and wildfire findings
made with it.
- B. Work with Nadya to learn how to run WRF-SFIRE on computer(s) of your
choice. Perhaps start with a case study that she already ran, to ensure that
you get the same answer as her. Finish this part well before the end of
April.
- C. Run the following idealized simulations using WRF-SFIRE. (But you are
allowed to ask Nadya or Rosie or me for help if you get stuck. You are also
allowed to negotiate with Stull on changing the specifications of this
exercise.) Keep a meticulous log of your work, so that you can write-it up
later.

Start with a statically-stable early morning ABL and tropospheric temperature
profile typical of summer in central BC. Run it for a few hours to create a
convective mixed layer that is roughly 1 km thick by noon. No Tstorms.
Use a forest fuel typical of the mountainous parts of BC that have frequent
wildfires. If there is a fire, assume the fire starts at local noon at a
small ignition point.

	1. Flat terrain:
	a. No mean wind in PBL.
	(1) No fire. Determine evolution of ABL height zi vs. time, and ABL
	TKE vs. time. This will serve as a reference “base state”.
	(2) Fire. See the pattern and speed of fire spread.
	b. Modest mean wind in PBL, typical of summer in BC.
	(1) No fire.
	(2) Fire.
	2. Sloping terrain. Assume half the domain is flat, other half is
	constant-slope (45°) ridge. Namely, terrain cross section looks like
	_/ . You will need to do a new spin-up of the PBL to develop the
	anabatic winds.
	a. No mean wind in PBL
	(1) No fire. Determine evolution of ABL height zi vs. time, and ABL
	TKE vs. time.
	(2) Fire started at a point near the base of the slope. See the
	pattern and speed of fire spread.
	b. Modest mean wind (typical of BC in summer) in PBL pointing upslope.
	(1) No fire.
	(2) Fire started at a point near the base of the slope.
	c. Modest mean wind (typical of BC in summer) in PBL pointing
	downslope.
	(1) No fire.
	(2) Fire started at a point near the base of the slope.
	d. Modest mean wind (typical of BC in summer) in PBL pointing along
	slope.
	(1) No fire.
	(2) Fire started at a point near the base of the slope.
	3. Optional: If possible turn off the parameterized fire-spread that was
	built into WRF-SFIRE, and re-run some of the cases above to see what
	portion of the spread was due to the parameterization, and what portion
	due to the resolved fire & atmospheric circulations.

- D. Write your results in the format of a draft journal paper. It can include
a supplementary section with larger images and animations if needed.
- E. Based on knowledge gained from these experiments, describe what types of
experiments you would set up in order to test and extend Nadya’s smokeplume-
rise theory for a fire on a uniformly sloping ridge. But do NOT do
these numerical experiments – only describe and justify your experiment
design.
- F. Present all your findings in a WFRT seminar (to which your committee
members also attend)

### WRF-SFIRE related readings
- Coen, J. Some Requirements for Simulating Wildland Fire Behavior Using
Insight from Coupled Weather—Wildland Fire Models. Fire 2018, 1, 6.
https://doi.org/10.3390/fire1010006
	- Simulating WeatherWell
	- NWP Model Design and Numerical Considerations
	- Configuration Considerations
		- Grid Aspect Ratio: Accuracy is
	best maintained when the vertical to horizontal grid aspect ratio is ~1, that is, the grid volume is
	an equal-sided cube. Good practice for convective-scale and turbulence modeling maintains the aspect
	ratio between 1 and 5 [38].
	- Specific Scenarios
	- Addressing Inadequate Outcomes

- Kochanski, A. K., Jenkins, M. A., Mandel, J., Beezley, J. D., Clements, C.
B., and Krueger, S.: Evaluation of WRF-SFIRE performance with field
observations from the FireFlux experiment, Geosci. Model Dev., 6, 1109–
1126, https://doi.org/10.5194/gmd-6-1109-2013, 2013.

- Mandel, J., Beezley, J. D., and Kochanski, A. K.: Coupled atmosphere-wildland
fire modeling with WRF 3.3 and SFIRE 2011, Geosci. Model Dev., 4, 591–610,
https://doi.org/10.5194/gmd-4-591-2011, 2011.

- Mandel, J., Amram, S., Beezley, J. D., Kelman, G., Kochanski, A. K.,
Kondratenko, V. Y., Lynn, B. H., Regev, B., and Vejmelka, M.: Recent
advances and applications of WRF–SFIRE, Nat. Hazards Earth Syst. Sci., 14,
2829–2845, https://doi.org/10.5194/nhess-14-2829-2014, 2014.

- Simpson, C. C., Sharples, J. J., and Evans, J. P.: Resolving vorticity-driven
lateral fire spread using the WRF-Fire coupled atmosphere–fire numerical
model, Nat. Hazards Earth Syst. Sci., 14, 2359–2371,
https://doi.org/10.5194/nhess-14-2359-2014, 2014.

- Simpson, Colin C., Jason J. Sharples, and Jason P. Evans. "Sensitivity of
atypical lateral fire spread to wind and slope." Geophysical Research
Letters 43.4 (2016): 1744-1751.
