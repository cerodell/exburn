import context
from datetime import datetime

import matplotlib.pyplot as plt
from metpy.plots import SkewT
import metpy.calc as mpcalc
from metpy.units import pandas_dataframe_to_unit_arrays, units
import numpy as np
from siphon.simplewebservice.wyoming import WyomingUpperAir
import matplotlib
from context import root_dir

dt = datetime(2021, 7, 1, 0)
station = "CWVK"

font = {"family": "normal", "weight": "bold", "size": 14}

matplotlib.rc("font", **font)

# Read remote sounding data based on time (dt) and station
df = WyomingUpperAir.request_data(dt, station)

# Create dictionary of united arrays
data = pandas_dataframe_to_unit_arrays(df)

# Isolate united arrays from dictionary to individual variables
p = data["pressure"]
T = data["temperature"]
Td = data["dewpoint"]
u = data["u_wind"]
v = data["v_wind"]

# Change default to be better for skew-T
fig = plt.figure(figsize=(9, 11))

# Initiate the skew-T plot type from MetPy class loaded earlier
skew = SkewT(fig, rotation=45)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, "r", linewidth=4, zorder=10)
skew.plot(p, Td, "g", linewidth=4, zorder=10)
skew.plot_barbs(p[::3], u[::3], v[::3], y_clip_radius=0.03)

# Set some appropriate axes limits for x and y
skew.ax.set_xlim(-20, 50)
skew.ax.set_ylim(1020, 100)

# Add the relevant special lines to plot throughout the figure
skew.plot_dry_adiabats(
    t0=np.arange(233, 533, 10) * units.K, alpha=0.25, color="orangered"
)
skew.plot_moist_adiabats(
    t0=np.arange(233, 400, 5) * units.K, alpha=0.25, color="tab:green"
)
skew.plot_mixing_lines(
    pressure=np.arange(1000, 99, -20) * units.hPa, linestyle="dotted", color="tab:blue"
)

lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
skew.plot(lcl_pressure, lcl_temperature, "ko", markerfacecolor="black")

# Calculate full parcel profile and add to plot as black line
prof = mpcalc.parcel_profile(p, T[0], Td[0]).to("degC")
skew.plot(p, prof, "k", linewidth=2)

# Shade areas of CAPE and CIN
skew.shade_cin(p, T, prof, Td)
skew.shade_cape(p, T, prof)

# Add some descriptive titles
plt.title("{} Sounding".format(station), loc="left")
plt.title("Valid Time: {}".format(dt), loc="right")

plt.savefig(str(root_dir) + "/img/skew-t.png")
