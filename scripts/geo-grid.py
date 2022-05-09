import context
import json
import numpy as np
import xarray as xr


from context import data_dir

# south_north_fire = slice(70, 100, None)
# west_east_fire = slice(73, 97, None)

south_north_fire = slice(76, 95, None)
west_east_fire = slice(77, 92, None)

with open(str(data_dir) + "/grid-sample.json") as f:
    config = json.load(f)


grid_ds = xr.open_zarr(str(data_dir) + "/grid.zarr")
grid_ds = grid_ds.sel(
    south_north=south_north_fire,
    west_east=west_east_fire,
)


s_XLONG = grid_ds["XLONG"].values
s_XLAT = grid_ds["XLAT"].values
shape = s_XLAT.shape


points = []
for i in range(shape[1]):
    for j in range(shape[0]):
        point = {
            "geometry": {"coordinates": [], "type": "Point"},
            "type": "Feature",
            "properties": {
                "marker-symbol": "point",
                "marker-color": "FF7F0E",
                "marker-size": "1",
                "class": "Marker",
            },
        }
        print([s_XLONG[j, i], s_XLAT[j, i], 0, 0])
        point["geometry"]["coordinates"] = [s_XLONG[j, i], s_XLAT[j, i], 0, 0]
        points.append(point)

main = {"type": "FeatureCollection", "features": []}

main["features"] = points

with open(str(data_dir) + "/unit4-fire-points.json", "w") as f:
    json.dump(main, f, indent=4, sort_keys=True)


grids = []
for i in range(shape[1]):
    for j in range(shape[0]):
        try:
            grid = {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [[]]},
            }
            ll = [s_XLONG[j, i], s_XLAT[j, i]]
            ul = [s_XLONG[j + 1, i], s_XLAT[j + 1, i]]
            lr = [s_XLONG[j, i + 1], s_XLAT[j, i + 1]]
            ur = [s_XLONG[j + 1, i + 1], s_XLAT[j + 1, i + 1]]
            locs = [ll, ul, ur, lr, ll]
            grid["geometry"]["coordinates"] = locs
            grids.append(grid)
        except:
            pass

main = {"type": "FeatureCollection", "features": []}

main["features"] = grids

with open(str(data_dir) + "/unit4-fire-grid.json", "w") as f:
    json.dump(main, f, indent=4, sort_keys=True)

# locs =[
#           [
#           -113.60469684478078, # west
#           55.79586793189076 # south
#         ],
#         [
#           -113.60469684478078, # west
#           55.79609276198169 # north
#         ],
#         [
#           -113.60429689369684, # east
#           55.79609276198169 # north
#         ],
#         [
#           -113.60429689369684, # east
#           55.79586793189076 # south
#         ],
#         [
#           -113.60469684478078, # west
#           55.79586793189076 # south
#         ]
#       ]


# locs = [ [
#   -113.56668934711838,
#   55.71117665884658
# ],
# [
#   -113.56670406813733,
#   55.71140106818565
# ],
# [
#   -113.56630657702557,
#   55.71140937819351
# ],
# [
#   -113.56629185828385,
#   55.71118496878542
# ],
# [
#   -113.56668934711838,
#   55.71117665884658
# ]
# ]
# main['features'] = [grid]
