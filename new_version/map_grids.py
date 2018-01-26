# Notebook init
import geopy.distance
import numpy as np
from tqdm import trange, tqdm_notebook, tqdm
import xarray as xr

# Load the data
ocean_map_file = xr.open_dataset("mapping_ocean.nc")
ice_map_file = xr.open_dataset("nhem_20km_xylatlon.nc")

ice_lon = ice_map_file["lon"].squeeze()
ice_lat = ice_map_file["lat"].squeeze()

ocean_lon = ocean_map_file["oces.lon"].squeeze()
ocean_lat = ocean_map_file["oces.lat"].squeeze()

# Allocate the arrays:
nearest_i_index_of_ocean_grid = np.zeros(ice_lon.shape)
nearest_j_index_of_ocean_grid = np.zeros(ice_lat.shape)
smallest_distances = np.zeros(ice_lat.shape)
distance = np.ones(ice_lat.shape)
distance *= 2*geopy.distance.EARTH_RADIUS

# Define a function to assign in a multiprocessor pool
def assign_index_to_arrays(params, test=False):
    ocean_index_i = params[0]
    ocean_index_j = params[1]
    ice_index_i = params[2]
    ice_index_j = params[3]
    if test: print(ocean_index_i, ocean_index_j, ice_index_i, ice_index_j)
    if not (np.isnan(ocean_lat[ocean_index_i, ocean_index_j].data) or \
            np.isnan(ocean_lon[ocean_index_i, ocean_index_j].data)):
        # Build the distance tuple:
        ocean_lat_lon = (ocean_lat[ocean_index_i, ocean_index_j].data, 
                         ocean_lon[ocean_index_i, ocean_index_j].data)
        ice_lat_lon = (ice_lat[ice_index_i, ice_index_j].data, 
                       ice_lon[ice_index_i, ice_index_j].data)
        current_distance = geopy.distance.GreatCircleDistance(ocean_lat_lon, ice_lat_lon).kilometers
        if test: print(current_distance)
        if current_distance < distance[ice_index_i, ice_index_j]:
            distance[ice_index_i, ice_index_j] = current_distance
            nearest_i_index_of_ocean_grid[ice_index_i, ice_index_j] = ocean_index_i
            nearest_j_index_of_ocean_grid[ice_index_i, ice_index_j] = ocean_index_j
            smallest_distances[ice_index_i, ice_index_j] = current_distance

for ice_index_i in trange(ice_lat.shape[0], desc="ice index i"):
    for ice_index_j in trange(ice_lon.shape[1], desc="ice index j"):
        for ocean_index_i in trange(ocean_lat.shape[0], desc="ocean index i"):
            for ocean_index_j in trange(ocean_lon.shape[1], desc="ocean index j"):
                assign_index_to_arrays((ocean_index_i, ocean_index_j, ice_index_i, ice_index_j))


xd_nearest_i_index_of_ocean_grid = xr.DataArray(nearest_i_index_of_ocean_grid)
xd_nearest_j_index_of_ocean_grid = xr.DataArray(nearest_j_index_of_ocean_grid)
xd_smallest_distances = xr.DataArray(smallest_distances)

xd_nearest_i_index_of_ocean_grid.to_netcdf("nearest_i_index_of_ocean_grid.nc")
xd_nearest_i_index_of_ocean_grid.to_netcdf("nearest_i_index_of_ocean_grid.nc")
xd_smallest_distances.to_netcdf("smallest_distances.nc")
