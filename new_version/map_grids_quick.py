#!/usr/bin/env python
# Notebook init
import geopy.distance
import numpy as np
from tqdm import trange
import xarray as xr
import math

#### FUNCTIONS

# Turn the math into a function
R = 6371 * 1000.
def x(phi, theta):
    return R * np.cos(theta) * np.cos(phi)
def y(phi, theta):
    return R * np.cos(theta) * np.sin(phi)
def z(theta):
    return R * np.sin(theta)


def distance_between_to_lon_lat_points(lon_grid_1, lat_grid_1, lon_grid_2, lat_grid_2):
    lon_grid_1, lat_grid_1 = math.radians(lon_grid_1), math.radians(lat_grid_1)
    lon_grid_2, lat_grid_2 = math.radians(lon_grid_2), math.radians(lat_grid_2)
    root_term = (
                (x(lon_grid_1, lat_grid_1) - x(lon_grid_2, lat_grid_2))**2 +
                (y(lon_grid_1, lat_grid_1) - y(lon_grid_1, lat_grid_1))**2 +
                (z(lat_grid_1) - z(lat_grid_2)**2)
                )**0.5
    D = 2*R * math.asin(root_term/(2*R))
    
    return D

def distance_between_to_lon_lat_points_straight_line(lon_grid_1, lat_grid_1, lon_grid_2, lat_grid_2):
    R = 6371 * 1000.
    lon_grid_1, lat_grid_1 = math.radians(lon_grid_1), math.radians(lat_grid_1)
    lon_grid_2, lat_grid_2 = math.radians(lon_grid_2), math.radians(lat_grid_2)
    def x(phi, theta):
        return R * math.cos(theta) * math.cos(phi)
    def y(phi, theta):
        return R * math.cos(theta) * math.sin(phi)
    def z(theta):
        return R * math.sin(theta)
    root_term = (
                (x(lon_grid_1, lat_grid_1) - x(lon_grid_2, lat_grid_2))**2 +
                (y(lon_grid_1, lat_grid_1) - y(lon_grid_1, lat_grid_1))**2 +
                (z(lat_grid_1) - z(lat_grid_2)**2)
                )**0.5
    D = root_term
    return D

def distance_between_cartersian_points(x_ice, y_ice, z_ice, x_ocean, y_ocean, z_ocean):
    return ((x_ice - x_ocean)**2+(y_ice-y_ocean)**2+(z_ice-z_ocean)**2)**0.5

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


def main():
    # Load the data
    ocean_map_file = xr.open_dataset("mapping_ocean.nc")
    ice_map_file = xr.open_dataset("nhem_20km_xylatlon.nc")

    ice_lon = ice_map_file["lon"].squeeze()
    ice_lat = ice_map_file["lat"].squeeze()

    ocean_lon = ocean_map_file["oces.lon"].squeeze()
    ocean_lat = ocean_map_file["oces.lat"].squeeze()

    # Transform into Cartesian Coordinates
    # This is suprisingly fast!
    ice_x = x(np.radians(ice_lat), np.radians(ice_lon))
    ice_y = y(np.radians(ice_lat), np.radians(ice_lon))
    ice_z = z(np.radians(ice_lat))

    ocean_x = x(np.radians(ocean_lat), np.radians(ocean_lon))
    ocean_y = y(np.radians(ocean_lat), np.radians(ocean_lon))
    ocean_z = z(np.radians(ocean_lat))

    # Allocate the arrays:
    nearest_i_index_of_ocean_grid = np.zeros(ice_lon.shape)
    nearest_j_index_of_ocean_grid = np.zeros(ice_lat.shape)
    smallest_distances = np.zeros(ice_lat.shape)
    all_distances = np.zeros((*ice_x.shape, *ocean_x.shape))
    distance = np.ones(ice_lat.shape)
    distance *= 2*geopy.distance.EARTH_RADIUS

    for i in tnrange(220, leave=False):
        for j in tnrange(256, leave=False):
            all_distances[:, :, i, j] = distance_between_cartersian_points(ice_x, ice_y, ice_z, ocean_x[i, j], ocean_y[i, j], ocean_z[i, j])

            xr_all_distances = xr.DataArray(all_distances)
            xr_all_distances.to_netcdf("all_distances.nc")

if __name__ == '__main__':
    main()


