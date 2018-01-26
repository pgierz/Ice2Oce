#!/usr/bin/env python
"""This program takes an ice grid (file1) and generates a map of (i, j) coordinate
pairs of the ice grid onto the corresponding points of the ocean grid (I, J).

Usage: gridpoint_map_ice_to_ocean.py [options] ICE_FILE OCEAN_FILE OUTPUT_FILE

Options:
-g, --ice-grid-orientation-as-ocean
-c, --restrict-coast
-i, --do-fortran-boundIJ
-b, --binary-erosion-coast

Paul Gierz, June 2017

"""
import docopt
import numpy as np
from scipy.io import netcdf
import scipy.ndimage
import logging
import os as os
# TODO: optional imports

# Switches:
do_fortran_boundIJ = True
do_grid1_2_same_orientation = True


def latlon_distance(lat1, lon1, lat2, lon2):
    """
    calculate the distance between points given as pair of lat/lon values

    Input
    lat1  : latitude point 1 (degree)
    lon1  : longitude point 1 (degree , -180..180)
    lat2  : latitude point 2 (degree)
    lon2  : longitude point 2 (degree , -180..180)

    Output
    dist  : distance between the points (meter)
    """
    # pi = 3.1415926535897932384626433832795028841971693993751058 # https://www.wolframalpha.com 2013-01-21
    # earthradius = 6.36751E6 #earth radius meters # https://www.wolframalpha.com 2013-01-21
    # deg2meter = 2.*pi*earthradius/360.
    deg2meter = 111134.014654  #meter (1 degree latitude equals this distance)
    # rad2deg = 57.2957795131 # rad2deg = 360./(2.*pi)
    deg2rad = 0.0174532925199  # rad2deg = (2.*pi)/360
    dlon0 = lon1 - lon2
    #dlon0 = N.where( N.abs(dlon0)> 180. ,  N.sign(dlon0) * (360.- N.abs(dlon0)) , dlon0 )
    if (np.abs(dlon0) > 180.):
        if (dlon0 < -180.):
            dlon0 = -1. * (360. + dlon0)
        else:
            dlon0 = -1. * (360. + dlon0)
    lat0 = (lat1 + lat2) * 0.5
    dlon = np.cos(
        lat0 * deg2rad) * dlon0  # CAUTION: cos expects rad Not degree
    dlat = (lat1 - lat2)
    # distance
    dist = np.sqrt(dlat * dlat + dlon * dlon) * deg2meter
    return dist


def read_ice(filename):
    with netcdf.netcdf_file(filename, mode="r") as fin_ice:
        lat1 = fin_ice.variables["lat"][:].copy().squeeze()
        lon1 = fin_ice.variables["lon"][:].copy().squeeze()
        x_ice = fin_ice.variables["x"][:].copy().squeeze()
        y_ice = fin_ice.variables["y"][:].copy().squeeze()
    return lat1, lon1, x_ice, y_ice


def read_ocean(filename):
    with netcdf.netcdf_file(filename, mode="r") as fin_oce:
        tmask = fin_oce.variables["oces.mask"][:].copy().squeeze()
        lat2 = fin_oce.variables["oces.lat"][:].copy().squeeze()
        lon2 = fin_oce.variables["oces.lon"][:].copy().squeeze()
    return tmask, lat2, lon2


def iscoast_explicit(ilen2, jlen2, isocean):
    iscoast = np.zeros(np.shape(isocean), dtype="bool")
    do_restrict2coast = True
    if do_restrict2coast:
        #
        # Corner points
        i2 = 0
        j2 = 0
        if (isocean[i2, j2] and
                not (isocean[i2 + 1, j2] and isocean[i2, j2 + 1])):
            iscoast[i2, j2] = True
        i2 = ilen2 - 1
        j2 = 0
        if (isocean[i2, j2] and
                not (isocean[i2 - 1, j2] and isocean[i2, j2 + 1])):
            iscoast[i2, j2] = True
        i2 = 0
        j2 = jlen2 - 1
        if (isocean[i2, j2] and
                not (isocean[i2 + 1, j2] and isocean[i2, j2 - 1])):
            iscoast[i2, j2] = True
        i2 = ilen2 - 1
        j2 = jlen2 - 1
        if (isocean[i2, j2] and
                not (isocean[i2 - 1, j2] and isocean[i2 - 1, j2])):
            iscoast[i2, j2] = True
        #
        # Edges along the i2=0 and i2=ilen2-1
        for i2 in range(1, ilen2 - 1):
            j2 = 0
            if (isocean[i2, j2] and
                    not (isocean[i2 - 1, j2] and isocean[i2 + 1, j2] and
                         isocean[i2, j2 + 1] and isocean[i2 - 1, j2 + 1] and
                         isocean[i2 + 1, j2 + 1])):
                iscoast[i2, j2] = True
            j2 = jlen2 - 1
            if (isocean[i2, j2] and
                    not (isocean[i2 - 1, j2] and isocean[i2 + 1, j2] and
                         isocean[i2, j2 - 1] and isocean[i2 - 1, j2 - 1] and
                         isocean[i2 + 1, j2 - 1])):
                iscoast[i2, j2] = True
        #
        # Edges along the j2=0 and j2=jlen2-1
        for j2 in range(1, jlen2 - 1):
            i2 = 0
            if (isocean[i2, j2] and
                    not (isocean[i2 + 1, j2] and isocean[i2, j2 - 1] and
                         isocean[i2, j2 + 1] and isocean[i2 + 1, j2 - 1] and
                         isocean[i2 + 1, j2 + 1])):
                iscoast[i2, j2] = True
                i2 = ilen2 - 1
            if (isocean[i2, j2] and
                    not (isocean[i2 - 1, j2] and isocean[i2, j2 - 1] and
                         isocean[i2, j2 + 1] and isocean[i2 - 1, j2 - 1] and
                         isocean[i2 - 1, j2 + 1])):
                iscoast[i2, j2] = True
        #
        # All inner points
        for i2 in range(1, ilen2 - 1):
            for j2 in range(1, jlen2 - 1):
                if (isocean[i2, j2] and not (
                        isocean[i2 - 1, j2] and isocean[i2 + 1, j2] and
                        isocean[i2, j2 - 1] and isocean[i2, j2 + 1] and
                        isocean[i2 - 1, j2 - 1] and isocean[i2 - 1, j2 + 1] and
                        isocean[i2 + 1, j2 - 1] and isocean[i2 + 1, j2 + 1])):
                    iscoast[i2, j2] = True
    else:
        iscoast = isocean
    return iscoast


def iscoast_binary_erosion(ilen2, jlen2, isocean):
    struct = scipy.ndimage.generate_binary_structure(2, 2)
    erode = scipy.ndimage.binary_erosion(isocean, struct, border_value=1)
    iscoast_quick = isocean ^ erode
    return iscoast_quick


def calculate_distances(ilen1, jlen1, ilen2, jlen2, lat1, lon1, lat2, lon2,
                        iscoast):
    size2D_ice = (ilen1, jlen1)
    size2D_ocean = (ilen2, jlen2)
    Distance = np.ones(size2D_ice, dtype="f") * -9.99999
    I_1to2 = np.ones(size2D_ice, dtype="i") * -9
    J_1to2 = np.ones(size2D_ice, dtype="i") * -9
    import progressbar
    bar = progressbar.ProgressBar(widgets=[
        ' [',
        progressbar.Timer(),
        '] ',
        progressbar.Bar(),
        ' (',
        progressbar.ETA(),
        ') ',
    ])
    for i1 in bar(range(ilen1)):
        for j1 in range(jlen1):
            distance = 9.9E24
            for i2 in range(ilen2):
                for j2 in range(jlen2):
                    if iscoast[i2, j2]:
                        dist = latlon_distance(lat1[i1, j1], lon1[i1, j1],
                                               lat2[i2, j2], lon2[i2, j2])
                        if dist < distance:
                            ii = i2
                            jj = j2
                            distance = dist
            Distance[i1, j1] = distance
            I_1to2[i1, j1] = ii
            J_1to2[i1, j1] = jj
            #if ic > 0:
            #    print "Identical Values %s times" % ic
    return Distance, I_1to2, J_1to2


def reorder_bad_points(ilen1, jlen1, I_bad, J_bad, I_good, J_good, I_1to2,
                       J_1to2):
    if (np.size(I_bad) == np.size(J_bad) and
            np.size(I_good) == np.size(J_good) and
            np.size(I_good) == np.size(I_bad)):
        ijlen = np.size(I_good)
        for ic in range(ijlen):
            for i1 in range(ilen1):
                for j1 in range(jlen1):
                    if (I_1to2[i1, j1] == I_bad[ic] and
                            J_1to2[i1, j1] == J_bad[ic]):
                        I_1to2[i1, j1] = I_good[ic]
                        J_1to2[i1, j2] = J_good[ic]
    else:
        print("** Input data arrays do not fit! Exiting...")
        exit()
    return I_1to2, J_1to2


def save_results(ofilename, ilen1, jlen1, ilen2, jlen2, x_ice, y_ice, lat1,
                 lon1, lat2, lon2, isocean, iscoast, I_1to2, J_1to2, Distance):
    fout = netcdf.netcdf_file(ofilename, "w")
    fout.instiution = "Alfred Wegener Institute for Polar and Marine Research"
    fout.address = "Bussesstrasse 24, 27570 Bremerhaven, Germany"
    fout.web = "http://www.awi.de"
    fout.working_directory = os.getcwd()
    fout.username = os.environ["USER"]

    #
    # Definition of the dimensions
    #
    print('    Create dimension')
    fout.createDimension('x1', ilen1)
    fout.createDimension('y1', jlen1)
    fout.createDimension('x2', ilen2)
    fout.createDimension('y2', jlen2)

    #
    # Define output variables and specify its attributes
    #
    print('    Define variables')

    # x-, y-arrays
    x1_var = fout.createVariable('x1', 'd', ('x1', ))
    x1_var.standard_name = 'projection_x_coordinate'
    x1_var.long_name = 'X-coordinate in Cartesian system'
    x1_var.unit = 'm'
    y1_var = fout.createVariable('y1', 'd', ('y1', ))
    y1_var.standard_name = 'projection_y_coordinate'
    y1_var.long_name = 'Y-coordinate in Cartesian system'
    y1_var.unit = 'm'

    x2_var = fout.createVariable('x2', 'd', ('x2', ))
    x2_var.standard_name = 'projection_x_coordinate'
    x2_var.long_name = 'X-coordinate in Cartesian system'
    y2_var = fout.createVariable('y2', 'd', ('y2', ))
    y2_var.standard_name = 'projection_y_coordinate'
    y2_var.long_name = 'Y-coordinate in Cartesian system'

    lat1_var = fout.createVariable('lat1', 'd', ('x1', 'y1', ))
    lat1_var.standard_name = 'latitude'
    lat1_var.long_name = 'latitude,  grid 1'
    lat1_var.unit = 'degree'
    lon1_var = fout.createVariable('lon1', 'd', ('x1', 'y1', ))
    lon1_var.standard_name = 'longitude'
    lon1_var.long_name = 'longitude, grid 1'
    lon1_var.unit = 'degree'

    lat2_var = fout.createVariable('lat2', 'd', ('x2', 'y2', ))
    lat2_var.standard_name = 'latitude'
    lat2_var.long_name = 'latitude, grid 2'
    lat2_var.unit = 'degree'
    lon2_var = fout.createVariable('lon2', 'd', ('x2', 'y2', ))
    lon2_var.standard_name = 'longitude'
    lon2_var.long_name = 'longitude, grid 2'
    lon2_var.unit = 'degree'

    isocean_var = fout.createVariable('isocean', 'd', ('x2', 'y2', ))
    isocean_var.long_name = 'ocean mask (1:ocean, 0:else)'

    iscoast_var = fout.createVariable('iscoast', 'd', ('x2', 'y2', ))
    iscoast_var.long_name = 'coast mask (1:ocean coast, 0:else)'

    I_1to2_var = fout.createVariable('I_1to2', 'd', ('x1', 'y1', ))
    if do_fortran_bound_IJ:
        I_1to2_var.long_name = 'i-coordinate of grid #1 in grid #2, I=i (start at one/1, Fortran-bound)'
    else:
        I_1to2_var.long_name = 'i-coordinate of grid #1 in grid #2, I=i (start at zero/0, C-bound)'

    J_1to2_var = fout.createVariable('J_1to2', 'd', ('x1', 'y1', ))
    if do_fortran_bound_IJ:
        J_1to2_var.long_name = 'j-coordinate of grid #1 in grid #2, J=j (start at one/1, Fortran-bound)'
    else:
        J_1to2_var.long_name = 'j-coordinate of grid #1 in grid #2, J=j (start at zero/0, C-bound)'

    TI_1to2_var = fout.createVariable('TI_1to2', 'd', ('x1', 'y1', ))
    if do_fortran_bound_IJ:
        TI_1to2_var.long_name = 'Switched i-coordinate of grid #1 in grid #2, I=j (start at one/1, Fortran-bound)'
    else:
        TI_1to2_var.long_name = 'Switched i-coordinate of grid #1 in grid #2, I=j (start at zero/0, C-bound)'

    TJ_1to2_var = fout.createVariable('TJ_1to2', 'd', ('x1', 'y1', ))
    if do_fortran_bound_IJ:
        TJ_1to2_var.long_name = 'Switched j-coordinate of grid #1 in grid #2, J=i (start at one/1, Fortran-bound)'
    else:
        TJ_1to2_var.long_name = 'Switched j-coordinate of grid #1 in grid #2, J=i (start at zero/0, C-bound)'

    dist_var = fout.createVariable('distance', 'd', ('x1', 'y1', ))
    dist_var.long_name = 'distance to next vaild point (grid#1 -> grid#2)'
    dist_var.unit = 'm'
    print('    Populate variables')
    x1_var[:] = x_ice
    y1_var[:] = y_ice
    x2_var[:] = range(ilen2)
    y2_var[:] = range(jlen2)
    lat1_var[:] = lat1
    lon1_var[:] = lon1
    lat2_var[:] = lat2
    lon2_var[:] = lon2
    isocean_var[:] = isocean
    iscoast_var[:] = iscoast
    if do_grid1_2_same_orientation:
        I_1to2_var[:] = I_1to2
        J_1to2_var[:] = J_1to2
        TI_1to2_var[:] = J_1to2
        TJ_1to2_var[:] = I_1to2
    else:
        I_1to2_var[:] = J_1to2
        J_1to2_var[:] = I_1to2
        TI_1to2_var[:] = I_1to2
        TJ_1to2_var[:] = J_1to2
    dist_var[:] = Distance
    ################
    print("Close all open files")
    fout.close()
    return fout


def main(args):
    ice_file = "test_data/pism_ini.nc"
    # TODO: Turn these prints into logging messages
    print("PROGRAM: gridpoint_map_ice_to_ocean.py")
    print(
        "Calculates distances of grid 1 to grid 2 and coordinates of grid 1 in grid 2"
    )
    print("Start!")

    print("Load Data files...")
    print("...ice data...")
    lat1, lon1, x_ice, y_ice = read_ice(args["ICE_FILE"])
    print("...ocean_data...")
    tmask, lat2, lon2 = read_ocean(args["OCEAN_FILE"])
    print("...finished!")

    print("Deriving some values...")
    lon1, lon2 = np.where(lon1 > 180.0, lon1 - 360.0, lon1), np.where(
        lon2 > 180.0, lon2 - 360.0, lon2)
    ilen1, jlen1 = np.size(x_ice), np.size(y_ice)
    ilen2, jlen2 = np.shape(lat2)
    isocean = np.where(tmask > 0, True, False)
    iscoast = make_coastline_mask(ilen2, jlen2, isocean)
    print("...finished!")

    print("Main loop...")
    Distance, I_1to2, J_1to2 = calculate_distances(
        ilen1, jlen1, ilen2, jlen2, lat1, lon1, lat2, lon2, iscoast)
    print("...finished!")

    print("Re-ordering bad points...")
    I_bad = [255, 257]
    J_bad = [244, 244]
    I_good = [255, 257]
    J_good = [242, 243]
    I_1to2, J_1to2 = reorder_bad_points(ilen1, jlen1, I_bad, J_bad, I_good,
                                        J_good, I_1to2, J_1to2)
    print("...finished!")

    print("Save the results to a netcdf file...")
    fout = save_results(args["OUTPUT_FILE"], ilen1, jlen1, ilen2, jlen2, x_ice,
                        y_ice, lat1, lon1, lat2, lon2, isocean, iscoast,
                        I_1to2, J_1to2, Distance)
    print("...finished!")


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    do_grid1_2_same_orientation = args["--ice-grid-orientation-as-ocean"] or False
    do_fortran_bound_IJ = args["--do-fortran-boundIJ"] or False
    #if args['--binary-erosion-coast']:
    make_coastline_mask = iscoast_binary_erosion
    #else:
    #    make_coastline_mask = iscoast_explicit
    main(args)
