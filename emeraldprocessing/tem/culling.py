# -*- coding: utf-8 -*-
import os
import typing
import pyproj
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

from .setup import allowed_moments
import copy

import emeraldprocessing.tem.dataIO

from .data_keys import inuse_dtype
from .data_keys import pos_keys
from .data_keys import dat_key_prefix, inuse_key_prefix, std_key_prefix, err_key_prefix
from .parameter_types import HiddenBool
from .parameter_types import Channel, ChannelAndGate, ChannelAndGateRange
from .parameter_types import ShapeUrl, DistanceDict
from .parameter_types import InversionModelUrls

from .utils import calculate_transient_slopes, calculate_transient_curvatures
from .utils import build_inuse_dataframe
from .utils import inuse_moment, estimateInlineSamplig, errKey, filtXYZ, get_flightline_bbox, read_shape_in_margins
from . import utils
from .. import pipeline

# import pydantic
# from emeraldprocessing import maptiler


def cull_roll_pitch_alt(processing: pipeline.ProcessingData,
                        channel: Channel = 1,
                        max_roll: float = 10.,
                        max_pitch: float = 10.,
                        max_alt: float = 110.,
                        min_alt: float = 5.,
                        verbose: bool = False,
                        save_filter_to_layer_data: HiddenBool = False):
    """
    This filter disables soundings where the roll, pitch, or altitude exceed the input values.

    Parameters
    ----------
    channel :
        Channel to disable soundings from
    max_roll :
        Maximum Tx roll angle deviation from 0, in degrees.
    max_pitch :
        Maximum Tx pitch angle deviation from 0, in degrees.
    max_alt :
        Maximum Tx altitude, in meters above terrain surface
    min_alt :
        Minimum Tx altitude, in meters above terrain surface
    verbose :
        If True, more output about what the filter is doing
    """
    start = time.time()
    print(f"  - Disabling channel {channel} based on roll (±{max_roll}°), pitch(±{max_pitch}°), and altitude ({max_alt} m) limits")
    data = processing.xyz

    roll_key = data.tilt_roll_column
    pitch_key = data.tilt_pitch_column
    alt_key = data.alt_column

    str_channel = f"0{channel}"[-2:]
    inuse_key = f"{inuse_key_prefix}{str_channel}"

    idx = pd.DataFrame(columns=["roll", "pitch", "alt"], dtype=bool)
    idx['roll'] = data.flightlines[roll_key].abs() > max_roll
    idx['pitch'] = data.flightlines[pitch_key].abs() > max_pitch
    idx['alt_max'] = data.flightlines[alt_key] >= max_alt
    idx['alt_min'] = data.flightlines[alt_key] < min_alt

    fl_filt =  idx.roll | idx.pitch | idx.alt_max | idx.alt_min

    # inuse_key_list = [key for key in data.layer_data.keys() if inuse_key_prefix in key]
    # for inuse_key in inuse_key_list:
    inuse_df = build_inuse_dataframe(data=data, channel=int(inuse_key.split(inuse_key_prefix)[1]))
    inuse_df.iloc[fl_filt, :] = 0

    filt = inuse_df == 0
    data.layer_data[inuse_key][filt] = 0

    if save_filter_to_layer_data:
        new_iu_key = f"disable_tilt_alt_{inuse_key}"
        data.layer_data[new_iu_key] = inuse_df
    
    data.flightlines.loc[idx.roll | idx.pitch, 'disable_reason'] = 'tilt'
    data.flightlines.loc[idx.alt_max | idx.alt_min, 'disable_reason'] = 'alt'

    pre_dat_stats = data.flightlines.loc[:, [roll_key, pitch_key, alt_key]].describe()
    pre_dat_stats.columns.name = "Pre-disable Tx_ Pitch, Alt stats"

    post_dat_stats = data.flightlines.loc[:, [roll_key, pitch_key, alt_key]].copy()
    post_dat_stats.iloc[fl_filt, :] = np.nan
    post_dat_stats = post_dat_stats.describe()
    post_dat_stats.columns.name = "Post-disable Tx_ Pitch, Alt stats"

    if verbose:
        print('')
        print(pre_dat_stats)
        print('')
        print(post_dat_stats)
        print('')
    else:
        print(f"  - Pre-disable statistics:")
        print(f"    - {roll_key}:     {pre_dat_stats.loc['mean', roll_key].round(2)} ± {pre_dat_stats.loc['std', roll_key].round(2)} °")
        print(f"    - {pitch_key}:    {pre_dat_stats.loc['mean', pitch_key].round(2)} ± {pre_dat_stats.loc['std', pitch_key].round(2)} °")
        print(f"    - {alt_key}: {pre_dat_stats.loc['mean', alt_key].round(2)} ± {pre_dat_stats.loc['std', alt_key].round(2)} m")
        print(f"  - Post-disable statistics:")
        print(f"    - {roll_key}:     {post_dat_stats.loc['mean', roll_key].round(2)} ± {post_dat_stats.loc['std', roll_key].round(2)} °")
        print(f"    - {pitch_key}:    {post_dat_stats.loc['mean', pitch_key].round(2)} ± {post_dat_stats.loc['std', pitch_key].round(2)} °")
        print(f"    - {alt_key}: {post_dat_stats.loc['mean', alt_key].round(2)} ± {post_dat_stats.loc['std', alt_key].round(2)} m")

    print(f"  - {np.round((np.sum(np.sum(filt.values)) / (inuse_df.shape[0] * inuse_df.shape[1])) * 100, 2)} % of the data have been deactivated by this filter")

    end = time.time()
    print(f"  - Time used to disable data based on tilt and altitude: {end - start} sec.\n")


def cull_on_geometry(processing: pipeline.ProcessingData,
                     shapefile: ShapeUrl = '/path/to/my/shape/file.shp',
                     distance_dict: DistanceDict = {'Gate_Ch01': {'first_gate': 30,
                                                                 'last_gate': 75},
                                                    'Gate_Ch02': {'first_gate': 50,
                                                                 'last_gate': 150}},
                     save_filter_to_layer_data: HiddenBool = False):
    """
    Function to disable all data within a safety distance from a geometric feature defined by a shape file

    Parameters
    ----------
    shapefile :
        DESCRIPTION. example: '/path/to/my/shape/file.shp'.
    distance_dict :
        All datapoints with a distance to shape smaller than safety distance will be disabled.
        You can define a trapeze-like disabling window by defining the distance for the first and last gate of the moment.
    """
    
    feature_name = os.path.basename(shapefile)
    
    safety_distance_dict = {}
    for moment in distance_dict.keys():
        safety_distance_dict[moment] = [distance_dict[moment]['first_gate'],
                                       distance_dict[moment]['last_gate']]
    
    data = processing.xyz
    data_crs = processing.crs
    print("  - Disabling data based on geometry")
    print(f"  - Shapefile:")
    print(f"      {shapefile}")
    bbox = get_flightline_bbox(data)
    df_shp, bbox = read_shape_in_margins(shapefile, bbox, data_crs)    
    if len(df_shp) > 0:
        cullGeometry(data, df_shp, safety_distance_dict, data_crs, QCplot=False, feature_name=feature_name, save_filter_to_layer_data=save_filter_to_layer_data)
    else:
        print("No obstructions from this file inside the survey area")


def cull_on_geometry_and_inversion_misfit(processing: pipeline.ProcessingData,
                                          inversion: InversionModelUrls,
                                          shapefile: ShapeUrl,
                                          maxRMS: float = 5.0,
                                          search_dist: float = 100,
                                          disable_earlier_gates: int = 0,
                                          disable_sounding_tails: bool = True,
                                          cleanup: bool = True,
                                          save_filter_to_layer_data: HiddenBool = False):
    """
    Function to disable data based on proximity to known infrastructure (described by .shp files) and high inversion misfits

    Parameters
    ----------
    inversion :
        model:
            the inverted model (your_prefix_inv.xyz)
            Example: path/to/my/inversion/data/inversion_inv.xyz
        fwd:
            the synthetic (forward) data for the inverted model (your_prefix_inv.xyz)
            Example: path/to/my/inversion/data/inversion_syn.xyz
        measured:
            the input data (your_prefix_dat.xyz)
            Example: path/to/my/inversion/data/inversion_dat.xyz
    shapefile :
        Example: /your/file/path/to/Pipelines.shp
    maxRMS :
        all data with RMS > maxRMS will be disabled
    search_dist :
        buffer around infrastructure defined by .shp file that will be considered
    disable_earlier_gates :
        The number of timegates earlier to disable when the filter finds a problem with the data
        This may deactivate gates earlier than the gate specified in channel_and_gate.
        Note: negative values are allowed and will disable the number of timegates after a problem is found
    disable_sounding_tails :
        If True, disables all timegates after the first gate that is disabled
    cleanup :
        if false, leave a misfit column in output
    """
    start = time.time()
    print('  - Disabling based on geometry and inversion misfit (gate-by-gate)')

    inv = emeraldprocessing.tem.dataIO.readWBinversionDataExportFull(
        inversion["measured"],
        inversion["fwd"],
        inversion["model"],
        processing.gex)

    utils.calcGateRelErr(inv.data.xyz, inv.synthetic.xyz)
    assign_InvError_to_Data(inv.data, processing)

    data = processing.xyz
    print('  - Using shp file:\n{}'.format(shapefile))

    # find distance between sounding locations and shape
    bbox = get_flightline_bbox(data)
    df_shp, bbox = read_shape_in_margins(shapefile, bbox, processing.crs)
    if len(df_shp) == 0:
        print("  - No obstructions from this file inside the survey area.")
    else:
        print("  - Calculating distance to geometry.")
        df_points = gpd.GeoDataFrame(data.flightlines[[pos_keys[0], pos_keys[1]]])
        df_points.set_geometry(
            gpd.points_from_xy(df_points[pos_keys[0]], df_points[pos_keys[1]]),
            inplace=True,
            crs=processing.crs)
        bbox = df_shp.total_bounds
        df_points['min_dist_to_lines'] = dist_points_shp(df_points, df_shp)

        # filter soundings that are within search_dist (geometry only)
        dist_filt = df_points['min_dist_to_lines'] < search_dist

        # assign inversion error to data
        if not ('relErr_Ch01' in data.layer_data.keys()):
            assign_InvError_to_Data(inv.data, data)
        else:
            print('  - Inversion error already assigned to data')

        # combine geometry and inversion error
        print('  - Combining geometry and inv-error.')
        for moment in allowed_moments:
            channel = int(moment.split(dat_key_prefix)[1])
            str_channel = f"0{channel}"[-2:]

            inuse_key = f"{inuse_key_prefix}{str_channel}"
            err_key = f"{err_key_prefix}{str_channel}"

            inuse_df = build_inuse_dataframe(data=data, channel=channel)

            # filter gates that have too high Err and are too close to infrastructure:
            for col in data.layer_data[err_key].columns:
                err_filt = data.layer_data[err_key].loc[:, col] > maxRMS
                n = 5
                cull_filt = (err_filt.rolling(n, center=True,
                                              min_periods=1).sum() > 0) & dist_filt  # additional rolling filter to also remove single soundings with low RMS
                inuse_df.loc[cull_filt, col] = 0  # assign inuse flags
                data.flightlines.loc[cull_filt, 'disable_reason'] = 'geom_err'

            if disable_sounding_tails:
                cull_sounding_tails(inuse_df)

            if disable_earlier_gates != 0:
                cull_earlier_gates(inuse_df, disable_earlier_gates)

            filt = inuse_df == 0
            data.layer_data[inuse_key][filt] = 0

            if save_filter_to_layer_data:
                new_iu_key = f"disable_geom_misfit_{inuse_key}"
                data.layer_data[new_iu_key] = inuse_df

        if cleanup:
            for moment in allowed_moments:
                del (data.layer_data[errKey(moment)])

    print(f"  - {np.round((np.sum(np.sum(filt.values)) / (inuse_df.shape[0] * inuse_df.shape[1])) * 100, 2)} % of the data have been deactivated by this filter")

    end = time.time()
    print(f"\nDisabling based on geometry and misfit took: {end - start} sec.\n")


# FIXME: 20240620 - BRB - I think the filter 'cullGeometryRMS' is replaced by
# FIXME:     'cull_on_geometry_and_inversion_misfit' (directly above)
# FIXME:     If not, documentation of inv_data and shape_dict needs to be done
def cullGeometryRMS(processing: pipeline.ProcessingData,
                    inv_data,
                    shape_dict):
    """
    What Do I do?!?

    Parameters
    ----------
    inv_data :
    shape_dict :
    """
    start = time.time()
    print("  - Disabling based on geometry and RMS")
    data = processing.xyz

    print(f"  - Shapefile:")
    print(f"      {shape_dict['file']}")
    # find distance between sounding locations and shape
    bbox = get_flightline_bbox(data)
    df_shp, bbox = read_shape_in_margins(shape_dict['file'], bbox, processing.crs)
    if len(df_shp) == 0:
        print("  - No obstructions from this file inside the survey area")
    else:
        df_points = gpd.GeoDataFrame(data.flightlines[[pos_keys[0], pos_keys[1]]])
        df_points.set_geometry(gpd.points_from_xy(df_points[pos_keys[0]], df_points[pos_keys[1]]),
                               inplace=True,
                               crs=processing.crs)
        bbox = df_shp.total_bounds
        df_points['min_dist_to_lines'] = dist_points_shp(df_points, df_shp)

        # filter soundings that are within search_dist
        dist_filt = df_points['min_dist_to_lines'] < shape_dict['search_dist']

        # find RMS in inverted dataset
        dist, closest_idx = cKDTree(inv_data.flightlines[["utmx", "utmy"]].values).query(
            data.flightlines[["UTMX", "UTMY"]].values)
        has_no_pair = dist > estimateInlineSamplig(data) * 1.5
        data.flightlines['resdata'] = inv_data.flightlines['resdata'].iloc[closest_idx].values
        data.flightlines.loc[has_no_pair, 'resdata'] = np.nan

        # filter soundings that have too high RMS:
        rms_filt = data.flightlines['resdata'] > shape_dict['maxRMS']

        cull_filt = rms_filt & dist_filt
        # additional rolling filter to also remove single soundings with low RMS
        n = 5
        cull_filt = (cull_filt.rolling(n, center=True, min_periods=1).sum() > 0) & dist_filt

        # FIXME: need to query for a channel
        for key in ['InUse_Ch01', 'InUse_Ch02']:
            if key in data.layer_data.keys():
                data.layer_data[key].loc[cull_filt, :] = 0

        data.flightlines.loc[cull_filt, 'disable_reason'] = 'geom_misfit'
    end = time.time()
    print(f"  - Time used for RMS based disabling: {end - start} sec.\n")


def cullGeometryOSM(data,
                    safety_distance_dict,
                    data_crs,
                    QCplot=True,
                    zoom=None,
                    save_filter_to_layer_data: HiddenBool = False):
    print("  - Disabling based on geometry")
    bbox = get_flightline_bbox(data)
    bbox = (pyproj.Transformer.from_crs(data_crs, 4326, always_xy=True).transform(bbox[0], bbox[1]) +
            pyproj.Transformer.from_crs(data_crs, 4326, always_xy=True).transform(bbox[2], bbox[3]))
    m = maptiler.Maptiler("https://openinframap.org/map.json")
    df_shp = pd.concat(m.tiles(*bbox + (zoom,)))
    df_shp = df_shp.set_crs(3857).to_crs(data_crs)
    return cullGeometry(data, df_shp, safety_distance_dict, data_crs, QCplot, save_filter_to_layer_data=save_filter_to_layer_data)


def dist_points_shp(df_points,
                    df_shp):
    # calculate distance between points and shp:
    gs_points = df_points['geometry']
    gs_shp = df_shp['geometry']
    min_dist = np.empty(gs_points.shape[0])
    for i, point in enumerate(gs_points):
        dist = [point.distance(feature) for feature in gs_shp]
        if len(dist) > 0:
            min_dist[i] = np.min(dist)
        else:
            raise Exception("Sorry, no obstructions from this file in the survey area")
    return min_dist


def cullGeometry(data,
                 df_shp,
                 safety_distance_dict,
                 data_crs,
                 QCplot=True,
                 feature_name='powerline',
                 save_filter_to_layer_data: HiddenBool = False):
    start = time.time()

    df_points = gpd.GeoDataFrame(data.flightlines[[pos_keys[0], pos_keys[1]]])
    df_points.set_geometry(
        gpd.points_from_xy(df_points[pos_keys[0]], df_points[pos_keys[1]]),
        inplace=True, 
        crs=data_crs)
    bbox = df_shp.total_bounds
    
    min_dist = dist_points_shp(df_points, df_shp)
    
    # facilitate gate dependent safety distance if True:
    
    for moment in safety_distance_dict:
        safety_distance = safety_distance_dict[moment]

        channel = int(moment.split(dat_key_prefix)[1])
        str_channel = f"0{channel}"[-2:]

        inuse_key = f"{inuse_key_prefix}{str_channel}"
        inuse_df = build_inuse_dataframe(data, channel)

        if len(safety_distance) > 1:
            gate_index = []
            channels = data.layer_data[moment].columns
            for n, dist in enumerate(safety_distance):
                gate_index.append( round((len(channels)-1) * (n/(len(safety_distance)-1))))
            f = interp1d(gate_index, safety_distance)
            safety_distance_i = f(np.arange(len(channels)))
            for n, channel in enumerate(channels):
                idx = min_dist < safety_distance_i[n]
                inuse_df.loc[idx, channel] = 0
                # data.layer_data[inuse_key].loc[idx, channel] = 0
                data.flightlines.loc[idx, 'disable_reason'] = 'geometry'
        else:  # just one safety distance for all gates
            idx = min_dist < safety_distance
            inuse_df.loc[idx, :] = 0
            # data.layer_data[inuse_key].loc[idx, :] = 0
            data.flightlines.loc[idx, 'disable_reason'] = 'geometry'

        filt = inuse_df == 0
        data.layer_data[inuse_key][filt] = 0

        if save_filter_to_layer_data:
            new_iu_key = f"disable_geom_{inuse_key}"
            data.layer_data[new_iu_key] = inuse_df

        print(f"  - {np.round((np.sum(np.sum(filt.values)) / (inuse_df.shape[0] * inuse_df.shape[1])) * 100, 2)} % of the data have been deactivated by this filter")

    if QCplot:
        fig, ax = plt.subplots(figsize=(10, 10))
        df_shp['geometry'].plot(ax=ax, color='black', edgecolor='black', label=feature_name)
        sc = ax.scatter(df_points[pos_keys[0]], df_points[pos_keys[1]], c=min_dist,
                         s=3,
                         cmap='viridis_r',
                         vmin=0,
                         vmax=max(safety_distance),
                         label = "AEM soundings - distance to {}".format(feature_name))
        ax.set_aspect('equal')
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
        plt.colorbar(sc, ax=ax, shrink=0.5)
        ax.legend()
        ax.set_title(' distance from geometry to data points')

    end = time.time()
    print(f"  - Time used for geometry based disabling: {end - start} sec.\n")


def cull_soundings_with_too_few_gates(processing: pipeline.ProcessingData,
                                      channel: Channel = 1,
                                      min_number_of_gates: int = 4,
                                      save_filter_to_layer_data: HiddenBool = False):
    """
    Disable the entire sounding if the sounding has fewer than min_number_of_gates in it.

    Parameters
    ----------
    channel :
        Channel to disable soundings from
    min_number_of_gates :
        Soundings with less than min_number_of_gates datapoints will be disabled entirely
    """
    start = time.time()
    print(f'  - Disabling soundings with too few timegates from channel-{channel}')
    data = processing.xyz

    str_channel = f"0{channel}"[-2:]
    inuse_key = f"{inuse_key_prefix}{str_channel}"

    n_gate_in_use = data.layer_data[inuse_key].sum(axis=1)
    filt = n_gate_in_use < min_number_of_gates

    inuse_df = build_inuse_dataframe(data=data, channel=channel)
    inuse_df.loc[filt, :] = 0

    filt = inuse_df == 0
    data.layer_data[inuse_key][filt] = 0

    if save_filter_to_layer_data:
        new_iu_key = f"too_few_gate_{inuse_key}"
        data.layer_data[new_iu_key] = inuse_df

    print(f"  - {np.round((np.sum(np.sum(filt.values)) / (inuse_df.shape[0] * inuse_df.shape[1])) * 100, 2)} % of the data have been deactivated by this filter")

    end = time.time()
    print(f"  - Time used for sounding disabling: {end - start} sec.\n")


def cull_sounding_tails(inuse_df):
    filt = (inuse_df.diff(axis=1) == -1).cumsum(axis=1) > 0
    inuse_df[filt] = 0


def cull_earlier_gates(inuse_df, num_earlier):
    if num_earlier != 0:
        filt = inuse_df == 0
        for earlier_gate in range(0, abs(num_earlier)):
            if num_earlier > 0:
                t_filt = filt.shift(-1, axis=1)
                t_filt.iloc[:, -1] = t_filt.iloc[:, -2]
            elif num_earlier < 0:
                t_filt = filt.shift(1, axis=1)
                t_filt.iloc[:, 0] = t_filt.iloc[:, 1]
            filt = filt | t_filt
        inuse_df[filt] = 0


def cull_std_threshold(processing: pipeline.ProcessingData,
                       channel_and_gate: ChannelAndGate = {'channel': 1,
                                                           'gate': 7},
                       std_threshold: float = 0.15,
                       disable_earlier_gates: int = 0,
                       disable_sounding_tails: bool = True,
                       save_filter_to_layer_data: HiddenBool = False):
    """
    Disable data where the STD of the datum is larger than the specified std_threshold.

    Parameters
    ----------
    channel_and_gate :
        The channel and first timegate that you would like evaluated.
         Note: All timegates after the first gate will be considered.
    std_threshold :
        Datapoints with STD's higher than std threshold will be disabled
    disable_earlier_gates :
        The number of timegates earlier to disable when the filter finds a problem with the data
        This may deactivate gates earlier than the gate specified in channel_and_gate.
        Note: negative values are allowed and will disable the number of timegates after a problem is found
    disable_sounding_tails :
        If True, disables the tails (later timegates) after the first disabled timegate
    """
    start = time.time()
    channel = channel_and_gate['channel']
    first_gate_to_consider = channel_and_gate['gate']

    print(f"  - Disabling channel-{channel} after gate-{first_gate_to_consider} based on {std_threshold} STD threshold")
    data = processing.xyz

    str_channel = f"0{channel}"[-2:]
    inuse_key = f"{inuse_key_prefix}{str_channel}"
    std_key   = f"{std_key_prefix}{str_channel}"

    if std_key not in data.layer_data.keys():
        print(f"  - no '{std_key}' in layer_data dictionary. No data have been deactivated by this filter.")
    else:
        inuse_df = build_inuse_dataframe(data=data, channel=channel)

        filt = data.layer_data[std_key] > std_threshold
        inuse_df[filt] = 0
        inuse_df.iloc[:, :first_gate_to_consider + 1] = 1

        if disable_sounding_tails:
            cull_sounding_tails(inuse_df)

        if disable_earlier_gates != 0:
            cull_earlier_gates(inuse_df, disable_earlier_gates)

        filt = inuse_df == 0

        data.layer_data[inuse_key][filt] = 0

        if save_filter_to_layer_data:
            new_iu_key = f"disable_STD_{inuse_key}"
            data.layer_data[new_iu_key] = inuse_df

        print(f"  - {np.round((np.sum(np.sum(filt.values)) / (inuse_df.shape[0] * inuse_df.shape[1])) * 100, 2)} % of the data have been deactivated by this filter")

    end = time.time()
    print(f"  - Time used for STD based disabling: {end - start} sec.\n")


def cull_negative_data(processing: pipeline.ProcessingData,
                       channel_and_gate: ChannelAndGate = {'channel': 1,
                                                           'gate': 7},
                       disable_earlier_gates: int = 0,
                       disable_sounding_tails: bool = True,
                       save_filter_to_layer_data: HiddenBool = False):
    """
    Processing filter that disables negative data (sets 'in-use' flag to 0 if dB/dt < 0) after the time-gate specified.

    Parameters
    ----------
    channel_and_gate :
        Which channel and first timegate to consider when looking for negative data
        note: All timegates will be considered after the timegate specified
    disable_earlier_gates :
        The number of timegates earlier to disable when the filter finds a problem with the data
        This may deactivate gates earlier than the gate specified in channel_and_gate.
        Note: negative values are allowed and will disable the number of timegates after a problem is found
    disable_sounding_tails :
        If True, disable sounding tails (later timegates) after the first negative data is detected.
    """
    start = time.time()
    channel = channel_and_gate['channel']
    first_gate_to_consider = channel_and_gate['gate']

    print(f'  - Disabling negative data from channel-{channel} after gate-{first_gate_to_consider}')
    data = processing.xyz

    str_channel = f"0{channel}"[-2:]
    data_key = f"{dat_key_prefix}{str_channel}"
    inuse_key = f"{inuse_key_prefix}{str_channel}"

    inuse_df = build_inuse_dataframe(data=data, channel=channel)

    filt = data.layer_data[data_key] < 0
    inuse_df[filt] = 0
    inuse_df.loc[:, :first_gate_to_consider + 1] = 1

    if disable_sounding_tails:
        cull_sounding_tails(inuse_df)

    if disable_earlier_gates != 0:
        cull_earlier_gates(inuse_df, disable_earlier_gates)

    filt = inuse_df == 0
    data.layer_data[inuse_key][filt] = 0

    if save_filter_to_layer_data:
        new_iu_key = f"disable_neg_{inuse_key}"
        data.layer_data[new_iu_key] = inuse_df

    print(f"  - {np.round((np.sum(np.sum(filt.values)) / (inuse_df.shape[0] * inuse_df.shape[1])) * 100, 2)} % of the data have been deactivated by this filter")

    end = time.time()
    print(f"  - Time used to disable negative data: {end - start} sec.\n")


def cull_max_slope(processing: pipeline.ProcessingData,
                   channel_and_gate: ChannelAndGate = {'channel': 1,
                                                       'gate': 7},
                   max_slope: float = 0.,
                   disable_earlier_gates: int = 0,
                   disable_sounding_tails: bool = True,
                   verbose: bool = False,
                   save_filter_to_layer_data: HiddenBool = False):
    """
    Function to disable datapoints in a transient curve where the slope between two datapoints
    is greater than (shallower than) the maximum_slope specified.
    Slope is d(log10(dBdt))/d(log10(t)), which is equivalent to t^(slope).
    For reference, the slope of a halfsapce is t^(-5/2) and the accepted slope of background noise is t^(-1/2).

    Interpreting slope values:
      Steeper Slope: Indicates a higher resistivity. This could suggest a more resistive layer or material
      Shallower Slope: Suggests a higher conductivity. This could suggest a more conductive layer or material.

    Parameters
    ----------
    channel_and_gate :
        the channel to evaluate and the first gate to consider for disabling.
        All timegates after the first gate and above the max_slope will be disabled
    max_slope :
        Datapoints of the transient curve with slope > max_slope are disabled.
    disable_earlier_gates :
        The number of timegates earlier to disable when the filter finds a problem with the data
        This may deactivate gates earlier than the gate specified in channel_and_gate.
        Note: negative values are allowed and will disable the number of timegates after a problem is found
    disable_sounding_tails :
        If True, disable timegates after the first slope that exceeds the max slope is encountered.
    verbose :
        If True, more output about what the filter is doing
    """
    start = time.time()
    channel = channel_and_gate['channel']
    first_gate_to_consider = channel_and_gate['gate']

    print(f"  - Disabling slopes greater than {max_slope} on channel-{channel} after gate-{first_gate_to_consider}")
    data = processing.xyz

    str_channel = f"0{channel}"[-2:]
    data_key = f"{dat_key_prefix}{str_channel}"
    inuse_key = f"{inuse_key_prefix}{str_channel}"

    inuse_df = build_inuse_dataframe(data=data, channel=channel)

    slope = calculate_transient_slopes(processing, data_key)

    max_filt = slope > max_slope
    max_filt[max_filt.isna()] = False  # diff returns nans in the begining of the trace

    inuse_df[max_filt] = 0
    inuse_df.loc[:, :first_gate_to_consider + 1] = 1

    if disable_sounding_tails:
        cull_sounding_tails(inuse_df)

    if disable_earlier_gates != 0:
        cull_earlier_gates(inuse_df, disable_earlier_gates)

    filt = inuse_df == 0
    data.layer_data[inuse_key][filt] = 0

    if save_filter_to_layer_data:
        new_iu_key = f"disable_max_slope_{inuse_key}"
        data.layer_data[new_iu_key] = inuse_df

    if not verbose:
        flat_slope = slope.loc[:, first_gate_to_consider:].to_numpy().flatten()
        print(f"  - Slope statistics for all soundings after gate-{first_gate_to_consider}:")
        print(f"      pre_mean_slope  = {np.round(np.nanmean(flat_slope), 3)} ± {np.round(np.nanstd(flat_slope), 3)}")
        slope[max_filt] = np.nan
        zero_filt = slope == 0
        iu_slope = slope.loc[:, first_gate_to_consider:] * inuse_df.loc[:, first_gate_to_consider:]
        iu_filt = iu_slope == 0
        iu_slope[iu_filt] = np.nan
        iu_slope[zero_filt] = 0
        flat_slope = iu_slope.to_numpy().flatten()
        print(f"      post_mean_slope = {np.round(np.nanmean(flat_slope), 3)} ± {np.round(np.nanstd(flat_slope), 3)}")
    else:
        describe_slope = slope.describe()
        describe_slope.loc['count', 'all'] = describe_slope.loc['count', :].sum()
        describe_slope.loc['mean', 'all'] = (describe_slope.loc['mean', :] * describe_slope.loc['count',
                                                                                     :]).sum() / describe_slope.loc[
                                                    'count', 'all']
        describe_slope.loc['std', 'all'] = (describe_slope.loc['std', :] * describe_slope.loc['count',
                                                                                   :]).sum() / describe_slope.loc[
                                                   'count', 'all']
        describe_slope.loc['min', 'all'] = describe_slope.loc['min', :].min()
        describe_slope.loc['max', 'all'] = describe_slope.loc['max', :].max()
        describe_slope = describe_slope.loc[['count', 'mean', 'std', 'min', 'max'], :].T.round(3)
        describe_slope.columns.name = "Pre-disable"
        print(f"  - Slope statistics for all gates:\n")
        print(f"{describe_slope}\n")
        slope[max_filt] = np.nan
        describe_slope = slope * inuse_df
        describe_filt = describe_slope == 0
        describe_slope[describe_filt] = np.nan
        describe_slope = describe_slope.describe()
        describe_slope.loc['count', 'all'] = describe_slope.loc['count', :].sum()
        describe_slope.loc['mean', 'all'] = (describe_slope.loc['mean', :] * describe_slope.loc['count',
                                                                                     :]).sum() / describe_slope.loc[
                                                    'count', 'all']
        describe_slope.loc['std', 'all'] = (describe_slope.loc['std', :] * describe_slope.loc['count',
                                                                                   :]).sum() / describe_slope.loc[
                                                   'count', 'all']
        describe_slope.loc['min', 'all'] = describe_slope.loc['min', :].min()
        describe_slope.loc['max', 'all'] = describe_slope.loc['max', :].max()
        describe_slope = describe_slope.loc[['count', 'mean', 'std', 'min', 'max'], :].T.round(3)
        describe_slope.columns.name = "Post-disable"
        print(f"{describe_slope}\n")

    print(f"  - {np.round((np.sum(np.sum(filt.values)) / (inuse_df.shape[0] * inuse_df.shape[1])) * 100, 2)} % of the data have been deactivated by this filter")

    end = time.time()
    print(f"  - Time used for maximum slope disabling: {end - start} sec.\n")


def cull_min_slope(processing: pipeline.ProcessingData,
                   channel_and_gate: ChannelAndGate = {'channel': 1,
                                                       'gate': 7},
                   min_slope: float = -5.,
                   disable_earlier_gates: int = 0,
                   disable_sounding_tails: bool = True,
                   verbose = False,
                   save_filter_to_layer_data: HiddenBool = False):
    """
    Function to disable datapoints in a transient curve where the slope between two datapoints
    is less than (steeper than) the minimum_slope specified.
    Slope is d(log10(dBdt))/d(log10(t)), which is equivalent to (t^(slope)).
    For reference, the slope of a halfsapce is t^(-5/2) and the accepted slope of background noise is t^(-1/2).

    Interpreting slope values:
      Steeper Slope: Indicates a higher resistivity. This could suggest a more resistive layer or material
      Shallower Slope: Suggests a higher conductivity. This could suggest a more conductive layer or material.

    Parameters
    ----------
    channel_and_gate :
        the channel to evaluate and the first gate to consider for disabling.
        All timegates after the specified timegate and below the min_slope will be disabled
    min_slope :
        Datapoints of the transient curve with slope < min_slope are disabled.
    disable_earlier_gates :
        The number of timegates earlier to disable when the filter finds a problem with the data.
        This may deactivate gates earlier than the gate specified in channel_and_gate.
        Note: negative values are allowed and will disable the number of timegates after a problem is found
    disable_sounding_tails :
        If True, disable timegates after the first slope that is less than the min slope is encountered.
    verbose :
        If True, more output about what the filter is doing
    """
    start = time.time()
    channel = channel_and_gate['channel']
    first_gate_to_consider = channel_and_gate['gate']

    print(f"  - Disabling slopes less than {min_slope} on channel-{channel} after gate-{first_gate_to_consider}")
    data = processing.xyz

    str_channel = f"0{channel}"[-2:]
    data_key = f"{dat_key_prefix}{str_channel}"
    inuse_key = f"{inuse_key_prefix}{str_channel}"

    inuse_df = build_inuse_dataframe(data=data, channel=channel)

    slope = calculate_transient_slopes(processing, data_key)

    min_filt =  slope < min_slope
    min_filt[min_filt.isna()] = False  # diff returns nans in the begining of the trace

    inuse_df[min_filt] = 0
    inuse_df.loc[:, :first_gate_to_consider + 1] = 1  # reset inuse flags earlier than first_gate_to_consider

    if disable_sounding_tails:
        cull_sounding_tails(inuse_df)

    if disable_earlier_gates != 0:
        cull_earlier_gates(inuse_df, disable_earlier_gates)

    filt = inuse_df == 0
    data.layer_data[inuse_key][filt] = 0

    if save_filter_to_layer_data:
        new_iu_key = f"disable_min_slope_{inuse_key}"
        data.layer_data[new_iu_key] = inuse_df

    if not verbose:
        flat_slope = slope.loc[:, first_gate_to_consider:].to_numpy().flatten()
        print(f"  - Slope statistics for all soundings after gate-{first_gate_to_consider}:")
        print(f"      pre_mean_slope  = {np.round(np.nanmean(flat_slope), 3)} ± {np.round(np.nanstd(flat_slope), 3)}")
        slope[min_filt] = np.nan
        zero_filt = slope == 0
        iu_slope = slope.loc[:, first_gate_to_consider:] * inuse_df.loc[:, first_gate_to_consider:]
        flat_filt = iu_slope == 0
        iu_slope[flat_filt] = np.nan
        iu_slope[zero_filt] = 0
        flat_slope = iu_slope.to_numpy().flatten()
        print(f"      post_mean_slope = {np.round(np.nanmean(flat_slope), 3)} ± {np.round(np.nanstd(flat_slope), 3)}")
    else:
        describe_slope = slope.describe()
        describe_slope.loc['count', 'all'] = describe_slope.loc['count', :].sum()
        describe_slope.loc['mean', 'all'] = (describe_slope.loc['mean', :] * describe_slope.loc['count',
                                                                                     :]).sum() / describe_slope.loc[
                                                    'count', 'all']
        describe_slope.loc['std', 'all'] = (describe_slope.loc['std', :] * describe_slope.loc['count',
                                                                                   :]).sum() / describe_slope.loc[
                                                   'count', 'all']
        describe_slope.loc['min', 'all'] = describe_slope.loc['min', :].min()
        describe_slope.loc['max', 'all'] = describe_slope.loc['max', :].max()
        describe_slope = describe_slope.loc[['count', 'mean', 'std', 'min', 'max'], :].T.round(3)
        describe_slope.columns.name = "Pre-disable"
        print(f"  - Slope statistics for all gates:\n")
        print(f"{describe_slope}\n")
        slope[min_filt] = np.nan
        describe_slope = slope * inuse_df
        describe_filt = describe_slope == 0
        describe_slope[describe_filt] = np.nan
        describe_slope = describe_slope.describe()
        describe_slope.loc['count', 'all'] = describe_slope.loc['count', :].sum()
        describe_slope.loc['mean', 'all'] = (describe_slope.loc['mean', :] * describe_slope.loc['count',
                                                                                     :]).sum() / describe_slope.loc[
                                                    'count', 'all']
        describe_slope.loc['std', 'all'] = (describe_slope.loc['std', :] * describe_slope.loc['count',
                                                                                   :]).sum() / describe_slope.loc[
                                                   'count', 'all']
        describe_slope.loc['min', 'all'] = describe_slope.loc['min', :].min()
        describe_slope.loc['max', 'all'] = describe_slope.loc['max', :].max()
        describe_slope = describe_slope.loc[['count', 'mean', 'std', 'min', 'max'], :].T.round(3)
        describe_slope.columns.name = "Post-disable"
        print(f"{describe_slope}\n")

    print(f"  - {np.round((np.sum(np.sum(filt.values)) / (inuse_df.shape[0] * inuse_df.shape[1])) * 100, 2)} % of the data have been deactivated by this filter")

    end = time.time()
    print(f"  - Time used for minimum slope disabling: {end - start} sec.\n")


def cull_max_curvature(processing: pipeline.ProcessingData,
                       channel_and_gate: ChannelAndGate = {'channel': 1,
                                                           'gate': 7},
                       max_curvature: float = 10.,
                       disable_earlier_gates: int = 0,
                       disable_sounding_tails: bool = True,
                       verbose: bool = False,
                       save_filter_to_layer_data: HiddenBool = False):
    """
    Function to disable datapoints in a transient curve where the curvature between three datapoints
    is greater than the maximum_curvature specified.
    Curvature is calculated with Central Difference with Finite Differencing.
      curvature_1 = (x_2 - 2 * x_1 + x_0) / (y_2 - y_0)^2
      ...
      curvature_(n-1) = (x_n - 2 * x_(n-1) + x_(n-2)) / (y_n - y_(n-2))^2

      Where: x is log10(dBdt), y is log10(timegates), and n is the number of timegates
        note: there is no curvature for the first and last time gates

    Interpreting curvature values:
      Positive curvature: Indicates a decreasing decay rate. This suggests a transition to a more conductive layer.
      Negative curvature: Indicates an increasing decay rate. This suggests a transition to a more resistive layer.
      Curvature close to zero: Suggests a relatively constant decay rate, implying ~ homogeneous subsurface layer.

    Parameters
    ----------
    channel_and_gate :
        the channel to evaluate and the first gate to consider for disabling.
        All timegates after the first gate and above the max_curvature will be disabled
    max_curvature :
        Datapoints of the transient curve with curvature > max_curvature are disabled.
    disable_earlier_gates :
        The number of timegates earlier to disable when the filter finds a problem with the data
        This may deactivate gates earlier than the gate specified in channel_and_gate.
        Note: negative values are allowed and will disable the number of timegates after a problem is found
    disable_sounding_tails :
        If True, disable timegates after the first curvature that exceeds the max curvature is encountered.
    verbose :
        If True, more output about what the filter is doing
    """
    start = time.time()
    channel = channel_and_gate['channel']
    first_gate_to_consider = channel_and_gate['gate']

    print(f"  - Disabling curvatures greater than {max_curvature} on channel-{channel} after gate-{first_gate_to_consider}")
    data = processing.xyz

    str_channel = f"0{channel}"[-2:]
    data_key = f"{dat_key_prefix}{str_channel}"
    inuse_key = f"{inuse_key_prefix}{str_channel}"

    inuse_df = build_inuse_dataframe(data=data, channel=channel)

    curvature = calculate_transient_curvatures(processing, data_key)

    max_filt = curvature > max_curvature
    max_filt[max_filt.isna()] = False  # diff returns nans in the begining of the trace

    inuse_df[max_filt] = 0
    inuse_df.loc[:, :first_gate_to_consider + 1] = 1

    if disable_sounding_tails:
        cull_sounding_tails(inuse_df)

    if disable_earlier_gates != 0:
        cull_earlier_gates(inuse_df, disable_earlier_gates)

    filt = inuse_df == 0
    data.layer_data[inuse_key][filt] = 0

    if save_filter_to_layer_data:
        new_iu_key = f"disable_max_curvature_{inuse_key}"
        data.layer_data[new_iu_key] = inuse_df

    if not verbose:
        flat_curvature = curvature.loc[:, first_gate_to_consider:].to_numpy().flatten()
        print(f"  - Curvature statistics for all soundings after gate-{first_gate_to_consider}:")
        print(f"      pre_mean_curvature  = {np.round(np.nanmean(flat_curvature), 3)} ± {np.round(np.nanstd(flat_curvature), 3)}")
        curvature[max_filt] = np.nan
        zero_filt = curvature == 0
        iu_curvature = curvature.loc[:, first_gate_to_consider:] * inuse_df.loc[:, first_gate_to_consider:]
        iu_filt = iu_curvature == 0
        iu_curvature[iu_filt] = np.nan
        iu_curvature[zero_filt] = 0
        flat_curvature = iu_curvature.to_numpy().flatten()
        print(f"      post_mean_curvature = {np.round(np.nanmean(flat_curvature), 3)} ± {np.round(np.nanstd(flat_curvature), 3)}")
    else:
        describe_curvature = curvature.describe()
        describe_curvature.loc['count', 'all'] = describe_curvature.loc['count', :].sum()
        describe_curvature.loc['mean', 'all'] = (describe_curvature.loc['mean', :] * describe_curvature.loc['count', :]).sum() / describe_curvature.loc['count', 'all']
        describe_curvature.loc['std', 'all'] =  (describe_curvature.loc['std', :]  * describe_curvature.loc['count', :]).sum() / describe_curvature.loc['count', 'all']
        describe_curvature.loc['min', 'all'] = describe_curvature.loc['min', :].min()
        describe_curvature.loc['max', 'all'] = describe_curvature.loc['max', :].max()
        describe_curvature = describe_curvature.loc[['count', 'mean', 'std', 'min', 'max'], :].T.round(3)
        describe_curvature.columns.name = "Pre-disable"
        print(f"  - Curvature statistics for all gates:\n")
        print(f"{describe_curvature}\n")
        curvature[max_filt] = np.nan
        describe_curvature = curvature * inuse_df
        describe_filt = describe_curvature == 0
        describe_curvature[describe_filt] = np.nan
        describe_curvature = describe_curvature.describe()
        describe_curvature.loc['count', 'all'] = describe_curvature.loc['count', :].sum()
        describe_curvature.loc['mean', 'all'] = (describe_curvature.loc['mean', :] * describe_curvature.loc['count', :]).sum() / describe_curvature.loc['count', 'all']
        describe_curvature.loc['std', 'all'] =  (describe_curvature.loc['std', :]  * describe_curvature.loc['count', :]).sum() / describe_curvature.loc['count', 'all']
        describe_curvature.loc['min', 'all'] = describe_curvature.loc['min', :].min()
        describe_curvature.loc['max', 'all'] = describe_curvature.loc['max', :].max()
        describe_curvature = describe_curvature.loc[['count', 'mean', 'std', 'min', 'max'], :].T.round(3)
        describe_curvature.columns.name = "Post-disable"
        print(f"{describe_curvature}\n")

    print(f"  - {np.round((np.sum(np.sum(filt.values)) / (inuse_df.shape[0] * inuse_df.shape[1])) * 100, 2)} % of the data have been deactivated by this filter")

    end = time.time()
    print(f"  - Time used for maximum curvature disabling: {end - start} sec.\n")


def cull_min_curvature(processing: pipeline.ProcessingData,
                       channel_and_gate: ChannelAndGate = {'channel': 1,
                                                           'gate': 7},
                       min_curvature: float = -10.,
                       disable_earlier_gates: int = 0,
                       disable_sounding_tails: bool = True,
                       verbose=False,
                       save_filter_to_layer_data: HiddenBool = False):
    """
    Function to disable datapoints in a transient curve where the curvature between three datapoints
    is less than the minimum_curvature specified.
    Curvature is calculated with Central Difference with Finite Differencing.
      curvature_1 = (x_2 - 2 * x_1 + x_0) / (y_2 - y_0)^2
      ...
      curvature_(n-1) = (x_n - 2 * x_(n-1) + x_(n-2)) / (y_n - y_(n-2))^2

      Where: x is log10(dBdt), y is log10(timegates), and n is the number of timegates
        note: there is no curvature for the first and last time gates

    Interpreting curvature values:
      Positive curvature: Indicates a decreasing decay rate. This suggests a transition to a more conductive layer.
      Negative curvature: Indicates an increasing decay rate. This suggests a transition to a more resistive layer.
      Curvature close to zero: Suggests a relatively constant decay rate, implying ~ homogeneous subsurface layer.

    Parameters
    ----------
    channel_and_gate :
        the channel to evaluate and the first gate to consider for disabling.
        All timegates after the specified timegate and below the min_curvature will be disabled
    min_curvature :
        Datapoints of the transient curve with curvature < min_curvature are disabled.
    disable_earlier_gates :
        The number of timegates earlier to disable when the filter finds a problem with the data.
        This may deactivate gates earlier than the gate specified in channel_and_gate.
        Note: negative values are allowed and will disable the number of timegates after a problem is found
    disable_sounding_tails :
        If True, disable timegates after the first curvature that is less than the min curvature is encountered.
    verbose :
        If True, more output about what the filter is doing
    """
    start = time.time()
    channel = channel_and_gate['channel']
    first_gate_to_consider = channel_and_gate['gate']

    print(f"  - Disabling curvatures less than {min_curvature} on channel-{channel} after gate-{first_gate_to_consider}")
    data = processing.xyz

    str_channel = f"0{channel}"[-2:]
    data_key = f"{dat_key_prefix}{str_channel}"
    inuse_key = f"{inuse_key_prefix}{str_channel}"

    inuse_df = build_inuse_dataframe(data=data, channel=channel)

    curvature = calculate_transient_curvatures(processing, data_key)

    min_filt = curvature < min_curvature
    min_filt[min_filt.isna()] = False  # diff returns nans in the begining of the trace

    inuse_df[min_filt] = 0
    inuse_df.loc[:, :first_gate_to_consider + 1] = 1  # reset inuse flags earlier than first_gate_to_consider

    if disable_sounding_tails:
        cull_sounding_tails(inuse_df)

    if disable_earlier_gates != 0:
        cull_earlier_gates(inuse_df, disable_earlier_gates)

    filt = inuse_df == 0
    data.layer_data[inuse_key][filt] = 0

    if save_filter_to_layer_data:
        new_iu_key = f"disable_min_curvature_{inuse_key}"
        data.layer_data[new_iu_key] = inuse_df

    if not verbose:
        flat_curvature = curvature.loc[:, first_gate_to_consider:].to_numpy().flatten()
        print(f"  - Curvature statistics for all soundings after gate-{first_gate_to_consider}:")
        print(f"      pre_mean_curvature  = {np.round(np.nanmean(flat_curvature), 3)} ± {np.round(np.nanstd(flat_curvature), 3)}")
        curvature[min_filt] = np.nan
        zero_filt = curvature == 0
        iu_curvature = curvature.loc[:, first_gate_to_consider:] * inuse_df.loc[:, first_gate_to_consider:]
        iu_filt = iu_curvature == 0
        iu_curvature[iu_filt] = np.nan
        iu_curvature[zero_filt] = 0
        flat_curvature = iu_curvature.to_numpy().flatten()
        print(f"      post_mean_curvature = {np.round(np.nanmean(flat_curvature), 3)} ± {np.round(np.nanstd(flat_curvature), 3)}")
    else:
        describe_curvature = curvature.describe()
        describe_curvature.loc['count', 'all'] = describe_curvature.loc['count', :].sum()
        describe_curvature.loc['mean', 'all'] = (describe_curvature.loc['mean', :] * describe_curvature.loc['count', :]).sum() / describe_curvature.loc['count', 'all']
        describe_curvature.loc['std', 'all'] =  (describe_curvature.loc['std', :]  * describe_curvature.loc['count', :]).sum() / describe_curvature.loc['count', 'all']
        describe_curvature.loc['min', 'all'] = describe_curvature.loc['min', :].min()
        describe_curvature.loc['max', 'all'] = describe_curvature.loc['max', :].max()
        describe_curvature = describe_curvature.loc[['count', 'mean', 'std', 'min', 'max'], :].T.round(3)
        describe_curvature.columns.name = "Pre-disable"
        print(f"  - Curvature statistics for all gates:\n")
        print(f"{describe_curvature}\n")
        curvature[min_filt] = np.nan
        describe_curvature = curvature * inuse_df
        describe_filt = describe_curvature == 0
        describe_curvature[describe_filt] = np.nan
        describe_curvature = describe_curvature.describe()
        describe_curvature.loc['count', 'all'] = describe_curvature.loc['count', :].sum()
        describe_curvature.loc['mean', 'all'] = (describe_curvature.loc['mean', :] * describe_curvature.loc['count', :]).sum() / describe_curvature.loc['count', 'all']
        describe_curvature.loc['std', 'all'] =  (describe_curvature.loc['std', :]  * describe_curvature.loc['count', :]).sum() / describe_curvature.loc['count', 'all']
        describe_curvature.loc['min', 'all'] = describe_curvature.loc['min', :].min()
        describe_curvature.loc['max', 'all'] = describe_curvature.loc['max', :].max()
        describe_curvature = describe_curvature.loc[['count', 'mean', 'std', 'min', 'max'], :].T.round(3)
        describe_curvature.columns.name = "Post-disable"
        print(f"{describe_curvature}\n")

    print(f"  - {np.round((np.sum(np.sum(filt.values)) / (inuse_df.shape[0] * inuse_df.shape[1])) * 100, 2)} % of the data have been deactivated by this filter")

    end = time.time()
    print(f"  - Time used for minimum curvature disabling: {end - start} sec.\n")


def cull_below_noise_level(processing: pipeline.ProcessingData,
                           channel_and_gate: ChannelAndGate = {'channel': 1,
                                                               'gate': 7},
                           noise_level_1ms: float = 1e-8,
                           noise_exponent: float = -0.5,
                           disable_earlier_gates: int = 0,
                           disable_sounding_tails: bool = True,
                           save_filter_to_layer_data: HiddenBool = False):
    """
    This filter will disable Rx-only normalized data that falls below an assumed noise floor defined by
        't^(noise_exponent)' with an intercept at 'noise_level_1ms', where t is time.
        Generally, the accepted slope, or noise_exponent, for the noise floor is -(1/2).
    
    Parameters
    ----------
    channel_and_gate :
        the channel to evaluate and the first gate to consider for disabling.
        All timegates after the specified timegate and below the noise floor will be disabled
    noise_level_1ms :
        Ambient noise level for measured data (not normalized by the TX dipole) at 1e-3 s, in V/(m^2)
    noise_exponent :
        Exponent describing the decay of the ambient noise as a function of gate-time. 
    disable_earlier_gates :
        The number of timegates earlier to disable when the filter finds a problem with the data
        This may deactivate gates earlier than the gate specified in channel_and_gate.
        Note: negative values are allowed and will disable the number of timegates after a problem is found
    disable_sounding_tails :
        If True, disables the sounding tails (later timegates) after the first timegate's amplitude is less than the
        noise floor is encountered.
    """
    start = time.time()
    channel = channel_and_gate['channel']
    first_gate_to_consider = channel_and_gate['gate']

    print(f"  - Disabling based on predefined noise levels for channel-{channel}")
    data = processing.xyz
    str_channel = f"0{channel}"[-2:]
    inuse_key = f"{inuse_key_prefix}{str_channel}"
    data_key = f"{dat_key_prefix}{str_channel}"

    noise_df = utils.make_noise_df(processing,
                                   channel=channel,
                                   noise_level_1ms=float(noise_level_1ms),
                                   noise_exponent=float(noise_exponent),
                                   norm_by_tx=True)

    inuse_df = build_inuse_dataframe(data=data, channel=channel)
    filt = data.layer_data[data_key].iloc[:, first_gate_to_consider:] <= noise_df.iloc[:, first_gate_to_consider:]
    inuse_df.iloc[:, first_gate_to_consider:][filt] = 0

    if disable_sounding_tails:
        cull_sounding_tails(inuse_df)

    if disable_earlier_gates != 0:
        cull_earlier_gates(inuse_df, disable_earlier_gates)

    filt = inuse_df == 0
    processing.xyz.layer_data[inuse_key][filt] = 0

    if save_filter_to_layer_data:
        new_iu_key = f"disable_noise_{inuse_key}"
        processing.xyz.layer_data[new_iu_key] = inuse_df

    print(f"  - {np.round((np.sum(np.sum(filt.values)) / (inuse_df.shape[0] * inuse_df.shape[1])) * 100, 2)} % of the data have been deactivated by this filter")

    end = time.time()
    print(f"  - Time used for error based disabling: {end - start} sec.\n")

# def cull_below_noise_dict(processing,
#                           channel,
#                           error_dict,
#                           disable_earlier_gates=0,
#                           disable_sounding_tails=True):
#     data = processing.xyz
#     # str_channel = f"0{channel}"[-2:]
#     # inuse_key = f"{inuse_key_prefix}{str_channel}"
#
#     print(f"error_dict =\n{error_dict}")
#
#     for moment in error_dict.keys():
#         for column_key in error_dict[moment].keys():
#             print(f"column_key ={column_key}")
#             idx = data.layer_data[moment][column_key].abs() < error_dict[moment][column_key]
#             inuse_df = build_inuse_dataframe(data=data, channel=channel)
#             inuse_df.loc[idx, column_key] = 0
#             idx2 = idx & (data.flightlines['disable_reason'] == 'none')
#             data.flightlines.loc[idx2, 'disable_reason'] = 'noise_level'
#
#         if disable_sounding_tails:
#             cull_sounding_tails(inuse_df)
#
#         if disable_earlier_gates != 0:
#             cull_earlier_gates(inuse_df, disable_earlier_gates)
#
#         return inuse_df

#
# FIXME: this is old code, just keeping it for backward compatibility for now
# FIXME: only works for WB xyz export of initial .skb  import
# FIXME: rather use copy_culling_use_position (below) instead
def cullPrevious_on_position(data,
                             culled_data):
    start = time.time()
    print("  - Disabling based on previously disabled data")
    print("  - Using positions")
    if ('InUse_Ch01' in data.layer_data.keys()) and ('InUse_Ch01' in culled_data.layer_data.keys()):
        print('  - Disabling gates from Ch01')
        data_ch_filt = data.flightlines.channel_no == 1
        culled_ch_filt = culled_data.flightlines.channel_no == 1
        dist, closest_idx = cKDTree(culled_data.flightlines.loc[culled_ch_filt, ["UTMX", "UTMY"]].values).query(data.flightlines.loc[data_ch_filt, ["UTMX", "UTMY"]].values)
        has_no_pair = dist > estimateInlineSamplig(culled_data)*1.2  # *1.2 to cover variances in inline sampling
        data.layer_data['InUse_Ch01'].loc[data_ch_filt] = culled_data.layer_data['InUse_Ch01'].loc[culled_ch_filt].iloc[closest_idx].set_index(data.layer_data['InUse_Ch01'].loc[data_ch_filt].index)
        data.layer_data['InUse_Ch01'].loc[data_ch_filt].iloc[has_no_pair, :] = 0
    if ('InUse_Ch02' in data.layer_data.keys()) and ('InUse_Ch02' in culled_data.layer_data.keys()):
        print('  - Disabling gates from Ch02')
        data_ch_filt = data.flightlines.channel_no == 2
        culled_ch_filt = culled_data.flightlines.channel_no == 2
        dist, closest_idx = cKDTree(culled_data.flightlines.loc[culled_ch_filt, ["UTMX", "UTMY"]].values).query(data.flightlines.loc[data_ch_filt, ["UTMX", "UTMY"]].values)
        has_no_pair = dist > estimateInlineSamplig(culled_data)*1.2  # *1.2 to cover variances in inline sampling
        data.layer_data['InUse_Ch02'].loc[data_ch_filt] = culled_data.layer_data['InUse_Ch02'].loc[culled_ch_filt].iloc[closest_idx].set_index(data.layer_data['InUse_Ch02'].loc[data_ch_filt].index)
        data.layer_data['InUse_Ch02'].loc[data_ch_filt].iloc[has_no_pair, :] = 0
    end = time.time()
    print(f"  - Time used for Disabling: {end - start} sec.\n")


def copy_culling_use_position(data,
                              culled_data):
    start = time.time()
    print("  - Copy previous disablings to new dataset")
    print("  - Using positions")
    for moment in allowed_moments:
        channel = int(moment.split('_Ch')[-1])
        inuse_key = inuse_moment(moment)
        if (inuse_key in data['layer_data'].keys()) and (inuse_key in culled_data['layer_data'].keys()):
            assert data['layer_data'][inuse_key].shape[1] == culled_data['layer_data'][inuse_key].shape[1], 'Original disabling has different number of gates'
            culled_ch_filt = culled_data['flightlines'].channel_no == channel
            # if 'channel_no' in data['flightlines'].columns:
            #     data_ch_filt = data['flightlines'].channel_no == channel
            # else:
            if 'channel_no' not in data['flightlines'].columns:
                data_ch_filt = pd.Series(np.ones(len(data['flightlines'])), dtype=bool)
                dist, closest_idx = cKDTree(culled_data['flightlines'].loc[culled_ch_filt, ["UTMX", "UTMY"]].values).query(data['flightlines'].loc[data_ch_filt, ["UTMX", "UTMY"]].values)
                has_no_pair = dist > estimateInlineSamplig(culled_data)*1.2  # *1.2 to cover variances in inline sampling
                data['layer_data'][inuse_key].loc[data_ch_filt] = culled_data['layer_data'][inuse_key].loc[culled_ch_filt].iloc[closest_idx].set_index(data['layer_data'][inuse_key].loc[data_ch_filt].index)
                data['layer_data'][inuse_key].loc[data_ch_filt].iloc[has_no_pair, :] = 0
    end = time.time()
    print(f"  - Time used for disabling: {end - start} sec.\n")


# FIXME: This function should probably live in corrections.py since no culling/disablings happen here
def assign_InvError_to_Data(inv_data,
                            processing,
                            verbose=False):
    data = processing.xyz
    # assign Err in inverted dataset to data, handle gate by gate
    print('  - Assigning inv-error to data.')
    for moment in allowed_moments:
        err_key = errKey(moment)
        data.layer_data[err_key] = data.layer_data[moment]*np.nan

        # only account for data rows from the respective moment
        filt = data.layer_data[moment].isna().sum(axis=1) < data.layer_data[moment].shape[1]
        moment_data = filtXYZ(data, filt, reset_index=False)
        moment_data.layer_data[err_key] = moment_data.layer_data[moment]*np.nan

        # only account for inv-data rows from the respective moment
        filt = inv_data.xyz.layer_data[moment].isna().sum(axis=1) < inv_data.xyz.layer_data[moment].shape[1]
        moment_inv_data = filtXYZ(inv_data.xyz, filt, reset_index=True)

        # find closest pairs
        dist, closest_idx = cKDTree(moment_inv_data.flightlines[["utmx", "utmy"]].values).query(moment_data.flightlines[["UTMX", "UTMY"]].values)
        has_no_pair = dist > estimateInlineSamplig(moment_data)*3.1  # *3.1 just to be safe

        # find matching gate times in data and inv-data
        for t in processing.GateTimes[moment]:
            if np.around(np.log10(t), decimals=3) in np.around(np.log10(inv_data.GateTimes[moment]), decimals=3):
                # print('Gate time: {}  found in both files'.format(t))
                col_inv = np.where(np.around(np.log10(t), decimals=3) == np.around(np.log10(inv_data.GateTimes[moment]), decimals=3))[0][0]
                col_data = np.where(t == processing.GateTimes[moment])[0][0]
                if verbose:
                    print(f'\n  - col_data: {col_data} ; col_inv: {col_inv}\n')
                moment_data.layer_data[err_key].loc[:, col_data] = moment_inv_data.layer_data[err_key].iloc[closest_idx, col_inv].values
            else:
                if verbose:
                    print(f'\n  - Gate time: {t} not found in both both files\n')

        # make sure we don't extrapolate too much
        moment_data.layer_data[err_key].iloc[has_no_pair, :] = np.nan

        # assign the error data back to the big dataframe
        data.layer_data[err_key].loc[moment_data.layer_data[err_key].index, :] = moment_data.layer_data[err_key]


def cull_data(data):
    for moment in allowed_moments:
        if moment in data.layer_data.keys():
            culled_key = 'Gate_culled_'+moment.split('_')[-1]
            # FIXME: Figure this out without calling 'moment'
            inuse_key = inuse_moment(moment)
            data.layer_data[culled_key] = data.layer_data[moment].copy()
            idx = (data.layer_data[inuse_key] == 0)
            data.layer_data[culled_key][idx] = np.nan
            
            scaled_key = f"Gate_scaled_{moment.split('_')[-1]}"
            
            if scaled_key in data.layer_data.keys():
                scaled_culled_key = 'Gate_scaled_culled_'+moment.split('_')[-1]
                data.layer_data[scaled_culled_key] = data.layer_data[scaled_key].copy()
                data.layer_data[scaled_culled_key][idx] = np.nan
            else:
                print('  - No scaled data found in data.layer_data')
            
            std_key = f"STD_{moment.split('_')[-1]}"
            if std_key in data.layer_data.keys():
                std_culled_key = 'STD_culled_'+moment.split('_')[-1]
                data.layer_data[std_culled_key] = data.layer_data[std_key].copy()
                data.layer_data[std_culled_key][idx] = np.nan


def enable_disable_time_gate(processing: pipeline.ProcessingData,
                             channel_and_gate: ChannelAndGate = {'channel': 1,
                                                                 'gate': 0},
                             action: typing.Literal['enable', 'disable'] = 'disable',
                             save_filter_to_layer_data: HiddenBool = False):
    """
    Function to enable or disable a timegate from the data.
    Note: This will overwrite any previous edits.

    Parameters
    ----------
    channel_and_gate :
        Dictionary defining which channel and gate to enable or disable
    action :
        Key to denote whether to 'enable' or 'disable' the timegate
    """
    start = time.time()
    if action == 'enable':
        action_print_key = 'Enabling'
    elif action == 'disable':
        action_print_key = 'Disabling'

    channel = channel_and_gate['channel']
    gate_index = channel_and_gate['gate']

    print(f'  - {action_print_key} gate-{gate_index} from channel-{channel}')
    data = processing.xyz

    str_channel = f"0{channel}"[-2:]
    inuse_key = f"{inuse_key_prefix}{str_channel}"

    inuse_df = build_inuse_dataframe(data=data, channel=channel)

    assert gate_index < len(inuse_df.columns), "gate index is invalid, please provide an index that is within the range for the channel specified"

    inuse_df.loc[:, gate_index] = 0
    filt = inuse_df == 0
    if action == "disable":
        action_value = 0
    elif action == "enable":
        action_value = 1

    data.layer_data[inuse_key][filt] = action_value

    print(f"  - {np.round((np.sum(np.sum(filt.values)) / (inuse_df.shape[0] * inuse_df.shape[1])) * 100, 2)} % of the data have been {action}d by this filter")

    if save_filter_to_layer_data:
        if action == 'enable':
            inuse_df = np.abs(inuse_df - 1)
        new_iu_key = f"{action}_gate{gate_index}_{inuse_key}"
        data.layer_data[new_iu_key] = inuse_df

    end = time.time()
    print(f"  - Time used to {action} timegate-{gate_index} from Channel{channel}: {end - start} sec.\n")


def apply_gex(processing: pipeline.ProcessingData,
              save_filter_to_layer_data: HiddenBool = False):
    """
    Function to disable early-time timegates, as specified in the gex (geometry file)
    """
    start = time.time()
    print('  - Applying GEX to the data')
    gex = processing.gex
    data = processing.xyz

    layer_data_keys = data.layer_data.keys()

    channels = []
    for ds in layer_data_keys:
        if dat_key_prefix in ds:
            channels.append(int(ds.split(dat_key_prefix)[1]))

    for ch in channels:
        RemoveInitialGates = int(gex.gex_dict[f"Channel{ch}"]['RemoveInitialGates'])

        str_channel = f"0{ch}"[-2:]
        inuse_key = f"{inuse_key_prefix}{str_channel}"

        inuse_df = build_inuse_dataframe(data, channel=ch)
        inuse_df.iloc[:, 0:RemoveInitialGates] = 0

        filt = inuse_df == 0
        data.layer_data[inuse_key][filt] = 0

        if save_filter_to_layer_data:
            new_iu_key = f"apply_gex_{inuse_key}"
            data.layer_data[new_iu_key] = inuse_df

        print(f"  - Gates {[x for x in range(0, RemoveInitialGates)]} from 'Ch{str_channel}' have been disabled")

    end = time.time()
    print(f"  - Time used for applying the gex to the in-use flags: {end - start} sec.\n")
