import numpy as np
import pandas as pd
import copy
import time
import typing
# import pydantic
# import geopandas as gpd
# import libaarhusxyz

from .. import pipeline
from . import utils
from . import sps

from .data_keys import pos_keys
from .data_keys import dat_key_prefix, std_key_prefix, inuse_key_prefix, err_key_prefix

from .parameter_types import FlightLineColumnName, LayerDataName
from .parameter_types import ChannelAndGateRange, Channel, ChannelAndGate
from .parameter_types import RasterUrl
from .parameter_types import FlightlinesList
from .parameter_types import MovingAverageFilterDict
from .parameter_types import FlightType, AvailableFlightTypeList
# from .parameter_types import HiddenString, HiddenBool
from .parameter_types import WeightedErrorCalcString, UnweightedErrorCalcString

def correct_altitude_and_topo(processing: pipeline.ProcessingData,
                              terrain_model: RasterUrl,
                              verbose: bool = True):
    """
    Correct altitude and topography

    Parameters
    ----------
    terrain_model :
        Path to digital-terrain-model (DTM) GeoTIFF, in meters-above-mean-sea-level (m.a.m.s.l.)
    verbose :
        If True, more output about what the filter is doing
   """
    start = time.time()
    print('  - Correcting altitude and topo')
    data = processing.xyz
    crs = processing.crs

    DEM_filename = terrain_model

    # topo_key = 'Topography'
    topo_key = processing.xyz.z_column
    orig_topo_key = f'orig_{topo_key}'
    dem_topo_key = f'dem_{topo_key}'
    diff_topo_key = f'o_d_diff_{topo_key}'

    # alt_key = 'TxAltitude'
    alt_key = processing.xyz.alt_column
    orig_alt_key = f'orig_{alt_key}'
    dem_alt_key = f'dem_{alt_key}'
    diff_alt_key = f'o_d_diff_{alt_key}'

    # FIXME: how to use the sps files for this here?
    #   need to build an importer for a zip file of sps files - I think %BRB_20240621
    sps_file_filter = None
    
    if sps_file_filter and crs:
        print('  - Recalculating Topo and Tx altitude based on SPS and DEM.')
        gps_data = sps.read_concat_DGPS_sps_files(sps_file_filter, crs)
        data.flightlines['datetime'] = pd.to_datetime(data.flightlines.Date.str.replace('/', '-') + ' ' + data.flightlines.Time)
        data.flightlines['epoch_time'] = pd.to_numeric(data.flightlines['datetime'])/1e6
        sps.interpolate_df2_on_df1(data.flightlines, 
                                   gps_data,
                                   common_key='epoch_time', 
                                   key2interpolate_df2='z_nmea', 
                                   key4interpolated_df1='gps_z')
        Tx_z = data.flightlines.gps_z
        data.flightlines.drop(['datetime', 'epoch_time'], axis=1, inplace=True)
    else:
        print('  - Recalculating Topo and Tx altitude based on the DEM only.')
        if 'TXZ' in data.flightlines.columns:
            print('  - Found Tx source elevation (TXZ) in data.')
            Tx_z = data.flightlines.TXZ
        elif 'TxZ' in data.flightlines.columns:
            print('  - Found Tx source elevation (TxZ) in data.')
            Tx_z = data.flightlines.TxZ
        elif 'Txz' in data.flightlines.columns:
            print('  - Found Tx source elevation (Txz) in data.')
            Tx_z = data.flightlines.Txz
        elif 'txz' in data.flightlines.columns:
            print('  - Found Tx source elevation (txz) in data.')
            Tx_z = data.flightlines.txz
        else:
            print(f"  - Did not find source elevation in data will calculate it from '{topo_key}' + '{alt_key}'")
            Tx_z = data.flightlines[topo_key] + data.flightlines[alt_key]

    data.flightlines[orig_topo_key] = data.flightlines[topo_key]
    data.flightlines[orig_alt_key] = data.flightlines[alt_key]

    utils.sampleDEM(DEM_filename,
                    df=data.flightlines,
                    poskeys=pos_keys,
                    z_key=dem_topo_key,
                    crs=crs)

    data.flightlines[topo_key] = data.flightlines[dem_topo_key]
    data.flightlines[diff_topo_key] = data.flightlines[orig_topo_key] - data.flightlines[dem_topo_key]
    data.flightlines[dem_alt_key] = Tx_z - data.flightlines[topo_key]
    data.flightlines[alt_key] = data.flightlines[dem_alt_key]
    data.flightlines[diff_alt_key] = data.flightlines[orig_alt_key] - data.flightlines[dem_alt_key]

    # FIXME: This is wrong. TxAltitude is height above topo, alt is above sea level!!!
    #   for backward compatibility
    #   data.flightlines['Alt']=data.flightlines.TxAltitude

    # FIXME: These Exception statements may not work right if the default of the sample DEM for mismatches is 0 and not NaN
    if data.flightlines[topo_key].isna().sum() > 0:
        raise Exception('¡¡¡¡ Something went wrong, computed topo has NaNs (DEM coverage?) !!!!')
    if data.flightlines[alt_key].isna().sum() > 0:
        raise Exception('¡¡¡¡ Something went wrong, computed altitude has NaNs (DEM and/or SPS coverage?) !!!!')

    fl_topo_stats = data.flightlines.loc[:, [topo_key, orig_topo_key, dem_topo_key, diff_topo_key]].describe()
    fl_topo_stats.columns.name = "Topo. stats"
    fl_alti_stats = data.flightlines.loc[:, [alt_key, orig_alt_key, dem_alt_key, diff_alt_key]].describe()
    fl_alti_stats.columns.name = "Alt. Stats "
    if verbose:
        print('')
        print(fl_topo_stats)
        print('')
        print(fl_alti_stats)
        print('')
    else:
        print(f"  - Topo. mean difference: {fl_topo_stats.loc['mean', diff_topo_key].round(2)} ± {fl_topo_stats.loc['std', diff_topo_key].round(2)} m")
        print(f"  - Alt. mean difference:  {fl_alti_stats.loc['mean', diff_alt_key].round(2)} ± {fl_alti_stats.loc['std', diff_alt_key].round(2)} m")

    end = time.time()
    print(f"  - Time used correcting the altitude and topography data: {end - start} sec.\n")


def select_lines(processing: pipeline.ProcessingData,
                 lines: FlightlinesList):
    """
    Function to select flightlines from the dataset to process.
    
    Parameters
    ----------
    lines :
        list of flightlines you would like to keep.
    """
    start = time.time()
    print('  - Reducing the dataset to the selected line numbers')

    filt = pd.Series(np.zeros(len(processing.xyz.flightlines)), dtype=bool)
    for line in lines:
        filt = filt | (processing.xyz.flightlines.Line == line)

    for key in processing.xyz.layer_data.keys():
        if key.startswith('InUse_'):
            processing.xyz.layer_data[key].loc[~filt, :] = 0
            
    # utils.drop_filt_XYZ(processing.xyz, ~filt)
    end = time.time()
    print(f"  - Time used reducing the dataset {end - start} sec.")


def moving_average_filter(processing: pipeline.ProcessingData,
                          filter_dict: MovingAverageFilterDict = {'Gate_Ch01': {'first_gate': 3,
                                                                                 'last_gate': 5},
                                                                  'Gate_Ch02': {'first_gate': 5,
                                                                                 'last_gate': 9}},
                          weighting_factor: float = 3,
                          error_calc_scheme: WeightedErrorCalcString = 'Weighted_SEM',
                          verbose: bool = False):
    """
    Moving average filter, averaging Gate values from neighboring soundings.
      If both data and error estimates exist than the data will be averaged using a rolling weighted averaging scheme,
        where the weights are the inverse of the square absolute error (1 / (ab_error^2)).
      If only data exists then the output will be a rolling average and error estimates will be from the
        unweighted standard error of the mean from the same rolling window.
    Results will always include error estimates in fractional percent (0.1 = 10%)

    Parameters
    ----------
    filter_dict :
        Dictionary describing the filter widths for the first and the last gate 
        of each moment/channel . The default is {'Gate_Ch01':[3, 5], 'Gate_Ch02':[5, 9]}.
    weighting_factor :
        Factor to adjust the weighting scheme, which is calculated like this:
            weights = 1 / (weighting_factor * ((data * fractional_err)**2))
        A factor of 1 would use only the inverse of the square of the absolute error.
        We find that 2 to 3 is a good value.
    error_calc_scheme :
        Method to calculate errors. Methods include:
            'Weighted_SEM' : Recommended. Weighted Standard Error of the Mean.
                error_calc = sqrt(weights / weights^2)
            'Balanced_1' :
                error_calc = (Weighted_SEM * Unweighted_SEM * Average * STD)^(1/4)
            'Average' :
                error_calc = average(error)
            'Balanced_2' : Not Recommended.
                error_calc = sqrt((Weighted_SEM^2)/4 + (Unweighted_SEM^2)/4 + (Average^2)/4 + (STD^2)/4)
            'STD' : Not Recommended.
                error_calc = std(data)
            'Unweighted_SEM' : Not Recommended. Unweighted Standard Error of the Mean
                error_calc = std(data) / sqrt(number_elements(data))
    verbose :
        If True, more output about what the filter is doing
    """
    
    start = time.time()
    print('  - Running a moving average filter  (line by line)')
    data = processing.xyz

    filter_list_dict = {}
    for moment in filter_dict.keys():
        filter_list_dict[moment] = [filter_dict[moment]['first_gate'],
                                    filter_dict[moment]['last_gate']]
    
    lines = utils.splitData_lines(data, line_key='Line')
    for line in lines.keys():
        if verbose: 
            print('Filtering line: {}'.format(line))
        movingAverageFilterLine(lines[line], filter_list_dict,
                                weighting_factor=weighting_factor,
                                error_calc_scheme=error_calc_scheme,
                                verbose=verbose)
    processing.xyz = utils.merge_lines(lines)
    end = time.time()
    print(f"  - Time used for moving average filter: {end - start} sec.")


def movingAverageFilterLine(lineData,
                            filter_dict,
                            weighting_factor=3,
                            error_calc_scheme='Weighted_SEM',
                            verbose=False):
    layer_data_keys = lineData.layer_data.keys()
    if sum([(std_key_prefix in key) for key in layer_data_keys]) > 0:
        print(f"  - Error estimates have been found!")
        for key in layer_data_keys:
            channels_number_str = []
            if 'Gate' in key:
                channels_number_str.append(key.split('_Ch')[-1])
        channels_number_str = sorted(channels_number_str)
        for channel_number_str in channels_number_str:
            dat_key = f"{dat_key_prefix}{channel_number_str}"
            std_key = f"{std_key_prefix}{channel_number_str}"
            inuse_key = f"{inuse_key_prefix}{channel_number_str}"

            if 'ChannelsNumber' in lineData.flightlines.columns:
                filt = lineData.flightlines.ChannelsNumber.astype(int) == int(channel_number_str)
            else:
                filt = lineData.flightlines.index

            if verbose: print(f'filtering: {dat_key} and {std_key} with filters {filter_dict[dat_key]}')

            if type(filter_dict[dat_key]) == int:
                # just one number -> box type filter
                rolling_lengths = utils.interpolate_rolling_size_for_all_gates([filter_dict[dat_key], filter_dict[dat_key]],
                                                                               lineData.layer_data[key])
            elif len(filter_dict[dat_key]) > 1:
                # trapeze type filter
                rolling_lengths = utils.interpolate_rolling_size_for_all_gates(filter_dict[dat_key],
                                                                               lineData.layer_data[key])
            else:
                raise Exception('filter lengths must be defined as:\n' +
                                    '    integer (box filter), or \n' +
                                    '    list, [width_at_first_gate, width_at_last_gate] (trapeze filter)')

            dBdt_df = copy.deepcopy(lineData.layer_data[dat_key].loc[filt, :])
            inuse_df = lineData.layer_data[inuse_key].loc[filt, :]
            std_df = copy.deepcopy(lineData.layer_data[std_key].loc[filt, :])

            dBdt_df[inuse_df == 0] = np.nan
            std_df[inuse_df == 0] = np.nan

            lineData.layer_data[dat_key].loc[filt, :], lineData.layer_data[std_key].loc[filt, :] = utils.rolling_weighted_mean_df(dBdt_df,
                                                                                                                                  std_df,
                                                                                                                                  rolling_lengths,
                                                                                                                                  weighting_factor=weighting_factor,
                                                                                                                                  error_calc_scheme=error_calc_scheme)
            lineData.layer_data[dat_key][lineData.layer_data[inuse_key] == 0] = np.nan
            lineData.layer_data[std_key][lineData.layer_data[inuse_key] == 0] = np.nan

    else:
        for key in layer_data_keys:
            if 'Gate' in key:
                print("  - Found the data but no error estimates. will calculate errors from the unweighted SEM")
                channel_number_str = key.split('_Ch')[-1]
                dat_key = dat_key_prefix + channel_number_str
                std_key = std_key_prefix + channel_number_str
                if 'ChannelsNumber' in lineData.flightlines.columns:
                    filt = lineData.flightlines.ChannelsNumber.astype(int) == int(channel_number_str)
                else:
                    filt = lineData.flightlines.index

                if verbose: print(f'filtering: {dat_key} with filters {filter_dict[dat_key]}')

                if type(filter_dict[dat_key]) == int:
                    # just one number -> box type filter
                    rolling_lengths = utils.interpolate_rolling_size_for_all_gates([filter_dict[dat_key], filter_dict[dat_key]], lineData.layer_data[dat_key])
                elif len(filter_dict[dat_key]) > 1:
                    # trapeze type filter
                    rolling_lengths = utils.interpolate_rolling_size_for_all_gates(filter_dict[dat_key], lineData.layer_data[dat_key])
                else:
                    raise Exception('filter lengths must be defined as:\n' +
                                    '    integer (box filter), or \n' +
                                    '    list, [width_at_first_gate, width_at_last_gate] (trapeze filter)')

            dBdt_df = copy.deepcopy(lineData.layer_data[dat_key].loc[filt, :])
            inuse_df = lineData.layer_data[utils.inuse_moment(dat_key)].loc[filt, :]
            dBdt_df[inuse_df == 0] = np.nan
            lineData.layer_data[dat_key].loc[filt, :], lineData.layer_data[std_key].loc[filt, :]  = utils.rolling_mean_df(dBdt_df, rolling_lengths, error_calc_scheme='Unweighted_SEM')

            lineData.layer_data[dat_key][lineData.layer_data[utils.inuse_moment(dat_key)] == 0] = np.nan
            lineData.layer_data[std_key][lineData.layer_data[utils.inuse_moment(dat_key)] == 0] = np.nan

def correct_data_tilt_for1D(processing: pipeline.ProcessingData,
                            verbose: bool = True):
    """
    Function to correct the data amplitudes according to the horizontal source dipole moment.
        Scaling by (cos(TxRoll) * cos(TxPitch))^2

    Parameters
    ----------
    verbose :
        If True, more output about what the filter is doing
    """
    start = time.time()
    print('  - 1D Data and Tilt correction of the data')
    data = processing.xyz

    roll_key = "TxRoll"
    pitch_key = "TxPitch"

    if (f'{roll_key}_orig' in data.flightlines) or (f'{pitch_key}_orig' in data.flightlines):
        print(f"  - Tilt correction has already been applied. '{roll_key}_orig' and/or '{pitch_key}_orig' in data")
        print(f"    - No tilt corrections will be performed.")
    else:
        rpdata = data.flightlines.loc[:, [roll_key, pitch_key]].describe()
        rpdata.columns.name = 'Pre-Correction'
        if verbose:
            print("")
            print(rpdata)
        else:
            print(f"  - Pre-Correction statistics:")
            print(f"    - {roll_key}:  {rpdata.loc['mean', roll_key].round(2)} ± {rpdata.loc['std', roll_key].round(2)} °")
            print(f"    - {pitch_key}: {rpdata.loc['mean', pitch_key].round(2)} ± {rpdata.loc['std', pitch_key].round(2)} °")

        cos_roll = np.cos(data.flightlines[roll_key]/180*np.pi)
        cos_pitch = np.cos(data.flightlines[pitch_key]/180*np.pi)
        for key in data.layer_data.keys():
            if 'Gate' in key:
                for col in data.layer_data[key].columns:
                    data.layer_data[key][col] = data.layer_data[key][col] / (cos_roll * cos_pitch)**2
        data.flightlines.insert(len(data.flightlines.columns), f'{pitch_key}_orig', data.flightlines.loc[:, pitch_key])
        data.flightlines.insert(len(data.flightlines.columns), f'{roll_key}_orig', data.flightlines.loc[:, roll_key])
        data.flightlines.loc[:, pitch_key] = 0.0
        data.flightlines.loc[:, roll_key] = 0.0

        rpdata = data.flightlines.loc[:, [roll_key, pitch_key]].describe()
        rpdata.columns.name = 'Post-Correction'
        if verbose:
            print("")
            print(rpdata)
            print("")
        else:
            print(f"  - Post-Correction statistics:")
            print(f"    - {roll_key}:  {rpdata.loc['mean', roll_key].round(2)} ± {rpdata.loc['std', roll_key].round(2)} °")
            print(f"    - {pitch_key}: {rpdata.loc['mean', pitch_key].round(2)} ± {rpdata.loc['std', pitch_key].round(2)} °")

    end = time.time()
    print(f"  - Time used for 1D correction of the data: {end - start} sec.\n")


def add_replace_gex_std_error(processing: pipeline.ProcessingData,
                              channel: Channel = 1):
    """
    Function to Replace STD error estimates with the estimates from the gex file.
        Note: often the errors reported alongside the data are measured from the raw transient
        stack and are individually reported for each sounding's timegates. Consult the data
        report for more information on how the STD data are created.
        Using this function will overwrite the STD estimates with a single value that comes from the gex file.

    Parameters
    ----------
    channel :
        Which channel to add or replace the std estimates from the gex file
    """
    start = time.time()
    print('  - Adding gex-based STD error to data')
    data = processing.xyz
    gex  = processing.gex
    gex_std = gex.gex_dict[f'Channel{channel}']['UniformDataSTD']

    str_channel = f"0{channel}"[-2:]
    data_key = f"{dat_key_prefix}{str_channel}"
    std_key = f"{std_key_prefix}{str_channel}"

    std_df = pd.DataFrame(np.ones(data.layer_data[data_key].shape),
                          dtype=float,
                          columns=data.layer_data[data_key].columns,
                          index=data.layer_data[data_key].index)
    std_df = std_df * gex_std

    data.layer_data[std_key] = std_df

    end = time.time()
    print(f"  - Time used to add gex-based STD errors to data: {end - start} sec.\n")


def add_std_error(processing: pipeline.ProcessingData,
                  channel_and_gates: ChannelAndGateRange = {'channel': 1,
                                                            'start_gate': 0,
                                                            'end_gate': 0},
                  error_fraction: float = 0.01):
    """
    Function to add fractional std error to all timegates.
        Ex: error_fraction = 0.02 = + 2%. Adds error to the gate range specified
        Ex: error_fraction = -0.01 = - 1%; Subtracts error to the gate range specified

    Parameters
    ----------
    channel_and_gates :
        The channel and gate range to add STD error to the data.
    error_fraction :
        Fraction of data to use as std error
    """
    start = time.time()
    print('  - Adding STD error to data')
    data = processing.xyz

    channel = channel_and_gates['channel']
    start_gate = channel_and_gates['start_gate']
    start_gate_index = start_gate
    end_gate = channel_and_gates['end_gate']

    str_channel = f"0{channel}"[-2:]

    existing_layer_data_keys = data.layer_data.keys()
    for key in existing_layer_data_keys:
        if std_key_prefix in key:
            if str(channel) in key:
                ld_std_key = key
        if dat_key_prefix in key:
            if str(channel) in key:
                ld_dat_key = key
                if "ld_std_key" not in locals():
                    ld_std_key = f"{std_key_prefix}{ld_dat_key.split(dat_key_prefix)[1]}"
                    print(ld_std_key)

    if end_gate is None or end_gate > len(data.layer_data[ld_dat_key].columns):
        end_gate = len(data.layer_data[ld_dat_key].columns) - 1  # Set to the length of the timegates for channel
    elif end_gate <= 0:
        end_gate = len(data.layer_data[ld_dat_key].columns) + end_gate - 1
    else:
        end_gate = end_gate

    assert start_gate < len(data.layer_data[ld_dat_key].columns-1), f"No STD vales Adjusted due to the start_gate being greater than then number of timegates available."
    assert end_gate > start_gate, f"Ending gate must come after starting gate"

    end_gate_index = end_gate + 1

    try:
        std_data = data.layer_data[ld_std_key]
    except:
        temp = np.zeros(data.layer_data[ld_dat_key].shape, )
        temp[data.layer_data[ld_dat_key].isna()] = np.nan
        temp = pd.DataFrame(temp,
                            index=  data.layer_data[ld_dat_key].index,
                            columns=data.layer_data[ld_dat_key].columns)
        std_data = temp

    cols = std_data.columns[start_gate_index: end_gate_index]

    std_data.loc[:, cols] = std_data.loc[:, cols] + error_fraction

    data.layer_data[ld_std_key] = std_data
    end = time.time()
    print(f"  - Time used to add STD errors to data: {end - start} sec.\n")


def classify_flightlines(processing: pipeline.ProcessingData,
                         lines: FlightlinesList,
                         flight_type: FlightType):
    """
    Function to apply classifications to the flightlines in the dataset. By default, all flightlines are considered
        'Production' until otherwise set.

    Parameters
    ----------
    lines :
        List of line numbers you like to assign a flight type too.
    flight_type :
        The type of flightline. Can be a custom key
    """
    start = time.time()
    print('  - Classifying flightlines')

    flightlines = processing.xyz.flightlines
    if 'flight_type' not in flightlines.columns:
        flightlines['flight_type'] = 'Production'

    filt = pd.Series(np.zeros(len(flightlines)), dtype=bool)
    for line in lines:
        filt = filt | (flightlines.Line == line)
    flightlines.loc[filt, 'flight_type'] = flight_type

    if True:
        u_fl_type = sorted(flightlines.flight_type.unique())
        print(f"  - Flight types in dataset: {u_fl_type}")

    end = time.time()
    print(f"  - Time used to classify flightlines: {end - start} sec.\n")


def select_flight_types(processing: pipeline.ProcessingData,
                        flight_types: AvailableFlightTypeList):
    """
    Function to reduce the dataset to the given list of flight-types. By default, all flightlines are considered
        'Production' until otherwise set.
        Hint: The processing filter "Classify flightlines" allows you to set the flight-types

    Parameters
    ----------
    flight_types :
        list of line numbers you like to keep in the dataset.
    """
    start = time.time()
    print('  - Reducing the dataset to the selected flight types')
    if 'flight_type' not in processing.xyz.flightlines.columns:
        processing.xyz.flightlines['flight_type'] = 'Production'

    filt = pd.Series(np.zeros(len(processing.xyz.flightlines)), dtype=bool)
    for f_type in flight_types:
        filt = filt | (processing.xyz.flightlines.flight_type == f_type)

    for key in processing.xyz.layer_data.keys():
        if key.startswith('InUse_'):
            processing.xyz.layer_data[key].loc[~filt, :] = 0

    # utils.drop_filt_XYZ(processing.xyz, ~filt)
    end = time.time()
    print(f"  - Time used to reduce the dataset: {end - start} sec.\n")


def copy_column(processing: pipeline.ProcessingData,
                orig: FlightLineColumnName,
                new: str):
    """
    Copy a column in the dataset (data.flightlines) to a new column in the dataset.
        These are generally attributes in the dataset that are a single value for the sounding, like:
            'tx_altitude', 'utm_x', 'utm_y', etc.
        Useful to work around data import and source issues.
        Warning: This will overwrite data if the new column exists

    Parameters
    ----------
    orig :
        The column to be copied
    new :
        The new name of the copied column
    """
    start = time.time()
    print('  - copying a column in the dataset')
    print(f"\t{orig} ––> {new}")

    if orig not in processing.xyz.flightlines.columns:
        raise ValueError(
            "Unknown orig column name '%s' not in [%s]" % (orig, ", ".join(processing.xyz.flightlines.columns)))
    processing.xyz.flightlines[new] = processing.xyz.flightlines[orig]

    end = time.time()
    print(f"  - Time used to copy a column in the dataset: {end - start} sec.\n")


def rename_column(processing: pipeline.ProcessingData,
                  orig: FlightLineColumnName,
                  new: str):
    """
    Rename a column in the dataset (data.flightlines).
        These are generally attributes in the dataset that are a single value for the sounding, like:
            'tx_altitude', 'utm_x', 'utm_y', etc.
        Useful to work around data import and source issues.
        Warning: This will overwrite data if the new column exists

    Parameters
    ----------
    orig :
        The column to be renamed
    new :
        The new name of the column
    """
    start = time.time()
    print('  - Renaming a column in the dataset')
    print(f"\t{orig} ––> {new}")

    if orig not in processing.xyz.flightlines.columns:
        raise ValueError(
            "Unknown orig column name '%s' not in [%s]" % (orig, ", ".join(processing.xyz.flightlines.columns)))
    processing.xyz.flightlines.rename(columns={orig: new}, inplace=True)

    end = time.time()
    print(f"  - Time used to rename a column in the dataset: {end - start} sec.\n")


def copy_data(processing: pipeline.ProcessingData,
                orig: LayerDataName,
                new: str):
    """
    Copy a group of data in the dataset (data.layer_data[<key>]).
        These are generally the per-timegate or per-layer dataframes
            "Gate_Ch01", "InUse_Ch01", 'STD_Ch01', 'relErr_Ch01', etc.
        Useful to work around data import and source issues.
        Warning: This will overwrite data if the new group exists

    Parameters
    ----------
    orig :
        The group of data to be copied
    new :
        The new name for the copied group of data

    """
    start = time.time()
    print('  - Renaming a group in the dataset')
    print(f"\t{orig} ––> {new}")

    if orig not in processing.xyz.layer_data:
        raise ValueError(
            "Unknown orig channel name '%s' not in [%s]" % (orig, ", ".join(processing.xyz.layer_data.keys())))
    processing.xyz.layer_data[new] = processing.xyz.layer_data[orig]

    end = time.time()
    print(f"  - Time used to rename a group in the dataset: {end - start} sec.\n")


def rename_data(processing: pipeline.ProcessingData,
                orig: LayerDataName,
                new: str):
    """
    Rename a group of data in the dataset (data.layer_data[<key>]).
        These are generally the per-timegate or per-layer dataframes
            "Gate_Ch01", "InUse_Ch01", 'STD_Ch01', 'relErr_Ch01'
        Useful to work around data import and source issues.
        Warning: This will overwrite data if the new group exists

    Parameters
    ----------
    orig :
        The group of data to be renamed
    new :
        The new name for the group of data

    """
    start = time.time()
    print('  - Renaming a group in the dataset')
    print(f"\t{orig} ––> {new}")

    if orig not in processing.xyz.layer_data:
        raise ValueError(
            "Unknown orig channel name '%s' not in [%s]" % (orig, ", ".join(processing.xyz.layer_data.keys())))
    processing.xyz.layer_data[new] = processing.xyz.layer_data.pop(orig)

    end = time.time()
    print(f"  - Time used to rename a group in the dataset: {end - start} sec.\n")

def auto_classify_high_altitude_flightlines(processing: pipeline.ProcessingData,
                                            height_threshold: float = 500.,
                                            verbose: bool = True):
    """
    Filter to find all flightlines where the mean altitude is above the height_threshold
    and label them as 'High-Altitude' test lines

    Parameters
    ----------
    height_threshold :
        The height threshold, in meters above land surface, for defining a high altitude flightline.
        The mean flight height for each flightline will be compared to this value.
    verbose :
        If True, more output about what the filter is doing
    """
    start = time.time()
    print(f"  - Classifying flightlines with mean altitude ≥ {height_threshold} as 'High-altitude'")

    flightlines = processing.xyz.flightlines

    fline_alt_df = flightlines.groupby('Line')['TxAltitude'].describe().reset_index(names='flightlines')
    fline_alt_df.columns.name = 'Altitude'
    fline_alt_df.sort_values(by=['mean', 'std'], ascending=False, inplace=True, ignore_index=True)

    filt = fline_alt_df['mean'] >= height_threshold
    HA_fl_list = fline_alt_df.flightlines[filt]

    if 'flight_type' not in flightlines.columns:
        flightlines['flight_type'] = 'Production'

    if len(HA_fl_list) >= 1:
        filt = pd.Series(np.zeros(len(flightlines)), dtype=bool)
        for ha_line in HA_fl_list:
            filt = filt | (flightlines.Line == ha_line)
        flightlines.loc[filt, 'flight_type'] = 'High-altitude'

        print(f"  - These lines have been labeled as high-altitude: {sorted(HA_fl_list.to_list())}.")
        print(f"    - All other lines without a previous flight_type classification have been labeled 'Production'.")
    else:
        print(f"  - No lines have been found as high altitude.")
        print(f"    - All lines without a previous flight_type classification have been labeled 'Production'.")
    if verbose:
        u_fl_type = sorted(flightlines.flight_type.unique())
        print(f"  - Flight types in dataset: {u_fl_type}")

    end = time.time()
    print(f"  - Time used to auto-classify the dataset: {end - start} sec.\n")
