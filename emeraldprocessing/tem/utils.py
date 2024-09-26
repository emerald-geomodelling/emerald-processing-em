# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from .setup import allowed_moments

from .data_keys import inuse_dtype
from .data_keys import dat_key_prefix, inuse_key_prefix, std_key_prefix, err_key_prefix

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from .sps import interpolate_df2_on_df1, calcUTMcoordinates, calcEpochTime, read_concat_DGPS_sps_files
import copy
import libaarhusxyz
import time


def build_inuse_dataframe(data, channel):
    """
    Function to build a fresh, fully in-use dataframe
    """
    str_channel = f"0{channel}"[-2:]
    data_key = f"{dat_key_prefix}{str_channel}"
    inuse_key = f"{inuse_key_prefix}{str_channel}"

    try:
        inuse_df = pd.DataFrame(np.ones(data.layer_data[inuse_key].shape),
                                dtype=data.layer_data[inuse_key].dtypes[0],
                                index=data.layer_data[inuse_key].index,
                                columns=data.layer_data[inuse_key].columns)
    except:
        inuse_df = pd.DataFrame(np.ones(data.layer_data[data_key].shape),
                                dtype=inuse_dtype,
                                index=data.layer_data[data_key].index,
                                columns=data.layer_data[data_key].columns)
    return inuse_df


def is_dual_moment(gex, verbose=False):
    if 'Channel2' in gex.gex_dict.keys():
        if verbose: print('looks like dual moment data.')
        dual_moment=True
    else:
        if verbose: print('looks like single moment data.')
        dual_moment=False
    return dual_moment


def get_flightline_bbox(Data, pos_keys=['UTMX','UTMY']):
    df_points = Data.flightlines
    return (df_points[pos_keys[0]].min(),
            df_points[pos_keys[1]].min(), 
            df_points[pos_keys[0]].max(), 
            df_points[pos_keys[1]].max())

def read_shape_in_margins(shapefile, bbox, crs, margin_x=2000, margin_y=2000):
    bbox=(bbox[0]-margin_x,
          bbox[1]-margin_y, 
          bbox[2]+margin_x, 
          bbox[3]+margin_y)
          
    # read shp into geopandas geodataframe:
    df_shp=gpd.read_file(shapefile, bbox=bbox)
    return df_shp.to_crs(crs), bbox

def getGateTimesFromGEX(gex, channel='Channel1'):
    NoGates=int(gex.gex_dict[channel]['NoGates'])
    if 'RemoveGatesFrom' in gex.gex_dict[channel].keys():
        RemoveGatesFrom=int(gex.gex_dict[channel]['RemoveGatesFrom'])
    else:
        RemoveGatesFrom=int(0)
    if not('MeaTimeDelay' in gex.gex_dict[channel].keys()):
        gex.gex_dict[channel]['MeaTimeDelay']=0.0
    gatetimes=(gex.gex_dict['General']['GateTime'][RemoveGatesFrom:NoGates,:] + gex.gex_dict[channel]['GateTimeShift'] + gex.gex_dict[channel]['MeaTimeDelay'] )
    return gatetimes

def inuse_moment(moment):
    return 'InUse_'+moment.split('_')[-1]

def errKey(moment):
    return 'relErr_'+moment.split('_')[-1]

def stdKey(moment):
    return 'STD_'+moment.split('_')[-1]


def resampleWaveform(gex):
    t=gex.General['WaveformLMPoint'][:,0]
    a=gex.General['WaveformLMPoint'][:,1]
    ti=np.linspace(t.min(), t.max(), 100000)
    f=interp1d(t, a, kind='linear')
    ai=f(ti)
    gex.General['WaveformLMPointInterpolated']=np.vstack([ti, ai]).T
    dti=  ti[:-1]+np.diff(ti)/2
    dai=np.diff(ai)
    gex.General['WaveformLMPointInterpolatedGrad']=np.vstack([dti, dai]).T
    d2ti=dti[:-1]+np.diff(dti)/2
    d2ai=np.diff(dai)
    gex.General['WaveformLMPointInterpolatedGradGrad']=np.vstack([d2ti, d2ai]).T
    return gex

def estimateInlineSamplig(data):
    if 'line' in data.flightlines.keys():
        line_key='line'
    elif 'line_no' in data.flightlines.keys():
        line_key='line_no'
    elif 'Line' in data.flightlines.keys():
        line_key='Line'
    lines = splitData_lines(data, line_key=line_key)
    med_dist=[]
    for key in lines.keys():
        if not('lineoffset' in lines[key].flightlines.columns):
            calc_lineOffset(lines[key])
        med_dist.append(lines[key].flightlines.lineoffset.diff().median())
    return np.median(np.array(med_dist))

def get_line(Data, line_no, line_key='line'):
    model_dict = {}
    key_list=list(Data.model_dict.keys())
    key_list.remove('layer_data')
    key_list.remove('flightlines')
    for key in key_list:
        model_dict[key]=Data.model_dict[key]
    # line spceific data:
    idx = Data.flightlines[line_key]==line_no
    model_dict["flightlines"] = Data.flightlines[idx]
    model_dict["layer_data"] = {}
    for key in Data.layer_data.keys():
        model_dict["layer_data"][key]=Data.layer_data[key][idx]
    return libaarhusxyz.XYZ(model_dict, normalize=False)

def split_lines(data, synth, model, line_key='line'):
    line_nos = np.unique(data.flightlines[line_key])
    lines={}
    for line_no in line_nos:
        line={"data" : get_line(data, line_no, line_key=line_key),
              "synth": get_line(synth, line_no, line_key=line_key),
              "model": get_line(model, line_no, line_key=line_key)}
        lines[str(line_no)] = line
    return lines


def splitData_lines(Data, line_key='line'):
    line_nos=Data.flightlines[line_key].unique()
    #line_nos = np.unique(Data.flightlines[line_key].values)
    Lines={}
    for line_no in line_nos:
        Lines[line_no] = get_line(Data, line_no, line_key=line_key)
    return Lines


def splitModel_lines(model, line_key='line'):
    return splitData_lines(model, line_key=line_key)


def merge_lines(lines_dict):
    return libaarhusxyz.XYZ(*lines_dict.values())

def concatXYZ(xyz1, xyz2):
    return libaarhusxyz.XYZ(xyz1, xyz2)

def calc_lineOffset(data):
    if ('UTMX' in data.flightlines.keys()) and ('UTMY' in data.flightlines.keys()):
        pos_keys=['UTMX', 'UTMY']
    elif ('utmx' in data.flightlines.keys()) and ('utmy' in data.flightlines.keys()):
        pos_keys=['utmx', 'utmy']
    else:
        raise Exception("Sorry, no coordinates with labels UTMX, UTMY or utmx, utmy, found in data") 
    for key in ['line', 'line_no', 'Line']:
        if key in data.flightlines.keys():
            line_key=key
    data.flightlines.insert(len(data.flightlines.columns), 'lineoffset', np.zeros( len(data.flightlines) ) )    
    for line in data.flightlines[line_key].unique():
        filt=data.flightlines[line_key]==line
        
        dx=data.flightlines.loc[filt, pos_keys[0]].diff()
        dx.iloc[0]=0
        dy=data.flightlines.loc[filt, pos_keys[1]].diff()
        dy.iloc[0]=0
        data.flightlines.loc[filt, 'lineoffset']=np.cumsum( np.sqrt(dx**2 + dy**2) )


def scaleData(processing, scalefactors=[1, 1]):
    data = processing.xyz
    for n, moment in enumerate(allowed_moments):
        if moment in data.layer_data.keys():
            scaled_key='Gate_scaled_'+moment.split('Gate_')[-1]
            data.layer_data[scaled_key]=data.layer_data[moment]*scalefactors[n]
        # if hasattr(processing, 'Noise'):
        #     if moment in processing.Noise.keys():
        #         processing.Noise[moment]=processing.Noise[moment]*scalefactors[n]
                

def unscaleData(data):
    for moment in allowed_moments:
        if moment in data.layer_data.keys():
            # scaling
            channel=moment.split('_')[-1]
            scaled_key='Gate_scaled_'+channel
            dipole_moment_key='DipoleMoment_'+channel
            dB_dt_df=data.layer_data[moment]
            M_df=pd.DataFrame(data=np.tile(data.flightlines[dipole_moment_key],[dB_dt_df.shape[1], 1]).T,
                              index=dB_dt_df.index,
                              columns=dB_dt_df.columns)
            data.layer_data[scaled_key]=dB_dt_df * data.model_info['scalefactor'] * M_df

def dBdt_to_rhoa(processing):
    data = processing.xyz
    mu0= 4 * np.pi * 1e-7
    for moment in allowed_moments:
        if moment in data.layer_data.keys():
            rhoa_key='Rhoa_' + moment.split('Gate_')[-1]
            dB_dt_df=data.layer_data[moment]
            gate_times_df=pd.DataFrame(data=np.tile(processing.GateTimes[moment], [dB_dt_df.shape[0],1]),
                                    index=dB_dt_df.index,
                                    columns=dB_dt_df.columns)
            M_df=pd.DataFrame(data=np.ones(data.layer_data[moment].shape) / data.model_info['scalefactor'],
                              index=dB_dt_df.index,
                              columns=dB_dt_df.columns)
            data.layer_data[rhoa_key] = 1/np.pi * (M_df / (20 * dB_dt_df)  )**(2/3) *   (mu0 / gate_times_df)**(5/3)


def sampleDEM(DEMfilename,
              df,
              poskeys=['utmx', 'utmy'],
              z_key='zdem_laser1',
              crs=None):
    dem = rasterio.open(DEMfilename, 'r')
    dem_xmin, dem_ymin, dem_xmax, dem_ymax = dem.bounds

    if crs is None:
        print('Assuming that the DTM projection is the same as the AEM data')
        crs = dem.crs

    if type(crs) is int:
        dst_crs = f"EPSG:{crs}"
    elif 'epsg:' in str(crs):
        dst_crs = crs.replace('epsg', 'EPSG')
    elif 'EPSG:' in str(crs):
        dst_crs = crs

    assert 'dst_crs' in locals(), f'There is something wrong with the supplied crs parameter ({crs}).'
    dst_crs = rasterio.crs.CRS.from_string(dst_crs)

    xmin = df[poskeys[0]].min()
    xmax = df[poskeys[0]].max()
    ymin = df[poskeys[1]].min()
    ymax = df[poskeys[1]].max()

    geometry = [Point(xy) for xy in zip(df[poskeys[0]], df[poskeys[1]])]
    geo_df = gpd.GeoDataFrame(df, crs=dst_crs, geometry=geometry)

    if dem.crs != crs:
        print(f"  - DTM projection ({dem.crs}) is not the same as specified projection ({dst_crs}).")
        print(f"      Re-projecting the data to {dem.crs} for sampling only.")
        geo_df.to_crs(epsg=dem.crs.to_epsg(), inplace=True)

        new_poskeys = copy.copy(poskeys)
        if ':' in str(dem.crs):
            new_poskeys[0] = f"{new_poskeys[0]}_{str(dem.crs).split(':')[1]}"
            new_poskeys[1] = f"{new_poskeys[1]}_{str(dem.crs).split(':')[1]}"
        else:
            new_poskeys[0] = f"{new_poskeys[0]}_{str(dem.crs)}"
            new_poskeys[1] = f"{new_poskeys[1]}_{str(dem.crs)}"

        df[new_poskeys[0]] = geo_df.geometry.x
        df[new_poskeys[1]] = geo_df.geometry.y

        xmin = df[new_poskeys[0]].min()
        xmax = df[new_poskeys[0]].max()
        ymin = df[new_poskeys[1]].min()
        ymax = df[new_poskeys[1]].max()

    for a, b in zip([    xmin,     ymin, dem_xmax, dem_ymax],
                    [dem_xmin, dem_ymin,     xmax,     ymax]):
        if a < b:
            bounds_comparison = pd.DataFrame([[dem_xmin, dem_ymin, dem_xmax, dem_ymax],
                                              [    xmin,     ymin,     xmax,     ymax]],
                                             index=['orig_dem', 'reprojected_dem', 'flightlines'],
                                             columns=['left', 'top', 'right', 'bottom'])
            print(bounds_comparison)
            raise Exception(f'coordinates outside raster bounds for projection: {dem.crs}')
        # else:
            # print(f'coordinates are inside raster bounds for projection: {dem.crs}')


    coord_list = [(x, y) for x, y in zip(geo_df["geometry"].x, geo_df["geometry"].y)]
    dtm_values = [x[0] for x in dem.sample(coord_list)]
    dem.close()

    df[z_key] = dtm_values


def build_l10_dBdt_time_df(processing, data_key):
    scalefactor = processing.xyz.model_info['scalefactor']
    data_df = copy.copy(processing.xyz.layer_data[data_key])
    gate_times = copy.copy(processing.GateTimes[data_key])

    dBdt_df = np.abs(data_df) * scalefactor
    l10_dBdt_df = np.log10(dBdt_df)

    # FIXME: This may need fixed better in the future
    gate_times[gate_times < 0] = np.nan
    l10_gate_times = np.log10(gate_times)
    l10_gate_times_df = pd.DataFrame(np.tile(l10_gate_times, (processing.xyz.layer_data[data_key].shape[0], 1)))

    return l10_dBdt_df, l10_gate_times_df

def calculate_transient_slopes(processing, data_key):
    l10_dBdt_df, l10_gate_times_df = build_l10_dBdt_time_df(processing, data_key)

    slope = (l10_dBdt_df.diff(axis=1) / l10_gate_times_df.diff(axis=1))
    return slope


def calculate_transient_curvatures(processing, data_key):
    def calculate_curvature(row):
        dBdt_row = row['l10_dBdt']
        time_row = row['l10_time']
        curvature_row = (dBdt_row.shift(-1) - 2 * dBdt_row + dBdt_row.shift(1)) / (time_row.shift(-1) - time_row.shift(1))**2
        curvature_row.iloc[0] = np.nan
        curvature_row.iloc[-1] = np.nan
        return curvature_row

    l10_dBdt_df, l10_gate_times_df = build_l10_dBdt_time_df(processing, data_key)

    dBdt_columns = pd.MultiIndex.from_product(      [l10_dBdt_df.columns, ['l10_dBdt']])
    gate_columns = pd.MultiIndex.from_product([l10_gate_times_df.columns, ['l10_time']])

    l10_dBdt_df.columns = dBdt_columns
    l10_gate_times_df.columns = gate_columns

    l10_dBdt_time_df = pd.concat([l10_dBdt_df, l10_gate_times_df], axis=1)

    l10_dBdt_time_stack = l10_dBdt_time_df.stack(level=0, future_stack=True)

    curvature_stack = l10_dBdt_time_stack.groupby(level=0).apply(calculate_curvature)

    curvature = curvature_stack.unstack()
    curvature = curvature.reset_index(drop=True)

    return curvature


def sampleDEM_reproject_DEM(DEMfilename,
              df,
              poskeys=['utmx', 'utmy'],
              z_key='zdem_laser1',
              crs=None,
              force_overwrite=False):
    xmin = df[poskeys[0]].min()
    xmax = df[poskeys[0]].max()
    ymin = df[poskeys[1]].min()
    ymax = df[poskeys[1]].max()

    coord_list = [(x, y) for x, y in zip(df[poskeys[0]], df[poskeys[1]])]

    dem = rasterio.open(DEMfilename, 'r')
    left, bottom, right, top = dem.bounds

    if crs is None:
        print('Assuming that the DTM projection is the same as the AEM data')
        crs = dem.crs

    if type(crs) is int:
        dst_crs = f"EPSG:{crs}"
    elif 'EPSG:' in str(crs):
        dst_crs = crs
    elif 'epsg:' in str(crs):
        dst_crs = crs.replace('epsg', 'EPSG')

    assert 'dst_crs' in locals(), f'There is something wrong with the supplied crs parameter ({crs}).'
    dst_crs = rasterio.crs.CRS.from_string(dst_crs)

    if dem.crs != crs:
        print(f"  - DTM projection ({dem.crs}) is not the same as specified projection ({dst_crs}).")
        print(f"      Re-projecting the DTM to {dst_crs}.")
        if str(dem.crs).split('EPSG:')[1]+'.tif' in DEMfilename:
            new_DEMfilename = DEMfilename.replace(str(dem.crs).split('EPSG:')[1], str(dst_crs).split('EPSG:')[1])
        else:
            new_DEMfilename = DEMfilename.replace('.tif', f"_{str(dst_crs).split('EPSG:')[1]}.tif")

        if os.path.isfile(new_DEMfilename) and not force_overwrite:
            print(f"\n****   Note: {new_DEMfilename} already exists, using this file   ****")
            reprojected_dem = rasterio.open(new_DEMfilename, 'r')
        else:
            print(f"{new_DEMfilename} does not exist. Writing file now")
            transform, width, height = calculate_default_transform(dem.crs,
                                                                   dst_crs,
                                                                   dem.width,
                                                                   dem.height,
                                                                   *dem.bounds)
            kwargs = dem.meta.copy()
            kwargs.update({'crs': dst_crs,
                           'transform': transform,
                           'width': width,
                           'height': height
                           })
            new_dem = rasterio.open(new_DEMfilename, 'w', **kwargs)
            with rasterio.open(new_DEMfilename, 'w', **kwargs) as new_dem:
                for i in range(1, dem.count + 1):
                    reproject(source=rasterio.band(dem, i),
                              destination=rasterio.band(new_dem, i),
                              src_transform=dem.transform,
                              src_crs=dem.crs,
                              dst_transform=transform,
                              dst_crs=dst_crs,
                              resampling=Resampling.nearest)

            dem.close()
            reprojected_dem = rasterio.open(new_DEMfilename, 'r')
    else:
        reprojected_dem = dem

    repro_dem_xmin, repro_dem_ymin, repro_dem_xmax, repro_dem_ymax = reprojected_dem.bounds

    for a, b in zip([          xmin,           ymin, repro_dem_xmax, repro_dem_ymax],
                    [repro_dem_xmin, repro_dem_ymin,           xmax,           ymax]):
        if a < b:
            bounds_comparison = pd.DataFrame([[left, bottom, right, top],
                                              [repro_dem_xmin, repro_dem_ymin, repro_dem_xmax, repro_dem_ymax],
                                              [xmin, ymin, xmax, ymax]],
                                             index=['orig_dem', 'reprojected_dem', 'flightlines'],
                                             columns=['left', 'top', 'right', 'bottom'])
            print(bounds_comparison)
            raise Exception('coordinates outside raster bounds')

    dtm_values = [x[0] for x in reprojected_dem.sample(coord_list)]
    df[z_key] = dtm_values
    reprojected_dem.close()
    dem.close()

def drop_lines_from_data(data, line_list, line_key='Line'):
    for line in line_list:
        filt=data.flightlines[line_key]==line
        data.flightlines = data.flightlines.drop(data.flightlines.iloc[filt.values].index)
        data.flightlines.reset_index(drop=True, inplace=True)
        for key in data.layer_data.keys():
            data.layer_data[key]=data.layer_data[key].drop(data.layer_data[key].iloc[filt.values].index)
            data.layer_data[key].reset_index(drop=True, inplace=True)

def calcGateRelErr(data, synth):
    print('calculating inversion error gate by gate, sounding by sounding.')
    for moment in allowed_moments:
        if moment in data.layer_data.keys():
            std_key='STD_'+moment.split('Gate_')[-1]
            relErr_key='relErr_'+moment.split('Gate_')[-1]
            err=data.layer_data[moment].abs()*data.layer_data[std_key]
            data.layer_data[relErr_key]=(data.layer_data[moment].abs()-synth.layer_data[moment].abs()) / err


def drop_filt_XYZ(data, filt, reset_index=True):
    data.flightlines=data.flightlines.drop(data.flightlines.loc[filt,:].index)
    if reset_index:
        data.flightlines.reset_index(inplace=True)
        data.flightlines.drop(['index'], axis=1, inplace=True)
    for key in data.layer_data.keys():
        data.layer_data[key]=data.layer_data[key].drop(data.layer_data[key].loc[filt,:].index)
        if reset_index:
            data.layer_data[key].reset_index(inplace=True)
            data.layer_data[key].drop(['index'], axis=1, inplace=True)

def filtXYZ(data, filt, reset_index=True):
    data_out=copy.deepcopy(data)
    drop_filt_XYZ(data_out, ~filt, reset_index=reset_index)
    return data_out


def substractSystemBias(data, System_bias_dict):
    for moment in allowed_moments:
        if moment in System_bias_dict.keys():
            if data.layer_data[moment].shape[1] == len(System_bias_dict[moment]):
                print('Correcting system bias for {}'.format(moment))
                Bias=pd.DataFrame(np.tile(System_bias_dict[moment].values, (len(data.layer_data[moment]), 1)))
                data.layer_data[moment]=data.layer_data[moment]-Bias
            else:
                raise Exception("Number of gates in system bias and data structure differ!") 
        else:
            print('Moment: {} not found in bias dictionary'.format(moment) )


def round_to_odd(f):
    return int(np.ceil(f/2)  * 2 - 1)

def interpolate_rolling_size_for_all_gates(filterlist, moment):
    ci=moment.columns.values.astype(int)
    c=[0, ci.max()]
    f=interp1d(c, filterlist)
    ni = f(ci)
    return [round_to_odd(n) for n in ni]

def get_min_periods(filter_length):
    if filter_length > 1:
        return np.ceil(filter_length/2).astype(int)
    else:
        return 1

def rolling_weighted_mean_df(df_dat, df_err_fp, rolling_lengths, weighting_factor=3, error_calc_scheme='Weighted_SEM'):
    assert weighting_factor > 0, "weighting_factor must be greater than 0. Suggested ranges are between 1 [Weights are only based on the errors - errors will be smaller] and 10 [errors will be bigger]"
    if len(rolling_lengths) == len(df_dat.columns):
        # Calculate absolute errors
        df_err_ab = df_dat * df_err_fp

        # Calculate weights
        # FIXME: I'm applying a factor of 3 here. This is purely determined experimentally. Since this is just a
        #   weighting function I think it's ok?
        weights_df = 1 / (weighting_factor * (df_err_ab**2))

        # Build weighted data df for the averaging.
        weighted_data = df_dat * weights_df

        # Prepare empty data frames
        ave_dat = df_dat * np.nan
        ave_err_abs_df = df_err_ab * np.nan
        std_err_df = df_err_ab * np.nan
        unweighted_SEM_df = df_err_ab * np.nan
        weighted_SEM_df = df_err_ab * np.nan

        for filter_length, col in zip(rolling_lengths, df_dat.columns):
            # Calculate the rolling mean of the absolute error
            ave_err_abs_df[col] = df_err_ab[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).mean()

            # Calculate the rolling STD error
            std_err_df[col] = df_dat[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).std()

            # Calculate the unweighted Standard Error of the Mean
            unweighted_SEM_df = df_dat[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).std() / np.sqrt(filter_length)

            # Calculate the weighted average of the data
            ave_dat[col] = weighted_data[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).sum() / \
                              weights_df[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).sum()

            # Calculate the weighted Standard Error of the Mean
            weighted_SEM_df[col] = (weights_df[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).sum() /
                                   (weights_df[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).sum()**2))**(1/2)

        # Balance absolute errors by the 1) the mean, 2) the STD, and 3) the weighted SEM 4) unweighted SEM
        # FIXME: I have a hard time to justify this balancing, but without it I feel that the errors from the SEM alone are too small
        unweighted_SEM_weight = 1
        weighted_SEM_weight = 1
        STD_weight = 1
        mean_weight = 1

        divide_by = unweighted_SEM_weight + weighted_SEM_weight + STD_weight + mean_weight

        balanced_abs_err1 = (unweighted_SEM_df * weighted_SEM_df * std_err_df * ave_err_abs_df) ** (1 / divide_by)

        balanced_abs_err2 = (unweighted_SEM_weight * (unweighted_SEM_df**2) / divide_by +
                             weighted_SEM_weight   * (weighted_SEM_df**2)   / divide_by +
                             STD_weight            * (std_err_df**2)        / divide_by +
                             mean_weight           * (ave_err_abs_df**2)    / divide_by  )**(1/2)

        # calculate the fractional error of the balanced absolute error
        weighted_SEM_frac_err =     np.abs(weighted_SEM_df / ave_dat)
        unweighted_SEM_frac_err = np.abs(unweighted_SEM_df / ave_dat)
        std_frac_err =                   np.abs(std_err_df / ave_dat)
        ave_frac_err =               np.abs(ave_err_abs_df / ave_dat)
        balanced_frac_err1 =      np.abs(balanced_abs_err1 / ave_dat)
        balanced_frac_err2 =      np.abs(balanced_abs_err2 / ave_dat)

        if error_calc_scheme == 'Weighted_SEM':
            return ave_dat, weighted_SEM_frac_err
        elif error_calc_scheme == 'Balanced_1':
            return ave_dat, balanced_frac_err1
        elif error_calc_scheme == 'Average':
            return ave_dat, ave_frac_err
        elif error_calc_scheme == 'Balanced_2':
            return ave_dat, balanced_frac_err2
        elif error_calc_scheme == 'STD':
            return ave_dat, std_frac_err
        elif error_calc_scheme == 'Unweighted_SEM':
            return ave_dat, unweighted_SEM_frac_err


    else:
        print(f'filter length: {len(rolling_lengths)}')
        print(f'number of data columns: {len(df_dat.columns)}')
        print(f'number of std columns: {len(df_err_fp.columns)}')
        raise Exception('number of rolling filter lengths differs from number of columns in dataframe ')


def rolling_mean_df(df, rolling_lengths):
    if len(rolling_lengths) == len(df.columns):
        ave_dat = copy.deepcopy(df) * np.nan
        ab_err =  copy.deepcopy(df) * np.nan

        for filter_length, col in zip(rolling_lengths, df.columns):
            ave_dat[col] = df[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).mean()
            ab_err[col] = df[col].rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).std()

        err_dat = ab_err / ave_dat  # error should be in fractional percent

        return ave_dat, err_dat

    else:
        print(f'filter length: {len(rolling_lengths)}')
        print(f'number of columns: {len(df.columns)}')
        raise Exception('number of rolling filter lengths differs from number of columns in dataframe ')


def rolling_square_root_sum_df(df, rolling_lengths):
    if len(rolling_lengths) == len(df.columns):
        df_out = copy.deepcopy(df)
        for filter_length, col in zip(rolling_lengths, df_out.columns):
            notna_length = df_out[col].notna().rolling(filter_length, center=True, min_periods=1).sum()
            # df_out[col] = np.sqrt(1/notna_length**2 * (df_out[col]*df_out[col]).rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).sum())
            df_out[col] = np.sqrt(1/notna_length * (df_out[col] * df_out[col]).rolling(filter_length, center=True, min_periods=get_min_periods(filter_length)).sum())
    else:
        # print(f'Filter length: {len(rolling_lengths)}')
        # print(f'Number of columns: {len(df.columns)}')
        raise Exception(f'Number of rolling filter lengths ({len(rolling_lengths)}) differs from number of columns in dataframe ({len(df.columns)})')
    return df_out


def make_noise_df(processing,
                  channel=1,
                  noise_level_1ms=1e-8,
                  noise_exponent=-0.5,
                  norm_by_tx: bool = True):
    """


    Parameters
    ----------
    channel :
        Which channel to use
    noise_level_1ms :
        amplitude of the noise floor, in V/m^2
    noise_exponent :
        Slope of the noise floor, in d(log10(V/m^2))/d(s), equivalent to t^slope
    norm_by_tx :
        Normalize by transmitter? If:
        True - the output unit is V/(A*m^4) [normalized by transmitter and receiver]
        False - the output unit is V/(m^2) [normalized by the receiver only]

    Returns
    -------
    noise_df : Pandas Dataframe
        A dataframe, the same shape as the data, with amplitudes representing the noise floor.
        if 'norm_by_tx' is False the output will be in V/(m^2), or normalized by the rx only.
        If 'norm_by_tx' is True (default) the output will be in V/(A*m^4), or normalized by the rx and transmitter moment
    """

    # build a dataframe that holds the noise levels for the individual gates:
    data = processing.xyz
    num_soundings = len(data.flightlines.index)

    str_channel = f"0{channel}"[-2:]
    data_key = f"{dat_key_prefix}{str_channel}"
    assert data_key in data.layer_data.keys(), "The channel requested does not exist"

    noise = ((processing.GateTimes[data_key] / 1e-3) ** noise_exponent) * noise_level_1ms

    noise_array = np.tile(noise, (num_soundings, 1))
    noise_df = pd.DataFrame(noise_array,
                            dtype=float,
                            columns=data.layer_data[data_key].columns,
                            index=data.layer_data[data_key].index)

    noise_df = noise_df / data.model_info['scalefactor']
    if norm_by_tx:
        noise_df = noise_df / processing.ApproxDipoleMoment[data_key]

    return noise_df


# def make_noise_dict(processing, channel=1, noise_level_1ms=1e8, noise_exponent=-0.5, unitDipole=False):
#     # build a dictionary that holds the noise levels for the individual gates:
#     data = processing.xyz
#
#     str_channel = f"0{channel}"[-2:]
#     data_key = f"{dat_key_prefix}{str_channel}"
#     assert data_key in data.layer_data.keys(), "The channel requested does not exist"
#
#     noise_dict = {}
#     noise = ((processing.GateTimes[data_key] * 1e3)**noise_exponent) * noise_level_1ms
#     noise_dict[data_key] = {}
#     for k, column in enumerate(data.layer_data[data_key].columns):
#         noise_dict[data_key][column] = noise[k]
#     if unitDipole:
#         for data_key in noise_dict.keys():
#             for key in noise_dict[data_key].keys():
#                 noise_dict[data_key][key] = noise_dict[data_key][key] / data.model_info['scalefactor'] / processing.ApproxDipoleMoment[data_key]
#
#     return noise_dict
#
# def add_noise_dict(processing, channel, noise_dict):
#     data = processing.xyz
#     str_channel = f"0{channel}"[-2:]
#     data_key = f"{dat_key_prefix}{str_channel}"
#
#     assert data_key in data.layer_data.keys(), "The channel requested does not exist"
#     assert 'scalefactor' in data.model_info.keys(), "data.model_info['scalefactor'] must be defined"
#
#     processing.Noise = {}
#
#     for key in noise_dict[data_key].keys():
#         processing.Noise[data_key] = np.array(list(noise_dict[data_key].values()))
                
def sortXYZ(data, sort_by_column):
    xyz=copy.deepcopy(data)
    for key in xyz.layer_data.keys():
        if sum(xyz.flightlines.index == xyz.layer_data[key].index)<len(xyz.flightlines.index):
            raise Exception('Indexes from flightlines and layer_data differ!')
    xyz.flightlines.sort_values(sort_by_column, inplace=True)
    for key in xyz.layer_data.keys():
        xyz.layer_data[key]=xyz.layer_data[key].loc[xyz.flightlines.index,:]
    return xyz

    

def scale_to_picoVolt(xyz):
    print("Warning: Ignoring existing scaling. This might give wrong results!")
    for moment in allowed_moments:
        xyz.layer_data[moment] = xyz.layer_data[moment]*1e12
        xyz.model_info['scalefactor']=1e-12

def add_inuse_flags(xyz, gex=None):
    for moment in allowed_moments:
        if moment in xyz.layer_data.keys():
            inuse_key=inuse_moment(moment)
            xyz.layer_data[inuse_key]=pd.DataFrame(np.ones(xyz.layer_data[moment].loc[:,:].shape).astype(int)).set_index(xyz.flightlines.index, drop=True)
            if gex:
                if type(gex)==str:
                    gex=libaarhusxyz.gex.parse(gex)
                gex_ch_key='Channel'+moment[-1]
                for col in range(int(gex[gex_ch_key]['RemoveInitialGates'])):
                    xyz.layer_data[inuse_key].loc[:,col]=0

def remove_empty_soundings(data):
    filt=pd.Series(np.ones(len(data.flightlines))).astype(bool)
    for moment in allowed_moments:
        if moment in data.layer_data.keys():
            no_data = data.layer_data[inuse_moment(moment)].sum(axis=1)==0
            print('{0} has {1} sounding positions without data'.format(moment, no_data.sum()))
            filt=filt & no_data
    drop_filt_XYZ(data, filt)

