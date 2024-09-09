#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:23:23 2023

@author: mp
"""

import numpy as np
import pandas as pd
import glob, os
from scipy.interpolate import splrep, splev

from emeraldprocessing.tem.sps import readAndPrepareGPSfromSPS, readAndPrepareLASERfromSPS, readSourceDataSPS
from emeraldprocessing.tem.sps import findSPSfiles, readLineFile
from emeraldprocessing.tem.sps import calcHeading, assign_lineNumber, interpolate_df2_on_df1
import copy
import libaarhusxyz
import time

from emeraldprocessing.tem.setup import allowed_moments
from emeraldprocessing.tem.utils import scale_to_picoVolt, concatXYZ
from emeraldprocessing.tem.dataIO import assignGateTimesDipoleMoments

import pymatreader




def calcUTMshiftsFromHeadingAuxGex(df, aux, poskeys=['utmx', 'utmy', 'z_nmea'], gps_position=[10.51,  3.95, -0.16]):
    # correct coordinates based on GPS position on the frame and frame heading
    if not('heading' in df.columns):
        calcHeading(df, poskeys)
    heading=df['heading']/180*np.pi
    
            
    df_tmp=copy.deepcopy(df)
    interpolate_df2_on_df1(df_tmp, aux['tl1'], common_key='epoch_time',  key2interpolate_df2='tilt_x', key4interpolated_df1='tilt_x1')
    interpolate_df2_on_df1(df_tmp, aux['tl2'], common_key='epoch_time',  key2interpolate_df2='tilt_x', key4interpolated_df1='tilt_x2')
    tilt_x=df_tmp.loc[:, ['tilt_x1', 'tilt_x2']].mean(axis=1)/180*np.pi
    
    #df['tilt_x']=tilt_x
    
    # interpolate_df2_on_df1(df_tmp, aux['tl1'], common_key='epoch_time',  key2interpolate_df2='tilt_y', key4interpolated_df1='tilt_y1')
    # interpolate_df2_on_df1(df_tmp, aux['tl2'], common_key='epoch_time',  key2interpolate_df2='tilt_y', key4interpolated_df1='tilt_y2')
    # tilt_y=df_tmp.loc[:, ['tilt_y1', 'tilt_y2']].mean(axis=1)/180*np.pi
    
    # coordinate shifts due to the position of the gps reative to center of frame
    dutmx_gps=np.sin(heading)*gps_position[0] + np.cos(heading)*gps_position[1]
    dutmy_gps=np.cos(heading)*gps_position[0] - np.sin(heading)*gps_position[1]
    dutmz_gps=gps_position[2] 
    
    # coordinate shift due to pitch (tilt_x)
    dutmz_tilt=gps_position[0]*np.sin(tilt_x)
    
    corr_keys=[]
    for key in poskeys:
        corr_keys.append(key+'_corr')
    df[corr_keys[0]]=df[poskeys[0]]-dutmx_gps
    df[corr_keys[1]]=df[poskeys[1]]-dutmy_gps
    df[corr_keys[2]]=df[poskeys[2]]-dutmz_gps+dutmz_tilt

def readAndProcessGPSdata(gpsfile, project_crs, gex, aux):
    gps = readAndPrepareGPSfromSPS(gpsfile, project_crs)
    for key in gps.keys():
        print('key: {0}, shape: {1}'.format(key, gps[key].shape))
        calcHeading(gps[key], poskeys=['utmx', 'utmy'])
        gps[key].drop(columns=['dx', 'dy'], inplace=True)
    
    if gps['gd1'].shape[0]>1:
        # dGPS data available
        calcUTMshiftsFromHeadingAuxGex(gps['gd1'], aux, gps_position=gex['General']['GPSPosition'][0])
    else:
        # only 2 regular GPS measurements available
        calcUTMshiftsFromHeadingAuxGex(gps['gp1'], aux, gps_position=gex['General']['GPSPosition'][0])
        calcUTMshiftsFromHeadingAuxGex(gps['gp2'], aux, gps_position=gex['General']['GPSPosition'][1])
    return gps


def calcAlt(df, gps_position=[11.68,  2.79, -0.16]):
    tilt_x=df.tilt_x/180*np.pi
    tilt_y=df.tilt_y/180*np.pi
    gps_vert_corr = np.sin(tilt_x)*gps_position[0] + np.sin(tilt_y)*gps_position[1]
    df['alt'] = (np.cos(tilt_x) * np.cos(tilt_y) * df.dist ) + gps_vert_corr

def interpolateTiltAngles(aux):
    for he_key, tl_key in zip(['he1', 'he2'],['tl1', 'tl2']):
        for tilt_key in ['tilt_x', 'tilt_y']:
            interpolate_df2_on_df1(aux[he_key], aux[tl_key], common_key='epoch_time', key2interpolate_df2=tilt_key, key4interpolated_df1=tilt_key)

def processAltitude(aux, gex):
    interpolateTiltAngles(aux)
    for key, gps_pos in zip(['he1', 'he2'], gex['General']['GPSPosition']):
        calcAlt(aux[key], gps_position=gps_pos)

def readAndProcessAUXdata(navsys_file, gex):
    aux=readAndPrepareLASERfromSPS(navsys_file)
    processAltitude(aux, gex)
    return aux


def readVoltageData(mat_file):
    mat=pymatreader.read_mat(mat_file)
    data={}
    for gate_key, moment_key in zip(allowed_moments,['LMZ', 'HMZ']):
        if moment_key in mat['EMdata'].keys():
            if 'PFC' in mat['EMdata'][moment_key].keys():
                key_to_read='PFC'
                print('Reading PFC corrected data for moment: {}'.format(moment_key))
            elif 'preFilt' in mat['EMdata'][moment_key].keys():
                key_to_read='preFilt'
                print('No PFC data found for moment: {}, will read preFilt'.format(moment_key))
            n_col=mat['EMdata'][moment_key][key_to_read]['dBdt'].shape[1]
            colnames=[]
            for n in range(n_col):
                colnames.append('{0}_{1:0>2d}'.format(gate_key,n+1))
            df=pd.DataFrame(mat['EMdata'][moment_key][key_to_read]['dBdt'], columns=colnames)
            df.insert(0, 'epoch_time', mat['EMdata'][moment_key][key_to_read]['datetime']*1000)  # datetime in ms
            data[gate_key]=df
    return data

def regression_based_std_for_df(df, order=1):
    s_out=df.drop(columns=['group']).mean()  # just to make a PDseries in the right format
    x=df['epoch_time'].values
    x=x-x.min()   # some conditioning
    for col in df.drop(columns=['epoch_time', 'group']).columns:
        if 'Gate_Ch' in col:
            y=df[col].values
            coeff = np.polyfit(x, y, order)
            p = np.poly1d(coeff)
            s_out[col]=np.std(y-p(x))
    return s_out

def stackVoltageData(data, regression_based_std=0, verbose=False):
    data_out={}
    for moment in allowed_moments:
        df = data[moment]
        
        # group all samples that belong to the same stack:
        df['epoch_time_diff']=df['epoch_time'].diff()
        cutoff=df['epoch_time_diff'].max()/4
        filt=df['epoch_time_diff']>cutoff
        df.loc[filt, 'group']=df.loc[filt].index
        df['group'].iloc[0]=df.index[0]    
        df['group']=df['group'].fillna(method='ffill')
        
        # caclulate the means for the stacks
        df_mean=df.groupby('group').mean()
        df_mean['epoch_time_stackmin']=df[['epoch_time', 'group']].groupby('group').min()
        df_mean['epoch_time_stackmax']=df[['epoch_time', 'group']].groupby('group').max()
        dt = df_mean['epoch_time'].diff().median() 
        df_mean['epoch_time_binmin']=df_mean['epoch_time']-dt/2
        df_mean['epoch_time_binmax']=df_mean['epoch_time']+dt/2
        
        data_out[moment]=df_mean.drop(columns=['epoch_time_diff'])
        
        # calculate the STDs for the stacks
        if regression_based_std:
            if verbose:
                print('estimating STDs after removing {} order trend'.format(regression_based_std))
            df_std=df.groupby('group').apply(regression_based_std_for_df, order=regression_based_std)
        else:
            df_std=df.groupby('group').std()
        
        df_std=df_std/df_mean.abs() # want std in percent not absolute
        df_std['epoch_time']=df_mean['epoch_time']
        col_rename_dict={}
        col_drop_list=[]
        for col in df_std.columns:
            if 'epoch_time' in col:
                col_drop_list.append(col)
            else:
                col_rename_dict[col]='STD_'+'_'.join(col.split('_')[1:])
        std_key='STD_'+moment.split('_')[-1]
        data_out[std_key]=df_std.drop(columns=col_drop_list).rename(columns=col_rename_dict)
    
    return data_out
    

def interpolateGPSdata(data_stacked, gps, cleanup=True):
    if gps['gd1'].shape[0] > 1:
        # dGPS data available
        for moment in allowed_moments:
            df=data_stacked[moment]
            interpolate_df2_on_df1(df, gps['gd1'], common_key='epoch_time',  key2interpolate_df2='utmx_corr', key4interpolated_df1='UTMX')
            interpolate_df2_on_df1(df, gps['gd1'], common_key='epoch_time',  key2interpolate_df2='utmy_corr', key4interpolated_df1='UTMY')
            interpolate_df2_on_df1(df, gps['gd1'], common_key='epoch_time',  key2interpolate_df2='z_nmea_corr', key4interpolated_df1='TxZ')
    else:
        # only 2 regular GPS measurements available
        gpskeys=['gp1', 'gp2']
        utmx_corr_keys=[]
        utmy_corr_keys=[]
        utmz_corr_keys=[]
        for gpskey in gpskeys:
            utmx_corr_keys.append('utmx_corr_'+gpskey)
            utmy_corr_keys.append('utmy_corr_'+gpskey)
            utmz_corr_keys.append('z_nmea_corr_'+gpskey)
            
        for moment in allowed_moments:
            df=data_stacked[moment]
            for gpskey, utmx_corr_key, utmy_corr_key, utmz_corr_key  in zip(gpskeys, utmx_corr_keys, utmy_corr_keys, utmz_corr_keys):
                interpolate_df2_on_df1(df, gps[gpskey], common_key='epoch_time',  key2interpolate_df2='utmx_corr', key4interpolated_df1=utmx_corr_key)
                interpolate_df2_on_df1(df, gps[gpskey], common_key='epoch_time',  key2interpolate_df2='utmy_corr', key4interpolated_df1=utmy_corr_key)
                interpolate_df2_on_df1(df, gps[gpskey], common_key='epoch_time',  key2interpolate_df2='z_nmea_corr', key4interpolated_df1=utmz_corr_key)
            df.insert(df.shape[1], 'UTMX', df[utmx_corr_keys].mean(axis=1) )
            df.insert(df.shape[1], 'UTMY', df[utmy_corr_keys].mean(axis=1) )
            df.insert(df.shape[1], 'TxZ', df[utmz_corr_keys].mean(axis=1) )
            
            if cleanup:
                df.drop(columns=utmx_corr_keys+utmy_corr_keys+utmz_corr_keys, inplace=True)




def mergeVoltageTiltAlt(data_stacked, aux, q=0.9, cleanup=True):
    for moment in allowed_moments:
        for n in [1, 2]:
            aux_key='he'+str(n)
            df=data_stacked[moment]
            he=aux[aux_key]
            df['group']=df.index
            merged = pd.merge_asof(he[['epoch_time', 'alt']], df[['epoch_time', 'group']], on='epoch_time', allow_exact_matches=True, direction='nearest').groupby(['group']).quantile(q=q, numeric_only=True)
            df['alt'+str(n)]=merged['alt']
            merged = pd.merge_asof(he[['epoch_time', 'tilt_x', 'tilt_y']], df[['epoch_time', 'group']], on='epoch_time', allow_exact_matches=True, direction='nearest').groupby(['group']).mean(numeric_only=True)
            df['tilt_x'+str(n)]=merged['tilt_x']
            df['tilt_y'+str(n)]=merged['tilt_y']
            df.drop(columns=['group'], inplace=True)
        df['TxPitch']=(df[['tilt_x1', 'tilt_x2' ]].mean(axis=1)).fillna(method='ffill', limit=2)
        df['TxRoll']=(df[['tilt_y1', 'tilt_y2' ]].mean(axis=1)).fillna(method='ffill', limit=2)
        df['TxAltitude']=df[['alt1', 'alt2' ]].mean(axis=1)
        if cleanup:
            df.drop(columns=['tilt_x1', 'tilt_x2', 'tilt_y1', 'tilt_y2', 'alt1', 'alt2'], inplace=True)


def mergeVoltageSource(data_stacked, source):
    for moment_key,source_key in zip(allowed_moments, ['TXD_LM', 'TXD_HM']):
        df=data_stacked[moment_key]
        txd=source[source_key].sort_values('epoch_time')
        df['group']=df.index
        merged = pd.merge_asof(txd[['epoch_time', 'MeanCurrent']], df[['epoch_time', 'group']], on='epoch_time', allow_exact_matches=True, direction='nearest').groupby(['group']).mean(numeric_only=True)
        current_key='Current_'+moment_key.split('_')[-1]
        df[current_key]=merged['MeanCurrent']
        df[current_key].fillna(method='ffill', limit=2, inplace=True)
        df.drop(columns=['group'], inplace=True)


def getColumnsWithPattern(df, pattern):
    columns=[]
    for col in df.columns:
        if pattern in col:
            columns.append(col)
    return columns

def getSTDkey(moment):
    return 'STD_'+moment.split('_')[-1]

def getConcatKey(moment):
    return 'Concat_'+moment.split('_')[-1]

def getFlightlineColumns(df):
    fl_cols=[]
    for col in ["Line", "Flight", "epoch_time", "Date",  "Time", "UTMX", "UTMY", "TxZ", "Topography", "TxAltitude",
                "TxPitch", "TxRoll",  "RxPitch", "RxRoll", "Current_Ch01", "Current_Ch02",
                "ChannelsNumber", "Magnetic", "PowerLineMonitor",
                "Misc1", "Misc2", "Misc3", "Misc4",
                "TxOffTime", "TxOnTime", "TxPeakTime", "TxRxHoriSep", "TxRxVertSep"]:
        if col in df.columns:
            fl_cols.append(col)
    return fl_cols

def getDataColumns(df):
    data_cols=[]
    for pattern in ['Gate_Ch01', 'Gate_Ch02', 'STD_Ch01', 'STD_Ch02']:
        data_cols=data_cols+getColumnsWithPattern(df, pattern)
    return data_cols


def concat_all(data_stacked):
    for gate_key, std_key, patch_gate, patch_std in zip(['Gate_Ch01', 'Gate_Ch02'], ['STD_Ch01', 'STD_Ch02'], 
                                                        ['Gate_Ch02', 'Gate_Ch01'], ['STD_Ch02', 'STD_Ch01']):
        concat_key=getConcatKey(gate_key)
        
        if len(data_stacked[gate_key]) == len(data_stacked[std_key]):
            std_colls_to_append=getColumnsWithPattern(data_stacked[std_key], 'STD')
            gate_colls_to_patch=getColumnsWithPattern(data_stacked[patch_gate], 'Gate')
            std_colls_to_patch=getColumnsWithPattern(data_stacked[patch_std], 'STD')
            list_to_concat=[data_stacked[gate_key].reset_index(drop=True),
                            data_stacked[std_key][std_colls_to_append].reset_index(drop=True),
                            pd.DataFrame(np.ones((len(data_stacked[gate_key]), len(gate_colls_to_patch)))*np.nan, columns=gate_colls_to_patch),
                            pd.DataFrame(np.ones((len(data_stacked[gate_key]), len(std_colls_to_patch)))*np.nan, columns=std_colls_to_patch)
                            ]
            data_stacked[concat_key]=pd.concat(list_to_concat, axis=1)
            channel_value=int(gate_key.split('_Ch')[-1])
            data_stacked[concat_key].insert(data_stacked[concat_key].shape[1], 'ChannelsNumber', np.ones(len(data_stacked[concat_key]))*channel_value)
            extra_curr_col_name='Current_'+patch_gate.split('_')[-1]
            data_stacked[concat_key].insert(data_stacked[concat_key].shape[1], extra_curr_col_name, np.ones(len(data_stacked[concat_key]))*np.nan )
            
        else:
            raise Exception('dataframes: {0} and {1} have differnt size'.format(gate_key, std_key))
    columns_out=getFlightlineColumns(data_stacked['Concat_Ch01']) + getDataColumns(data_stacked['Concat_Ch01'])
    data_stacked['Concat_all'] = (pd.concat([data_stacked['Concat_Ch01'][columns_out], data_stacked['Concat_Ch02'][columns_out]], axis=0)).sort_values('epoch_time').reset_index(drop=True)
    for key in ['Concat_Ch01', 'Concat_Ch02']:
        del data_stacked[key]

def calc_Date_Time_from_epochtime(df):
    df['datetime']=df.epoch_time.astype('datetime64[ms]')
    df[['Date', 'Time']]=df['datetime'].astype(str).str.split(pat=' ', expand=True)
    df['Date'] = df.Date.str.replace('-', '/')
    df.drop(columns=['datetime'], inplace=True)


def makeXYZ(df_concat_all):
    filt=df_concat_all.Line>0
    all_valid_data=df_concat_all.loc[filt, :].reset_index(drop=True)
    if not('Topography' in all_valid_data.columns):
        print('did not find topography data in stacked data, will calculate from TxZ and TxAltitude')
        all_valid_data['Topography']=all_valid_data.TxZ-all_valid_data.TxAltitude
    xyz={'flightlines':all_valid_data[getFlightlineColumns(all_valid_data)]}
    xyz['layer_data']={}
    for key in ['Gate_Ch01', 'Gate_Ch02', 'STD_Ch01', 'STD_Ch02']:
        df=all_valid_data[getColumnsWithPattern(df_concat_all, key)]
        rename_dict={}
        for n, col in enumerate(df.columns):
            rename_dict[col]=n
        xyz['layer_data'][key]=df.rename(columns=rename_dict)
    xyz['model_info']={}
    return xyz

def findMATfile(mat_dir):
    files=glob.glob(os.path.join(mat_dir, '*.mat'))
    if len(files)>1:
        raise Exception('Found filtiple files: {} \n make sure there is just one'.format(files))
    else:
        mat_file=files[0]
        print('found the follwing matfile file: \n {}'.format(mat_file))
    return mat_file


def correctGEX(gex):
    gex['General']['GPSDifferentialPosition'][:]=0.0
    gex['General']['GPSPosition'][:]=0.0
    gex['General']['AltimeterPosition'][:]=0.0
    gex['General']['InclinometerPosition'][:]=0.0
    gex['General']['CalculateRawDataSTD']=0
    gex['Channel1']['UniformDataSTD']=0.03
    gex['Channel2']['UniformDataSTD']=0.03

def correctGateFactor(xyz, gex):
    xyz['layer_data']['Gate_Ch01']=xyz['layer_data']['Gate_Ch01']*gex['Channel1']['GateFactor']
    gex['Channel1']['GateFactor']=1.0
    if 'Gate_Ch02' in xyz['layer_data'].keys():
        xyz['layer_data']['Gate_Ch02']=xyz['layer_data']['Gate_Ch02']*gex['Channel2']['GateFactor']
        gex['Channel2']['GateFactor']=1.0


def normalize_with_TxDipoleMoment_XYZ(data, gex):
    for moment in allowed_moments:
        if moment in data['layer_data'].keys():
            # scaling
            channel=moment.split('_')[-1]
            current_key='Current_'+channel
            dB_dt_df=data['layer_data'][moment]
            if 'Ch01' in moment:
                nTurns=gex['General']['NumberOfTurnsLM']
            elif 'Ch02' in moment:
                nTurns=gex['General']['NumberOfTurnsHM']
            else:
                raise Exception('No number of turns found for moment: {}'.format(moment))
            
            M=data['flightlines'][current_key] * gex['General']['TxLoopArea'] * nTurns
            M_df=pd.DataFrame(data=np.tile(M.values,[dB_dt_df.shape[1], 1]).T,
                              index=dB_dt_df.index,
                              columns=dB_dt_df.columns)
            data['layer_data'][moment]=dB_dt_df / M_df


def exportSkyTEMdata(xyz_in, gex_in, fileprefix_out):
    # use this function when storing data from .mat/.sps reading/stacking/resampleing
    xyz=copy.deepcopy(xyz_in)
    gex=copy.deepcopy(gex_in)
    correctGateFactor(xyz, gex)
    xyz['flightlines']=xyz['flightlines'][getFlightlineColumns(xyz['flightlines'])]
    xyz['flightlines']['Current_Ch01'].fillna(method='ffill', inplace=True)  # workbench requirement 
    xyz['flightlines']['Current_Ch02'].fillna(method='bfill', inplace=True)  # workbench requirement
    xyz_filename=fileprefix_out+'.xyz'
    alc_filename=fileprefix_out+'.alc'
    gex_filename=fileprefix_out+'.gex'
    print('Writing:')
    print(xyz_filename)
    print(alc_filename)
    print(gex_filename)
    normalize_with_TxDipoleMoment_XYZ(xyz, gex)
    scale_to_picoVolt(xyz)
    xyz['model_info']['data type']='DTSKYTEMMIN1'
    libaarhusxyz.dump(xyz, xyz_filename, alcfile=alc_filename)
    correctGEX(gex)
    libaarhusxyz.dump_gex(gex, gex_filename)


def resample_df(df_in, sampling_rate=1, time_key='epoch_time', bspline=False, std_scaling=3, min_std=None, verbose=False):
    df=copy.deepcopy(df_in.drop(columns=['ChannelsNumber']))
    t_min=np.ceil(df[time_key].min()/1e3)*1e3
    t_max=np.floor(df[time_key].max()/1e3)*1e3   
    t = np.arange(t_min, t_max, sampling_rate*1000)
    df.insert(df.shape[-1], 'interpolated', np.zeros(df.shape[0]).astype(bool))
    column_names=df.columns
    time_df=pd.DataFrame(data=np.ones((len(t), df.shape[1]))*np.nan, columns=column_names)
    time_df[time_key]=t
    calc_Date_Time_from_epochtime(time_df)
    time_df['interpolated'].loc[:]=True
    df_out=pd.concat([df, time_df], axis=0).sort_values(time_key).reset_index(drop=True)
    column_names_to_nanfill=['Flight', 'Line']
    column_names_to_drop=['Date', 'Time', 'interpolated', time_key]
    column_names_to_interpolate=column_names.drop(column_names_to_nanfill+column_names_to_drop)
    df_out.loc[:,column_names_to_nanfill] = df_out.loc[:,column_names_to_nanfill].fillna(method='ffill')
    for col in column_names_to_interpolate:
        if ('Gate_Ch' in col) and bspline:
            # The following will do cross-sounding b-spline interpolation
            # not accounting for other gates (experimental) 
            # therefore default not active
            if verbose: 
                print('doing B-spline interpolation on: {}'.format(col))
            # b-spline fitting
            not_nan= ~(df_out.loc[:,time_key].isna()) & ~(df_out.loc[:,col].isna()) 
            t = df_out.loc[not_nan,time_key].values
            y = df_out.loc[not_nan,col].values
            std_key = 'STD_'+col.split('Gate_')[-1]
            stds=df_out.loc[not_nan,std_key] 
            if min_std:
                if verbose:
                    print('assume minimum std: {}'.format(min_std))
                filt=stds<min_std
                stds.loc[filt]=min_std
            uncertainties = stds/std_scaling * y
            n = len(t)
            tck = splrep(t, y, w=1./uncertainties, s=n, k=3)
            ti = df_out.loc[df_out['interpolated'],time_key]
            df_out.loc[df_out['interpolated'], col] = splev(ti, tck)
        else:
            cols = [time_key, col] 
            df_out.loc[:, col]= df_out.loc[:,cols].set_index(time_key).interpolate(method='linear').values
    return df_out[df_out.interpolated].drop(columns=['interpolated'])



def read_raw_data_from_mat(gex_file, lin_file, sps_file_dir, project_crs, verbose=True, 
                           regression_based_std=0,
                           sampling_rate=None,
                           bspline_interpolation=False):
    if verbose: print('\nreading gex file:\n{}'.format(gex_file))
    gex=libaarhusxyz.gex.parse(gex_file)
    if verbose: print('\nreading lin file:\n{}'.format(lin_file))
    lin=readLineFile(lin_file)
    if verbose: print('\nsearching for sps files.')
    gpsfile, sourcefile, navsysfile = findSPSfiles(sps_file_dir)
    mat_file = findMATfile(sps_file_dir)
    if verbose: print('\nreading aux file:\n{}'.format(navsysfile))
    aux=readAndProcessAUXdata(navsysfile, gex)
    if verbose: print('\nreading gps file:\n{}'.format(gpsfile))
    gps = readAndProcessGPSdata(gpsfile, project_crs, gex, aux)
    if verbose: print('\nreading source file:\n{}'.format(sourcefile))
    source=readSourceDataSPS(sourcefile)
    if verbose: print('\nreading .mat file:\n{}'.format(mat_file))
    data=readVoltageData(mat_file)
    if verbose: print('\nstacking the data.')
    data_stacked=stackVoltageData(data, regression_based_std=regression_based_std, verbose=verbose)
    if verbose: print('\ninterpolating gps data.')
    interpolateGPSdata(data_stacked, gps)
    if verbose: print('\nprocessing altitude data.')
    mergeVoltageTiltAlt(data_stacked, aux, cleanup=False)
    if verbose: print('\nmerging source data.')
    mergeVoltageSource(data_stacked, source)
    if verbose: print('\npreparing xyz data structure.')
    concat_all(data_stacked)
    calc_Date_Time_from_epochtime(data_stacked['Concat_all'])
    assign_lineNumber(data_stacked['Concat_all'], lin, line_key='Line', assign_flightNR=True, flight_key='Flight')
    if sampling_rate:
        if verbose: print(f'\nresampling the data to sampling rate: {sampling_rate} s.')
        data_stacked['resampled'] = resample_df(data_stacked['Concat_all'], sampling_rate=sampling_rate, bspline=bspline_interpolation, verbose=verbose)
    else:
        if verbose: print('\nnot resampling the data, keeping raw stacks.')
        data_stacked['resampled']=data_stacked['Concat_all']
    xyz = makeXYZ(data_stacked['resampled'])
    assignGateTimesDipoleMoments(xyz, gex)
    if verbose: print('\nDone!')
    return xyz, gex

def read_raw_data_from_mat_files(gex_file, line_file, sps_file_dirs, project_crs,
                                 regression_based_std=0,
                                 sampling_rate=None, 
                                 bspline_interpolation=False, 
                                 verbose=False):
    # function to read an entire survey from a list of sps dirs that hold both .sps and .mat files
    xyz={}
    start = time.time()
    for sps_file_dir in sps_file_dirs:
        print('\n**************  Reading sps dir:  ****************\n{}'.format(sps_file_dir))
        tmp, gex = read_raw_data_from_mat(gex_file, line_file, sps_file_dir, project_crs, 
                                          sampling_rate=sampling_rate, 
                                          regression_based_std=regression_based_std,
                                          bspline_interpolation=bspline_interpolation,
                                          verbose=verbose)
        end = time.time()
        xyz = concatXYZ(xyz, tmp)
    print("\n\nTime used for reading all data {} sec.".format(end-start))
    return xyz, gex
