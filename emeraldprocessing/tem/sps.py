# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import copy
import os
import glob
from pyproj import transform, Proj
import time
import warnings
from scipy.interpolate import interp1d
import libaarhusxyz


def calcUTMcoordinates(gps, proj_crs, poskeys=['lon_nmea', 'lat_nmea']):
    epsg_out = f'epsg:{proj_crs}'
    gps['long_deg'] = np.trunc(pd.to_numeric(gps[poskeys[0]])/100)
    gps['long_min'] = pd.to_numeric(gps[poskeys[0]])-(gps['long_deg']*100)
    gps['long_dd'] = gps['long_deg'] + gps['long_min']/60

    gps['lat_deg'] = np.trunc(pd.to_numeric(gps[poskeys[1]])/100)
    gps['lat_min'] = pd.to_numeric(gps[poskeys[1]])-(gps['lat_deg']*100)
    gps['lat_dd'] = gps['lat_deg'] + gps['lat_min']/60
    
    inProj = Proj('epsg:4326')  # Spherical Mercator
    outProj = Proj(epsg_out)
    x1, y1 = pd.to_numeric(gps['lat_dd']), pd.to_numeric(gps['long_dd'])
    gps['utmx'], gps['utmy'] = transform(inProj, outProj, x1, y1)

def read_concat_DGPS_sps_files(sps_file_filter, proj_crs):
    sps_data_root_dir = sps_file_filter.split('*')[0]
    gps_file_ending = sps_file_filter.split('*')[-1]

    sps_dir = {}

    for root, dirs, files in os.walk(sps_data_root_dir):
        for file in files:
            if file.endswith(gps_file_ending):
                print('reading: {}'.format(os.path.join(root, file)))
                sps_dir[file] = readGPSfromSPS(os.path.join(root, file))
    
    gps_df = pd.DataFrame()
    for file in sps_dir.keys():
        gps_df = pd.concat([gps_df, sps_dir[file]['gd1']])
        if len(sps_dir[file]['gd2']) > 0:
            gps_df = pd.concat([gps_df, sps_dir[file]['gd2']])
    calcUTMcoordinates(gps_df, proj_crs,  poskeys=['lon_nmea', 'lat_nmea'])
    calcEpochTime(gps_df)    
    
    return gps_df

def readLASERfromSPS(fullfilename_sps):
    dic = {'yy': [],
           'mo': [],
           'dd': [],
           'hh': [],
           'mi': [],
           'ss': [],
           'ms': []}
    
    he1 = copy.deepcopy(dic)
    he1['dist'] = []
    he2 = copy.deepcopy(he1)
    
    tl1 = copy.deepcopy(dic)
    tl1['tilt_x'] = []
    tl1['tilt_y'] = []
    tl2 = copy.deepcopy(tl1)
    
    with open(fullfilename_sps) as sps:
        nlines = 0
        for line in sps:
            nlines = nlines+1
            words = line.split()
            if 'HE1' in words[0]:
                for n, k in enumerate(dic.keys()):
                    he1[k].append(int(words[n+1]))
                he1['dist'].append(float(words[8]))
            
            if 'HE2' in words[0]:
                for n, k in enumerate(dic.keys()):
                    he2[k].append(int(words[n+1]))
                he2['dist'].append(float(words[8]))
            
            if 'TL1' in words[0]:
                for n, k in enumerate(dic.keys()):
                    tl1[k].append(int(words[n+1]))
                tl1['tilt_x'].append(float(words[8]))
                tl1['tilt_y'].append(float(words[9]))
            
            if 'TL2' in words[0]:
                for n, k in enumerate(dic.keys()):
                    tl2[k].append(int(words[n+1]))
                tl2['tilt_x'].append(float(words[8]))
                tl2['tilt_y'].append(float(words[9]))                
            
    return {
            'he1': pd.DataFrame(he1),
            'he2': pd.DataFrame(he2),
            'tl1': pd.DataFrame(tl1),
            'tl2': pd.DataFrame(tl2),
            }

def readAndPrepareLASERfromSPS(fullfilename_sps):
    laserdata = readLASERfromSPS(fullfilename_sps)

    filt = laserdata['he1']['dist'] >= 99999
    laserdata['he1']['dist'].loc[filt] = np.nan
    filt = laserdata['he2']['dist'] >= 99999
    laserdata['he2']['dist'].loc[filt] = np.nan

    # Establish common time base for interpolation
    for key in laserdata.keys():
        calcEpochTime(laserdata[key])

    # Average doublicated measurements
    for key in ['he1', 'he2']:
        laserdata[key] = laserdata[key].groupby(by=['datetime'], as_index=False).mean()

    return laserdata

def readSourceDataSPS(fullfilename_sps):
    dic = {'yy': [],
           'mo': [],
           'dd': [],
           'hh': [],
           'mi': [],
           'ss': [],
           'ms': []}

    txd = copy.deepcopy(dic)

    # below comes from SkyTEM system guide v2.5 (2011)
    txd_keys = {'NumberOfShots': 'int',
                'NumberOfProtemErrors': 'int',
                'FirstProtemError': 'int',
                'Nplus': 'int',
                'Nminus': 'int',
                'VoltageOn': 'float',
                'VoltageOff': 'float',
                'TxTemperature': 'float',
                'VersionNo': 'int',
                'NumberOfDatasets': 'int',
                'NumberOfSeries': 'int',
                'MeanCurrent': 'float',
                'MaxCurrent': 'float',
                'MinCurrent': 'float',
                'RMSCurrent': 'float'}

    for key in txd_keys.keys():
        txd[key] = []

    sof = []
    ver = []
    mrk = []

    with open(fullfilename_sps) as sps:
        nlines = 0
        for line in sps:
            nlines = nlines+1
            words = line.split()
            if len(words) > 0:
                if 'SOF' in words[0]:
                    sof.append(line)
                if 'MRK' in words[0]:
                    mrk.append(line)
                if 'VER' in words[0]:
                    ver.append(line)
                if 'TXD' in words[0]:
                    for n, k in enumerate(dic.keys()):
                        txd[k].append(int(words[n+1]))
                    for n, key in enumerate(txd_keys.keys()):
                        txd[key].append(words[8+n])
    
    TXD = pd.DataFrame(txd)
    for key in txd_keys.keys():
        TXD[key] = TXD[key].astype(txd_keys[key])
    
    calcEpochTime(TXD)
    
    # Split low and high moment (LM/HM)
    if TXD.MeanCurrent.loc[0::2].mean() > TXD.MeanCurrent.loc[1::2].mean():
        TXD_HM = TXD.loc[0::2, :].reset_index(drop=True)
        TXD_LM = TXD.loc[1::2, :].reset_index(drop=True)
    else:
        TXD_HM = TXD.loc[1::2, :].reset_index(drop=True)
        TXD_LM = TXD.loc[2::2, :].reset_index(drop=True)
    
    return {'VER': ver,
            'MRK': mrk,
            'SOF': sof,
            'TXD': TXD,
            'TXD_LM': TXD_LM,
            'TXD_HM': TXD_HM
            }

def readGPSfromSPS(fullfilename_sps):
    template = {'yy': [],
               'mo': [],
               'dd': [],
               'hh': [],
               'mi': [],
               'ss': [],
               'ms': [],
               'lat_nmea': [],
               'lon_nmea': [],
               'z_nmea': [],
               }
    
    gps_keys = ['gp1', 'gp2', 'gd1', 'gd2']
    output = {}
    for key in gps_keys:
        output[key] = copy.deepcopy(template)
    
    with open(fullfilename_sps) as sps:
        nlines = 0
        for line in sps:
            nlines = nlines+1
            words = line.split()
            
            for key in gps_keys:
                if key.upper() in words[0]:
                    for n, k in enumerate(['yy', 'mo', 'dd']):
                        output[key][k].append(words[n+1])
                    hh   = int(words[4])
                    mi   = int(words[5])
                    sec  = int(words[6])
                    msec = int(words[7])
                    output[key]['hh'].append(hh)
                    output[key]['mi'].append(mi)
                    output[key]['ss'].append(sec)
                    output[key]['ms'].append(msec)
                    output[key]['lat_nmea'].append(float(words[8]))
                    output[key]['lon_nmea'].append(float(words[9]))
                    output[key]['z_nmea'].append(float(words[17]))
    for key in gps_keys:
        output[key] = pd.DataFrame(output[key])
        
    return output

def readAndPrepareGPSfromSPS(fullfilename_sps, proj_crs):
    gpsdata = readGPSfromSPS(fullfilename_sps)

    for key in ['gp1', 'gp2', 'gd1', 'gd2']:
        calcUTMcoordinates(gpsdata[key], proj_crs, poskeys=['lon_nmea', 'lat_nmea'])

    for key in gpsdata.keys():
        calcEpochTime(gpsdata[key])

    return gpsdata
    
def calcEpochTime(df):
    if not('datetime' in df.columns):
        date = df.yy.astype(str) + '-' + df.mo.astype(str) + '-' + df.dd.astype(str)
        sec = df.ss.astype(str) + '.' + df.ms.astype(str)
        times = df.hh.astype(str) + ':' + df.mi.astype(str) + ':' + sec 
        df['datetime'] = pd.to_datetime(date + ' ' + times)
    df['epoch_time'] = pd.to_numeric(df['datetime'])/1e6

def interpolate_df2_on_df1(df1, df2, common_key='epoch_time', key2interpolate_df2='utmx', key4interpolated_df1='utmx_1'):
    f = interp1d(df2[common_key], df2[key2interpolate_df2], bounds_error=False, fill_value=np.nan)
    df1[key4interpolated_df1] = f(df1[common_key])

def calcSpeed(df, poskeys=['utmx_1', 'utmy_1'], timekey='epoch_time'):
    dx = df[poskeys[0]].diff()
    dy = df[poskeys[1]].diff()
    ds = (dx**2 + dy**2)**.5
    dt = df[timekey].diff()
    df['speed'] = ds/(dt/1000)


def calcXdist(df, poskeys=['utmx_1', 'utmy_1']):
    if not('dx' in df.columns) or not('dy' in df.columns):
        df['dx'] = df[poskeys[0]].diff()
        df['dy'] = df[poskeys[1]].diff()
    df['xdist'] = ((df.dx**2+df.dy**2)**.5).cumsum()


def calcHeading(df, poskeys=['utmx_1', 'utmy_1']):
    # frame heading
    df['dx'] = df[poskeys[0]].diff()
    df['dy'] = df[poskeys[1]].diff()
    heading = np.arctan2(df['dx'], df['dy'])
    heading[0] = 0
    df['heading'] = heading/np.pi*180


def calcLaserPointPos(df, 
                      poskeys=['utmx_1', 'utmy_1', 'z_nmea'], 
                      tiltkeys=['tilt_x1', 'tilt_y1'], 
                      distkey='dist1', 
                      outkeys=['utmx_laser1', 'utmy_laser1', 'z_laser1'],
                      calcaux=False):
    
    # frame heading
    calcHeading(df, poskeys)
    heading = df['heading']/180*np.pi

    tilt_x = df[tiltkeys[0]]/180*np.pi
    tilt_y = df[tiltkeys[1]]/180*np.pi
    dist = df[distkey]
    
    # relative coordinate changes
    dz_r = dist * np.cos(tilt_x) * np.cos(tilt_y)  # relative vertical distance
    dx_r =  np.sin(tilt_x) * dist  # relative inline distance
    dy_r = -np.sin(tilt_y) * dist  # relative xline distance
    dh_r = (dx_r**2 + dy_r**2)**.5   # relative horizontal distance
    head_r = np.arctan2(dy_r, dx_r)  # local heading angle
    
    dutmx = np.sin(heading+head_r)*dh_r
    dutmy = np.cos(heading+head_r)*dh_r
    
    if calcaux:
        df['dx_r'] = dx_r
        df['dy_r'] = dy_r
        df['dh_r'] = dh_r
        df['head_r'] = head_r/np.pi*180
        df['dutmx'] = dutmx
        df['dutmy'] = dutmy
        df['dz_r'] = dz_r
    
    df[outkeys[0]] = df[poskeys[0]] + dutmx
    df[outkeys[1]] = df[poskeys[1]] + dutmy
    df[outkeys[2]] = df[poskeys[2]] - dz_r
    
    df.drop(index=df.index[0], axis=0, inplace=True)


def readLineFile(fullfilename_lin):
    datetime = []
    line_no = []
    easting = []
    northing = []
    label = []
    flight = []
    comments = []
    with open(fullfilename_lin) as sps:
        nlines = 0
        for line in sps:
            nlines = nlines+1
            words = line.split(' ')
            date = '-'.join(words[0].split('-')[::-1])
            datetime.append(' '.join([date, words[1]]))
            line_no.append(words[2])
            easting.append(words[3])
            northing.append(words[4])
            label.append(words[5])
            flight.append(words[6])
            comments.append(' '.join(words[7:])[:-1])

    lin_file_df = pd.DataFrame({'datetime': np.array(datetime).astype('datetime64[s]'),
                               'line_no': np.array(line_no).astype(float),
                               'easting': np.array(easting).astype(float),
                               'northing': np.array(northing).astype(float),
                               'label': np.array(label),
                               'flight': np.array(flight).astype(float),
                               'comments': np.array(comments)
                              })  
    return lin_file_df

def assign_lineNumber(df, lin_file_df, verbose=False, line_key='line_no', assign_flightNR=False, flight_key='Flight'):
    linekey = 'line_no'
    flightkey = 'flight'
    epoch_time_key = 'epoch_time'
    if line_key not in df.columns:
        df.insert(0, line_key, np.zeros(len(df)))
    if assign_flightNR and not(flight_key in df.columns):
        df.insert(0, flight_key, np.zeros(len(df)))
    
    if not(epoch_time_key in df.columns):
        calcEpochTime(df)
    if not(epoch_time_key in lin_file_df.columns):
        calcEpochTime(lin_file_df)
    
    for line in lin_file_df[linekey].unique():
        filt = lin_file_df[linekey] == line
        start = lin_file_df.loc[filt, epoch_time_key].min()
        end = lin_file_df.loc[filt, epoch_time_key].max()
        filt2 = (df[epoch_time_key] > start) & (df[epoch_time_key] < end)
        if verbose:
            print('line:{0}, start:{1}, end:{2}, n:{3}'.format(line, start, end, filt2.sum()))
        df.loc[filt2, line_key] = line
        if assign_flightNR:
            df.loc[filt2, flight_key] = lin_file_df.loc[filt, flightkey].iloc[0]


def merge_laser_gps_source(laserdata, gpsdata, sourcedata, gps_key='gd1'):
    df = laserdata['he1'].rename(columns={'dist': 'dist1'})
    interpolate_df2_on_df1(df, laserdata['he2'], common_key='epoch_time', key2interpolate_df2='dist', key4interpolated_df1='dist2')

    interpolate_df2_on_df1(df, laserdata['tl1'], common_key='epoch_time', key2interpolate_df2='tilt_x', key4interpolated_df1='tilt_x1')
    interpolate_df2_on_df1(df, laserdata['tl1'], common_key='epoch_time', key2interpolate_df2='tilt_y', key4interpolated_df1='tilt_y1')

    interpolate_df2_on_df1(df, laserdata['tl2'], common_key='epoch_time', key2interpolate_df2='tilt_x', key4interpolated_df1='tilt_x2')
    interpolate_df2_on_df1(df, laserdata['tl2'], common_key='epoch_time', key2interpolate_df2='tilt_y', key4interpolated_df1='tilt_y2')
    
    for key in ['utmx', 'utmy', 'z_nmea']:
        interpolate_df2_on_df1(df, gpsdata[gps_key], common_key='epoch_time', key2interpolate_df2=key, key4interpolated_df1=key)
    
    for key in ['VoltageOn', 'VoltageOff', 'TxTemperature', 'MeanCurrent', 'MaxCurrent', 'MinCurrent']:
        for moment in ['LM', 'HM']:
            interpolate_df2_on_df1(df, sourcedata['TXD_'+moment], common_key='epoch_time', key2interpolate_df2=key, key4interpolated_df1=key+'_'+moment)
    
    return df


def findSPSfiles(dirname_sps):
    spsfiles = glob.glob(os.path.join(dirname_sps, '*.sps'))
    for file in spsfiles:
        if 'GP1' in file:
            gpsfile = file
        elif 'PaPc' in file:
            sourcefile = file
        elif 'NavSys' in file:
            navsysfile = file
    if not('gpsfile' in locals()):
        print('No gps file found, will use NavSys file')
        gpsfile = navsysfile
    print('found the follwing files:')
    print('navsysfile: {}'.format(navsysfile))
    print('gpsfile: {}'.format(gpsfile))
    print('sourcefile: {}'.format(sourcefile))
    return gpsfile, sourcefile, navsysfile

def readAndMergeSPSfiles(dirname_sps, linfile, proj_crs, verbose=True):
    if verbose: print('searching for sps files.')
    gpsfile, sourcefile, navsysfile = findSPSfiles(dirname_sps)
    
    if not('gpsfile' in locals()):
        raise Exception("didn't find a gps SPS file (*_GP1.sps)")
    if not('sourcefile' in locals()):
        raise Exception("didn't find a source SPS file (*_PaPc.sps)")
    if not('navsysfile' in locals()):
        raise Exception("didn't find NavSys SPS file (*NavSys*.sps)")
    
    if verbose: print('reading laser data.')
    laserdata = readAndPrepareLASERfromSPS(navsysfile)
    
    if verbose: print('reading gps data.')
    gpsdata = readAndPrepareGPSfromSPS(gpsfile, proj_crs=proj_crs)
    if gpsdata['gd1'].shape[0] > 0:
        gps_key = 'gd1'
    elif gpsdata['gd2'].shape[0] > 0:
        gps_key = 'gd2'
    elif gpsdata['gp1'].shape[0] > 0:
        gps_key = 'gp1'
    elif gpsdata['gp2'].shape[0] > 0:
        gps_key = 'gp2'
    else:
        raise Exception('Did not find any gps data in file')
    print('Using gpskey: {}'.format(gps_key))
    
    if verbose: print('reading source data.')
    sourcedata = readSourceDataSPS(sourcefile)
    
    if verbose: print('merging data.')
    df = merge_laser_gps_source(laserdata, gpsdata, sourcedata, gps_key=gps_key)
    
    if verbose: print('doing some math.')
    filt = df['utmx'].isna() | df['utmy'].isna()
    df.drop(index=df.loc[filt].index, inplace=True)

    calcSpeed(df, poskeys=['utmx', 'utmy'])
    calcLaserPointPos(df, 
                      poskeys=['utmx', 'utmy', 'z_nmea'], 
                      tiltkeys=['tilt_x1', 'tilt_y1'], 
                      distkey='dist1', 
                      outkeys=['utmx_laser1', 'utmy_laser1', 'z_laser1'],
                      calcaux=True)

    calcLaserPointPos(df, 
                      poskeys=['utmx', 'utmy', 'z_nmea'], 
                      tiltkeys=['tilt_x2', 'tilt_y2'], 
                      distkey='dist2', 
                      outkeys=['utmx_laser2', 'utmy_laser2', 'z_laser2'],
                      calcaux=True)
    
    if verbose: print('reading line file.')
    lin_file_df = readLineFile(linfile)
    
    if verbose: print('assigning line numbers.')
    assign_lineNumber(df, lin_file_df)
    
    if verbose: print('some cleanup.')
    df.drop(columns=["yy", "mo", "dd", "hh", "mi", "ss", "ms"], inplace=True)
    
    df = df.rename(columns={"utmx_1": "utmx", "utmy_1": "utmy"})
    df.insert(df.shape[1], 'tilt_x', df[['tilt_x1', 'tilt_x2']].mean(axis=1))
    df.insert(df.shape[1], 'tilt_y', df[['tilt_y1', 'tilt_y2']].mean(axis=1))
    df.insert(df.shape[1], 'topo', df[['z_laser1', 'z_laser2']].mean(axis=1))
    df.insert(df.shape[1], 'tx_altitude', df.z_nmea - df.topo)
    df["date"] = df.datetime.dt.strftime("%Y-%m-%d")
    df["time"] = df.datetime.dt.strftime("%H:%M:%S.%f")
    
    return df
    

def readAndMergeSPSfilesXYZ(dirname_sps, linfile, proj_crs=32632, verbose=True):
    df = readAndMergeSPSfiles(dirname_sps, linfile, proj_crs=proj_crs, verbose=verbose)
    df.rename(columns={'tilt_x': 'TxPitch',
                       'tilt_y': 'TxRoll',
                       'topo': 'Topography',
                       'tx_altitude': 'TxAltitude',
                       'utmx': 'UTMX',
                       'utmy': 'UTMY',
                       'line_no': 'Line',
                       'time': 'Time'
                       }, inplace=True)
    df['Date'] = df['date'].str.replace('-', '/')
    return libaarhusxyz.XYZ({"flightlines": df, "layer_data": {}, "model_info": {}})

def raiseExceptionOrWarning(message, raiseException):
    if raiseException: 
        raise Exception(message)
    else:
        print(message)

def check_linefile(linfile, verbose=True, raiseException=True):
    if verbose: print('Checking line file:\n{}\n'.format(linfile))
    line_df = readLineFile(linfile)
    calcEpochTime(line_df)
    filt = line_df.epoch_time.diff() <= 0
    if filt.sum() > 0:
        idx = line_df.loc[filt].index
        raiseExceptionOrWarning('Found non increasing time step in line: {}'.format(line_df.loc[idx]), raiseException)
        
    lines_checked = []
    for line in line_df.line_no.unique():
        filt = line_df.line_no == line
        df = line_df.loc[filt, :]
        if len(df) > 2:
            raiseExceptionOrWarning('Line: {0} is defined multiple times:\n{1}'.format(line, df), raiseException)
        elif df.epoch_time.iloc[0] > df.epoch_time.iloc[1]:
            raiseExceptionOrWarning('Start-time before End-time', raiseException)
        else:
            lines_checked.append(line)
            if verbose: print('\nLine: {0} Start: {1}, End: {2}'.format(line, df.datetime.iloc[0], df.datetime.iloc[1]))
        if not('start' in df.comments.iloc[0].lower()):
            raiseExceptionOrWarning('Start for line: {0} is not properly defined in comments: \n{1}'.format(line, df), raiseException)
        elif not('end' in df.comments.iloc[1].lower()):
            raiseException('End for line: {0} is not properly defined in comments: \n{1}'.format(line, df), raiseException)
