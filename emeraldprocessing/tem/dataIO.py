# -*- coding: utf-8 -*-

import libaarhusxyz
import numpy as np
import pandas as pd
import geopandas as gpd
from copy import deepcopy
from .utils import getGateTimesFromGEX, inuse_moment, is_dual_moment
from .data_keys import inuse_dtype
from .setup import allowed_moments
import os
import time

def remove_initial_gates(Data, gex):  # this is essentially they apply "Apply Gex" filter, but maybe better formulated?
    for moment in allowed_moments:
        if moment in Data.layer_data.keys():
            ch_key = 'Channel'+moment.split('Gate_Ch0')[-1]
            Data.layer_data[inuse_moment(moment)].iloc[:, 0:int(gex.gex_dict[ch_key]['RemoveInitialGates'])] = 0

def makeGateTimesDipoleMoments(processing):
    gex = processing.gex
    gex_dict = gex.gex_dict
    Data = processing.xyz
    
    if 'dipolemoment_ch01' in Data.flightlines.columns:
        Data.flightlines.drop(columns=['dipolemoment_ch01'], inplace=True)
    if 'dipolemoment_ch02' in Data.flightlines.columns:
        Data.flightlines.drop(columns=['dipolemoment_ch02'], inplace=True)
    # add some fields that are needed for the processing:
    processing.GateTimes={}
    processing.ApproxDipoleMoment={}
    #Data['DipoleMoment']={}
    processing.GateTimes['Gate_Ch01']=getGateTimesFromGEX(gex, 'Channel1')[:,0]
    processing.ApproxDipoleMoment['Gate_Ch01']=gex_dict["General"]["NumberOfTurnsLM"] * gex_dict["General"]["TxLoopArea"] * gex_dict["Channel1"]["TxApproximateCurrent"]
    Data.flightlines['DipoleMoment_Ch01']=gex_dict["General"]["NumberOfTurnsLM"] * gex_dict["General"]["TxLoopArea"] * Data.flightlines['Current_Ch01']
    
    if 'Channel2' in gex_dict.keys():
        processing.GateTimes['Gate_Ch02']=getGateTimesFromGEX(gex, 'Channel2')[:,0]
        processing.ApproxDipoleMoment['Gate_Ch02']=gex_dict["General"]["NumberOfTurnsHM"] * gex_dict["General"]["TxLoopArea"] * gex_dict["Channel2"]["TxApproximateCurrent"]
        Data.flightlines['DipoleMoment_Ch02']=gex_dict["General"]["NumberOfTurnsHM"] * gex_dict["General"]["TxLoopArea"] * Data.flightlines['Current_Ch02']
        
def readWBxyz(xyz_file, gex_file, dummi_value='*', removeInititalGates=False, tilt_angle_reference=0):
    gex=libaarhusxyz.GEX(gex_file)
    
    Data = libaarhusxyz.XYZ(xyz_file, normalize=False)
    
    if 'dummy' in Data.model_info.keys():
        dummy_value=Data.model_info['dummy']
        print('dummi value found in xyz file, will override input' )
        print('using: {} from xyz file'.format(dummy_value))
    else:
        dummy_value=dummi_value
        print('no dummy value found in xyz file, will use input: {}'.format(dummy_value) )
        
    
    # fix  dummi values
    for col in Data.flightlines.columns:
        filt=Data.flightlines[col]==dummy_value
        Data.flightlines.loc[filt,col]=np.nan
    
    for key in Data.layer_data.keys():
        filt=Data.layer_data[key]==dummy_value
        Data.layer_data[key][filt]=np.nan
    
    
    # scale data to be in picovolt
    for key in ['dbdt_ch1gt', 'dbdt_ch2gt']:
        if key in Data.layer_data.keys():
            # FIXME: Scale factor here is hard coded
            Data.layer_data[key] = Data.layer_data[key] * 1e12  # the data in the xyz file is usually stored in picovolt (1e-12 V)
    Data.model_info['scalefactor'] = 1e-12
    
    #Add inuse flag if not present:
    if not('dbdt_inuse_ch1gt' in Data.layer_data.keys()):
        # FIXME: what is the right key? 'dbdt_ch1gt' or 'Gate_Ch01'?!?
        Data.layer_data['dbdt_inuse_ch1gt'] = pd.DataFrame(data=np.ones(Data.layer_data['Gate_Ch01'].shape),
                                                           columns=Data.layer_data['Gate_Ch01'].columns,
                                                           dtype=inuse_dtype,
                                                           # dtype='int16',
                                                           index=Data.layer_data['Gate_Ch01'].index)
        # in case we have data with two dipole moments, we need an additional inuse-dataframe for the second moment:
        if 'Gate_Ch02' in Data.layer_data.keys():
            Data.layer_data['dbdt_inuse_ch2gt'] = pd.DataFrame(data=np.ones(Data.layer_data['Gate_Ch02'].shape),
                                                               columns=Data.layer_data['Gate_Ch02'].columns,
                                                               dtype=inuse_dtype,
                                                               # dtype='int16',
                                                               index=Data.layer_data['Gate_Ch02'].index)
    elif ('dbdt_inuse_ch1gt' in Data.layer_data.keys()) and ('dbdt_inuse_ch2gt' in Data.layer_data.keys()):
        for col in Data.layer_data['dbdt_inuse_ch1gt'].columns:
            Data.layer_data['dbdt_inuse_ch1gt'].loc[:, col] = Data.layer_data['dbdt_inuse_ch1gt'][col].fillna(0).astype(inuse_dtype)
            # Data.layer_data['dbdt_inuse_ch1gt'].loc[:, col] = Data.layer_data['dbdt_inuse_ch1gt'][col].fillna(0).astype('int16')
        for col in Data.layer_data['dbdt_inuse_ch2gt'].columns:
            Data.layer_data['dbdt_inuse_ch2gt'].loc[:, col] = Data.layer_data['dbdt_inuse_ch2gt'][col].fillna(0).astype(inuse_dtype)
            # Data.layer_data['dbdt_inuse_ch2gt'].loc[:, col] = Data.layer_data['dbdt_inuse_ch2gt'][col].fillna(0).astype('int16')
    
    # make sure some flightline columns to be int
    # FIXME: line_no doesn't have to be an integer
    # for col in ['line_no', 'channel_no', 'fieldpolarity', 'numgates']:
    for col in ['channel_no', 'fieldpolarity', 'numgates']:
        if col in Data.flightlines.columns:
            Data.flightlines[col] = Data.flightlines[col].astype(inuse_dtype)
            # FIXME: numgates will be under a hundred in general, field polarity is is just a sign, so +1 or -1, channel_no will be <10, so why int32
            # Data.flightlines[col] = Data.flightlines[col].astype('int32')
    
    # make individual current columns for each channel
    Data.flightlines['Current_Ch01']=Data.flightlines['current']
    filt=Data.flightlines['channel_no']==2
    #Data.flightlines.loc[filt, 'Current_Ch01'] = Data.flightlines['Current_Ch01'].loc[~filt].median()
    Data.flightlines.loc[filt, 'Current_Ch01'] = np.nan
    
    
    Data.flightlines['Current_Ch02']=Data.flightlines['current']
    filt=Data.flightlines['channel_no']==1
    #Data.flightlines.loc[filt, 'Current_Ch02'] = Data.flightlines['Current_Ch02'].loc[~filt].median()
    Data.flightlines.loc[filt, 'Current_Ch02'] = np.nan
    
    # recalculate tilt and pitch to vary around 0 
    if tilt_angle_reference!=0:
        #(not around270 like in the WB export, asssuming difffernt cooridnate system)
        Data.flightlines.loc[:,'tilt_x']=270-tilt_angle_reference-Data.flightlines.loc[:,'tilt_x']
        Data.flightlines.loc[:,'tilt_y']=270-tilt_angle_reference-Data.flightlines.loc[:,'tilt_y']
    
    # make sure there is no tilt coorection of Gate Values
    # WB is asumin uncorrected Gate values
    # so we manually un-correct the gate values for Tx and Rx pitc and roll
    cos_tilt_x=np.cos(Data.flightlines.tilt_x/180*np.pi)
    cos_tilt_y=np.cos(Data.flightlines.tilt_y/180*np.pi)
    for key in ['dbdt_ch1gt', 'dbdt_ch2gt']:
        if key in Data.layer_data.keys():
            for col in Data.layer_data[key].columns:
                Data.layer_data[key].loc[:,col] = Data.layer_data[key].loc[:,col] * (cos_tilt_x * cos_tilt_y)**2
    
    
    
    ## make sure we write vertical seperation Rx-Tx
    #Data.flightlines['TxRxVertSep']=Data.flightlines.rx_altitude - Data.flightlines.tx_altitude
    
    # fix datum format:
    for n in range(Data.flightlines.shape[0]):
        Data.flightlines['date'].iloc[n] = Data.flightlines['date'].iloc[n].replace('-','/')
    
    # add some fields that are needed for the processing:    
    
    # rename columns to be consistent with WB internal naming (ALC file requirements)
    flightline_reaname_dict={'date':'Date',
                         'time':'Time',
                         'line_no':'Line',
                         'elevation':'Topography',
                         'tx_altitude':'TxAltitude',
                         'utmx':'UTMX', 
                         'utmy':'UTMY',
                         'tilt_x':"TxPitch",
                         'tilt_y':"TxRoll",
                        }
    Data.flightlines = Data.flightlines.rename(columns=flightline_reaname_dict)
    
    # rename keys in Data.layer_data].keys() to be consistent with WB naming (ALC file requirements)
    data_rename_dict={'dbdt_ch1gt':'Gate_Ch01', 
                  'dbdt_ch2gt':'Gate_Ch02', 
                  'dbdt_std_ch1gt':'STD_Ch01', 
                  'dbdt_std_ch2gt':'STD_Ch02',
                  'dbdt_inuse_ch1gt':'InUse_Ch01', 
                  'dbdt_inuse_ch2gt':'InUse_Ch02'
                 }
    for key in data_rename_dict.keys():
        Data.layer_data[data_rename_dict[key]] = Data.layer_data.pop(key)
    
    if removeInititalGates:
        remove_initial_gates(Data, gex)
    
    # Add fields for later visualization of cullings and culling reason
    Data.flightlines.insert(len(Data.flightlines.columns), "disable_reason", ['none' for k in range(len(Data.flightlines))] )
    # FIXME: is int16 the right dtype here?
    Data.flightlines.insert(len(Data.flightlines.columns), "coverage", np.int16( np.zeros( len(Data.flightlines) ) ) )

    return Data, gex


def splitMoments(data, gex):    
    dataKey='data'
    stdKey='datastd'
    
    # some cleanup:
    filt=data.xyz.layer_data[dataKey]==data.xyz.model_info['dummy']
    data.xyz.layer_data[dataKey][filt]=np.nan
    
    if stdKey in data.xyz.layer_data.keys():
        filt=data.xyz.layer_data[stdKey]==data.xyz.model_info['dummy']
        data.xyz.layer_data[stdKey][filt]=np.nan
    
    # gate times:
    for key in data.xyz.model_info.keys():
        if 'times' in key.lower():
            times_key=key
    data.xyz.model_info['gate_times']=np.array(data.xyz.model_info[times_key])
    
    # indices for splitting
    idxLM=np.zeros(data.xyz.model_info['gate_times'].shape, dtype=bool)
    idxHM=np.zeros(data.xyz.model_info['gate_times'].shape, dtype=bool)
    
    for n, gate in enumerate(data.xyz.model_info['gate_times']):
        if np.round(np.log10(gate), decimals=4) in np.round(np.log10(getGateTimesFromGEX(gex, channel='Channel1')[:,0]), decimals=4):
            idxLM[n]=True
        elif np.round(np.log10(gate), decimals=3) in np.round(np.log10(getGateTimesFromGEX(gex, channel='Channel2')[:,0]), decimals=3) and is_dual_moment(gex):
            idxHM[n]=True
            
    data.GateTimes={'Gate_Ch01':data.xyz.model_info['gate_times'][idxLM]}
    if is_dual_moment(gex):
        data.GateTimes['Gate_Ch02'] = data.xyz.model_info['gate_times'][idxHM]

    data.xyz.layer_data['Gate_Ch01']=data.xyz.layer_data[dataKey].iloc[:,idxLM]
    if is_dual_moment(gex):
        data.xyz.layer_data['Gate_Ch02']=data.xyz.layer_data[dataKey].iloc[:,idxHM]
    
    if stdKey in data.xyz.layer_data.keys():
        data.xyz.layer_data['STD_Ch01']=data.xyz.layer_data[stdKey].iloc[:,idxLM]
        if is_dual_moment(gex):
            data.xyz.layer_data['STD_Ch02']=data.xyz.layer_data[stdKey].iloc[:,idxHM]
    


def readWBinversionDataExport(dirname_xyz, fileprefix, gex_pointer):
    fullfilename_xyz_dat = os.path.join(dirname_xyz, fileprefix+'_dat.xyz')
    fullfilename_xyz_syn = os.path.join(dirname_xyz, fileprefix+'_syn.xyz')
    fullfilename_xyz_inv = os.path.join(dirname_xyz, fileprefix+'_inv.xyz')

    return readWBinversionDataExportFull(
        fullfilename_xyz_dat,
        fullfilename_xyz_syn,
        fullfilename_xyz_inv,
        gex_pointer)

class InvData(libaarhusxyz.Survey):
    pass

class InversionResult(object):
    def __init__(self, data, synthetic, model):
        self.data = data
        self.synthetic = synthetic
        self.model = model

def readWBinversionDataExportFull(
        fullfilename_xyz_dat,
        fullfilename_xyz_syn,
        fullfilename_xyz_inv,
        gex_pointer):
    print('Reading inversion data:\n'
          + "\n".join([fullfilename_xyz_dat,
                       fullfilename_xyz_syn,
                       fullfilename_xyz_inv]))
    res = InversionResult(
        InvData(xyz=fullfilename_xyz_dat, gex=gex_pointer, normalize=False),
        InvData(xyz=fullfilename_xyz_syn, gex=gex_pointer, normalize=False),
        InvData(xyz=fullfilename_xyz_inv, gex=gex_pointer, normalize=False))

    if "gate_ch01" in res.data.xyz.layer_data:
        # This is not a workbench inversion, already normalized
        for dataset in (res.data, res.synthetic):
            dataset.GateTimes={'Gate_Ch01':dataset.xyz.model_info['gate times for channel 1']}
            if 'gate times for channel 2' in dataset.xyz.model_info:
                dataset.GateTimes['Gate_Ch02'] = dataset.xyz.model_info['gate times for channel 2']
            dataset.xyz.layer_data["Gate_Ch01"] = dataset.xyz.layer_data.pop("gate_ch01")
            if "gate_ch02" in dataset.xyz.layer_data:
                dataset.xyz.layer_data["Gate_Ch02"] = dataset.xyz.layer_data.pop("gate_ch02")
            dataset.xyz.layer_data["STD_Ch01"] = dataset.xyz.layer_data.pop("std_ch01")
            if "std_ch02" in dataset.xyz.layer_data:
                dataset.xyz.layer_data["STD_Ch02"] = dataset.xyz.layer_data.pop("std_ch02")
    else:        
        splitMoments(res.data, res.data.gex)
        splitMoments(res.synthetic, res.synthetic.gex)
    
        if is_dual_moment(res.data.gex):
            stdkeys=['STD_Ch01', 'STD_Ch02']
        else:
            stdkeys=['STD_Ch01']

        for stdkey in stdkeys:
            if res.data.xyz.layer_data[stdkey].min(axis=1).min()>1:
                res.data.xyz.layer_data[stdkey]= res.data.xyz.layer_data[stdkey]-1
    
    return res


def readSkyTEMxyz(xyz_file, alc_file=None, gex_file=None, removeInititalGates=True):
    assert gex_file is not None, "Please specify gex_file"
    start = time.time()
    print('=============== Reading SkyTEM xyz data ===============')
    print('  - Reading gex file.')
    gex = libaarhusxyz.GEX(gex_file)
    print('  - Reading xyz file.')
    Data = libaarhusxyz.XYZ(xyz_file, alcfile=alc_file, naming_standard="alc", normalize=True)
    
    print('  - building xyz dictionary')
    if 'scalefactor' not in Data.model_info:
        Data.model_info['scalefactor']=1e-12  # the data in the xyz file is usually stored in picovolt (1e-12 V)
        
    #Add inuse flag if not present:
    if not('InUse_Ch01' in Data.layer_data.keys()):
        Data.layer_data['InUse_Ch01'] = pd.DataFrame(data=np.ones(Data.layer_data['Gate_Ch01'].shape),
                                                     columns=Data.layer_data['Gate_Ch01'].columns,
                                                     dtype=inuse_dtype,
                                                     # dtype='int16',
                                                     index=Data.layer_data['Gate_Ch01'].index)
        # in case we have data with two dipole moments, we need an additional inuse-dataframe for the second moment:
        # assume that if the inuse doesn't exist for one moment than it probably doesn't exist for the other if it exists either
        if 'Gate_Ch02' in Data.layer_data.keys():
            Data.layer_data['InUse_Ch02'] = pd.DataFrame(data=np.ones(Data.layer_data['Gate_Ch02'].shape),
                                                         columns=Data.layer_data['Gate_Ch02'].columns,
                                                         dtype=inuse_dtype,
                                                         # dtype='int16',
                                                         index=Data.layer_data['Gate_Ch02'].index)
    
    if removeInititalGates:
        remove_initial_gates(Data, gex)

    if "disable_reason" not in Data.flightlines.columns:
        Data.flightlines.insert(len(Data.flightlines.columns), "disable_reason", ['none' for k in range(len(Data.flightlines))] )
    if "coverage" not in Data.flightlines.columns:
        # FIXME: is int16 the right dtype here?
        Data.flightlines.insert(len(Data.flightlines.columns), "coverage", np.int16( np.zeros( len(Data.flightlines) ) ) )
    
    if 'Alt' in Data.flightlines.columns:
        Data.flightlines.insert(len(Data.flightlines.columns), "TxZ", Data.flightlines['Alt'])
    
    print('  - Done reading the data package!')
    end = time.time()
    print(f"  - Time used to read and import the data package: {end - start} sec.\n")
    return Data, gex

def dumpDataframe2shape(df, outputShapeFileName, crs=None):
    if ('UTMX' in df.columns) and  ('UTMY' in df.columns):
        poskeys=[ 'UTMX', 'UTMY']
    if ('utmx' in df.columns) and  ('utmy' in df.columns):
        poskeys=[ 'utmx', 'utmy']
    else:
        raise Exception('no UTM coordinates found  (utmx/utmy)')
    gdf = gpd.GeoDataFrame(df)
    gdf.set_geometry(
        gpd.points_from_xy(gdf[poskeys[0]], gdf[poskeys[1]]),
        inplace=True, 
        crs=crs)
    #gdf.drop([pos_keys[0], pos_keys[1]], axis=1, inplace=True)  # optional
    gdf.to_file(outputShapeFileName)


def writeSkyTEMxyz(DataOut, xyz_file, alc_file=None, shp_file=None, crs=None):
    Data=deepcopy(DataOut)
    if shp_file:
        Data.flightlines['coverage']=Data.layer_data['InUse_Ch01'].sum(axis=1)
        if 'InUse_Ch02' in Data.layer_data.keys():
            Data.flightlines['coverage']+=Data.layer_data['InUse_Ch02'].sum(axis=1)
        dumpDataframe2shape(Data.flightlines, shp_file, crs=crs)
    
    # do some cleanup prior to dumping to file
    columns_to_strip = ['disable_reason', 'coverage', 'geometry']
    for column in columns_to_strip:
        if column in Data.flightlines.columns:
            Data.flightlines.drop(column, inplace=True, axis=1)  # strip axuliary columns from dataframe before writing to disk

    Data.dump(xyz_file, alcfile=alc_file)
