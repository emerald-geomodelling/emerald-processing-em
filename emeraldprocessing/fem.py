#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:22:34 2022


"""
import numpy as np
import pandas as pd
import geopandas as gpd
import time
import matplotlib.pyplot as plt
from emeraldprocessing import tempfilename
import os

import copy


def fileFixer_in(filename_in, filename_out):
    fid_in=open(filename_in,"r")
    lines_in = fid_in.readlines()
    fid_in.close()
    fid_out = open(filename_out, "w")
    flights=[]
    dates=[]
    flightlines=[]
    for line in lines_in:
        if line.startswith("//Flight"):
            flights.append(line)
        elif line.startswith("//Date"):
            dates.append(line)
        elif line.startswith("Line"):
            flightlines.append(line)
        elif line.startswith("/"):
            fid_out.write(line[1:])
        else:
            fid_out.write(line)
    fid_out.close()
    return flights, dates, flightlines
    
def fileFixer_out(filename_in, filename_out, flights, dates, flightlines, sep=",", lc=1):
    fid_in=open(filename_in,"r")
    lines_in = fid_in.readlines()
    fid_in.close()
    fid_out = open(filename_out, "w")
    lastwords=["0", "0", "0", "0"]
    for ii,line in enumerate(lines_in):
        words=line.split(sep)
        # print(words)
        if ii > 0 and words[lc] != lastwords[lc]:
            searchstring='Line  '+words[lc].split(".")[0]+'\n'
            k = flightlines.index(searchstring)
            print("fixing flightline: {}".format(flightlines[k]))
            fid_out.write(flights[k])
            fid_out.write(dates[k])
            fid_out.write(flightlines[k])
            fid_out.write(line)
        else:
            fid_out.write(line)
        lastwords = words
    fid_out.close()


def cullRollPitchYawAlt(data, max_roll=15, max_pitch=15, max_alt=90):
    start = time.time()
    print("=============== Culling based on roll, pitch and altitude limits ===============\n")
    idx=pd.DataFrame(columns=["roll", "pitch", "alt", "combined"], dtype=bool)
    idx.loc[:, 'roll'] = data['flightlines'].tx_roll.abs() > max_roll
    idx.loc[:, 'pitch'] = data['flightlines'].tx_pitch.abs() > max_pitch
    idx.loc[:, 'alt'] = data['flightlines'].alt_tx_laser > max_alt
    
    idx.loc[:, "combined"] =  idx.roll | idx.pitch | idx.alt
    data['inuse'].loc[idx.combined, :] = 0
    
    data['flightlines'].loc[idx.roll & idx.pitch, 'reason']='roll_pitch'
    data['flightlines'].loc[idx.alt, 'reason']='alt'
    end=time.time()
    print("time culling roll, pitch, yaw, altitude: {} sec.".format(end-start))
    print('\nRemoved {0} out of {1} sounding positions'.format(data['inuse'].shape[0]-data['inuse'].any(axis=1).sum(),
                                                 data['inuse'].shape[0]))


def cullNoise(data, error_dict):
    start = time.time()
    print("=============== Culling based on predefined noise levels ===============\n")
    
    for key in error_dict.keys():
        idx=data['data'][key] < error_dict[key]
        data['inuse'].loc[idx, key]=0
        idx2=idx & (data['flightlines'].reason=='none')
        data['flightlines'].loc[idx2, 'reason'] = 'amplitude'
    
    end=time.time()
    print("time used for error based culling: {} sec.".format(end-start))

def cullGeometry(data, shapefile, safety_distance=150, QCplot=True):
   
    start = time.time()
    print("=============== Culling based on geometry ===============\n")
    n_validSoundings_in = data['inuse'].any(axis=1).sum()
    # Make a geopandas geoframe:
    df_points = gpd.GeoDataFrame(data['flightlines'][[data['pos_keys'][0], data['pos_keys'][1]]])
    df_points.set_geometry(
        gpd.points_from_xy(df_points[data['pos_keys'][0]], df_points[data['pos_keys'][1]]),
        inplace=True, 
        crs='EPSG:31981')
    
    
    margin_x=3000
    margin_y=3000
    bbox=(df_points[data['pos_keys'][0]].min()-margin_x,
          df_points[data['pos_keys'][1]].min()-margin_y, 
          df_points[data['pos_keys'][0]].max()+margin_x, 
          df_points[data['pos_keys'][1]].max()+margin_y)
          
    # read powerlines into geopandas geodataframe:      
    
    df_powerlines=gpd.read_file(shapefile, bbox=bbox)
    df_powerlines.head()
    
    
    # calculate distance between points and powerlines:
    gs_points=df_points['geometry']
    gs_powerlines=df_powerlines['geometry']
    
    
    min_dist=np.empty(gs_points.shape[0])
    for i, point in enumerate(gs_points):
        min_dist[i] = np.min([point.distance(powerline) for powerline in gs_powerlines])
    df_points['min_dist_to_lines'] = min_dist
    
    idx = min_dist<safety_distance
    
    
    data['inuse'].loc[idx,:]=0 # for now, remove all datapoints at these locations
    data['flightlines'].loc[idx, 'reason']='geometry'
    
    if QCplot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharex=True, sharey=True)
        gs_powerlines.plot(ax=ax[0], color='black', edgecolor='black', label='power lines')
        sc=ax[0].scatter(df_points[data['pos_keys'][0]], 
                         df_points[data['pos_keys'][1]], 
                         c=min_dist,  cmap='inferno_r', 
                         label = "AEM soundings - distance to powerline" )
        ax[0].set_aspect('equal')
        ax[0].set_xlim(bbox[0], bbox[2] )
        ax[0].set_ylim(bbox[1], bbox[3] )
        plt.colorbar(sc, ax=ax[0])
        ax[0].legend()
        ax[0].set_title('Powerlines and distance to powerlines')
        
        gs_powerlines.plot(ax=ax[1], color='black', edgecolor='black', label='power lines')
        ax[1].plot(df_points.loc[~idx, data['pos_keys'][0]],
                   df_points.loc[~idx, data['pos_keys'][1]] ,
                   '.' , markersize=0.5,
                   label = "valid AEM soundings" )
        ax[1].set_aspect('equal')
        ax[1].set_xlim(bbox[0], bbox[2] )
        ax[1].set_ylim(bbox[1], bbox[3] )
        ax[1].legend()
        ax[1].set_title('Powerlines and valid datapoints')
    
    end=time.time()
    print("time used for error based culling: {} sec.".format(end-start))
    print('\nRemoved {0} out of {1} remaining sounding positions'.format(n_validSoundings_in-data['inuse'].any(axis=1).sum(),
                                                 n_validSoundings_in))

def cullPrevious(data, header_dict, oldculled_filename):
    start = time.time()
    print("=============== Culling based on previous culling levels ===============")
    print(".. using positions")
    print("... and individuall frequencies")
    
    flightline_columns=data['flightlines'].columns
    inuse_columns=data['inuse'].columns
    
    
    data_old=readAahrusWorkbenchFEMdata(oldculled_filename, nan_values=['*', -9999,-9999.9,-9999.99,-9999.999,'-9999.9',])
    rename_dict={'LINE_NO':'line', 'FIDUCIAL':'fid'}
    data_old['flightlines'].rename(columns = rename_dict, inplace = True)
    
    #concat data and flightline details again
    join_columns=['line', 'fid']
    df_olddata_concat=pd.concat([data_old['flightlines'], data_old['data']], axis=1)
    df_data_concat=pd.concat([data['flightlines'], data['inuse']], axis=1)

    # joind dataframes based on line number and fiducial
    df_joined = df_data_concat.join(df_olddata_concat.set_index(join_columns), 
                                   on=join_columns, 
                                   how='left', rsuffix='_old')

    for key in header_dict.keys():
        cullkey = header_dict[key]
        idx_culled = df_joined[cullkey].isna()
        #print("{0} rows are NAN for key {1} and cullkey {2}".format(idx_culled.sum(), key, cullkey))
        df_joined.loc[idx_culled, key]=0
        #idx_reason=idx_culled & df_joined['reason']=='none'
        df_joined.loc[idx_culled, 'reason'] = 'manual'
    
    data['inuse']=df_joined[inuse_columns]
    data['flightlines'] = df_joined[flightline_columns]
    
    end=time.time()
    print("time used for position/frequency culling:{}".format(end-start))
    #return df_olddata_concat, df_data_concat, df_joined


def find_headerlines(workbenchFEMfilename, header_prefix='/', maxlines=100):
    n_headerlines=0
    with open(workbenchFEMfilename) as fid:
        for x in range(maxlines):
            headerline = fid.readline()
            if headerline[0] == '/':
                n_headerlines += 1
                last_headerline=headerline
    return n_headerlines, last_headerline[1:]

def readAahrusWorkbenchFEMdata(workbenchFEMfilename, nan_values=['*', -9999,-9999.9,-9999.99,-9999.999,'-9999.9',]):
    n_headerlines, headerline = find_headerlines(workbenchFEMfilename)
    data_df = pd.read_csv(workbenchFEMfilename,
                          sep='\s+', na_values=nan_values,
                          skiprows=n_headerlines,
                          names=headerline.split())
    real_columns=[]
    imag_columns=[]
    for c in data_df.columns:
        if "REAL" in c:
            real_columns.append(c)
        elif "IMAG" in c:
            imag_columns.append(c)
    
    data_columns=[]
    for c1, c2 in zip(real_columns, imag_columns):
        data_columns.append(c1)
        data_columns.append(c2)
        
    data={'data_columns': data_columns,
         'filename': workbenchFEMfilename,
         'nan_values': nan_values,
         'flightlines': data_df.drop(columns=data_columns),
         'data': data_df[data_columns]
         }
    
    data['inuse']=(~data['data'].isna()).astype(int)
    return data




def readXcaliburFEMdata(DataFileName_orig, data_columns=['CPI140K', 'CPQ140K', 'CPI40K', 'CPQ40K'], nan_values=[-9999,-9999.9,-9999.99,-9999.999,'-9999.9','*']):
    data={'data_columns' : data_columns,
          'filename': DataFileName_orig,
          'nan_values': nan_values}
    tmpfile=tempfilename()
    data['flights'], data['dates'], data['lines'] = fileFixer_in(DataFileName_orig, tmpfile)
    df=pd.read_csv(tmpfile, sep="\s+", na_values=nan_values)
    os.remove(tmpfile)
    df=df[df.any(axis=1)] # remove rows with only nans
    data['flightlines']=df.drop(data_columns, axis=1)
    data['data']=df[data_columns]
    data['inuse']=data['data'].copy().astype(int)
    data['inuse'].loc[:,:]=1  # use all to start with
    
    data['flightlines']['coverage']=data['data'].notna().sum(axis=1)
    data['flightlines']['reason']='none'
    
    int_dict={'line': int,
              'flight': int, 
              'date': int}
    data['flightlines']=data['flightlines'].astype(int_dict)
    
    return data

def dumpXcaliburFEMdata(outputDataFileName, data,  sep="\t", nan_rep='*'):
    print("\n writing to output file: {} ".format(outputDataFileName))
    if data['data'].shape == data['inuse'].shape and data['flightlines'].shape[0] == data['data'].shape[0]:
        df=pd.concat([data['flightlines'], data['data']], axis=1)
    else:
        print('something went wrong, shapes of dataframes are different!')
        
    
    tmpfile=tempfilename()
    df.rename(columns={'fid':'/fid'}, inplace=True)
    print(df.columns)
    df.to_csv(tmpfile, index=False, sep=sep, na_rep=nan_rep)
    fileFixer_out(tmpfile, outputDataFileName,
                  data['flights'], data['dates'], data['lines'],
                  sep=sep)
    os.remove(tmpfile)

def dumpCulling2shape(data, outputShapeFileName, pos_keys, crs='EPSG:31981'):
    gdf = gpd.GeoDataFrame(data['flightlines'])
    gdf.set_geometry(
        gpd.points_from_xy(gdf[pos_keys[0]], gdf[pos_keys[1]]),
        inplace=True, 
        crs=crs)
    #gdf.drop([pos_keys[0], pos_keys[1]], axis=1, inplace=True)  # optional
    gdf.to_file(outputShapeFileName)
    

def drop_empty_columns(data):
    idx_drop=data['inuse'].sum(axis=1)==0
    for key in ['flightlines', 'data', 'inuse']:
        data[key].drop(data[key][idx_drop].index, inplace=True)

def applyCulling(data):
    culled_data=copy.deepcopy(data)
    for key in culled_data['inuse'].columns:
        idx=culled_data['inuse'][key]==0
        culled_data['data'].loc[idx, key] = np.nan # set data top be removed to nan
    drop_empty_columns(culled_data)
    return culled_data
