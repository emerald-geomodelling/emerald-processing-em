from .utils import splitData_lines, concatXYZ, calc_lineOffset
import matplotlib.pyplot as plt
import os
import yaml
from .setup import allowed_moments
import numpy as np
import pandas as pd
from .. import pipeline


def extract_high_alt_lines(xyz, min_alt=500, verbose=True, qc_plot=True):
    if 'lineoffset' not in xyz.flightlines.columns:
        calc_lineOffset(xyz)
    Lines = splitData_lines(xyz, line_key='Line')
    
    high_alt_lines = []
    for line in Lines.keys():
        min_alt_line = Lines[line].flightlines['TxAltitude'].min()
        date = Lines[line].flightlines.Date.iloc[0]
        if verbose: 
            print('line: {0}  date: {1} min. alt: {2}'.format(line, date, min_alt_line))
        if min_alt_line > min_alt:
            high_alt_lines.append(line)
    if verbose:
        print('high alt lines: {}'.format(high_alt_lines))
    
    data = {}
    for line in high_alt_lines:
        data = concatXYZ(data, Lines[line])
        
    if qc_plot:
        for line in high_alt_lines:    
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(Lines[line].flightlines.lineoffset, Lines[line].flightlines.TxAltitude)
            ax.set_ylabel('altitude [m]')
            ax.set_title('line: {0}, date:{1}'.format(line, Lines[line].flightlines.Date.iloc[0]))
            ax.set_xlabel('lineoffset [m]')
            ax.grid()
    
    return data

def average_data(data, layer_data_key='Gate_Ch02', valid_gate_time_interval=[]):
    avg_trace_key = layer_data_key+'_avg'
    time_key = '_'.join(layer_data_key.split('_scaled_'))
    data[avg_trace_key] = data.layer_data[layer_data_key].mean(axis=0)
    if len(valid_gate_time_interval) > 0:
        filt = (data['GateTimes'][time_key] < valid_gate_time_interval[0]) | (data['GateTimes'][time_key] > valid_gate_time_interval[1])
        data[avg_trace_key].iloc[filt] = 0
    
def subtract_system_bias(processing: pipeline.ProcessingData,
                         system_bias_file: str):
    """
    Function to remove the system bias from the data. The system bias should be 
    estimated on beforehand and stored as a dictionary in a .yaml file

    WARNING: This filter should not be used on data that has already been corrected for system bias.
        For example: SkyTEM final XYZ data deliveries

    Parameters
    ----------
    system_bias_file : str
        Path to .yaml file holding the dictionary describing the system bias.

    """
    print('  - Subtracting system bias')

    data = processing.xyz
    if not type(system_bias_file) == dict:
        with open(os.path.join(system_bias_file), "r") as infile:
            system_bias_dict = yaml.safe_load(infile)
    else:
        system_bias_dict = system_bias_file
    for moment in allowed_moments:
        if moment in system_bias_dict.keys():
            if not data.layer_data[moment].shape[1] == len(system_bias_dict[moment]):
                raise Exception("Number of gates in system bias and data structure differ!") 
            print('Correcting system bias for {}'.format(moment))
            bias = np.array(list(system_bias_dict['Gate_Ch02'].values()))
            bias_df = pd.DataFrame(np.tile(bias, (len(data.layer_data['Gate_Ch02']), 1)))
            data.layer_data[moment] = data.layer_data[moment]-bias_df
        else:
            print('Moment: {} not found in bias dictionary'.format(moment))
