#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 15:30:08 2023

@author: mp
"""

import matplotlib.pyplot as plt
import rasterio
import contextily
from .utils import sampleDEM
import numpy as np
import os


def _add_Tx_z(df):
    if 'z_nmea' in df.columns:
        df['Tx_z'] = df.z_nmea
    else:
        df.insert(df.shape[1], 'Tx_z', df['Topography']+df['TxAltitude'])
        print('did not find Tx_z in data, will calculate it from topo and alt')

def _calc_tilt_size(tilt_factor, max_size=100):
    size = 3+((1/tilt_factor)**100)
    filt = size > max_size
    size.loc[filt] = max_size
    return size

def _calc_scatter_size(series, clim, max_size=100):
    mean = (clim[0]+clim[1])/2
    size = 4+((np.abs(series-mean)/mean)*100)
    filt = size > max_size
    size.loc[filt] = max_size
    return size

class FlightParameterPlotter(object):
    
    def _plot_flightParameter_overview(self, df, poskeys=['utmx', 'utmy'], dem=None):
        Line = df.line_id[0].copy()
        df.rename(columns={'topo': 'Topography'}, inplace=True)
        
        if 'Tx_z' not in df.columns:
            _add_Tx_z(df)
        
        if dem:
            DEM = rasterio.open(dem)
            sampleDEM(DEM, df, poskeys=poskeys, z_key='topo_dem')
            df.insert(df.shape[1], 'TxAltitude_DEM', df['Tx_z']-df['topo_dem'])
        
        fig = plt.figure(figsize=(self.figsize[0]*1.3, self.figsize[1]*0.8))
        
        ax = fig.subplot_mosaic(
            """
            ABCDEFH
            GGGGGGH
            GGGGGGH
            """
        )
        
        # ax={'A':fig.add_subplot(2,6,1),
        #     'B':fig.add_subplot(2,6,2),
        #     'C':fig.add_subplot(2,6,3),
        #     'D':fig.add_subplot(2,6,4),
        #     'E':fig.add_subplot(2,6,5),
        #     'F':fig.add_subplot(2,6,6),
        #     'G':fig.add_subplot(2,1,2)
        #     }
        
        if ('MeanCurrent_LM' in df.columns) and ('MeanCurrent_HM' in df.columns):
            current_keys = ['MeanCurrent_LM', 'MeanCurrent_HM']
        elif 'MeanCurrent_LM' in df.columns:
            current_keys = ['MeanCurrent_LM']
        elif ('Current_Ch01' in df.columns) and ('Current_Ch02' in df.columns):
            current_keys = ['Current_Ch01', 'Current_Ch02']
        elif 'Current_Ch01' in df.columns:
            current_keys = ['Current_Ch01']
        else:
            raise Exception("did not find any current data in flightline dataframe")
        
        if dem:
            keys = ['TxPitch', 'TxRoll',  'TxAltitude_DEM', 'Speed']+current_keys
        else:
            keys = ['TxPitch', 'TxRoll',  'TxAltitude', 'Speed']+current_keys
        units = ['deg', 'deg', 'm', 'km/h', 'A', 'A']
        for key, unit, axs in zip(keys,
                                units,
                                ['A', 'B', 'C', 'D', 'E', 'F']):
            ax[axs].hist(df[key], bins=100)
            mean = df[key].mean()
            y_lim = ax[axs].get_ylim()
            ax[axs].plot([mean, mean], y_lim, 'k--', label='mean={0:3.1f} {1}'.format(mean, unit))
            ax[axs].legend()
            
            title_words = key.split('_')
            title_words.append('('+unit+')')
            ax[axs].set_xlabel(' '.join(title_words))
            ax[axs].set_ylabel('count')

        if dem:
            ax['G'].plot(df['xdist'], df['topo_dem'], 'k-', label='DEM')
            tx_altitude_key = 'TxAltitude_DEM'
        else:
            tx_altitude_key = 'TxAltitude'
        
        sc = ax['G'].scatter(df['xdist'], df['Tx_z'], c=df[tx_altitude_key], 
                             s=3,
                             cmap='jet',
                             # vmin=self.TargetAltitude*(1-(2.*self.altfact)),
                             vmin=self.TargetAltitude*(1-(1.*self.altfact)),
                             vmax=self.TargetAltitude*(1+(2.*self.altfact)),
                             label=tx_altitude_key)
        ax['G'].text(df['xdist'].iloc[-1], df['Tx_z'].iloc[-1], tx_altitude_key)
        
        ax['G'].plot(df['xdist'], df['Topography'], ':', color='grey', label='Topography (laser)')
        cax = ax['G'].inset_axes([0.0, -0.2, 0.15, 0.02])
        cb = plt.colorbar(sc, shrink=0.25, cax=cax, orientation='horizontal')
        cb.set_label('Altitude\n[m]', rotation=0)        
        ax['G'].legend()

        ylim = ax['G'].get_ylim()
        dy = (ylim[1]-ylim[0]) / 25
        shifty = (ylim[1]-ylim[0]) / 10
        y0 = np.ones(len(df)) * df['Tx_z'].max()
        
        y = y0+shifty+(1*dy)
        sc = ax['G'].scatter(df['xdist'], y, c=df['Speed'], s=3,
                             cmap='turbo',
                             # vmin=self.TargetSpeed*(1-(2.*self.speedfact)),
                             vmin=self.TargetSpeed * (1 - (1. * self.speedfact)),
                             vmax=self.TargetSpeed*(1+(2.*self.speedfact)))
        cax = ax['G'].inset_axes([0.2, -0.2, 0.15, 0.02])
        cb = plt.colorbar(sc, shrink=0.25, cax=cax, orientation='horizontal')
        cb.set_label('Speed [km/h]', rotation=0)
        ax['G'].text(df['xdist'].iloc[-1], y[-1], 'Speed')
        
        tilt_factor = (np.cos(df['TxRoll']/180*np.pi) * np.cos(df['TxPitch']/180*np.pi))**2
        y = y0+shifty+(2*dy)
        sc = ax['G'].scatter(df['xdist'], y, c=tilt_factor, s=_calc_tilt_size(tilt_factor),
                             cmap='turbo_r', 
                             vmin=0.95, 
                             vmax=1.0)
        cax = ax['G'].inset_axes([0.4, -0.2, 0.15, 0.02])
        cb = plt.colorbar(sc, shrink=0.25, cax=cax, orientation='horizontal')
        cb.set_label('TiltFactor [1]', rotation=0)
        ax['G'].text(df['xdist'].iloc[-1], y[-1], 'TiltFactor')

        for n, current_key in enumerate(current_keys): 
            y = y0+shifty+((3+n)*dy)
            mean_current = df[current_key].mean()
            # clim=[self.TargetCurrent[n]*(1-(2*self.current_fact)), self.TargetCurrent[n]*(1+(2*self.current_fact))]
            clim = [mean_current*(1-(2*self.current_fact)), mean_current*(1+(2*self.current_fact))]
            
            sc = ax['G'].scatter(df['xdist'], y,
                                 c=df[current_key],
                                 s=_calc_scatter_size(df[current_key], clim),
                                 cmap='turbo_r',
                                 vmin=clim[0], vmax=clim[1])
            cax = ax['G'].inset_axes([0.6+(n*0.2), -0.2, 0.15, 0.02])
            cb = plt.colorbar(sc, shrink=0.25, cax=cax, orientation='horizontal')
            # cax = ax['G'].inset_axes([1.06+0.075*(3+n), 0.2, 0.01, 0.5])
            # cb=plt.colorbar(sc, shrink=0.5, cax=cax)
            cb_title = current_key + ' [A]'
            cb.set_label(cb_title, rotation=0)
            ax['G'].text(df['xdist'].iloc[-1], y[-1], current_key)
        
        ax['G'].grid()
        ax['G'].set_xlabel('Line offset [m]')
        ax['G'].set_ylabel('Elevation [m]')

        ax['H'].plot(df[poskeys[0]].values, df[poskeys[1]].values, label='flightline')
        ax['H'].plot(df[poskeys[0]].values[0], df[poskeys[1]].values[0], 'r.', label='SOL')
        # ax['H'].legend()
        ax['H'].yaxis.set_label_position("right")
        ax['H'].set_aspect(1)
        ax['H'].set_xlabel('easting')
        ax['H'].set_ylabel('northing')
        if self.basemap:
            xlim = ax['H'].get_xlim()
            ax['H'].set_xlim([xlim[0]-self.map_buffer, xlim[1]+self.map_buffer])
            ylim = ax['H'].get_ylim()
            ax['H'].set_ylim([ylim[0]-self.map_buffer, ylim[1]+self.map_buffer])
            contextily.add_basemap(ax['H'], crs=self.data_crs, attribution=False, source=self.basemap)

        fig.suptitle('Flight stats line: {0}  - Date: {1}'.format(Line, df.Date.iloc[0])) 
    
        plt.tight_layout()
        
        if self.save_fig:
            dirpath = os.path.join(self.PlotDirectory, 'FlightParamQC')
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath)
            date = df.Date.iloc[0].replace('/', '_')
            filename = os.path.join(dirpath, f'FlightParamQC_line{Line}_{date}.png')
            print('Saving plot: {}'.format(filename))
            plt.savefig(filename, facecolor='white')
        if 'show' not in self.ShowPlot:
            plt.close('all')
            
    def _plot_flight_parameter_map(self, df, poskeys=['utmx', 'utmy'], basemap=None, crs=None, buffer=500, dem=None):
        
        fig, ax = plt.subplots(2, 3, figsize=(self.figsize[0]*1.2, self.figsize[1]*0.8), sharex=True, sharey=True)
        x_col, y_col = poskeys
        
        _add_Tx_z(df)
        
        if dem:
            DEM = rasterio.open(dem)
            sampleDEM(DEM, df, poskeys=poskeys, z_key='topo_dem')
            df.insert(df.shape[1], 'TxAltitude_DEM', df['Tx_z']-df['topo_dem'])
        
        if 'MeanCurrent_LM' in df.columns:
            current_keys = ['MeanCurrent_LM', 'MeanCurrent_HM']
        else:
            current_keys = ['Current_Ch01', 'Current_Ch02']
        
        if dem:
            keys = ['TxPitch', 'TxRoll',  'TxAltitude_DEM', 'Speed']+current_keys
        else:
            keys = ['TxPitch', 'TxRoll',  'TxAltitude', 'Speed']+current_keys
        units = ['deg', 'deg', 'm', 'km/h', 'A', 'A']
        
        vmins = [-10,
                 -10,
                 self.TargetAltitude*(1-(2.*self.altfact)),
                 self.TargetSpeed*(1-(2*self.speedfact)),
                 self.TargetCurrent[0]*(1-(2*self.current_fact)),
                 self.TargetCurrent[1]*(1-(2*self.current_fact))]
        vmaxs = [10,
                 10,
                 self.TargetAltitude*(1+(2.*self.altfact)),
                 self.TargetSpeed*(1+(2*self.speedfact)),
                 self.TargetCurrent[0]*(1+(2*self.current_fact)),
                 self.TargetCurrent[1]*(1+(2*self.current_fact))]
        
        for key, unit, axs, vmin, vmax in zip(keys, units, ax.flatten(), vmins, vmaxs):
            sc = axs.scatter(df[x_col], df[y_col], 
                             c=df[key],
                             cmap='jet',
                             vmin=vmin,
                             vmax=vmax,
                             s=4,
                             label=key)
            axs.plot(df[x_col].iloc[0], df[y_col].iloc[0], 'ko', fillstyle='none', label='SOL')
            axs.legend()
            axs.set_aspect(1)
            axs.set_xlim([df[x_col].min()-buffer,
                         df[x_col].max()+buffer])
            axs.set_ylim([df[y_col].min()-buffer,
                         df[y_col].max()+buffer])
            plt.colorbar(sc)
            
            title_words = key.split('_')
            title_words.append('('+unit+')')
            axs.set_xlabel(' '.join(title_words))
            axs.set_ylabel('count')
            if basemap:
                if not crs:
                    raise Exception('if basemap is given, data CRS must be defined')
                else:
                    contextily.add_basemap(axs, crs=crs, attribution=False, source=basemap)
        Line = df.line_id.unique()
        fig.suptitle('Flight stats line(s): {}'.format(Line))
        plt.tight_layout()
        
        if self.save_fig:
            self.move_old_files('FlightParameterMapQC', f'FlightParameterMap{Line}*.png')
            plt.savefig(os.path.join(self.PlotDirectory, 'FlightParameterMap', f'FlightParameterMap{Line}_{self.now()}.png'),
                        facecolor='white')
        if 'show' not in self.ShowPlot:
            plt.close('all')
    
    def _plot_FlightLines_map(self, flightlines, poskeys=['utmx', 'utmy'], axs=None, lines=None,  basemap=None, crs=None,):
        if axs is None:
            fig, axs = plt.subplots(figsize=self.figsize)
        
        x_col, y_col = poskeys
        
        if lines is None:
            lines = flightlines.Line.unique()
        for line in lines:
            if line == 0:
                tflightlines = flightlines.copy()
                tflightlines.Line.loc[tflightlines.Line != line] = np.NaN
            else:
                tflightlines = flightlines.loc[flightlines.Line == line]
    
            axs.plot(tflightlines[x_col], tflightlines[y_col], label=f'L{line}')
            axs.plot(tflightlines[x_col].iloc[0], tflightlines[y_col].iloc[0], 'k.')
        
        axs.set_xlim([flightlines[x_col].min()-self.map_buffer,
                     flightlines[x_col].max()+self.map_buffer])
        axs.set_ylim([flightlines[y_col].min()-self.map_buffer,
                     flightlines[y_col].max()+self.map_buffer])
        axs.set_aspect(1)
        axs.legend()
        axs.set_xlabel('easting')
        axs.set_ylabel('northing')
        
        if basemap:
            if not crs:
                raise Exception('if basemap is given, data CRS must be defined')
            else:
                contextily.add_basemap(axs, crs=crs, attribution=False, source=basemap)
        
        if self.save_fig:
            self.move_old_files('FlightlineMapQC', f'FlightlineMap{line}*.png')
            plt.savefig(os.path.join(self.PlotDirectory, 'FlightlineMapQC', f'FlightlineMap.png'),
                        facecolor='white')
        if 'show' not in self.ShowPlot:
            plt.close('all')
    