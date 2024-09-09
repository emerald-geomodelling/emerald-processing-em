#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:32:39 2023

@author: mp
"""

import numpy as np
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import os.path


from .utils import *

class SpeedQCplotter(object):
    def _plot_speed_stats(self, MyFlightData):
        STDnumSigma = self.STDnumSigma
        TargetSpeed = self.TargetSpeed
        MaxSpeed = self.MaxSpeed
        SpeedBuffer = self.SpeedBuffer
        ThePlottingOpacity = self.ThePlottingOpacity
        TargSpeedColor = self.TargSpeedColor
        MaxSpeedColor = self.MaxSpeedColor
        MeanSpeedColor = self.MeanSpeedColor
        STDSpeedColor = self.STDSpeedColor
        HistSpeedColor = self.HistSpeedColor
        
        Line = MyFlightData.line_id[0].copy()
        MyLinexdist = MyFlightData.xdist.copy()
        MyLineBirdAlt = MyFlightData.TxAltitude.copy()
        MyLineBirdElev = MyFlightData.TxAltitude.copy() + MyFlightData.topo.copy()
        MyLineSpeed = MyFlightData.Speed.copy()
        MyLineSpeed.loc[MyLineSpeed == 0] = np.nan
        MyLineMeanSpeed = MyFlightData.Speed.mean()
        MyLineStdSpeed = MyFlightData.Speed.std()
        MyLineRangeSpeed = MyFlightData.Speed.max() - MyFlightData.Speed.min()

        'Plot Speed Statistics'
        # set fonts for greek symbols
        rcParams.update({'font.size': self.fontsize})
        #rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        #rc('text', usetex=True)

        # start new figure
        fig, axs = plt.subplots(2, 1, figsize=(self.figsize[0],  self.figsize[1]*(4/5)), gridspec_kw={'height_ratios': [2, 2]})

        # calculate statistics on histogram and get bin range
        SpeedBin = np.int64(np.floor(MyLineRangeSpeed))
        if SpeedBin < 1:
            SpeedBin = 1
        SpeedBincounts, dum = np.histogram(MyLineSpeed[~np.isnan(MyLineSpeed)], bins=SpeedBin)

        # get max y limit
        axs0YMaxLim=np.nanmax([SpeedBincounts])

        # plot items with labels
        lowX  = MyLineMeanSpeed - (STDnumSigma * MyLineStdSpeed)
        highX = MyLineMeanSpeed + (STDnumSigma * MyLineStdSpeed)
        TempX = np.array([lowX, lowX, highX, highX])

        lowY  = np.float64(0.0)
        highY = axs0YMaxLim * (1.0 + SpeedBuffer)
        TempY = np.array([lowY, highY, highY, lowY])

        axs[0].fill_between(TempX, TempY, color=STDSpeedColor, alpha=ThePlottingOpacity)
        del TempX; del TempY

        axs[0].hist(MyLineSpeed, bins=SpeedBin, color=HistSpeedColor)
        axs[0].plot([TargetSpeed, TargetSpeed], [0-SpeedBuffer, axs0YMaxLim * (1.0 + SpeedBuffer)], color=TargSpeedColor)
        axs[0].plot([MaxSpeed,    MaxSpeed],    [0-SpeedBuffer, axs0YMaxLim * (1.0 + SpeedBuffer)], color=MaxSpeedColor)
        axs[0].plot([MyLineMeanSpeed, MyLineMeanSpeed], [0-SpeedBuffer, axs0YMaxLim * (1.0 + SpeedBuffer)],
                    color=MeanSpeedColor)

        # create legend
        handles0 = ([Line2D((0, 0), (0, 0), color=c) for c in [TargSpeedColor, MaxSpeedColor, MeanSpeedColor]] +
                    [Rectangle((0, 0), 1, 1, color=c[0], ec=c[0], alpha=c[1])
                     for c in [(STDSpeedColor, ThePlottingOpacity), (HistSpeedColor, 1)]])
        labels0 = [f'Target_Speed ({np.round(TargetSpeed, 1)} Km/hr)',
                   f'Max_Speed ({np.round(MaxSpeed, 1)} Km/hr)',
                   f'Mean Flight Speed (MFS :  {np.round(MyLineMeanSpeed, 1)} Km/hr)',
                   fr'{STDnumSigma} * $\sigma$ ± MFS (± {np.round(MyLineStdSpeed, 1)} Km/hr)',
                   'Distribution of calculated Flight Speeds']

        # plot legend
        axs[0].legend(handles0, labels0)
        del handles0; del labels0

        # Set labels
        axs[0].set_xlabel('Ground Speed (Km/hr)')
        axs[0].set_ylabel('counts')
        axs[0].set_title(f'Speed statistics for line {Line}')
        axs[0].set_ylim([0, axs0YMaxLim])

        #
        [axs[1].plot([dld, dld], [0, max(MyLineSpeed)*1.5], color='lightgrey', linestyle='dotted')
         for dld in range(1000, np.int64(np.ceil(MyLinexdist.max()/1000)*1000)+1, 1000)]
        axs[1].fill_between([0,
                             np.nanmax(MyLinexdist),
                             np.nanmax(MyLinexdist),
                             0,
                             0], [MyLineMeanSpeed - (STDnumSigma * MyLineStdSpeed),
                                  MyLineMeanSpeed - (STDnumSigma * MyLineStdSpeed),
                                  MyLineMeanSpeed + (STDnumSigma * MyLineStdSpeed),
                                  MyLineMeanSpeed + (STDnumSigma * MyLineStdSpeed),
                                  MyLineMeanSpeed - (STDnumSigma * MyLineStdSpeed)],
                            color=STDSpeedColor, alpha=ThePlottingOpacity)
        axs[1].plot([0, np.nanmax(MyLinexdist)], [TargetSpeed,     TargetSpeed],     color=TargSpeedColor)
        axs[1].plot([0, np.nanmax(MyLinexdist)], [MaxSpeed,        MaxSpeed],        color=MaxSpeedColor)
        axs[1].plot([0, np.nanmax(MyLinexdist)], [MyLineMeanSpeed, MyLineMeanSpeed], color=MeanSpeedColor)
        axs[1].plot(MyLinexdist, MyLineSpeed, color=HistSpeedColor)

        # create legend
        handles1 = ([Line2D((0, 0), (0, 0), color=c[0], alpha=c[1])
                     for c in [(TargSpeedColor, 1), (MaxSpeedColor, 1), (HistSpeedColor, 1), (MeanSpeedColor, 1)]] +
                    [Rectangle((0, 0), 1, 1, color=c[0], ec=c[0], alpha=c[1])
                     for c in [(STDSpeedColor, ThePlottingOpacity)]])
        labels1 = [f'Target_Speed ({np.round(TargetSpeed,1)} Km/hr)',
                   f'Max_Speed ( {np.round(MaxSpeed,1)} Km/hr)',
                   f'Calculated_Flight_Speed',
                   f'Mean Flight Speed (MFS : {np.round(MyLineMeanSpeed,1)} Km/hr)',
                   fr'{STDnumSigma} * $\sigma$ ± MFS (± {np.round(MyLineStdSpeed,1)} Km/hr)']

        # plot legend
        axs[1].legend(handles1, labels1)
        del handles1; del labels1

        # Set labels
        axs[1].set_ylabel('Calculated Flight Speed (Km/Hr)')
        axs[1].set_xlabel('Downline distance (m)')
        axs[1].set_title(f'Flight Speed for line {Line}')
        axs[1].set_xlim([0, np.nanmax(MyLinexdist)])
        ybuff = 0.1
        axs[1].set_ylim([np.nanmin(MyLineSpeed[1:]) * (1-ybuff), np.nanmax(MyLineSpeed[1:])*(1+ybuff/4)])

        fig.tight_layout()
        plt.show
        
        if self.save_fig:
            self.move_old_files('SpeedQC', f'SpeedStatistics_L{Line}*.png')
            plt.savefig(os.path.join(self.PlotDirectory, 'SpeedQC', f'SpeedStatistics_L{Line}_{self.now()}.png'),
                        facecolor='white')
        if 'show' not in self.ShowPlot:
            plt.close('all')

        return fig, axs
