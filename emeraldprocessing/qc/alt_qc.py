#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:28:11 2023

@author: mp
"""

import numpy as np
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from datetime import datetime
import os.path
import glob
import contextily

from .utils import *



class AltQCplotter(object):
    def _plot_altitude_stats(self, MyFlightData):
        
        STDnumSigma = self.STDnumSigma
        TargetAltitude = self.TargetAltitude
        MaxAltitude = self.MaxAltitude
        AltBuffer = self.AltBuffer
        ThePlottingOpacity = self.ThePlottingOpacity
        Topo_Color = self.Topo_Color
        HistBirdAltColor = self.HistBirdAltColor
        TargAltColor = self.TargAltColor
        MaxAltColor = self.MaxAltColor
        MeanBirdAltColor = self.MeanBirdAltColor
        STDBirdAltColor = self.STDBirdAltColor
        
        
        """Plot Altitude Statistics
        Fill out more Doc
        """

        Line = MyFlightData.line_id[0].copy()
        MyLinexdist = MyFlightData.xdist.copy()
        MyLineBirdAlt = MyFlightData.TxAltitude.copy()
        MyLineMeanBirdAlt = MyFlightData.TxAltitude.mean()
        MyLineStdBirdAlt = MyFlightData.TxAltitude.std()
        MyLineRangeBirdAlt = MyFlightData.TxAltitude.max() - MyFlightData.TxAltitude.min()

        # set fonts for greek symbols
        rcParams.update({'font.size': self.fontsize})
        rcParams.update({'axes.titlesize': self.axes_titlesize})
        
        #rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        #rc('text', usetex=True)

        # start new figure
        fig, axs = plt.subplots(3, 1, figsize=(self.figsize[0],  self.figsize[1]), gridspec_kw={'height_ratios': [2, 2, 1]})

        # calculate statistics on histogram and get bin range
        BirdAltBin = np.int64(np.floor(MyLineRangeBirdAlt))
        if BirdAltBin < 1:
            BirdAltBin = 1
        BirdBincounts, dum = np.histogram(MyLineBirdAlt[~np.isnan(MyLineBirdAlt)], bins=BirdAltBin)

        # get max y limit
        axs0YMaxLim = BirdBincounts.max()

        # plot items with labels.
        axs[0].fill_between([MyLineMeanBirdAlt - (STDnumSigma * MyLineStdBirdAlt),
                             MyLineMeanBirdAlt - (STDnumSigma * MyLineStdBirdAlt),
                             MyLineMeanBirdAlt + (STDnumSigma * MyLineStdBirdAlt),
                             MyLineMeanBirdAlt + (STDnumSigma * MyLineStdBirdAlt)], [0,
                                                                                     axs0YMaxLim * (1.0+AltBuffer),
                                                                                     axs0YMaxLim * (1.0+AltBuffer),
                                                                                     0],
                            color=STDBirdAltColor, alpha=ThePlottingOpacity)
        axs[0].hist(MyLineBirdAlt, bins=BirdAltBin, color=HistBirdAltColor)
        axs[0].plot([TargetAltitude, TargetAltitude], [0-AltBuffer, axs0YMaxLim*(1.0+AltBuffer)], color=TargAltColor)
        axs[0].plot(   [MaxAltitude,    MaxAltitude], [0-AltBuffer, axs0YMaxLim*(1.0+AltBuffer)], color=MaxAltColor)
        axs[0].plot([MyLineMeanBirdAlt, MyLineMeanBirdAlt], [0-AltBuffer, axs0YMaxLim*(1.0+AltBuffer)],
                    color=MeanBirdAltColor)

        # create legend
        handles0 = ([Line2D((0, 0), (0, 0), color=c) for c in [TargAltColor, MaxAltColor, MeanBirdAltColor]] +
                    [Rectangle((0, 0), 1, 1, color=HistBirdAltColor, ec=HistBirdAltColor)] +
                    [Rectangle((0, 0), 1, 1, color=STDBirdAltColor, ec=STDBirdAltColor, alpha=ThePlottingOpacity)])
        labels0 = [f'Target_Bird_Alt ( {np.round(TargetAltitude, 1)} m)',
                   f'Max_Bird_Alt ( {np.round(MaxAltitude, 1)} m)',
                   f'Mean_Bird_Alt (MBA : {np.round(MyLineMeanBirdAlt, 1)} m)',
                   f'Bird_Alt',
                   fr'STD_Bird_Alt {STDnumSigma} * $\sigma$ ± MBA (± {np.round(STDnumSigma*MyLineStdBirdAlt, 1)} m)']

        # plot legend
        axs[0].legend(handles0, labels0)
        del handles0; del labels0

        # Set labels
        axs[0].set_xlabel('Altitude above ground (m)')
        axs[0].set_ylabel('counts')
        axs[0].set_title(f'Altitude statistics for line {Line}')
        axs[0].set_ylim([0, axs0YMaxLim*(1+AltBuffer/2)])

        #
        MyLineTopo = MyFlightData.topo
        MyLineBirdElev = MyLineBirdAlt + MyLineTopo

        [axs[1].plot([dld, dld], [0, max(MyLineBirdElev)*1.5], color='lightgrey', linestyle='dotted')
         for dld in range(1000, np.int64(np.ceil(MyLinexdist.max()/1000)*1000)+1, 1000)]
        axs[1].fill_between([0,
                             np.nanmax(MyLinexdist),
                             np.nanmax(MyLinexdist),
                             0,
                             0], [MyLineMeanBirdAlt - (STDnumSigma * MyLineStdBirdAlt),
                                  MyLineMeanBirdAlt - (STDnumSigma * MyLineStdBirdAlt),
                                  MyLineMeanBirdAlt + (STDnumSigma * MyLineStdBirdAlt),
                                  MyLineMeanBirdAlt + (STDnumSigma * MyLineStdBirdAlt),
                                  MyLineMeanBirdAlt - (STDnumSigma * MyLineStdBirdAlt)],
                            color=STDBirdAltColor, alpha=ThePlottingOpacity)
        axs[1].plot([0, np.nanmax(MyLinexdist)], [   TargetAltitude,    TargetAltitude], color=TargAltColor)
        axs[1].plot([0, np.nanmax(MyLinexdist)], [      MaxAltitude,       MaxAltitude], color=MaxAltColor)
        axs[1].plot([0, np.nanmax(MyLinexdist)], [MyLineMeanBirdAlt, MyLineMeanBirdAlt], color=STDBirdAltColor)
        axs[1].plot(MyLinexdist, MyLineBirdAlt, color=HistBirdAltColor)

        # create legend
        handles1 = ([Line2D((0, 0), (0, 0), color=c[0],alpha=c[1])
                     for c in [(HistBirdAltColor, 1), (TargAltColor, 1), (MaxAltColor, 1), (STDBirdAltColor, 1)]] +
                    [Rectangle((0, 0), 1, 1, color=c[0], ec=c[0], alpha=c[1])
                     for c in [(STDBirdAltColor, ThePlottingOpacity)]])
        labels1 = ['Bird_Alt',
                   f'Target_Bird_Alt ( {np.round(TargetAltitude, 1)} m)',
                   f'Max_Bird_Alt ( {np.round(MaxAltitude, 1)} m)',
                   f'Mean_Bird_Alt (MBA :  {np.round(MyLineMeanBirdAlt, 1)} m)',
                   fr'{STDnumSigma} * $\sigma$ ± MBA (± {np.round(STDnumSigma*MyLineStdBirdAlt, 1)} m)']

        # plot legend
        axs[1].legend(handles1, labels1)
        del handles1; del labels1

        # Set labels
        axs[1].set_ylabel('Altitude above ground (m)')
        axs[1].set_xlabel('Downline distance (m)')
        axs[1].set_title(f'Flight altitude above ground for line {Line}')
        axs[1].set_xlim([0, np.nanmax(MyLinexdist)])
        ybuff=0.1
        axs[1].set_ylim([np.nanmin([MyLineBirdAlt])*(1-ybuff),
                         np.nanmax([MyLineBirdAlt])*(1+ybuff/4)])

        #
        MyLineBirdElevBAD = np.copy(MyLineBirdElev)
        MyLineBirdElevBAD[(MyLineBirdElev < MaxAltitude+MyLineTopo)] ='NaN'

        [axs[2].plot([dld, dld], [0, np.nanmax(MyLineBirdElev)*1.5], color='lightgrey', linestyle='dotted')
         for dld in range(1000, np.int64(np.ceil(MyLinexdist.max()/1000)*1000) + 1, 1000)]
        axs[2].plot(MyLinexdist, MyLineTopo,        color=Topo_Color)
        axs[2].plot(MyLinexdist, MyLineBirdElev,    color=HistBirdAltColor, alpha=0.3)
        axs[2].plot(MyLinexdist, MyLineBirdElevBAD, color=HistBirdAltColor)

        # Set limits
        axs[2].set_xlim([0, np.nanmax(MyLinexdist)])
        axs[2].set_ylim([np.nanmin(MyLineTopo) * (1-ybuff/2), np.nanmax(MyLineBirdElev)*(1+ybuff/4)])

        axs2xrange = np.nanmax(MyLinexdist) - 0
        axs2yrange = (np.nanmax(MyLineBirdElev) * (1+ybuff/4)) - (np.nanmin(MyLineBirdElev) * (1-ybuff))
        dum, dum, axs2W, axs2H = axs[2].get_position().bounds
        nWidth = axs2xrange/axs2W
        nHeight = axs2yrange/axs2H
        axs2VE = np.round(nWidth/nHeight)

        # create legend
        handles2 = ([Line2D((0, 0), (0, 0), color=c[0], alpha=c[1])
                     for c in [(Topo_Color, 1), (HistBirdAltColor, 0.3), (HistBirdAltColor, 1)]] +
                    [Line2D((0, 0), (0, 0), color=c[0], alpha=c[1]) for c in [('white', 0)]])
        labels2 = ['topo', 'Bird_Alt',
                   f'Bird_Alt ABOVE MaxBirdAlt ( {MaxAltitude} m)',
                   f'Vertical exaggeration: {axs2VE} : 1']

        # plot legend
        axs[2].legend(handles2,labels2)
        del handles2; del labels2

        # Set labels
        axs[2].set_ylabel('Altitude above ground (m)')
        axs[2].set_xlabel('Downline distance (m)')
        axs[2].set_title(f'Flight altitude above ground for line {Line}')

        fig.tight_layout()
        plt.show

        
        if self.save_fig:
            self.move_old_files('AltitudeQC', f'AltitudeStatistics_L{Line}*.png')
            plt.savefig(os.path.join(self.PlotDirectory, 'AltitudeQC', f'AltitudeStatistics_L{Line}_{self.now()}.png'),
                        facecolor='white')
        if 'show' not in self.ShowPlot:
            plt.close('all')
        
        return fig, axs
    
    def _plot_alt_vs_speed(self, MyFlightData):
        STDnumSigma = self.STDnumSigma
        TargetAltitude = self.TargetAltitude
        MaxAltitude = self.MaxAltitude
        TargetSpeed = self.TargetSpeed
        MaxSpeed = self.MaxSpeed
        AltBuffer = self.AltBuffer
        ThePlottingOpacity = self.ThePlottingOpacity
        TargAltColor = self.TargAltColor
        MaxAltColor = self.MaxAltColor
        TargSpeedColor = self.TargSpeedColor
        MaxSpeedColor = self.MaxSpeedColor

        """
        Plot Altitude Vs. Speed
        """

        Line = MyFlightData.line_id[0]
        MyLineBirdAlt = MyFlightData.TxAltitude.copy()
        MyLineSpeed = MyFlightData.Speed.copy()

        # set fonts for greek symbols
        rcParams.update({'font.size': self.fontsize})
        #rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        #rc('text', usetex=True)

        fig, axs = plt.subplots(1, 1, figsize=(self.figsize[0]//2, self.figsize[1]//2), gridspec_kw={'height_ratios': [1]})

        speedplot = MyLineSpeed.values.copy()
        speedplot = speedplot[MyLineSpeed > 0]

        altplotfull = MyLineBirdAlt.values.copy()
        altplot = altplotfull
        altplot = altplot[MyLineSpeed > 0] #remove all NaN's and negative speed

        MaxcolorVal = 1.0

        MyRed = np.array(altplot/max(altplot) * MaxcolorVal).reshape(len(altplot), 1)
        MyBlue = np.array(speedplot/max(speedplot) * MaxcolorVal).reshape(len(speedplot), 1)
        MyGreen = np.array(np.abs((np.sqrt((MyRed)**2 + (MyBlue)**2) /
                                   max(np.sqrt((MyRed)**2 + (MyBlue)**2)) * MaxcolorVal) - MaxcolorVal)).reshape(len(speedplot), 1)
        #MyGreen = np.array(np.matlib.repmat(MaxcolorVal, len(speedplot), 1))

        MyColors = np.concatenate((MyRed, MyGreen, MyBlue), axis=1)

        axs.plot([0, np.nanmax(speedplot)*2], [TargetAltitude, TargetAltitude], color=TargAltColor, linestyle='dashed')
        axs.plot([0, np.nanmax(speedplot)*2], [MaxAltitude, MaxAltitude], color=MaxAltColor, linestyle='dashed')
        axs.plot([TargetSpeed, TargetSpeed], [0, np.nanmax(altplot)*2], color=TargSpeedColor, linestyle='dotted')
        axs.plot([MaxSpeed, MaxSpeed], [0, np.nanmax(altplot)*2], color=MaxSpeedColor, linestyle='dotted')
        [axs.plot(speedplot[PltPair], altplot[PltPair], color=list(np.round(MyColors[PltPair, 0:], 3)),
                  marker='o', linestyle='none', ms=2)
         for PltPair in range(0, len(altplot))]

        # Set labels
        axs.set_ylabel('Bird Alt. above ground (m)')
        axs.set_xlabel('Calc Flight Speed (Km/Hr)')
        axs.set_title(f'Flight altitude Vs. Flight Speed for line {Line}')
        PBuff = 0.1
        axs.set_xlim([np.nanmin(speedplot)*(1-PBuff), np.nanmax(speedplot)*(1+PBuff/4)])
        axs.set_ylim([np.nanmin(altplot)*(1-PBuff), np.nanmax(altplot)*(1+PBuff/4)])

        fig.tight_layout()
        plt.show
        
        if self.save_fig:
            self.move_old_files('AltVsSpeedQC', f'Alt_vs_Speed_L{Line}*.png')
            plt.savefig(os.path.join(self.PlotDirectory, 'AltVsSpeedQC', f'Alt_vs_Speed_L{Line}_{self.now()}.png'),
                        facecolor='white')
        if 'show' not in self.ShowPlot:
            plt.close('all')
        
        return fig, axs
