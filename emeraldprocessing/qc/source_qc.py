#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:38:40 2023

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

import contextily


class SourceQCplotter(object):
    
    def _plot_source_stats(self, df):
        
        if not(any(['VoltageOn' in c for c in df.columns])):
            raise Exception('No source statistics available, try reading sps data')
        
        fig=plt.figure(figsize=(self.figsize[0]*1.5,self.figsize[1]/1.5))
        ax = fig.subplot_mosaic(
        """
        ABCDE
        FGHIJ
        """
        )
        
        for moment in ['LM', 'HM']:
            if 'LM' in moment:
                axes=['A', 'B', 'C', 'D', 'E']
            else:
                axes=['F', 'G', 'H', 'I', 'J']
            for key,unit,axs in zip(['TxTemperature_'+moment, 'MeanCurrent_'+moment, 'MaxCurrent_'+moment, 'MinCurrent_'+moment, 'VoltageOn_'+moment],
                                    ['deg', 'A', 'A', 'A', 'V', 'V'],
                                    axes):
                ax[axs].hist(df[key], bins=100)
                mean = df[key].mean()
                y_lim=ax[axs].get_ylim()
                ax[axs].plot([mean, mean], y_lim, 'k--', label='mean={0:3.1f} {1}'.format(mean, unit))
                ax[axs].legend()
                title_words=key.split('_')
                title_words.append('('+unit+')')
                ax[axs].set_xlabel(' '.join(title_words))
                ax[axs].set_ylabel('count')
        Line=df.line_id.unique()
        fig.suptitle('Flight stats line(s): {}'.format(Line))
        plt.tight_layout()
    
    def _plot_source_stats_map(self, df, poskeys=['utmx', 'utmy'], basemap=None, crs=None, buffer=500):
        
        if not(any(['VoltageOn' in c for c in df.columns])):
            raise Exception('No source statistics available, try reading sps data or use plot_current_stats')
        
        x_col, y_col = poskeys
        fig, ax = plt.subplots(2, 5, figsize=(self.figsize[0]*1.5,self.figsize[1]), sharex=True, sharey=True)
           
        for moment in ['LM', 'HM']:
            if 'LM' in moment:
                axes=[ax[0,0], ax[0,1], ax[0,2], ax[0,3], ax[0,4]]
            else:
                axes=[ax[1,0], ax[1,1], ax[1,2], ax[1,3], ax[1,4]]
            for key,unit,axs in zip(['TxTemperature_'+moment, 'MeanCurrent_'+moment, 'MaxCurrent_'+moment, 'MinCurrent_'+moment, 'VoltageOn_'+moment],
                                    ['deg', 'A', 'A', 'A', 'V', 'V'],
                                    axes):
                sc = axs.scatter(df[x_col], df[y_col], c=df[key], s=3,
                                    cmap='jet',
                                    vmin=df[key].min(),
                                    vmax=df[key].max())
                axs.set_aspect(1)
                axs.set_xlim([df[x_col].min()-buffer,
                             df[x_col].max()+buffer])
                axs.set_ylim([df[y_col].min()-buffer,
                             df[y_col].max()+buffer])
                plt.colorbar(sc)
                title_words=key.split('_')
                title_words.append('('+unit+')')
                axs.set_title(' '.join(title_words))
                if basemap:
                    if not(crs):
                        raise Exception('if basemap is given, data CRS must be defined')
                    else:
                        contextily.add_basemap(axs, crs=crs, attribution=False, source=basemap)
        Line=df.line_id.unique()
        fig.suptitle('Flight stats line(s): {}'.format(Line))
        plt.tight_layout()
        
        if self.save_fig:
            self.move_old_files('SourceQC', f'SourceQC{Line}*.png')
            plt.savefig(os.path.join(self.PlotDirectory, 'SourceQC', f'SourceQC{Line}_{self.now()}.png'),
                        facecolor='white')
        if 'show' not in self.ShowPlot:
            plt.close('all')
    
    def _plot_current_stats(self, df, poskeys=['utmx', 'utmy'], basemap=None, crs=None, buffer=500):
        fig, ax = plt.subplots(2, 2, figsize=(self.figsize[0],self.figsize[1]))
        ax[1,0].sharex(ax[0,0])
        ax[1,0].sharey(ax[0,0])
        
        x_col, y_col = poskeys
        
        if any(['Current_Ch' in c for c in df.columns]):
            moments=['Current_Ch01', 'Current_Ch02']
        elif any(['MeanCurrent' in c for c in df.columns]):
            moments=['MeanCurrent_LM', 'MeanCurrent_HM']
        else:
            raise Exception(' found no current data in  dataset')
            
        for moment in moments:
            if ('Ch01' in moment) or ('LM' in moment):
                axs=[ax[0,0], ax[0,1]]
            else:
                axs=[ax[1,0], ax[1,1]]
            
            sc = axs[0].scatter(df[x_col], df[y_col], c=df[moment], s=3,
                                cmap='jet',
                                vmin=df[moment].min(),
                                vmax=df[moment].max())
            axs[0].set_aspect(1)
            axs[0].set_xlim([df[x_col].min()-buffer,
                         df[x_col].max()+buffer])
            axs[0].set_ylim([df[y_col].min()-buffer,
                         df[y_col].max()+buffer])
            cb=plt.colorbar(sc)
            title_words=moment.split('_')
            title_words.append(' [A]')
            cb.set_label(' '.join(title_words))
            if basemap:
                if not(crs):
                    raise Exception('if basemap is given, data CRS must be defined')
                else:
                    contextily.add_basemap(axs[0], crs=crs, attribution=False, source=basemap)
        
            axs[1].hist(df[moment], bins=100)
            axs[1].set_xlabel(' '.join(title_words))
            axs[1].set_ylabel('count')
        
        Line=df.line_id.unique()
        fig.suptitle('Flight stats line(s): {}'.format(Line))
        plt.tight_layout()    
        if self.save_fig:
            self.move_old_files('CurrrentQC', f'CurrentQC{Line}*.png')
            plt.savefig(os.path.join(self.PlotDirectory, 'CurrrentQC', f'CurrrentQC{Line}_{self.now()}.png'),
                        facecolor='white')
        if 'show' not in self.ShowPlot:
            plt.close('all')