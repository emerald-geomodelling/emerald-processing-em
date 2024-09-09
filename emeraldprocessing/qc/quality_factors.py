#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:48:42 2023

@author: mp
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import numpy.matlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

from .utils import *

import os.path


class QualityFactorPlotter(object):

    def _calc_quality_factors(self, MyFlightData):
        return SkyTEM_quality_calculator(
            MyFlightData=MyFlightData,
            Alt_QualBreakPts=self.Alt_QualBreakPts,     Alt_QualWeight=self.Alt_QualWeight,
            Speed_QualBreakPts=self.Speed_QualBreakPts, Speed_QualWeight=self.Speed_QualWeight,
            Roll_QualBreakPts=self.Roll_QualBreakPts,   Roll_QualWeight=self.Roll_QualWeight,
            Pitch_QualBreakPts=self.Pitch_QualBreakPts, Pitch_QualWeight=self.Pitch_QualWeight)

    def _assign_quality_factors(self, xyz):
        rename_dict={'UTMX':'utmx', 
                     'UTMY':'utmy',
                     'Line':'line_id',
                     'speed':'Speed'}
        flightlines=xyz.flightlines.rename(columns=rename_dict)
        if not 'FlightData_index' in flightlines.columns:
            flightlines['FlightData_index']=flightlines.index
        quality_factors = self._calc_quality_factors(flightlines)
        quality_factors.set_index(quality_factors['FlightData_index'], drop=True, inplace=True)
        return pd.concat([flightlines, quality_factors], axis=1)
    
    def _save_quality_factors_to_shp(self, flightlines, shpfilename):
        columns_to_drop=['datetime', 'dist1', 'dist2', 'tilt_x1','tilt_y1', 'tilt_x2', 'tilt_y2', 
                         'VoltageOn_LM', 'VoltageOn_HM', 'VoltageOff_LM', 'VoltageOff_HM',
                         'MaxCurrent_LM', 'MaxCurrent_HM', 'MinCurrent_LM', 'MinCurrent_HM',
                         'dutmx', 'dutmy', 'dz_r', 'utmx_laser1', 'utmy_laser1',
                         'z_laser1', 'utmx_laser2', 'utmy_laser2', 'z_laser2', 'FlightData_index']
        rename_dict={'line_id':'Line'}
        gdf_points = gpd.GeoDataFrame(flightlines.drop(columns=columns_to_drop).rename(columns=rename_dict),
                                      geometry=gpd.points_from_xy(flightlines.utmx, flightlines.utmy, flightlines.Topography.fillna(-9999.99), 
                                      crs=self.data_crs))
            
        gdf_points.to_file(shpfilename.split('.shp')[0]+'_Point.shp')
        
        gdf_lines = gpd.GeoDataFrame(geometry=gdf_points.groupby('Line')['geometry'].apply(lambda geometry: shapely.geometry.LineString(geometry.tolist())))
        gdf_lines.to_file( shpfilename.split('.shp')[0]+'_Line.shp')
        
    
    def _plot_quality_factors(self, MyFlightData):
        """
        Quality factors Plots

        """

        NumQualityFactors = self.NumQualityFactors
        Alt_QualBreakPts = self.Alt_QualBreakPts
        Speed_QualBreakPts = self.Speed_QualBreakPts
        Roll_QualBreakPts = self.Roll_QualBreakPts
        Pitch_QualBreakPts = self.Pitch_QualBreakPts
        Topo_Color = self.Topo_Color
        goodColor = self.goodColor
        okColor = self.okColor
        badColor = self.badColor
        cullColor = self.cullColor

        quality_grade_df = self._calc_quality_factors(MyFlightData)
        # quality_grade_df = SkyTEM_quality_calculator(
        #     MyFlightData=MyFlightData,
        #     Alt_QualBreakPts=self.Alt_QualBreakPts,     Alt_QualWeight=self.Alt_QualWeight,
        #     Speed_QualBreakPts=self.Speed_QualBreakPts, Speed_QualWeight=self.Speed_QualWeight,
        #     Roll_QualBreakPts=self.Roll_QualBreakPts,   Roll_QualWeight=self.Roll_QualWeight,
        #     Pitch_QualBreakPts=self.Pitch_QualBreakPts, Pitch_QualWeight=self.Pitch_QualWeight)
        
        legendFont = mpl.font_manager.FontProperties(family='monospace',
                                                     weight='normal',
                                                     style='normal',
                                                     size=9)

        Line = MyFlightData.line_id[0]
        MyLinexdist = MyFlightData.xdist.copy()
        MyLineTopo = MyFlightData.topo.copy()
        MyLineBirdElev = MyFlightData.TxAltitude.copy() + MyFlightData.topo.copy()
        MyLineBirdAlt = MyFlightData.TxAltitude.copy()

        MyLineAltQuality   = quality_grade_df.MyLineAltQuality
        MyLineSpeedQuality = quality_grade_df.MyLineSpeedQuality
        MyLineRollQuality  = quality_grade_df.MyLineRollQuality
        MyLinePitchQuality = quality_grade_df.MyLinePitchQuality
        #MyLinePLMQuality   = quality_grade_df.MyLinePLMQuality
        MyLineTOTALQuality = quality_grade_df.MyLineTOTALQuality
        MyLineGrade        = quality_grade_df.MyLineGrade


        # set fonts for greek symbols
        rcParams.update({'font.size': self.fontsize})
        #rc('font', **{'family':'serif','serif':['Palatino']})
        #rc('text', usetex=True)

        #fill arrays with bird elev and then remove values that don't fit the quality needed for a feature
        MyLineAltbad =    np.copy(MyLineBirdElev);      MyLineAltbad[MyLineAltQuality != -1] = np.nan
        MyLineAltQElev0 = np.copy(MyLineBirdElev);   MyLineAltQElev0[MyLineAltQuality != 0]  = np.nan
        MyLineAltQElev1 = np.copy(MyLineBirdElev);   MyLineAltQElev1[MyLineAltQuality != 1]  = np.nan
        MyLineAltQElev2 = np.copy(MyLineBirdElev);   MyLineAltQElev2[MyLineAltQuality != 2]  = np.nan

        MyLineSpeedQElev0 = np.copy(MyLineBirdElev); MyLineSpeedQElev0[MyLineSpeedQuality != 0] = np.nan
        MyLineSpeedQElev1 = np.copy(MyLineBirdElev); MyLineSpeedQElev1[MyLineSpeedQuality != 1] = np.nan
        MyLineSpeedQElev2 = np.copy(MyLineBirdElev); MyLineSpeedQElev2[MyLineSpeedQuality != 2] = np.nan

        MyLineRollbad =    np.copy(MyLineBirdElev);     MyLineRollbad[MyLineRollQuality != -1] = np.nan
        MyLineRollQElev0 = np.copy(MyLineBirdElev);  MyLineRollQElev0[MyLineRollQuality != 0]  = np.nan
        MyLineRollQElev1 = np.copy(MyLineBirdElev);  MyLineRollQElev1[MyLineRollQuality != 1]  = np.nan
        MyLineRollQElev2 = np.copy(MyLineBirdElev);  MyLineRollQElev2[MyLineRollQuality != 2]  = np.nan

        MyLinePitchbad =    np.copy(MyLineBirdElev);    MyLinePitchbad[MyLinePitchQuality != -1] = np.nan
        MyLinePitchQElev0 = np.copy(MyLineBirdElev); MyLinePitchQElev0[MyLinePitchQuality != 0]  = np.nan
        MyLinePitchQElev1 = np.copy(MyLineBirdElev); MyLinePitchQElev1[MyLinePitchQuality != 1]  = np.nan
        MyLinePitchQElev2 = np.copy(MyLineBirdElev); MyLinePitchQElev2[MyLinePitchQuality != 2]  = np.nan

        # MyLinePLMQElev0 = np.copy(MyLineBirdElev);   MyLinePLMQElev0[MyLinePLMQuality!=0] = np.nan
        # MyLinePLMQElev1 = np.copy(MyLineBirdElev);   MyLinePLMQElev1[MyLinePLMQuality!=1] = np.nan
        # MyLinePLMQElev2 = np.copy(MyLineBirdElev);   MyLinePLMQElev2[MyLinePLMQuality!=2] = np.nan

        start = 0
        end = 1
        interval = 0.1
        vals = np.arange(start, end, interval)
        QualCriteria = np.zeros((MyLineTOTALQuality.shape[0], len(vals)), np.bool)

        for col, val in enumerate(vals):
            QualCriteria[:, col] = ~np.logical_and(MyLineTOTALQuality >= val, MyLineTOTALQuality <= val+interval)

        MyLineTOTQElev = np.column_stack(len(vals)*[np.copy(MyLineBirdElev)])
        MyLineTOTQElev[QualCriteria] = np.nan

        base = 4
        crazyarray = np.matlib.repmat(1, NumQualityFactors+1, 1); crazyarray[-1] = 2
        fig, axs = plt.subplots(NumQualityFactors+1, 1, figsize=(3*base, (NumQualityFactors+2)*base), gridspec_kw={'height_ratios': crazyarray})
        del crazyarray

        ii = -1
        # Alt Quality Plot
        ii = ii+1
        [axs[ii].plot([dld, dld], [0, max(MyLineBirdElev)*1.5], color='lightgrey', linestyle='dotted')
         for dld in range(1000, np.int64(np.ceil(MyLinexdist.max()/1000)*1000)+1, 1000)]
        axs[ii].plot(MyLinexdist, MyLineTopo,      color=Topo_Color)
        axs[ii].plot(MyLinexdist, MyLineAltbad,    color=cullColor, linewidth=2)
        axs[ii].plot(MyLinexdist, MyLineAltQElev0, color=badColor,  linewidth=6)
        axs[ii].plot(MyLinexdist, MyLineAltQElev1, color=okColor,   linewidth=4)
        axs[ii].plot(MyLinexdist, MyLineAltQElev2, color=goodColor, linewidth=4)

        # Set limits
        axs[ii].set_xlim([0, np.nanmax(MyLinexdist)])
        ybuff = 0.1
        axs[ii].set_ylim([np.nanmin(MyLineTopo)*(1-ybuff/2), np.nanmax(MyLineBirdElev)*(1+ybuff/4)])

        axsxrange = np.nanmax(MyLinexdist)-0
        axs0yrange = (np.nanmax(MyLineBirdElev)*(1+ybuff/4))-(np.nanmin(MyLineBirdElev)*(1-ybuff/2))
        dum, dum, axs0W, axs0H = axs[0].get_position().bounds
        nWidth = axsxrange / axs0W
        nHeight = axs0yrange/axs0H
        axs0VE = np.round(nWidth/nHeight)

        # create legend
        handles = ([Line2D((0, 0), (0, 0), color=c[0], alpha=c[1])
                    for c in [(Topo_Color, 1), (goodColor, 1), (okColor, 1), (badColor, 1), (cullColor, 1)]] +
                    [Line2D((0, 0), (0, 0),color=c[0], alpha=c[1]) for c in [('white', 0)]])
        labels = ([ 'topo',
                  r'High_Quality_Altitude : (         Alt $\leq$ {:5.1f} [m])'.format(                        Alt_QualBreakPts[0]),
                  r'Med_Quality_Altitude  : ({:5.1f} $<$ Alt $\leq$ {:5.1f} [m])'.format(Alt_QualBreakPts[0], Alt_QualBreakPts[1]),
                  r'Low_Quality_Altitude  : ({:5.1f} $<$ Alt $\leq$ {:5.1f} [m])'.format(Alt_QualBreakPts[1], Alt_QualBreakPts[2]),
                  r'Unusable_Altitude     : ({:5.1f} $<$ Alt          [m])'.format(      Alt_QualBreakPts[2]),
                  f'Vertical exaggeration [{np.round(axs0VE,1)} : 1]'])

        # plot legend
        axs[ii].legend(handles, labels, prop=legendFont)
        del handles; del labels

        # Speed Quality plot
        ii = ii+1
        [axs[ii].plot([dld, dld], [0, max(MyLineBirdElev)*1.5], color='lightgrey', linestyle='dotted')
         for dld in range(1000, np.int64(np.ceil(MyLinexdist.max()/1000)*1000)+1, 1000)]
        axs[ii].plot(MyLinexdist, MyLineTopo,        color=Topo_Color)
        axs[ii].plot(MyLinexdist, MyLineSpeedQElev0, color=badColor,  linewidth=6)
        axs[ii].plot(MyLinexdist, MyLineSpeedQElev1, color=okColor,   linewidth=4)
        axs[ii].plot(MyLinexdist, MyLineSpeedQElev2, color=goodColor, linewidth=4)

        # Set limits
        axs[ii].set_xlim([0, np.nanmax(MyLinexdist)])
        axs[ii].set_ylim([np.nanmin(MyLineTopo)*(1-ybuff/2), np.nanmax(MyLineBirdElev)*(1+ybuff/4)])

        axs1yrange = (np.nanmax(MyLineBirdElev)*(1+ybuff/4))-(np.nanmin(MyLineBirdElev)*(1-ybuff/2))
        dum, dum, axs1W, axs1H = axs[1].get_position().bounds
        nWidth = axsxrange/axs1W
        nHeight = axs1yrange/axs1H
        axs1VE = np.round(nWidth/nHeight)

        # create legend
        handles = ([Line2D((0,0),(0,0),color=c[0], alpha=c[1])
                    for c in [(Topo_Color, 1), (goodColor, 1), (okColor, 1), (badColor, 1)]] +
                   [Line2D((0,0),(0,0),color=c[0], alpha=c[1]) for c in [('white', 0)]])
        labels = (['topo',
                  r'High_Quality_Speed : (         Speed $\leq$ {:5.1f} [km/hr])'.format(                          Speed_QualBreakPts[0]),
                  r'Med_Quality_Speed  : ({:5.1f} $<$ Speed $\leq$ {:5.1f} [km/hr])'.format(Speed_QualBreakPts[0], Speed_QualBreakPts[1]),
                  r'Low_Quality_Speed  : ({:5.1f} $<$ Speed          [km/hr])'.format(Speed_QualBreakPts[1]),
                  f'Vertical exaggeration [{np.round(axs0VE,1)} : 1]'])

        # plot legend
        axs[ii].legend(handles, labels, prop=legendFont)
        del handles; del labels

        # Roll Quality plot
        ii = ii+1
        [axs[ii].plot([dld, dld], [0, max(MyLineBirdElev)*1.5], color='lightgrey', linestyle='dotted')
         for dld in range(1000, np.int64(np.ceil(MyLinexdist.max()/1000)*1000)+1, 1000)]
        axs[ii].plot(MyLinexdist, MyLineTopo,       color=Topo_Color)
        axs[ii].plot(MyLinexdist, MyLineRollbad,    color=cullColor, linewidth=2)
        axs[ii].plot(MyLinexdist, MyLineRollQElev0, color=badColor,  linewidth=6)
        axs[ii].plot(MyLinexdist, MyLineRollQElev1, color=okColor,   linewidth=4)
        axs[ii].plot(MyLinexdist, MyLineRollQElev2, color=goodColor, linewidth=4)

        # Set limits
        axs[ii].set_xlim([0, np.nanmax(MyLinexdist)])
        axs[ii].set_ylim([np.nanmin(MyLineTopo)*(1-ybuff/2), np.nanmax(MyLineBirdElev)*(1+ybuff/4)])

        axs2yrange = (np.nanmax(MyLineBirdElev)*(1+ybuff/4))-(np.nanmin(MyLineBirdElev)*(1-ybuff/2))
        dum, dum, axs2W, axs2H=axs[2].get_position().bounds
        nWidth = axsxrange/axs2W
        nHeight = axs2yrange/axs2H
        axs2VE = np.round(nWidth/nHeight)

        # create legend
        handles = ([Line2D((0, 0), (0, 0), color=c[0], alpha=c[1])
                    for c in [(Topo_Color, 1), (goodColor, 1), (okColor, 1), (badColor, 1), (cullColor, 1)]] +
                    [Line2D((0, 0), (0, 0), color=c[0], alpha=c[1]) for c in [('white', 0)]])
        labels = (['topo',
                   r'High_Quality_Roll : (        Roll $\leq$ {:4.1f} [deg])'.format(                          Roll_QualBreakPts[0]),
                   r'Med_Quality_Roll  : ({:4.1f} $<$ Roll $\leq$ {:4.1f} [deg])'.format(Roll_QualBreakPts[0], Roll_QualBreakPts[1]),
                   r'Low_Quality_Roll  : ({:4.1f} $<$ Roll $\leq$ {:4.1f} [deg])'.format(Roll_QualBreakPts[1], Roll_QualBreakPts[2]),
                   r'Unusable_Roll     : ({:4.1f} $<$ Roll         [deg])'.format(       Roll_QualBreakPts[2]),
                   f'Vertical exaggeration [{np.round(axs0VE, 1)} : 1]'])

        # plot legend
        axs[ii].legend(handles, labels, prop=legendFont)
        del handles; del labels

        # Pitch Quality plot
        ii=ii+1
        [axs[ii].plot([dld, dld], [0, max(MyLineBirdElev)*1.5], color='lightgrey', linestyle='dotted')
         for dld in range(1000, np.int64(np.ceil(MyLinexdist.max()/1000)*1000)+1, 1000)]
        axs[ii].plot(MyLinexdist, MyLineTopo,        color=Topo_Color)
        axs[ii].plot(MyLinexdist, MyLinePitchQElev0, color=badColor,  linewidth=6)
        axs[ii].plot(MyLinexdist, MyLinePitchQElev1, color=okColor,   linewidth=4)
        axs[ii].plot(MyLinexdist, MyLinePitchQElev2, color=goodColor, linewidth=4)

        # Set limits
        axs[ii].set_xlim([0, np.nanmax(MyLinexdist)])
        axs[ii].set_ylim([np.nanmin(MyLineTopo)*(1-ybuff/2), np.nanmax(MyLineBirdElev)*(1+ybuff/4)])

        axs3yrange = (np.nanmax(MyLineBirdElev)*(1+ybuff/4))-(np.nanmin(MyLineBirdElev)*(1-ybuff/2))
        dum, dum, axs3W, axs3H = axs[3].get_position().bounds
        nWidth = axsxrange/axs3W
        nHeight = axs3yrange/axs3H
        axs3VE = np.round(nWidth/nHeight)

        # create legend
        handles = ([Line2D((0,0),(0,0),color=c[0], alpha=c[1]) for c in [(Topo_Color, 1), (goodColor, 1), (okColor, 1), (badColor, 1), (cullColor, 1)]] +
                    [Line2D((0,0),(0,0),color=c[0], alpha=c[1]) for c in [('white', 0)]])
        labels = (['topo',
                   r'High_Quality_Pitch : (        Pitch $\leq$ {:4.1f} [deg])'.format(                           Pitch_QualBreakPts[0]),
                   r'Med_Quality_Pitch  : ({:4.1f} $<$ Pitch $\leq$ {:4.1f} [deg])'.format(Pitch_QualBreakPts[0], Pitch_QualBreakPts[1]),
                   r'Low_Quality_Pitch  : ({:4.1f} $<$ Pitch $\leq$ {:4.1f} [deg])'.format(Pitch_QualBreakPts[1], Pitch_QualBreakPts[2]),
                   r'Unusable_Pitch     : ({:4.1f} $<$ Pitch         [deg])'.format(       Pitch_QualBreakPts[2]),
                   f'Vertical exaggeration [{np.round(axs0VE, 1)} : 1]'])


        # plot legend
        axs[ii].legend(handles, labels, prop=legendFont)
        del handles; del labels

        # # PLM Quality plot
        # ii = ii+1
        # [axs[ii].plot([dld, dld], [0,max(MyLineBirdElev)*1.5], color='lightgrey', linestyle='dotted')
        #  for dld in range(1000, np.int64(np.ceil(MyLinexdist.max()/1000)*1000)+1, 1000)]
        # axs[ii].plot(MyLinexdist, MyLineTopo,      color=Topo_Color)
        # axs[ii].plot(MyLinexdist, MyLinePLMQElev0, color=badColor,  linewidth=6)
        # axs[ii].plot(MyLinexdist, MyLinePLMQElev1, color=okColor,   linewidth=4)
        # axs[ii].plot(MyLinexdist, MyLinePLMQElev2, color=goodColor, linewidth=4)

        # #Set limits
        # axs[ii].set_xlim([0, np.nanmax(MyLinexdist)])
        # axs[ii].set_ylim([np.nanmin(MyLineTopo)*(1-ybuff/2), np.nanmax(MyLineBirdElev)*(1+ybuff/4)])

        # axs4yrange = (np.nanmax(MyLineBirdElev)*(1+ybuff/4))-(np.nanmin(MyLineBirdElev)*(1-ybuff/2))
        # dum, dum, axs4W, axs4H = axs[4].get_position().bounds
        # nWidth = axsxrange/axs4W
        # nHeight = axs4yrange/axs4H
        # axs4VE = np.round(nWidth/nHeight)

        # #create legend
        # handles = ([Line2D((0, 0), (0, 0), color=c[0], alpha=c[1])
        #             for c in [(Topo_Color, 1), (goodColor, 1), (okColor, 1), (badColor, 1)]] +
        #             [Line2D((0, 0), (0, 0), color=c[0], alpha=c[1]) for c in [('white', 0)]])
        # labels = (['topo',
        #           'High_Quality_PLM : (' + str(PLM_QualBreakPts[0]) + r' $\geq$ PLM [deg])',
        #           'Med_Quality_PLM : (' + str(PLM_QualBreakPts[0]) + r' < PLM $\geq$ ' + str(PLM_QualBreakPts[1]) + ' [deg])',
        #           r'Low_Quality_PLM : (PLM < ' + str(PLM_QualBreakPts[1]) + ' [deg])',
        #           'Vertical exageration: ' + str(axs4VE) + ' : 1'])

        # #plot legend
        # axs[ii].legend(handles,labels)
        # del handles; del labels

        # Total Quality plot
        #colorstartind = 2
        mycmap = plt.cm.get_cmap('turbo_r')
        QualCol = mycmap(np.linspace(0, 0.8, len(MyLineTOTQElev[1, :])))

        ii = ii+1
        [axs[ii].plot([dld, dld], [0, max(MyLineBirdElev)*1.5], color='lightgrey', linestyle='dotted')
         for dld in range(1000, np.int64(np.ceil(MyLinexdist.max() / 1000) * 1000) + 1, 1000)]
        axs[ii].plot(MyLinexdist, MyLineTopo, color=Topo_Color)
        #for pp in range(len(MyLineTOTQElev[1, :])):
        for pp in range(len(MyLineTOTQElev[1, :])-1, -1, -1):
            if pp < 8:
                #axs[ii].plot(MyLinexdist, MyLineTOTQElev[:, pp], color=np.array(QualCol(pp + colorstartind))[0:3], linewidth=6)
                axs[ii].plot(MyLinexdist, MyLineTOTQElev[:, pp], color=QualCol[pp, 0:3], linewidth=6)
            else:
                #axs[ii].plot(MyLinexdist, MyLineTOTQElev[:, pp], color=np.array(QualCol(pp + colorstartind))[0:3], linewidth=4)
                axs[ii].plot(MyLinexdist, MyLineTOTQElev[:, pp], color=QualCol[pp, 0:3], linewidth=4)

        # Set limits
        axs[ii].set_xlim([0, np.nanmax(MyLinexdist)])
        axs[ii].set_ylim([np.nanmin(MyLineTopo)*(1-ybuff/2), np.nanmax(MyLineBirdElev)*(1+ybuff/4)])

        axs7yrange=(np.nanmax(MyLineBirdElev)*(1+ybuff/4))-(np.nanmin(MyLineBirdElev)*(1-ybuff/2))
        dum, dum, axs7W, axs7H = axs[ii].get_position().bounds
        nWidth = axsxrange/axs7W
        nHeight = axs7yrange/axs7H
        axs7VE = np.round(nWidth/nHeight)

        # create legend
        handles = ([Line2D((0, 0), (0, 0), color=c[0], alpha=c[1]) for c in [(Topo_Color, 1)]] +
                   [Line2D((0, 0), (0, 0), color=QualCol[pp, 0:3], alpha=1) for pp in range(len(QualCol)-1, -1, -1)] +
                   [Line2D((0, 0), (0, 0), color=c[0], alpha=c[1]) for c in [('white', 0), ('white', 0)]])
        labels = (['topo',
                  fr'TOT $\geq$ {0.9} : High_Quality ',
                  fr'TOT $\geq$ {0.8}',
                  fr'TOT $\geq$ {0.7}',
                  fr'TOT $\geq$ {0.6}',
                  fr'TOT $\geq$ {0.5}',
                  fr'TOT $\geq$ {0.4}',
                  fr'TOT $\geq$ {0.3}',
                  fr'TOT $\geq$ {0.2}',
                  fr'TOT $\geq$ {0.1}',
                  fr'TOT = {0.0} : Low_Quality',
                  f'Total cumulative grade: {np.round(MyLineGrade[0],3)}',
                  f'Vertical exaggeration: {np.round(axs7VE,1)} : 1'])

        # plot legend
        axs[ii].legend(handles, labels, prop=legendFont)
        del handles; del labels

        # Set labels
        ii = -1
        for title_name in ['Altitude', 'Speed', 'Roll', 'Pitch', #'PLM',
                           'TOTAL']:
            ii = ii+1
            axs[ii].set_ylabel('Elevation (m.a.s.l)')
            axs[ii].set_title(f'Flight {title_name} quality for line {Line}')
        del ii; del title_name
        axs[NumQualityFactors].set_xlabel('Downline distance (m)')

        fig.tight_layout()
        plt.show

        if self.save_fig:
            self.move_old_files('QualityQC', f'QualityPlots_L{Line}*.png')
            plt.savefig(os.path.join(self.PlotDirectory, 'QualityQC', f'QualityPlots_L{Line}_{self.now()}.png'),
                        facecolor='white')
        if 'show' not in self.ShowPlot:
            plt.close('all')

        return fig, axs

