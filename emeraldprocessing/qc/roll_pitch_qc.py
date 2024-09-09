#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:45:40 2023

@author: mp
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import os.path


class RollPitchQCplotter(object):
    
    def _plot_roll_pitch_stats(self, MyFlightData):
        """Plot Roll and Pitch Statistics"""

        Roll_QualBreakPts = self.Roll_QualBreakPts
        Pitch_QualBreakPts = self.Pitch_QualBreakPts
        STDnumSigma = self.STDnumSigma
        RollPitchBuff = self.RollPitchBuff
        ThePlottingOpacity = self.ThePlottingOpacity
        HistRollColor = self.HistRollColor
        MeanRollColor = self.MeanRollColor
        STDRollColor = self.STDRollColor
        HistPitchColor = self.HistPitchColor
        MeanPitchColor = self.MeanPitchColor
        STDPitchColor = self.STDPitchColor
        RPgoodColor = self.RPgoodColor
        RPokayColor = self.RPokayColor
        RPbadColor = self.RPbadColor
        
        Line = MyFlightData.line_id[0]
        MyLinexdist = MyFlightData.xdist.copy()
        MyLineRoll = MyFlightData.TxRoll.copy()
        MyLineMeanRoll = MyFlightData.TxRoll.mean()
        MyLineStdRoll = MyFlightData.TxRoll.std()
        MyLineRangeRoll = MyFlightData.TxRoll.max() - MyFlightData.TxRoll.min()
        MyLinePitch = MyFlightData.TxPitch.copy()
        MyLineMeanPitch = MyFlightData.TxPitch.mean()
        MyLineStdPitch = MyFlightData.TxPitch.std()
        MyLineRangePitch = MyFlightData.TxPitch.max() - MyFlightData.TxPitch.min()

        # set fonts for greek symbols
        rcParams.update({'font.size': self.fontsize})
        #rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        #rc('text', usetex=True)

        # start new figure
        fig, axs = plt.subplots(3, 2, figsize=(self.figsize[0],  self.figsize[1] * (6 / 5)), gridspec_kw={'height_ratios': [2, 2, 2]})

        if MyLineRoll[~np.isnan(MyLineRoll)].size > 0 or MyLinePitch[~np.isnan(MyLinePitch)].size > 0:
            # calculate statistics on histogram and get bin range
            RollBin = np.int64(np.floor(MyLineRangeRoll))
            if RollBin < 1:
                RollBin = 1
            RollBincounts, dum = np.histogram(MyLineRoll[~np.isnan(MyLineRoll)], bins=RollBin)

            PitchBin = np.int64(np.floor(MyLineRangePitch))
            if PitchBin < 1:
                PitchBin = 1
            PitchBincounts, dum = np.histogram(MyLinePitch[~np.isnan(MyLinePitch)], bins=PitchBin)

            # plot items with labels.
            axs[0, 0].fill_between([MyLineMeanRoll - STDnumSigma * MyLineStdRoll, MyLineMeanRoll - STDnumSigma * MyLineStdRoll,
                                    MyLineMeanRoll + STDnumSigma * MyLineStdRoll, MyLineMeanRoll + STDnumSigma * MyLineStdRoll],
                                   [0, RollBincounts.max() * (1.0 + RollPitchBuff), RollBincounts.max() * (1.0 + RollPitchBuff),
                                    0],
                                   color=STDRollColor, alpha=ThePlottingOpacity)
            axs[0, 0].plot([MyLineMeanRoll, MyLineMeanRoll], [0 - RollPitchBuff, RollBincounts.max() * (1.0 + RollPitchBuff)],
                           color=MeanRollColor)
            axs[0, 0].plot([Roll_QualBreakPts[0], Roll_QualBreakPts[0]],
                           [0 - RollPitchBuff, RollBincounts.max() * (1.0 + RollPitchBuff)], color=RPgoodColor)
            axs[0, 0].plot([Roll_QualBreakPts[0] * -1, Roll_QualBreakPts[0] * -1],
                           [0 - RollPitchBuff, RollBincounts.max() * (1.0 + RollPitchBuff)], color=RPgoodColor)
            axs[0, 0].plot([Roll_QualBreakPts[1], Roll_QualBreakPts[1]],
                           [0 - RollPitchBuff, RollBincounts.max() * (1.0 + RollPitchBuff)], color=RPokayColor)
            axs[0, 0].plot([Roll_QualBreakPts[1] * -1, Roll_QualBreakPts[1] * -1],
                           [0 - RollPitchBuff, RollBincounts.max() * (1.0 + RollPitchBuff)], color=RPokayColor)
            axs[0, 0].plot([Roll_QualBreakPts[2], Roll_QualBreakPts[2]],
                           [0 - RollPitchBuff, RollBincounts.max() * (1.0 + RollPitchBuff)], color=RPbadColor)
            axs[0, 0].plot([Roll_QualBreakPts[2] * -1, Roll_QualBreakPts[2] * -1],
                           [0 - RollPitchBuff, RollBincounts.max() * (1.0 + RollPitchBuff)], color=RPbadColor)
            axs[0, 0].hist(MyLineRoll, bins=RollBin, color=HistRollColor)

            # create legend
            handles00 = ([Line2D((0, 0), (0, 0), color=c) for c in [MeanRollColor, RPgoodColor, RPokayColor, RPbadColor]] +
                         [Rectangle((0, 0), 1, 1, color=c[0], ec=c[0], alpha=c[1])
                          for c in [(STDRollColor, ThePlottingOpacity), (HistRollColor, 1)]])
            labels00 = [f'Mean Flight Roll (MFR :  {np.round(MyLineMeanRoll, 1)} degrees)',
                        fr'    Good Roll angles (                            Roll $\leq$ ±{Roll_QualBreakPts[0]})',
                        fr'    Okay Roll angles (±{Roll_QualBreakPts[0]} $<$ Roll $\leq$ ±{Roll_QualBreakPts[1]})',
                        fr'Mediocre Roll angles (±{Roll_QualBreakPts[1]} $<$ Roll $\leq$ ±{Roll_QualBreakPts[2]})',
                        fr'{STDnumSigma} * $\sigma$ ± MFR (± {np.round(MyLineStdRoll, 1)} Degrees)',
                        f'Distribution of Measured Bird Roll']

            # plot legend
            axs[0, 0].legend(handles00, labels00, loc=1)
            del handles00; del labels00

            # Set labels
            axs[0, 0].set_xlabel('Bird Roll (degrees)')
            axs[0, 0].set_ylabel('counts')
            axs[0, 0].set_title(f'Bird Roll statistics for line {Line}')
            axs[0, 0].set_ylim([0, RollBincounts.max() * (1.0 + RollPitchBuff)])

            #
            axs[0, 1].fill_between(
                [MyLineMeanPitch - STDnumSigma * MyLineStdPitch, MyLineMeanPitch - STDnumSigma * MyLineStdPitch,
                 MyLineMeanPitch + STDnumSigma * MyLineStdPitch, MyLineMeanPitch + STDnumSigma * MyLineStdPitch],
                [0, PitchBincounts.max() * (1.0 + RollPitchBuff), PitchBincounts.max() * (1.0 + RollPitchBuff), 0],
                color=STDPitchColor, alpha=ThePlottingOpacity)
            axs[0, 1].plot([MyLineMeanPitch, MyLineMeanPitch],
                           [0 - RollPitchBuff, PitchBincounts.max() * (1.0 + RollPitchBuff)], color=MeanPitchColor)
            axs[0, 1].plot([Pitch_QualBreakPts[0], Pitch_QualBreakPts[0]],
                           [0 - RollPitchBuff, PitchBincounts.max() * (1.0 + RollPitchBuff)], color=RPgoodColor)
            axs[0, 1].plot([Pitch_QualBreakPts[0] * -1, Pitch_QualBreakPts[0] * -1],
                           [0 - RollPitchBuff, PitchBincounts.max() * (1.0 + RollPitchBuff)], color=RPgoodColor)
            axs[0, 1].plot([Pitch_QualBreakPts[1], Pitch_QualBreakPts[1]],
                           [0 - RollPitchBuff, PitchBincounts.max() * (1.0 + RollPitchBuff)], color=RPokayColor)
            axs[0, 1].plot([Pitch_QualBreakPts[1] * -1, Pitch_QualBreakPts[1] * -1],
                           [0 - RollPitchBuff, PitchBincounts.max() * (1.0 + RollPitchBuff)], color=RPokayColor)
            axs[0, 1].plot([Pitch_QualBreakPts[2], Pitch_QualBreakPts[2]],
                           [0 - RollPitchBuff, PitchBincounts.max() * (1.0 + RollPitchBuff)], color=RPbadColor)
            axs[0, 1].plot([Pitch_QualBreakPts[2] * -1, Pitch_QualBreakPts[2] * -1],
                           [0 - RollPitchBuff, PitchBincounts.max() * (1.0 + RollPitchBuff)], color=RPbadColor)
            axs[0, 1].hist(MyLinePitch, bins=PitchBin, color=HistPitchColor)

            # create legend
            handles01 = ([Line2D((0, 0), (0, 0), color=c) for c in [MeanPitchColor, RPgoodColor, RPokayColor, RPbadColor]] +
                         [Rectangle((0, 0), 1, 1, color=c[0], ec=c[0], alpha=c[1])
                          for c in [(STDPitchColor, ThePlottingOpacity), (HistPitchColor, 1)]])
            labels01 = [f'Mean Flight Pitch (MFP :  {np.round(MyLineMeanPitch, 1)} degrees)',
                        fr'    Good Pitch angles (                             Pitch $\leq$ ±{Pitch_QualBreakPts[0]})',
                        fr'    Okay Pitch angles (±{Pitch_QualBreakPts[0]} $<$ Pitch $\leq$ ±{Pitch_QualBreakPts[1]})',
                        fr'Mediocre Pitch angles (±{Pitch_QualBreakPts[1]} $<$ Pitch $\leq$ ±{Pitch_QualBreakPts[2]})',
                        fr'{STDnumSigma} * $\sigma$ ± MFP (± {np.round(MyLineStdPitch, 1)} Degrees)',
                        f'Distribution of Measured Bird Pitch']

            # plot legend
            axs[0, 1].legend(handles01, labels01, loc=1)
            del handles01; del labels01

            # Set labels
            axs[0, 1].set_xlabel('Bird Pitch (degrees)')
            axs[0, 1].set_ylabel('counts')
            axs[0, 1].set_title(f'Bird Pitch statistics for line {Line}')
            axs[0, 1].set_ylim([0, PitchBincounts.max() * (1.0 + RollPitchBuff)])

            #
            gs = axs[1, 0].get_gridspec()
            # remove the underlying axes
            for ax in axs[1, :]:
                ax.remove()
            axs10 = fig.add_subplot(gs[1, :])

            axs10.fill_between([0, np.nanmax(MyLinexdist), np.nanmax(MyLinexdist), 0],
                               [MyLineMeanRoll - STDnumSigma * MyLineStdRoll, MyLineMeanRoll - STDnumSigma * MyLineStdRoll,
                                MyLineMeanRoll + STDnumSigma * MyLineStdRoll, MyLineMeanRoll + STDnumSigma * MyLineStdRoll],
                               color=STDRollColor, alpha=ThePlottingOpacity)
            [axs10.plot([dld, dld], [-90, 90], color='lightgrey', linestyle='dotted') for dld in
             range(1000, np.int64(np.ceil(MyLinexdist.max() / 1000) * 1000) + 1, 1000)]
            axs10.plot([0, np.nanmax(MyLinexdist)], [MyLineMeanRoll, MyLineMeanRoll], color=MeanRollColor)
            axs10.plot([0, np.nanmax(MyLinexdist)], [Roll_QualBreakPts[0],      Roll_QualBreakPts[0]],      color=RPgoodColor)
            axs10.plot([0, np.nanmax(MyLinexdist)], [Roll_QualBreakPts[0] * -1, Roll_QualBreakPts[0] * -1], color=RPgoodColor)
            axs10.plot([0, np.nanmax(MyLinexdist)], [Roll_QualBreakPts[1],      Roll_QualBreakPts[1]],      color=RPokayColor)
            axs10.plot([0, np.nanmax(MyLinexdist)], [Roll_QualBreakPts[1] * -1, Roll_QualBreakPts[1] * -1], color=RPokayColor)
            axs10.plot([0, np.nanmax(MyLinexdist)], [Roll_QualBreakPts[2],      Roll_QualBreakPts[2]],      color=RPbadColor)
            axs10.plot([0, np.nanmax(MyLinexdist)], [Roll_QualBreakPts[2] * -1, Roll_QualBreakPts[2] * -1], color=RPbadColor)
            axs10.plot(MyLinexdist, MyLineRoll, color=HistRollColor)

            # create legend
            handles10 = ([Line2D((0, 0), (0, 0), color=c[0], alpha=c[1]) for c in
                          [(HistRollColor, 1), (RPgoodColor, 1), (RPokayColor, 1), (RPbadColor, 1), (MeanRollColor, 1)]] +
                         [Rectangle((0, 0), 1, 1, color=c[0], ec=c[0], alpha=c[1])
                          for c in [(STDRollColor, ThePlottingOpacity)]])
            labels10 = [f'Measured Bird Roll',
                        fr'    Good Roll angles (                            Roll $\leq$ ±{Roll_QualBreakPts[0]})',
                        fr'    Okay Roll angles (±{Roll_QualBreakPts[0]} $<$ Roll $\leq$ ±{Roll_QualBreakPts[1]})',
                        fr'Mediocre Roll angles (±{Roll_QualBreakPts[1]} $<$ Roll $\leq$ ±{Roll_QualBreakPts[2]})',
                        f'Mean Bird Roll (MBR :  {np.round(MyLineMeanRoll, 1)} deg.)',
                        fr'{STDnumSigma} * $\sigma$ ± MBR (± {np.round(MyLineStdRoll, 1)} deg.)']

            # plot legend
            axs10.legend(handles10, labels10)
            del handles10; del labels10

            # Set labels
            axs10.set_ylabel('Bird Roll (degrees)')
            axs10.set_xlabel('Downline distance (m)')
            axs10.set_title(f'Bird Roll for line {Line}')
            axs10.set_xlim([0, np.nanmax(MyLinexdist)])
            axs10.set_ylim(
                [np.nanmin(np.append([np.nanmin(MyLineRoll)], [np.array(Roll_QualBreakPts) * -1.])) * (1 + RollPitchBuff),
                 np.nanmax(np.append([np.nanmax(MyLineRoll)], [np.array(Roll_QualBreakPts) * 1.])) * (1 + RollPitchBuff)])

            gs = axs[2, 0].get_gridspec()
            # remove the underlying axes
            for ax in axs[2, :]:
                ax.remove()
            axs20 = fig.add_subplot(gs[2, :])

            axs20.fill_between([0, np.nanmax(MyLinexdist), np.nanmax(MyLinexdist), 0],
                               [MyLineMeanPitch - STDnumSigma * MyLineStdPitch, MyLineMeanPitch - STDnumSigma * MyLineStdPitch,
                                MyLineMeanPitch + STDnumSigma * MyLineStdPitch, MyLineMeanPitch + STDnumSigma * MyLineStdPitch],
                               color=STDPitchColor, alpha=ThePlottingOpacity)
            [axs20.plot([dld, dld], [-90, 90], color='lightgrey', linestyle='dotted') for dld in
             range(1000, np.int64(np.ceil(MyLinexdist.max() / 1000) * 1000) + 1, 1000)]
            axs20.plot([0, np.nanmax(MyLinexdist)], [MyLineMeanPitch, MyLineMeanPitch], color=MeanPitchColor)
            axs20.plot([0, np.nanmax(MyLinexdist)], [Pitch_QualBreakPts[0],      Pitch_QualBreakPts[0]],      color=RPgoodColor)
            axs20.plot([0, np.nanmax(MyLinexdist)], [Pitch_QualBreakPts[0] * -1, Pitch_QualBreakPts[0] * -1], color=RPgoodColor)
            axs20.plot([0, np.nanmax(MyLinexdist)], [Pitch_QualBreakPts[1],      Pitch_QualBreakPts[1]],      color=RPokayColor)
            axs20.plot([0, np.nanmax(MyLinexdist)], [Pitch_QualBreakPts[1] * -1, Pitch_QualBreakPts[1] * -1], color=RPokayColor)
            axs20.plot([0, np.nanmax(MyLinexdist)], [Pitch_QualBreakPts[2],      Pitch_QualBreakPts[2]],      color=RPbadColor)
            axs20.plot([0, np.nanmax(MyLinexdist)], [Pitch_QualBreakPts[2] * -1, Pitch_QualBreakPts[2] * -1], color=RPbadColor)
            axs20.plot(MyLinexdist, MyLinePitch, color=HistPitchColor)

            # create legend
            handles20 = ([Line2D((0, 0), (0, 0), color=c[0], alpha=c[1]) for c in
                          [(HistPitchColor, 1), (RPgoodColor, 1), (RPokayColor, 1), (RPbadColor, 1), (MeanPitchColor, 1)]] +
                         [Rectangle((0, 0), 1, 1, color=c[0], ec=c[0], alpha=c[1]) for c in
                          [(STDPitchColor, ThePlottingOpacity)]])
            labels20 = [f'Measured Bird Pitch',
                        fr'    Good Pitch angles (                             Pitch $\leq$ ±{Pitch_QualBreakPts[0]})',
                        fr'    Okay Pitch angles (±{Pitch_QualBreakPts[0]} $<$ Pitch $\leq$ ±{Pitch_QualBreakPts[1]})',
                        fr'Mediocre Pitch angles (±{Pitch_QualBreakPts[1]} $<$ Pitch $\leq$ ±{Pitch_QualBreakPts[2]})',
                        f'Mean Bird Pitch (MBP :  {np.round(MyLineMeanPitch, 1)} deg.)',
                        fr'{STDnumSigma} * $\sigma$ ± MBP (± {np.round(MyLineStdPitch, 2)} deg.)']

            # plot legend
            axs20.legend(handles20, labels20)
            del handles20; del labels20

            # Set labels
            axs20.set_ylabel('Bird Pitch (degrees)')
            axs20.set_xlabel('Downline distance (m)')
            axs20.set_title(f'Bird Pitch for line {Line}')
            axs20.set_xlim([0, np.nanmax(MyLinexdist)])
            axs20.set_ylim(
                [np.nanmin(np.append([np.nanmin(MyLinePitch)], [np.array(Pitch_QualBreakPts) * -1.])) * (1 + RollPitchBuff),
                 np.nanmax(np.append([np.nanmax(MyLinePitch)], [np.array(Pitch_QualBreakPts) * 1.])) * (1 + RollPitchBuff)])

            fig.tight_layout()
            plt.show

        else:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print(f'%    Roll:       There are: {MyLineRoll[np.isnan(MyLineRoll)].size} NaN values')
            print(f'%        Out of a total of: {MyLineRoll.size}')
            print('% ')
            print(f'%    Pitch:      There are: {MyLinePitch[np.isnan(MyLinePitch)].size} NaN values')
            print(f'%        Out of a total of: {MyLinePitch.size}')
            print('% ')
            print('%    No Roll Pitch File printed')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        if self.save_fig:
            self.move_old_files('RollPitchQC', f'RollPitch_L{Line}*.png')
            plt.savefig(os.path.join(self.PlotDirectory, 'RollPitchQC', f'RollPitch_L{Line}_{self.now()}.png'),
                        facecolor='white')
        if 'show' not in self.ShowPlot:
            plt.close('all')

            
        return fig, axs
