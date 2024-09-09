
from datetime import datetime
import os.path
import glob

import pandas as pd

from .utils import downlinedist_calc, speed_calc


from .alt_qc import AltQCplotter
from .speed_qc import SpeedQCplotter
from .source_qc import SourceQCplotter
from .roll_pitch_qc import RollPitchQCplotter
from .quality_factors import QualityFactorPlotter
from .plotter_properties import PlotterProperties
from .flight_parameter_qc import FlightParameterPlotter

import yaml

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import contextily


class QCPlotter(PlotterProperties,
                SourceQCplotter,
                SpeedQCplotter,
                AltQCplotter,
                RollPitchQCplotter,
                QualityFactorPlotter,
                FlightParameterPlotter):

    def __init__(self, filename=None, **kw):
        if filename is not None:
            with open(filename) as fid:
                kw = yaml.safe_load(fid)
        for key, value in kw.items():
            setattr(self, key, value)

    def now(self):
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def move_old_files(self, dirname, filepattern):
        dirpath = os.path.join(self.PlotDirectory, dirname)
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)

        oldfile = glob.glob(os.path.join(dirpath, filepattern))
        if oldfile:
            oldfile = str(oldfile[0])
            oldfile = oldfile.split(os.path.sep)[-1]
            olddir = os.path.join(dirpath, 'old')
            if not os.path.isdir(olddir):
                os.makedirs(olddir)
            os.rename(
                os.path.join(dirpath, oldfile),
                os.path.join(olddir, oldfile))

    def _get_data(self, xyz=None, line=None):
        MyFlightData = xyz.flightlines.loc[xyz.flightlines[xyz.line_id_column] == line, :].copy()
        MyFlightData.rename(columns={xyz.x_column: "utmx",
                                     xyz.y_column: "utmy",
                                     xyz.z_column: "topo",
                                     xyz.line_id_column: "line_id"},
                            inplace=True)
        MyFlightData.reset_index(names='FlightData_index', inplace=True)
        MyFlightData['xdist'], MyFlightData['SoundDist'] = downlinedist_calc(MyFlightData)
        MyFlightData['Speed'] = speed_calc(MyFlightData)
        return MyFlightData
            
    def plot(self, xyz, lines=None):
        if lines is None: lines = xyz.flightlines.line_no.unique()
        for line in lines:
            MyFlightData = self._get_data(xyz=xyz, line=line)
            
            self._plot_altitude_stats(MyFlightData)
            self._plot_speed_stats(MyFlightData)
            self._plot_alt_vs_speed(MyFlightData)
            self._plot_roll_pitch_stats(MyFlightData)
            self._plot_quality_factors(MyFlightData)

    def plot_source_stats(self, xyz, lines=None):
        if lines is None: lines = xyz.flightlines.line_no.unique()
        for line in lines:
            MyFlightData = self._get_data(xyz=xyz, line=line)
            self._plot_source_stats(MyFlightData)
        
    def plot_current_stats(self, xyz, lines=None):
        if lines is None: lines = xyz.flightlines.line_no.unique()
        for line in lines:
            MyFlightData = self._get_data(xyz=xyz, line=line)
            self._plot_current_stats(MyFlightData, poskeys=['utmx', 'utmy'], basemap=self.basemap, crs=self.data_crs, buffer=self.map_buffer)
    
    def plot_source_stats_map(self, xyz, lines=None):
        if lines is None: lines = xyz.flightlines.line_no.unique()
        for line in lines:
            MyFlightData = self._get_data(xyz=xyz, line=line)
            self._plot_source_stats_map(MyFlightData, poskeys=['utmx', 'utmy'], basemap=self.basemap, crs=self.data_crs, buffer=self.map_buffer)

    def plot_flightParameter_overview(self, xyz, lines=None, dem=None):
        if lines is None: lines = xyz.flightlines.line_no.unique()
        for line in lines:
            MyFlightData = self._get_data(xyz=xyz, line=line)
            self._plot_flightParameter_overview(MyFlightData, poskeys=['utmx', 'utmy'], dem=dem)

    def plot_flight_parameter_map(self, xyz, lines=None, dem=None):
        if lines is None: lines = xyz.flightlines.line_no.unique()
        for line in lines:
            MyFlightData = self._get_data(xyz=xyz, line=line)
            self._plot_flight_parameter_map(MyFlightData, poskeys=['utmx', 'utmy'], basemap=self.basemap, crs=self.data_crs, buffer=self.map_buffer, dem=dem)
    
    def plot_FlightLine_map(self, xyz, ax=None, lines=None):
        rename_dict = {'UTMX': 'utmx',
                       'UTMY': 'utmy'}
        self._plot_FlightLines_map(xyz.flightlines.rename(columns=rename_dict), axs=ax, poskeys=['utmx', 'utmy'], lines=lines, basemap=self.basemap, crs=self.data_crs)

    def plot_altitude_stats(self, xyz, lines=None):
        if lines is None: lines = xyz.flightlines.line_no.unique()
        for line in lines:
            MyFlightData = self._get_data(xyz=xyz, line=line)
            self._plot_altitude_stats(MyFlightData)

    def plot_speed_stats(self, xyz, lines=None):
        if lines is None: lines = xyz.flightlines.line_no.unique()
        for line in lines:
            MyFlightData = self._get_data(xyz=xyz, line=line)
            self._plot_speed_stats(MyFlightData)

    def plot_alt_vs_speed(self, xyz, lines=None):
        if lines is None: lines = xyz.flightlines.line_no.unique()
        for line in lines:
            MyFlightData = self._get_data(xyz=xyz, line=line)
            self._plot_alt_vs_speed(MyFlightData)

    def plot_roll_pitch_stats(self, xyz, lines=None):
        if lines is None: lines = xyz.flightlines.line_no.unique()
        for line in lines:
            MyFlightData = self._get_data(xyz=xyz, line=line)
            self._plot_roll_pitch_stats(MyFlightData)

    def calc_quality_factors(self, xyz, lines=None):
        if lines is None: lines = xyz.flightlines.line_no.unique()
        Quality = []
        for line in lines:
            MyFlightData = self._get_data(xyz=xyz, line=line)
            Quality.append(self._calc_quality_factors(MyFlightData))
        return pd.concat(Quality)

    def assign_quality_factors(self, xyz):
        return self._assign_quality_factors(xyz)
    
    def save_quality_factors_to_shp(self, xyz, shp_file):
        flightlines = self._assign_quality_factors(xyz)
        self._save_quality_factors_to_shp(flightlines, shp_file)

    def plot_quality_factors(self, xyz, lines=None):
        if lines is None: lines = xyz.flightlines.line_no.unique()
        for line in lines:
            MyFlightData = self._get_data(xyz=xyz, line=line)
            self._plot_quality_factors(MyFlightData)
    
    
def PlotFlightLinesFromSPSXYZ(xyz, axs, crs, lines=None, basemap='default'):
    """
    xyz is an object where xyz.flightlines is a Dataframe
    axs is an axis object
    crs is the coordinate reference for the project
    lines, if given, is a list of lines desired to plot
    basemap options are 'default', None, or an url to a wms server as a string
    """
    if basemap == 'default':
        basemap = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    FlightDF = xyz.flightlines
    if lines is None:
        lines = sorted(FlightDF.Line.unique())
    _PlotFlightLinesFromSPS(FlightDF, axs, crs, lines, basemap)

def PlotFlightLinesFromQualitySPSXYZ(QualityXYZ, axs, crs, lines=None, basemap='default'):
    """
    QualityXYZ is a Dataframe
    axs is an axis object
    crs is the coordinate reference for the project
    lines, if given, is a list of lines desired to plot
    basemap options are 'default', None, or an url to a wms server as a string
    """
    if basemap == 'default':
        basemap = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    FlightDF = QualityXYZ
    if lines is None:
        lines = sorted(FlightDF.Line.unique())
    _PlotFlightLinesFromSPS(FlightDF, axs, crs, lines, basemap)


def _PlotFlightLinesFromSPS(FlightDF, axs, crs, lines, basemap):
    legendFont = mpl.font_manager.FontProperties(family='monospace',
                                                 weight='normal',
                                                 style='normal',
                                                 size=9)
    for line in lines:
        if line == 0:
            tflightlines = FlightDF.copy()
            tflightlines.Line.loc[tflightlines.Line != line] = np.NaN
        else:
            tflightlines = FlightDF.loc[FlightDF.Line == line]

        axs.plot(tflightlines.UTMX, tflightlines.UTMY, label=f'L{line}')
    if 0 in lines:
        zeroline = FlightDF.loc[FlightDF.Line == 0]
        axs.plot(zeroline.UTMX.iloc[0], zeroline.UTMY.iloc[0], marker='x', markersize=7, color='red',
                 label=f'Start X:  {zeroline.UTMX.iloc[0].round(2)}\n      Y: {zeroline.UTMY.iloc[0].round(2)}')
        axs.plot(zeroline.UTMX.iloc[-1], zeroline.UTMY.iloc[-1], marker='o', markersize=7, color='red', fillstyle='none',
                 label=f'  End X:  {zeroline.UTMX.iloc[-1].round(2)}\n      Y: {zeroline.UTMY.iloc[-1].round(2)}')
    axs.axis('equal')
    axs.legend(prop=legendFont)
    # plt.show()
    if basemap is not None:
        contextily.add_basemap(ax=axs, crs=crs, attribution=False, source=basemap)
    plt.show()
