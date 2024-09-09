#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:57:42 2023

@author: mp
"""

import numpy as np
import numpy.matlib




class PlotterProperties(object):
    PlotDirectory = "."

    ################ Define target values and plotting################
    ## Altitudue
    TargetAltitude = 40. #meters
    altfact = 0.375 # alt factor; decimal percentage (0.2 = 20%) 
    @property
    def MaxAltitude(self):
        return self.TargetAltitude*(1+self.altfact) #meters

    ## Speed 
    TargetSpeed = 90 #Km/Hr
    speedfact = 0.15 #speed factor; decimal percentage (0.2 = 20%) 
    @property
    def MaxSpeed(self):
        return self.TargetSpeed*(1+self.speedfact) # km/hr

    ################ Define Quality Factors for grading the data ################
    NumQualityFactors = 4 # 1) Alt, 2) Speed, 3) Roll, 4) pitch

    #QualityFactor1 Alt
    ## Quality factors ; 3 fields per factor, [f1, f2, f3] ; [val]<=f1 == 2 , f1<[val]<=f2 == 1 , f2<[val]<=f3 == 0 , f2<[val] ==  UNUSED; higher scores are better
    @property
    def Alt_QualBreakPts(self):
        return  [np.round(self.TargetAltitude*(1+(self.altfact)), 2), np.round(self.TargetAltitude*(1+(2.*self.altfact)), 2), 100] # Quality break points for 3 value system
    Alt_QualWeight = 2 # 0 : 1 : weight of factor

    #QualityFactor2 Speed
    ## Quality factors ; 2 fields per factor, [f1, f2] ; [val]<=f1 == 2 , f1<[val]<=f2 == 1 , f2<[val] ==  0; higher scores are better
    @property
    def Speed_QualBreakPts(self): 
        return [np.round(self.TargetSpeed*(1+(self.speedfact)), 2), np.round(self.TargetSpeed*(1+(2.*self.speedfact)), 2)] # m # Quality break points for 3 value system
    Speed_QualWeight = 1 # 0 : 1 : weight of factor

    #QualityFactor3 Roll
    ## Quality factors ; 3 fields per factor, [f1, f2, f3] ; [val]<=f1 == 2 , f1<[val]<=f2 == 1 , f2<[val]<=f3 == 0 , f2<[val] ==  UNUSED; higher scores are better
    Roll_QualBreakPts = [5, 7, 10] # ± deg. # Quality break points for 3 value system
    Roll_QualWeight = 2

    #QualityFactor4 Pitch
    ## Quality factors ; 3 fields per factor, [f1, f2, f3] ; [val]<=f1 == 2 , f1<[val]<=f2 == 1 , f2<[val]<=f3 == 0 , f2<[val] ==  UNUSED; higher scores are better
    Pitch_QualBreakPts = [5, 7, 10] # ± deg. # Quality break points for 3 value system
    Pitch_QualWeight = 2 # 0 : 1 : weight of factor
    
    #QualityFactor5 Source-current
    current_fact: 0.01  # varaitions over +/- 5% will be highlighted 
    TargetCurrent=[9.0, 109.0] # desired current output for LM/HM

    # #QualityFactor5 PLM
    # PLM_QualBreakPts = [0.15, 0.3] # STD values of whole dataset # Quality break points for 3 value system
    # PLM_QualWeight = 0.5 # 0 : 1 : weight of factor

    ################ Set Plotting properties ##################
    ## set number of std to apply
    STDnumSigma = 1 # number of STD sigmas for plotting 

    ## Set plotting buffers
    AltBuffer = 0.1 # %
    SpeedBuffer = 0.05 # %
    RollPitchBuff = 0.05 # %

    ## Colors etc
    
    ThePlottingOpacity=0.3
    Topo_Color='black'

    HistBirdAltColor='dodgerblue'
    TargAltColor='green'
    MaxAltColor='red'
    MeanBirdAltColor='purple'
    STDBirdAltColor='purple'
    
    TargSpeedColor='aquamarine'
    MaxSpeedColor='sienna'
    MeanSpeedColor='violet'
    STDSpeedColor='violet'
    HistSpeedColor='deepskyblue'

    HistRollColor='magenta'
    MeanRollColor='royalblue'
    STDRollColor='royalblue'
    
    HistPitchColor='Peru'
    MeanPitchColor='forestgreen'
    STDPitchColor='forestgreen'
    
    RPgoodColor='cyan'
    RPokayColor='darkseagreen'
    RPbadColor='firebrick'

    goodColor='green'
    okColor='purple'
    badColor='red'
    cullColor='darkgrey'
    
    map_buffer=500
    data_crs=None
    basemap=None
        
    ## Write plot to file and then 'show' or 'close'.
    ShowPlot = 'show'
    #ShowPlot = 'close'
    save_fig=True
    
    figsize=(10,9)
    fontsize=9
    axes_titlesize='medium'
