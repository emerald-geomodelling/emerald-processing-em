#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:07:58 2023

@author: mp
"""

import numpy as np
from datetime import datetime
import pandas as pd

def downlinedist_calc(MyFlightData):
    SoundDist = np.insert((  (MyFlightData.utmx.iloc[1:].values - MyFlightData.utmx.iloc[:-1].values)**2
                           + (MyFlightData.utmy.iloc[1:].values - MyFlightData.utmy.iloc[:-1].values)**2)**0.5, 0,0)
    xdist = np.cumsum(SoundDist)
    return xdist, SoundDist


def speed_calc(MyFlightData):
    tarr = np.zeros([len(MyFlightData), 1])
    for jj in range(1, len(MyFlightData)):
        ii = jj - 1
        time1 = datetime.strptime(f'{MyFlightData.Date.loc[ii]} {MyFlightData.Time[ii]}', "%Y/%m/%d %H:%M:%S.%f")
        time2 = datetime.strptime(f'{MyFlightData.Date.loc[jj]} {MyFlightData.Time[jj]}', "%Y/%m/%d %H:%M:%S.%f")
        dtime = time2-time1
        tarr[jj] = dtime.total_seconds()
    MyFlightData['Time_step'] = tarr
    Speed = (MyFlightData.SoundDist/1000)/(MyFlightData.Time_step/60/60) # km/hr   # m->km / s->hr
    Speed[0] = 0
    return Speed


def SkyTEM_quality_calculator(MyFlightData,
                              Alt_QualBreakPts,   Alt_QualWeight,
                              Speed_QualBreakPts, Speed_QualWeight,
                              Roll_QualBreakPts,  Roll_QualWeight,
                              Pitch_QualBreakPts, Pitch_QualWeight):#,
                              #PLM_QualBreakPts,   PLM_QualWeight):

    """
    Calculate the Quality of a feature and the sounding as a whole
    """
    MyLineBirdAlt = MyFlightData.TxAltitude.copy()
    MyLineSpeed = MyFlightData.Speed.copy()
    MyLineRoll = MyFlightData.TxRoll.copy()
    MyLinePitch = MyFlightData.TxPitch.copy()

    MyLineAltQuality = np.zeros(MyLineBirdAlt.size)
    MyLineAltQuality[MyLineBirdAlt >= Alt_QualBreakPts[2]] = -1  # lose point for being too high
    MyLineAltQuality[MyLineBirdAlt <  Alt_QualBreakPts[2]] += 0  # no point for being below the high cutoff
    MyLineAltQuality[MyLineBirdAlt <= Alt_QualBreakPts[1]] += 1  # 1 point for being below the mid cutoff
    MyLineAltQuality[MyLineBirdAlt <= Alt_QualBreakPts[0]] += 1  # 1 point for being below the low cutoff

    MyLineSpeedQuality = np.zeros(len(MyLineSpeed))
    MyLineSpeedQuality[MyLineSpeed <= Speed_QualBreakPts[1]] += 1
    MyLineSpeedQuality[MyLineSpeed <= Speed_QualBreakPts[0]] += 1

    MyLineRollQuality = np.zeros(MyLineRoll.size)
    MyLineRollQuality[np.abs(MyLineRoll) >= Roll_QualBreakPts[2]] = -1
    MyLineRollQuality[np.abs(MyLineRoll) <  Roll_QualBreakPts[2]] += 0
    MyLineRollQuality[np.abs(MyLineRoll) <= Roll_QualBreakPts[1]] += 1
    MyLineRollQuality[np.abs(MyLineRoll) <= Roll_QualBreakPts[0]] += 1

    MyLinePitchQuality = np.zeros(MyLinePitch.size)
    MyLinePitchQuality[np.abs(MyLinePitch) >= Pitch_QualBreakPts[2]] = -1
    MyLinePitchQuality[np.abs(MyLinePitch) <  Pitch_QualBreakPts[2]] += 0
    MyLinePitchQuality[np.abs(MyLinePitch) <= Pitch_QualBreakPts[1]] += 1
    MyLinePitchQuality[np.abs(MyLinePitch) <= Pitch_QualBreakPts[0]] += 1

    #MyLinePLMQuality=np.zeros(MyLinePLM.size)
    #MyLinePLMQuality[MyLinePLM <= PLM_QualBreakPts[1]] = MyLinePLMQuality[MyLinePLM <= PLM_QualBreakPts[1]] + 1
    #MyLinePLMQuality[MyLinePLM <= PLM_QualBreakPts[0]] = MyLinePLMQuality[MyLinePLM <= PLM_QualBreakPts[0]] + 1

    # build temporary quality arrays. If Alt, roll, or pitch are -1 (unaccaptable) then 0 all quality measurements
    # as the sounding will be culled. Use as 0's so that the sounding is still factored into the final total quality
    TempAltQ = MyLineAltQuality.copy();       TempAltQ[(MyLineAltQuality == -1) | (MyLineRollQuality == -1) | (MyLinePitchQuality == -1)] = 0
    TempSpeedQ = MyLineSpeedQuality.copy(); TempSpeedQ[(MyLineAltQuality == -1) | (MyLineRollQuality == -1) | (MyLinePitchQuality == -1)] = 0
    TempRollQ = MyLineRollQuality.copy();    TempRollQ[(MyLineAltQuality == -1) | (MyLineRollQuality == -1) | (MyLinePitchQuality == -1)] = 0
    TempPitchQ = MyLinePitchQuality.copy(); TempPitchQ[(MyLineAltQuality == -1) | (MyLineRollQuality == -1) | (MyLinePitchQuality == -1)] = 0
    #TempPLMQ = MyLinePLMQuality.copy();      TempPLMQ[(MyLineAltQuality == -1) | (MyLineRollQuality == -1) | (MyLinePitchQuality == -1)] = 0

    MyLineTOTALQualityMaster = ((TempAltQ   * Alt_QualWeight) +
                                (TempSpeedQ * Speed_QualWeight) +
                                (TempRollQ  * Roll_QualWeight) +
                                (TempPitchQ * Pitch_QualWeight)) / (((len(Alt_QualBreakPts)-1)   * Alt_QualWeight) +
                                                                    (len(Speed_QualBreakPts)     * Speed_QualWeight) +
                                                                    ((len(Roll_QualBreakPts)-1)  * Roll_QualWeight) +
                                                                    ((len(Pitch_QualBreakPts)-1) * Pitch_QualWeight))

    MyLineTOTALQuality=0*MyLineTOTALQualityMaster.copy()+10
    MyLineTOTALQuality[MyLineTOTALQualityMaster < 1.0] += -1
    MyLineTOTALQuality[MyLineTOTALQualityMaster < 0.9] += -1
    MyLineTOTALQuality[MyLineTOTALQualityMaster < 0.8] += -1
    MyLineTOTALQuality[MyLineTOTALQualityMaster < 0.7] += -1
    MyLineTOTALQuality[MyLineTOTALQualityMaster < 0.6] += -1
    MyLineTOTALQuality[MyLineTOTALQualityMaster < 0.5] += -1
    MyLineTOTALQuality[MyLineTOTALQualityMaster < 0.4] += -1
    MyLineTOTALQuality[MyLineTOTALQualityMaster < 0.3] += -1
    MyLineTOTALQuality[MyLineTOTALQualityMaster < 0.2] += -1
    MyLineTOTALQuality[MyLineTOTALQualityMaster == 0] = 0

    TempMyLineGrade = np.cumsum(MyLineTOTALQualityMaster)[-1]/MyLineTOTALQualityMaster.size

    MyLineGrade = np.ones(len(MyLineTOTALQuality))*TempMyLineGrade
    quality_grade_dict = {'FlightData_index'  : MyFlightData.FlightData_index,
                          'MyLineAltQuality'  : MyLineAltQuality,
                          'MyLineSpeedQuality': MyLineSpeedQuality,
                          'MyLineRollQuality' : MyLineRollQuality,
                          'MyLinePitchQuality': MyLinePitchQuality,
                          #'MyLinePLMQuality'  : MyLinePLMQuality,
                          #'MyLineTOTALQuality': MyLineTOTALQuality,
                          'MyLineTOTALQuality': MyLineTOTALQualityMaster,
                          'MyLineGrade'       : MyLineGrade}
    quality_grade_df = pd.DataFrame.from_dict(quality_grade_dict)
    #print('\nquality_grade_df = ')
    #print(quality_grade_df)

    return quality_grade_df

def sampleDEM(DEM, df, poskeys=['utmx_laser1', 'utmy_laser1'], z_key='zdem_laser1'):
    filt=df[poskeys].isna().sum(axis=1)==0
    coord_list = [(x,y) for x,y in zip(df.loc[filt, poskeys[0]] , df.loc[filt, poskeys[1]])]
    df.loc[filt, z_key]=[x for x in DEM.sample(coord_list)]