import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .utils import calc_lineOffset, resampleWaveform, getGateTimesFromGEX, splitData_lines
from .setup import allowed_moments
import contextily
import rasterio
from rasterio.plot import show
import copy
import os

def plotdBdt(data,
             gate_times,
             noise=None,
             ax=None,
             scaled=True,
             plotSTD=False,
             **kw):
    legend2display = []
    if ax is None:
        fig, ax  = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()
    for moment in allowed_moments:
        channel = moment.split('_')[-1]
        if channel == 'Ch01':
            color = '#1f77b4'
        else:
            color = '#ff7f0e'
        
        if scaled:
            channel_key = 'Gate_scaled_' + channel
            culled_key = 'Gate_scaled_culled_' + channel
        else:
            channel_key = 'Gate_' + channel
            culled_key = 'Gate_culled_' + channel
        if (culled_key in data.layer_data.keys()) and (channel_key in data.layer_data.keys()):
            ax.plot(gate_times[moment],
                    np.abs(data.layer_data[channel_key].values.T),
                    color='lightgrey',
                    label='{} culled'.format(channel),
                    **kw)
            handles, labels = ax.get_legend_handles_labels()
            legend2display.append(len(handles)-1)
            if not plotSTD:
                ax.plot(gate_times[moment],
                        np.abs(data.layer_data[culled_key].values.T),
                        '.-', ms=2,
                        color=color,
                        label='{} used'.format(channel),
                        **kw)
            else:
                std_key = 'STD_'+channel
                # ax.set_prop_cycle(None)
                for n in range(len(data.layer_data[culled_key])):
                    ax.errorbar(gate_times[moment],
                                np.abs(data.layer_data[culled_key].loc[n, :].values),
                                yerr=np.abs(data.layer_data[culled_key].loc[n, :].values) * np.abs(data.layer_data[std_key].loc[n, :].values),
                                color=color,
                                label='{} used'.format(channel),
                                **kw)
                # ax.errorbar(np.tile(gate_times[moment], (data.layer_data[culled_key].shape[0],1)),
                #             np.abs(data.layer_data[culled_key].values),
                #             yerr=np.abs(data.layer_data[culled_key].values) * np.abs(data.layer_data[std_key].values),
                #             color=color,
                #             label='{} used'.format(channel),
                #             **kw)
            
            handles, labels = ax.get_legend_handles_labels()
            legend2display.append(len(handles)-1)
        elif channel_key in data.layer_data.keys():
            ax.plot(gate_times[moment],
                    np.abs(data.layer_data[channel_key].values.T),
                    '.-', ms=2,
                    color=color,
                    label=channel_key,
                    **kw)
            handles, labels = ax.get_legend_handles_labels()
            legend2display.append(len(handles)-1)
        else:
            print('Nothing to plot for {}:-('.format(channel))
        
        if noise is not None:
            for moment in allowed_moments:
                if moment in noise.keys():
                    ax.plot(gate_times[moment],
                                noise[moment],
                                '--',
                                color='grey',
                                label='noise level')
            handles, labels = ax.get_legend_handles_labels()
            legend2display.append(len(handles)-1)
    
    ax.legend([handle for i, handle in enumerate(handles) if i in legend2display],
           [label for i, label in enumerate(labels) if i in legend2display], loc = 'best')
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()
    # ax.set_title('AEM responses in subset')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('dB/dt [V/Am$^2$]')
    return fig, ax

def plotRhoa(data, gate_times):
    
    legend2display = []
    fig, ax = plt.subplots(figsize=(9, 8))
    for moment in allowed_moments:
        channel = moment.split('_')[-1]
        if channel == 'Ch01':
            color = '#1f77b4'
        else:
            color = '#ff7f0e'
        rhoa_key = 'Rhoa_'+channel
        if rhoa_key in data.layer_data.keys():
            ax.plot(gate_times[moment],
                    np.abs(data.layer_data[rhoa_key].values.T),
                    '.-',
                    color=color,
                    lw=1, ms=2,
                    label=channel)
            handles, labels = ax.get_legend_handles_labels()
            legend2display.append(len(handles)-1)
        else:
            print('Nothing to plot for {}:-('.format(channel))
        
    ax.legend([handle for i, handle in enumerate(handles) if i in legend2display],
    [label for i, label in enumerate(labels) if i in legend2display], loc = 'best')
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()
    ax.set_xlabel('time [s]')
    ax.set_ylabel('dB/dt [V/Am$^2$]')
    return fig, ax


def plotGates(data, gate_times):
    
    legend2display = []
    fig, ax  = plt.subplots(figsize=(9, 8))
    for moment in allowed_moments:
        print(moment)
        channel = moment.split('_')[-1]
        if channel == 'Ch01':
            color = '#1f77b4'
        else:
            color = '#ff7f0e'
        if moment in data.layer_data.keys():
            ax.plot(gate_times[moment],
                    np.abs(data.layer_data[moment].values.T),
                    '.-',
                    color=color,
                    lw=1, ms=2,
                    label=channel)
            handles, labels = ax.get_legend_handles_labels()
            legend2display.append(len(handles)-1)
        else:
            print('Nothing to plot for {}:-('.format(channel))
        
    ax.legend([handle for i, handle in enumerate(handles) if i in legend2display],
    [label for i, label in enumerate(labels) if i in legend2display], loc = 'best')
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()
    # ax.set_title('AEM responses in subset')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('dB/dt [V/Am$^2$]')
    return fig, ax


def plotSingelMomentProfile(data, moment, ax, xkey='lineoffset', plot_scaled=True, plotSTD=True, **kw):
    std_factor = 1
    channel = moment.split('_')[-1]
    if plot_scaled:
        chan_key = 'Gate_scaled_' + channel
        culled_key = 'Gate_scaled_culled_' + channel
    else:
        chan_key = 'Gate_' + channel
        culled_key = 'Gate_culled_' + channel
    std_key = 'STD_' + channel
    std_culled_key = 'STD_culled_' + channel
    
    if not(culled_key in data.layer_data.keys()):
        print('not plotting culled data, no culled key found')
        filt = ~data.layer_data[chan_key].isna().all(axis=1)
        ax.plot(data.flightlines[xkey].loc[filt],
                np.abs(data.layer_data[chan_key].loc[filt].values),
                label='_nolegend_',
                **kw)
        if std_culled_key in data.layer_data.keys() and plotSTD:
            ax.set_prop_cycle(None)
            for c in data.layer_data[std_culled_key].columns:
                ax.errorbar(data.flightlines[xkey].loc[filt],
                            np.abs(data.layer_data[culled_key].loc[filt, c].values),
                            yerr=np.abs(data.layer_data[culled_key].loc[filt, c].values) * (std_factor*data.layer_data[std_culled_key].loc[filt, c].values),
                            label='_nolegend_',
                            **kw)
    else:
        print('plotting culled data')
        filt = ~data.layer_data[chan_key].isna().all(axis=1)
        filt_culled = ~data.layer_data[chan_key].isna().all(axis=1)
        ax.plot(data.flightlines[xkey].loc[filt],
                np.abs(data.layer_data[chan_key].loc[filt].values),
                color='lightgrey',
                label='_nolegend_',
                **kw)
        ax.plot(data.flightlines[xkey].loc[filt_culled],
                np.abs(data.layer_data[culled_key].loc[filt_culled].values),
                label='_nolegend_',
                **kw)
        if std_culled_key in data.layer_data.keys() and plotSTD:
            ax.set_prop_cycle(None)
            for c in data.layer_data[std_culled_key].columns:
                ax.errorbar(data.flightlines[xkey].loc[filt_culled],
                            np.abs(data.layer_data[culled_key].loc[filt_culled, c].values),
                            yerr=np.abs(data.layer_data[culled_key].loc[filt_culled, c].values) * (std_factor*data.layer_data[std_culled_key].loc[filt_culled, c].values),
                            label='_nolegend_',
                            **kw)
    
    ax.set_yscale('log')
    ax.grid()
    ax.set_title(moment)
    ax.set_xlabel(xkey)
    ax.set_ylabel('dB/dt [V/Am$^2$]')


def plotSingleErrorProfile(data, err_key, ax, xkey='lineoffset', clim=[0, 6], **kw):
    for col in data.layer_data[err_key].columns:
        if data.layer_data[err_key].loc[:, col].notna().sum() > 0:
            scaled_key = 'Gate_'+err_key.split('_')[-1]
            sc = ax.scatter(data.flightlines[xkey].loc[:],
                       data.layer_data[scaled_key].loc[:, col].abs(),
                       c=data.layer_data[err_key].loc[:, col],
                       s=3,
                      cmap='jet',
                      vmin=clim[0],
                      vmax=clim[1],
                      zorder=3)
    ax.set_yscale('log')
    ax.set_title(err_key)
    ax.grid()
    plt.colorbar(sc)


def plotErrorProfile(data, ax=[None, None], xkey='utmy'):
    if not(ax[0]):
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    errKeys = ['relErr_Ch01', 'relErr_Ch02']
    if (errKeys[0] in data.layer_data.keys()) and (errKeys[1] in data.layer_data.keys()):
        plotSingleErrorProfile(data, errKeys[0], ax[0], xkey=xkey)
        plotSingleErrorProfile(data, errKeys[1], ax[1], xkey=xkey)
    elif errKeys[0] in data.layer_data.keys():
        plotSingleErrorProfile(data, errKeys[0], ax, xkey=xkey)
    else:
        raise Exception('found no errors to plot in data')


def plotdBdtProfile(data, ax=None, xkey='lineoffset', plot_scaled=True, **kw):
    if xkey == 'lineoffset' and not( 'lineoffset' in data.flightlines.columns):
        calc_lineOffset(data)
    
    if ('Gate_Ch01' in data.layer_data.keys()) and ('Gate_Ch02' in data.layer_data.keys()):
        single_moment = False
    else:
        single_moment = True
    
    if ax is None:
        if single_moment:
            fig, ax = plt.subplots(figsize=(13, 4))
        else:
            fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        noInputAxis = True
    else:
        noInputAxis = False
    
    if single_moment:
        if len(ax) > 1:
            axs = ax[0]
        else:
            axs = ax
        plotSingelMomentProfile(data, allowed_moments[0], axs, xkey=xkey, plot_scaled=plot_scaled, **kw)
        axs.set_title('Single moment data')
        # if plot_scaled:
        #     ax.set_ylim([1e-10, 1e-4])
    else:
        plotSingelMomentProfile(data, allowed_moments[0], ax[0], xkey=xkey, plot_scaled=plot_scaled, **kw)
        ax[0].set_title('LM data', fontsize=11)
        ax[0].set_xlabel('')
        plotSingelMomentProfile(data, allowed_moments[1], ax[1], xkey=xkey, plot_scaled=plot_scaled, **kw)
        ax[1].set_title('HM data', fontsize=11)
        # if plot_scaled:
        #     ax[0].set_ylim([1e-10, 1e-3])
        #     ax[1].set_ylim([1e-10, 1e-4])

    if noInputAxis:
        return fig, ax


def pcolormesh_with_gaps(ax, local_x, depth, image_data, clim, cmap='jet', shading='auto'):
    x_col = pd.DataFrame({'x': local_x}).reset_index(drop=True)
    data = pd.DataFrame(image_data.T)
    df = pd.concat([x_col, data], axis=1).reset_index(drop=True)
    
    depth = pd.DataFrame(depth.T)
    df_depth = pd.concat([x_col, depth], axis=1).reset_index(drop=True)
    
    df['dx'] = df.x.diff()
    dx_median = df.dx.median()
    filt = df.dx > 3 * dx_median
    for idx in df.loc[filt].index:
        rows = copy.deepcopy(df.loc[idx-1:idx])
        rows.iloc[:, 1:] = np.nan
        rows.loc[idx-1, 'x'] = rows.loc[idx-1, 'x']+dx_median
        rows.loc[idx, 'x'] = rows.loc[idx, 'x']-dx_median
        df = pd.concat([df, rows], axis=0, ignore_index=True)
        
        depth_rows = copy.deepcopy(df_depth.loc[idx-1:idx])
        depth_rows.loc[idx-1, 'x'] = rows.loc[idx-1, 'x']+dx_median
        depth_rows.loc[idx, 'x'] = rows.loc[idx, 'x']-dx_median
        df_depth = pd.concat([df_depth, depth_rows], axis=0, ignore_index=True)
    
    df.sort_values('x', inplace=True)
    df_depth.sort_values('x', inplace=True)
    
    local_x = df['x'].values
    image_data = df.drop(columns=['x', 'dx']).values.T
    depth = df_depth.drop(columns=['x']).values.T
    pm = ax.pcolormesh(local_x, depth, image_data, cmap=cmap, shading=shading, vmin=clim[0], vmax=clim[1])
    return pm


def cbcalc_cbar_lin_tick_and_labels_for_logscale(clim=[0, 4]):
    mult = 0.1
    dum = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    cblabel = []
    while mult <= 10000:
        cblabel.extend(dum * mult)
        mult *= 10
    cblabel.append(100000)
    cblabel.append(10 ** (clim[0]))
    cblabel.append(10 ** (clim[1]))
    cblabel = pd.Series(np.sort(np.unique(np.round(cblabel, 1))))
    cblabel = cblabel.drop(cblabel.loc[cblabel < 10 ** (clim[0] - 0.01)].index)
    cblabel = cblabel.drop(cblabel.loc[cblabel > 10 ** (clim[1] + 0.01)].index)
    cbtick = np.log10(cblabel).round(4)
    cblabel = cblabel.to_list()
    cbtick = cbtick.to_list()
    for inder in range(0, len(cblabel)):
        tstring = str(cblabel[inder])
        tstring = tstring.split('.0')[0]
        cblabel[inder] = tstring
    return cbtick, cblabel

def plot_model_section(model, ax, keyx="utmy", res_key='rho_i', elev_key='elevation', doi_key='doi_standard',
                       cmap="jet", clim=[-1, 4], hideBelowDOI=True, attr='log_res', cb_orientation='vertical',
                       showRMS=False, plot_invalt=True, cb_shrink=2):  # cb_shrink=0.33):
    # some preparations
    if ('lineoffset' in keyx) and not('lineoffset' in model.flightlines.columns):
        calc_lineOffset(model)
    
    if attr == 'log_res':
        image_data = np.log10(model.layer_data[res_key].values.T)
        # cb_label = r'Resistivity [log10($\Omega$m)]'
        cb_label = r'Resistivity [$\Omega$m]'
    elif attr == 'res':
        image_data = model.layer_data[res_key].values.T
        cb_label = r'Resistivity [$\Omega$m]'
    elif attr == 'cond':
        image_data = 1./model.layer_data[res_key].values.T
        cb_label = r'Conductivity [S/m]'
    elif attr == 'log_cond':
        image_data = np.log10(1./model.layer_data[res_key].values.T)
        cb_label = r'Conductivity [log10(S/m)]'
    elif attr == 'log_cond_mSm':
        image_data = np.log10(1./model.layer_data[res_key].values.T*1000)
        cb_label = r'Conductivity [log10(mS/m)]'
    
    local_x = model.flightlines[keyx]
    
    depth_top = model.layer_data['dep_top'].values.T
    if model.layer_data['dep_top'].shape[1] == model.layer_data['dep_bot'].shape[1]+1:
        df = model.layer_data['dep_bot']
        ncol = df.shape[1]
        maxDepth = df.loc[:, ncol-1]+50
        model.layer_data['dep_bot'].insert(ncol, df.columns[-1]+1, maxDepth)
    depth_bot = model.layer_data['dep_bot'].values.T
    depth = model.flightlines[elev_key].values - (depth_top+depth_bot)/2
    
    # actual plotting
    if 'lineoffset' in keyx:
        pm = pcolormesh_with_gaps(ax, local_x, depth, image_data, clim, cmap=cmap, shading='auto')
    else:
        pm = ax.pcolormesh(local_x, depth, image_data, cmap=cmap, shading='auto', vmin=clim[0], vmax=clim[1])
    ax.set_ylabel('Elevation [m]')
    ax.set_xlabel(keyx)
    cb = plt.colorbar(pm, ax=ax, orientation=cb_orientation, shrink=cb_shrink)
    if attr == 'log_res':
        cbtick, cblabel = cbcalc_cbar_lin_tick_and_labels_for_logscale(clim)
        cb.set_ticks(cbtick)
        cb.set_ticklabels(cblabel, rotation=45)
    cb.set_label(cb_label)
    ax.grid(linestyle='--', linewidth=0.5)
    ax.plot(local_x, model.flightlines[elev_key] + model.flightlines.alt, 'b-', label="alt.", lw=.5)
    if plot_invalt:
        ax.plot(local_x, model.flightlines[elev_key] + model.flightlines.invalt, 'g--', label="inv. alt.", lw=.5)
    # ax.plot(local_x, model.flightlines[elev_key] - model.flightlines[doi_key], "k-", label="DOI", lw=.5)
    ax.legend()
    if hideBelowDOI:
        z = (model.flightlines[elev_key] - model.flightlines[doi_key]).values
        # zmax=(model.flightlines[doi_key] - model.layer_data['dep_bot'].max(axis=1)).values-100
        zmax = (model.flightlines[elev_key] - model.layer_data['dep_bot'].max(axis=1)).values-50
        poly_coords = []
        xbuffer = 0
        for n, x in enumerate(local_x):
            if n == 1:
                poly_coords.append((x+xbuffer, z[n]))
            elif n == len(local_x)-1:
                poly_coords.append((x-xbuffer, z[n]))
            else:
                poly_coords.append((x, z[n]))
        
        if local_x.iloc[0] < local_x.iloc[-1]:
            poly_coords.append( (local_x.max(), np.max(zmax)))
            poly_coords.append( (local_x.min(), np.max(zmax)))
        else:
            poly_coords.append( (local_x.min(), np.max(zmax)))
            poly_coords.append( (local_x.max(), np.max(zmax)))
        ax.add_patch(plt.Polygon(poly_coords, color='white', alpha=0.5))
    if showRMS:
        ax2 = ax.twinx()
        ax2.plot(local_x, model.flightlines.resdata)
        ax2.set_ylabel('data RMS')
        ax2.set_ylim([-6, 3])
    ax.set_xlim([local_x.min(), local_x.max()])
    

def demQCplot(dem_file, xyz, poskeys=['UTMX', 'UTMY']):
    fig, ax = plt.subplots(figsize=(8, 8))
    dem = rasterio.open(dem_file)
    s = show(dem.read(1), ax=ax, transform=dem.transform, vmin=0, vmax=300, cmap='jet')
    im = s.get_images()[0]
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('altitude [m]')
    ax.plot(xyz.flightlines[poskeys[0]],
           xyz.flightlines[poskeys[1]],
           'k.', ms=2,
           label='flightlines')
    ax.set_aspect('equal')
    ax.set_xlabel('easting')
    ax.set_ylabel('northing')
    ax.legend()
    ax.set_title('DEM:\n{}'.format(os.path.basename(dem_file)))


def plotRMSmap(model, ax=None, basemap=None, crs=None, buffer=500):
    if 'utmx' in model.flightlines.columns:
        poskeys = ['utmx', 'utmy']
    elif 'UTMX' in model.flightlines.columns:
        poskeys = ['UTMX', 'UTMY']
    else:
        raise Exception('not utmx/utmy or UTMX/UTMY keys found')
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(model.flightlines[poskeys[0]], 
                    model.flightlines[poskeys[1]], 
                    c=model.flightlines.resdata,
                    s=20,
                    cmap='jet',
                    vmin=0, vmax=6,
                    label='RMS line')
    ax.plot(model.flightlines[poskeys[0]].values[0], 
            model.flightlines[poskeys[1]].values[0],
            'ko', 
            markerfacecolor='none',
            ms=8, label='SOL')
    ax.legend()
    
    cb = plt.colorbar(sc, ax=ax, orientation='horizontal', shrink=0.5)
    cb.set_label('RMS')
    
    ax.set_xlim([model.flightlines[poskeys[0]].min()-buffer,
                 model.flightlines[poskeys[0]].max()+buffer])
    ax.set_ylim([model.flightlines[poskeys[1]].min()-buffer,
                 model.flightlines[poskeys[1]].max()+buffer])
    
    if basemap:
        if crs is None:
            raise Exception('if basemap is given, data CRS must be defined')
        else:
            contextily.add_basemap(ax, crs=crs, attribution=False, source=basemap)

    ax.set_aspect('equal')
    ax.set_xlabel('easting')
    ax.set_ylabel('nothing')
    

def plot_gate_times(gate_times, ax):
    for k in np.arange(gate_times.shape[0]):
        ax.plot(gate_times[k, 1:], np.zeros(2)+1e-3, '.-', label='gate{0:02d}'.format(k))


def plotGEX(gex):
    
    if 'WaveformLMPointInterpolated' not in gex.General.keys():
        gex = resampleWaveform(gex)
    
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(gex.General['WaveformLMPoint'][:, 0], gex.General['WaveformLMPoint'][:, 1], '.-', label='LM')
    ax[0].plot(gex.General['WaveformLMPointInterpolated'][:, 0], gex.General['WaveformLMPointInterpolated'][:, 1], '-', label='LM interpolated')
    # ax[0].semilogx(gex.General['WaveformHMPoint'][:,0],gex.General['WaveformHMPoint'][:,1], label='HM')
    ax[0].grid()
    ax[0].set_title('Waveform')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Current I [1]')
    # ax[0].set_ylim([gex.General['WaveformLMPoint'][:,1].min(), gex.General['WaveformLMPoint'][:,1].max()])
    plt.legend()
    
    ax[1].plot(gex.General['WaveformLMPointInterpolatedGrad'][:, 0],
              gex.General['WaveformLMPointInterpolatedGrad'][:, 1],
              '-')
    # ax[1].semilogx(gex.General['WaveformHMPoint'][:,0],gex.General['WaveformHMPoint'][:,1], label='HM')
    ax[1].grid()
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('d(I)/dt')
    # ax[1].set_ylim([gex.General['WaveformLMPoint'][:,1].min(), gex.General['WaveformLMPoint'][:,1].max()])
    
    ax[2].plot(gex.General['WaveformLMPointInterpolatedGradGrad'][:, 0],
              gex.General['WaveformLMPointInterpolatedGradGrad'][:, 1],
              '-')
    # ax[2].semilogx(gex.General['WaveformHMPoint'][:,0],gex.General['WaveformHMPoint'][:,1], label='HM')
    ax[2].grid()
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('d^2(I)/dt^2')
    # ax[2].set_ylim([gex.General['WaveformLMPoint'][:,1].min(), gex.General['WaveformLMPoint'][:,1].max()])
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    idx = gex.General['WaveformLMPoint'][:, 0] >= 0
    ax.semilogx(gex.General['WaveformLMPoint'][idx, 0], gex.General['WaveformLMPoint'][idx, 1], '.-', label='LM')
    Gt = getGateTimesFromGEX(gex, channel='Channel1')
    plot_gate_times(Gt, ax)
    ax.grid()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude [1]')
    # ax.set_ylim([gex.General['WaveformLMPoint'][:,1].min(), gex.General['WaveformLMPoint'][:,1].max()])
    ax.set_title('Rampdown and gate times ch01')
    ax.legend(loc='upper right')
    
    if 'WaveformHMPoint' in gex.General.keys():
        fig, ax = plt.subplots(figsize=(10, 6))
        idx = gex.General['WaveformHMPoint'][:, 0] >= 0
        ax.semilogx(gex.General['WaveformHMPoint'][idx, 0], gex.General['WaveformHMPoint'][idx, 1], '.-', label='LM')
        # ax.semilogx(gex.General['WaveformHMPoint'][:,0],gex.General['WaveformHMPoint'][:,1], label='HM')
        Gt = getGateTimesFromGEX(gex, channel='Channel2')
        plot_gate_times(Gt, ax)
        ax.grid()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude [1]')
        # ax.set_ylim([gex.General['WaveformLMPoint'][:,1].min(), gex.General['WaveformLMPoint'][:,1].max()])
        ax.set_title('Rampdown and gate times ch02')
        plt.legend()


def plotSR(sr, gex=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sr['system_response'][:, 0], sr['system_response'][:, 1], '.-', lw=1, ms=1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude [1]')
    ax.grid()
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sr['system_response'][:, 0], sr['system_response'][:, 1], '.-', lw=1, ms=1, label='Sys. response')
    ax.set_xlabel('Time [s]')
    ax.set_xscale('log')
    ax.set_ylabel('Amplitude [1]')
    ax.set_xlim([1e-7, sr['system_response'][:, 0].max()])
    ax.grid()
    plt.tight_layout()
    
    if gex:
        Gt = getGateTimesFromGEX(gex, channel='Channel1')
        plot_gate_times(Gt, ax)
        ax.legend(loc='upper right')


def plot_flightline(data, ax=None, legend=True, basemap=None, crs=None, label='', sol=True, buffer=200):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    if ('UTMX' in data.flightlines.columns) and ('UTMY' in data.flightlines.columns):
        poskeys = ['UTMX', 'UTMY']
    elif ('utmx' in data.flightlines.columns) and ('utmy' in data.flightlines.columns):
        poskeys = ['utmx', 'utmy']
    else:
        raise Exception('no utmx/utmy found in flightline columns')
    
    p = ax.plot(data.flightlines[poskeys[0]],
              data.flightlines[poskeys[1]], 
              '.',  markersize=3.0, label = 'line:{}'.format(label))
    if sol:
        ax.text(data.flightlines[poskeys[0]].iloc[0], 
                  data.flightlines[poskeys[1]].iloc[0], 
                  'SOL:{}'.format(label),
                  color=p[-1].get_color(),
                  fontsize=9)
    
    if buffer > 0:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim([xlim[0] - buffer, xlim[1] + buffer])
        ax.set_ylim([ylim[0] - buffer, ylim[1] + buffer])
    if basemap:
        if crs is None:
            raise Exception('if basemap is given, data CRS must be defined')
        else:
            contextily.add_basemap(ax, crs=crs, attribution=False, source=basemap)
    if legend:
        ax.legend()
    ax.set_aspect(1)
    ax.grid()
    ax.set_xlabel('Easting')
    ax.set_ylabel('Northing')


def plot_flightlines(data, line_key='Line', ax=None, legend=True, basemap=None, crs=None, sol=False, buffer=200):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        
    Lines = splitData_lines(data, line_key=line_key)
    
    for line in Lines.keys():
        plot_flightline(Lines[line], ax=ax, legend=False, basemap=None, crs=None, label=line, sol=sol, buffer=0)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim([xlim[0]-buffer, xlim[1]+buffer])
    ax.set_ylim([ylim[0]-buffer, ylim[1]+buffer])
    
    if basemap:
        if crs is None:
            raise Exception('if basemap is given, data CRS must be defined')
        else:
            contextily.add_basemap(ax, crs=crs, attribution=False, source=basemap)
    if legend:
        ax.legend()
    
    ax.set_aspect(1)


def dataQCplot(data, xkey='lineoffset', basemap=None, crs=None, label='', legend=True, plot_scaled=True, plotSTD=True):
    fig = plt.figure(figsize=(17, 10))
    ax = fig.subplot_mosaic(
        """
        AAE
        AAE
        BBE
        BBE
        CCE
        DDE
        """
    )
    ax['B'].sharex(ax['A'])
    ax['C'].sharex(ax['A'])
    ax['D'].sharex(ax['A'])
    
    data_ax = [ax['A'], ax['B']]

    plotdBdtProfile(data, ax=data_ax, xkey=xkey, plot_scaled=plot_scaled, plotSTD=plotSTD)
            
    ax['C'].plot(data.flightlines[xkey], data.flightlines.Topography, label='topo')
    ax['C'].plot(data.flightlines[xkey], data.flightlines.Topography + data.flightlines.TxAltitude, label='topo+alt')
    ax['C'].legend()
    ax['C'].grid()
    ax['C'].set_ylabel('Elevation [m]')

    ax['D'].plot(data.flightlines[xkey], data.flightlines.TxAltitude, label='alt')
    ax['D'].legend()
    ax['D'].grid()
    ax['D'].set_ylabel('Altitude [m]')
    
    plot_flightline(data, ax=ax['E'], basemap=basemap, crs=crs, label='', legend=legend)

    plt.tight_layout()
    return fig, ax

def inversionQCplot(model, data, synth, xkey='lineofffset', basemap=None, crs=None, clim=[0, 4], cmap='jet', plot_doi=True):
    fig = plt.figure(figsize=(17, 9))
    ax = fig.subplot_mosaic(
        """
        AAD
        BBD
        EED
        CCD
        CCD
        """
    )
    ax['B'].sharex(ax['A'])
    ax['C'].sharex(ax['A'])
    ax['E'].sharex(ax['A'])
    
    data_ax = [ax['A'], ax['B']]
    plotdBdtProfile(data, ax=data_ax, xkey=xkey, marker='.', linestyle='', plot_scaled=False, ms=2)
    for a in data_ax:
        a.set_prop_cycle(None)
    plotdBdtProfile(synth, ax=data_ax, xkey=xkey,  marker='', linestyle='-', plot_scaled=False,  ms=3, lw=1)
    for a in data_ax:
        a.set_xlabel(None)
    
    if 'rho' in model.layer_data.keys():
        res_key = 'rho'
    elif 'rho_i' in model.layer_data.keys():
        res_key = 'rho_i'

    plot_model_section(model, ax['C'], clim=clim, cb_orientation='horizontal', 
                       keyx=xkey, res_key=res_key, showRMS=False, cmap=cmap, 
                       hideBelowDOI=plot_doi)
    
    if plot_doi:
        ax['C'].set_ylim([model.flightlines.elevation.min()-model.flightlines.doi_standard.max()-5,
                          model.flightlines.elevation.max()+80]
                          )
    if 'numdata' in data.flightlines.keys():
        num_key = 'numdata'
    else:
        num_key = 'NUMDATA'
    
    line_data_rms = np.sqrt((model.flightlines.resdata**2 * model.flightlines[num_key]).sum() / model.flightlines[num_key].sum())
    ax['C'].set_title('data RMS for line: {0:2.2f}'.format(line_data_rms))
    
    plotRMSmap(model, ax=ax['D'], basemap=basemap, crs=crs, buffer=1000)
    ax['D'].grid()
    
    ax['E'].plot(model.flightlines[xkey], model.flightlines.resdata)
    ax['E'].set_ylabel('Data error [1]')
    ax['E'].grid()
    
    plt.tight_layout()
    return fig, ax
