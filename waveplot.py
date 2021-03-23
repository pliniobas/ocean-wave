


def plot_spec_dire1(df, waveout, filename):
    """
    Plotagem do espectro direcional
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(waveout['sn'][:,0], waveout['sn'][:,1])
    ax1.set_title(str(df.index[0])[:-10])
    ax1.set_ylabel('Energia (m²/Hz)')
    ax1.grid()
    ax2 = fig.add_subplot(212)
    ax2.plot(waveout['sn'][:,0], waveout['dire1'])
    ax2.set_xlabel('Frequência (Hz)')
    ax2.set_ylabel('Direção (graus)')            
    ax2.grid()
    fig.savefig(path_fig + 'specdire_{}.png'.format(filename), bbox_inches='tight')
    plt.close('all')
    return


    import os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
import wafo.spectrum.models as wsm
import wafo.objects as wo
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.stats import gaussian_kde
plt.close('all')

def interp_spectra(p):
    """
    interpolacao para um vetor de frequencia fixo
    """
    freq_interp = np.linspace(0, 0.5, 50)
    spec1d_interp = []
    meandir_interp = []
    spread_interp = []
    for i in range(len(p['spec1d'])):
        spec1d_interp.append(np.interp(freq_interp, p['freq'].iloc[i,:], p['spec1d'].iloc[i,:]))
        meandir_interp.append(np.interp(freq_interp, p['freq'].iloc[i,:], p['meandir'].iloc[i,:]))
        spread_interp.append(np.interp(freq_interp, p['freq'].iloc[i,:], p['spread'].iloc[i,:]))
    spec1d_interp = pd.DataFrame(spec1d_interp, index=p['spec1d'].index)
    meandir_interp = pd.DataFrame(meandir_interp, index=p['meandir'].index)
    spread_interp = pd.DataFrame(spread_interp, index=p['spread'].index)
    return freq_interp, spec1d_interp, meandir_interp, spread_interp

def calc_spec2d_matrix(s, d, spr):
    """
    Calcula matriz do espectro 2d
    """
    thetas = np.linspace(0, 360, 360)
    d2 = np.zeros((len(thetas), len(freqi)))
    for i in range(len(freqi)):
        Dt = norm.pdf(thetas, d[i], spr[i]/2.35)
        d2[:,i] = s[i] * Dt
    return d2, thetas

def plot_posicao(param_wav, date):
    """
    plota mapa com as posições da waverider
    """
    fig = plt.figure(figsize=(8,6))
    map_proj = ccrs.Mercator()
    ax = fig.add_subplot(1,1,1, projection=map_proj)
    ax.set_title('WaveriderAzul\n{}'.format(date))
    ax.coastlines()
    ax.set_extent([param_wav['longitude'].iloc[-1]-2, param_wav['longitude'].iloc[-1]+2,
                   param_wav['latitude'].iloc[-1]-2, param_wav['latitude'].iloc[-1]+2],
                   crs=ccrs.PlateCarree())
    # ax.set_extent([-44, -42, -22, -25], crs=ccrs.PlateCarree())
    ax.plot(param_wav.longitude.values, param_wav.latitude.values, 'g-o', markersize=2,
               alpha=0.5, transform=ccrs.PlateCarree())
    # ax.scatter(x=gps_0x380.longitude.values, y=gps_0x380.latitude.values, color="green", s=10,
    #            alpha=0.5, transform=ccrs.PlateCarree())
    ax.scatter(x=param_wav.longitude.values[-1], y=param_wav.latitude.values[-1], color="red", s=50,
               alpha=1, transform=ccrs.PlateCarree())    
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
    # ax.set_xticks(gps_0x380['longitude'].iloc[::10], crs=ccrs.PlateCarree())
    # ax.set_yticks(gps_0x380['latitude'].iloc[::10], crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter(number_format='.1f',
    #                                    degree_symbol='',
    #                                    dateline_direction_label=True)
    # lat_formatter = LatitudeFormatter(number_format='.1f',
    #                                   degree_symbol='')
    # ax.xaxis.set_major_formatter(lon_formatter)
    # ax.yaxis.set_major_formatter(lat_formatter)
    return fig

def plot_spec1d_meandir_spread(freq, s, d, spr, date):
    """
    plotagem dos espectros
    """
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Frequência [Hz]')
    ax1.set_ylabel('S(f) [m²/Hz]', color=color)
    # ax1.plot(t, data1, color=color)
    ax1.fill(freq, s, alpha=0.5, color=color)#, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_xlim(freq.iloc[0], freq1d.iloc[-1])
    ax1.set_ylim(0, np.round(s.max(),1)+.5)
    ax1.set_title(date)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.plot(freq, d, alpha=1, color=color)
    ax2.plot(freq, d + spr/2, '--', color=color)
    ax2.plot(freq, d - spr/2, '--', color=color)
    ax2.set_ylabel('Direção', color=color)  # we already handled the x-label with ax1
    ax2.set_ylim(0, 360)
    plt.yticks(np.arange(0, 360+45, 45), ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'))
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    return fig

def plot_evolspec(freq, spec1d, date):
    spec1d[spec1d<0.5] = 0
    fig = plt.figure(figsize=(11,5))
    ax1 = fig.add_subplot(111)
    ax1.set_title('Evolução do Espectro 1D\n{}'.format(date))
    # lvls = [0.05, 0.1, 0.15, 0.3, 0.6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
    # lvls = np.linspace(0.5, max(spec1d), 50)
    CS = ax1.contourf(spec1d.index, freq, spec1d.values.T, levels=np.arange(0,11,1),
                      locator=None, colors=None, linewidths=1.2, cmap='jet')
    CS2 = ax1.contour(spec1d.index, freq, spec1d.values.T, levels=6,
                      locator=None, colors='black', linewidths=1, cmap=None)
    # plt.clabel(CS, inline=True, fontsize=8)
    # plt.imshow(spec1d.values.T, extent=[0, 5, 0, 5], origin='lower', cmap='jet', alpha=0.5)
    cbar = fig.colorbar(CS, orientation="vertical", pad=0.02)
    cbar.ax.set_ylabel('S(f) [m²/Hz]')
    # Add the contour line levels to the colorbar
    # cbar.add_lines(CS2)
   # plt.colorbar();
    ax1.set_ylabel('Frequência (Hz)')
    ax1.grid('on')
    ax1.set_ylim(0, 0.3)
    # ax1.set_xlim(bmop_spec1.index[0], bmop_spec1.index[-1])
    # ax1.set_ylim(0.05, .5)
    plt.xticks(rotation=15)
    return fig

def plot_spec2d(freqi, thetas, d2, date):
    """
    """
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    plt.title(date, fontsize=14, y=1.11)
    cc = ax.contourf(freqi, thetas, d2, levels = 100, cmap='jet', 
                     vmin=np.min(d2), vmax=np.max(d2))
    ax.set_xlabel('Frequência [Hz]')
    ax.set_ylabel('Direção [deg]')
    cbar = fig.colorbar(cc, pad=0.08, shrink=0.7, orientation='vertical')
    cbar.set_label('S(f,\u03B8) [m²/Hz/deg]', rotation=90, fontsize=12, labelpad=20)
    fig.tight_layout()
    return fig

def plot_polar_spec2d(freqi, thetas, d2, hm0, tp, dp, date):
    """
    Plota espectro 2d na projecao polar
    """
    r, theta = np.meshgrid(freqi, np.deg2rad(thetas))
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    cc = ax.contourf(theta, r, d2, levels = 100, cmap='jet', 
                    vmin=np.min(d2), vmax=np.max(d2))
    ax.set_ylim(0, 0.3)
    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[-1] = labels[-1] + ' Hz'
    ax.set_yticklabels(labels)
    plt.title('{}\nHm0: {:.1f} m, Tp: {:.1f} s, Dp: {:.0f}\u00b0'.format(date, hm0, tp, dp), fontsize=14, y=1.11)
    cbar = fig.colorbar(cc, pad=0.08, shrink=0.7, orientation='vertical')
    cbar.set_label('S(f,\u03B8) [m²/Hz/deg]', rotation=90, fontsize=12, labelpad=20)
    fig.tight_layout()
    return fig

def plot_distribuicao_conjunta(df):
    """
    """
    fig = plt.figure(figsize=(10,3))
    ax1 = fig.add_subplot(131)
    x, y = df.H_s.values, df.T_p.values
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    ax1.scatter(x, y, c=z)
    ax1.set_xlabel('Hs (m)')
    ax1.set_ylabel('Tp (s)')

    ax2 = fig.add_subplot(132)
    x, y = df.H_s.values, df.theta_p.values
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    ax2.scatter(x, y, c=z)
    ax2.set_xlabel('Hs (m)')
    ax2.set_ylabel('Dp (deg)')

    ax3 = fig.add_subplot(133)
    x, y = df.T_p.values, df.theta_p.values
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    ax3.scatter(x, y, c=z)
    ax3.set_xlabel('Tp (s)')
    ax3.set_ylabel('Dp (deg)')

    fig.tight_layout()
    return fig

def plot_serie_wind_hs_tp_dp(date, ws, wd, hs, tp, dp, title=None):
    """
    """
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(511)
    ax1.set_title(title)
    ax1.plot(ws)
    ax1.set_ylabel('WS [m/s]')
    ax1.grid()
    ax2 = fig.add_subplot(512)
    ax2.plot(wd)
    ax2.set_ylabel('WD [deg]')
    ax2.grid()
    ax3 = fig.add_subplot(513)
    ax3.plot(hs)
    ax3.set_ylabel('Hs [m]')
    ax3.grid()
    ax4 = fig.add_subplot(514)
    ax4.plot(tp)
    ax4.set_ylabel('Tp [s]')
    ax4.grid()
    ax5 = fig.add_subplot(515)
    ax5.plot(dp)
    ax5.set_yticks(np.arange(0, 360+45, 45))
    ax5.set_ylabel('Dp [deg]')
    ax5.grid()
    return fig

def subplots_T(df):
    """
    Cria subplot com todos os periodos
    """
    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(111)
    df[['T_i', 'T_e', 'T_1', 'T_z', 'T_3', 'T_c', 'T_dw', 'T_p']].plot(ax=ax1)
    ax1.legend(ncol=4, loc=1)
    ax1.set_ylabel('Período [s]')
    ax1.set_xlabel('')
    ax1.grid()
    fig.tight_layout()
    return fig

if __name__ == '__main__':

    # leitura do arquivo pickle da waverider
    p = pd.read_pickle('/home/hp/Dropbox/prooceano/waverider.pkl')

    # retira NaN do Tp para plotar distribuicao conjunta (retirar esta linha no operacional)
    p['param_wav'].T_p.iloc[5] = p['param_wav'].T_p.iloc[4]

    # dataframe com parametros de onda
    df = p['param_wav'].drop(['checksum', 'message_stamp', 'battery_time_remaining',
                              'O_v', 'O_x', 'O_y'], axis=1)

    # vetores interpolados
    freqi, spec1di, meandiri, spreadi = interp_spectra(p)

    # loop para todos os registros
    for i in range(len(spec1di))[50:51]:

        date = str(spec1di.index[i])

        # valores de freq, direcao e spread de uma hora
        s = spec1di.iloc[i,:].values
        d = meandiri.iloc[i,:].values
        spr = spreadi.iloc[i,:].values

        d2, thetas = calc_spec2d_matrix(s, d, spr)

        fig = plot_posicao(p['param_wav'], date)
        fig.savefig('posicao.png', bbox_inches='tight')

        fig = plot_spec1d_meandir_spread(freqi, s, d, spr, date)
        fig.savefig('spec1d_meandir_spread.png', bbox_inches='tight')

        fig = plot_spec2d(freqi, thetas, d2, date)
        fig.savefig('spec2d.png', bbox_inches='tight')

        fig = plot_polar_spec2d(freqi, thetas, d2, date)
        fig.savefig('polar_spec2d.png', bbox_inches='tight')

    fig = plot_evolspec(freqi, spec1di, date)
    fig.savefig('evolspec.png', bbox_inches='tight')

    fig = plot_distribuicao_conjunta(df)
    fig.savefig('distribuicao_conjunta.png', bbox_inches='tight')

    fig = plot_serie_hs_tp_dp(df)
    fig.savefig('serie_hs_tp_dp.png', bbox_inches='tight')

    fig = subplots_T(df)
    fig.savefig('subplots_T.png', bbox_inches='tight')

    # df.to_csv('tabela.csv', float_format='%.2f')

    plt.show()
