# Processamento dos dados espectrais da boia
# de Fortaleza do PNBOIA
# - Calcula parâmetros de onda no tempo e frequencia


import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import mlab
from scipy.interpolate import interp1d
from pandas.plotting import register_matplotlib_converters
from glob import glob
register_matplotlib_converters()
plt.close('all')


def read_axys_hne(filename):
    """
    Leitura dos dados do sensor inercial da NavCon
    """
    df = pd.read_csv(filename, skiprows=12, sep='\s+', names=['time','hv','dn','de'])
    return df

def espec1(x, nfft, fs):
    """
    Calculo do espectro 1d
    """
    s, f = mlab.psd(x=x, NFFT=int(nfft), Fs=fs, detrend=mlab.detrend_mean,
                  window=mlab.window_hanning, noverlap=nfft/2)

    #degrees of freedom
    dof = len(x) / nfft * 2
    
    #confidence interval 95%
    ici = s * dof / 26.12
    ics = s * dof / 5.63
    
    # matriz de saida
    aa = np.array([f,s,ici,ics]).T

    return aa

def espec2(x, y, nfft, fs):
    """    
    Calcula o espectro cruzado entre duas series reais
    
    Dados de entrada: x = serie real 1 (potencia de 2)
                      y = serie real 2 (potencia de 2)
                      nfft - Numero de pontos utilizado para o calculo da FFT
                      fs - frequencia de amostragem
    
    Dados de saida: [aa2] - col 0: vetor de frequencia
                            col 1: amplitude do espectro cruzado
                            col 2: co-espectro
                            col 3: quad-espectro
                            col 4: espectro de fase
                            col 5: espectro de coerencia
                            col 6: intervalo de confianca inferior do espectro cruzado
                            col 7: intervalo de confianca superior do espectro cruzado
                            col 8: intervalo de confianca da coerencia
    
    Infos:    detrend - mean
              window - hanning
              noverlap - 50%    
    """

    #cross-spectral density - welch method (complex valued)
    s, f = mlab.csd(x, y, NFFT=nfft, Fs=fs, detrend=mlab.detrend_mean, window=mlab.window_hanning, noverlap=nfft/2)

    # graus de liberdade    
    dof = len(x) / nfft * 2

    #co e quad espectro (real e imag) - verificar com parente
    co = np.real(s)
    qd = np.imag(s)
    
    #phase (angle function)
    ph = np.angle(s, deg=True)
    
    #ecoherence between x and y (0-1)
    coer = mlab.cohere(x, y, NFFT=nfft, Fs=fs, detrend=mlab.detrend_mean, window=mlab.window_hanning, noverlap=nfft/2)[0]
    # coer = coer[0][1:]
    
    #intervalo de confianca para a amplitude do espectro cruzado - 95%
    ici = s * dof /26.12
    ics = s * dof /5.63
    
    #intervalo de confianca para coerencia
    icc = np.zeros(len(s))
    icc[:] = 1 - (0.05 ** (1 / (14 / 2.0 - 1)))
    
    # matriz de saida
    aa = np.array([f, s ,co, qd, ph, coer, ici, ics, icc]).T

    return aa

def numeronda(h, df, reg):
    '''
    Calcula o numero de onda (k)
    
    Dados de entrada: h - profundidade
                        deltaf - vetor de frequencia
                        reg - comprimento do vetor de frequencia
    
    Dados de saida: k - vetor numero de onda
    '''

    #gravidade
    g = 9.8

    #vetor numero de onda a ser criado
    k = []

    #k anterior
    kant = 0.001

    #k posterior
    kpos = 0.0011

    for j in range(reg):
        sigma = (2 * np.pi * df[j] ) ** 2
        while abs(kpos - kant) > 0.001:
            kant = kpos
            dfk = g * kant * h * (1 / np.cosh(kant*h)) ** 2 + g + np.tanh(kant * h)
            fk = g * kant * np.tanh(kant * h) - sigma
            kpos = kant - fk / dfk
        kant = kpos - 0.002
        k.append(kpos)
    return k

def ondat(t, eta, h):
    '''
    Calcula parametros de onda no dominio do tempo
    
    Dados de entrada: t - vetor de tempo  
                      eta - vetor de elevacao
                      h - profundidade
    
    Dados de saida: pondat = [Hs,H10,Hmax,THmax,Tmed,]
                      Hs - altura significativa
                    H10 - altura de 1/10 das maiores
                    Hmax - altura maxima
                    THmax - periodo associado a altura maxima
                    Tmed - periodo medio
    '''

    #retira a media
    eta = eta - np.mean(eta)

    #criando os vetores H(altura),Cr(crista),Ca(cavado),T (periodo)
    Cr = []
    Ca = []
    H = []
    T = []

    #acha os indices que cruzam o zero
    z = np.where(np.diff(np.sign(eta)))[0]

    #zeros ascendentes e descendentes
    zas=z[0::2]
    zde=z[1::2]

    #calcula ondas individuas
    for i in range(len(zas)-1):
        onda = eta[zas[i]:(zas[i+1])+1]
        cr = np.max(onda)
        # Cr.append(cr)
        ca = np.min(onda)
        # Ca.append(ca)
        H.append(cr + np.abs(ca))
        T.append(t[zas[i+1]] - t[zas[i]])

    #coloca as alturas em ordem crescente
    Hss = np.sort(H)
    Hss = np.flipud(Hss)

    #calcula a altura significativa (H 1/3)
    div = int(len(Hss) / 3.0)
    hs = np.mean(Hss[0:div+1])
    
    #calcula a altura das 1/10 maiores (H 1/10)
    div1 = int(len(Hss) / 10.0)
    h10 = np.mean(Hss[0:div1+1]) #altura da media das um decimo maiores
    
    #altura maxima
    hmax = np.max(H)
    
    #periodo medio
    tz = np.mean(T)
    
    #calcula periodo associado a altura maxima
    ind = np.where(H == hmax)[0][0]
    thmax = T[ind]

    #parametros de onda no tempo
    # pondat = np.array([Hs,H10,Hmax,Tmed,THmax])

    return H, T, hs, h10, hmax, tz, thmax

def ondaf(eta, etax, etay, h, nfft, fs):
    """
    Calcula parametros de onda no dominio da frequencia
    
    Dados de entrada: eta - vetor de elevacao
                      etax - vetor de deslocamento em x
                      etay - vetor de deslocamento em y
                      h - profundidade
                      nfft - Numero de pontos utilizado para o calculo da FFT
                      fs - frequencia de amostragem
                      az - azimute da boia
    
    Dados de saida: pondaf = [hm0 tp dp]
    """

    #espectro simples
    sn = espec1(eta,nfft,fs)
    snx = espec1(etax,nfft,fs)
    sny = espec1(etay,nfft,fs)

    #espectros cruzados
    snn = espec2(eta,eta,nfft,fs)
    snnx = espec2(eta,etax,nfft,fs)
    snny = espec2(eta,etay,nfft,fs)
    snxny = espec2(etax,etay,nfft,fs)
    snxnx = espec2(etax,etax,nfft,fs)
    snyny = espec2(etay,etay,nfft,fs)

    #vetor de frequencia
    f = sn[:,0]

    #deltaf
    df = f[1] - f[0]

    #calculo do numero de onda
    k = np.array(numeronda(h,f,len(f)))

    #calculo dos coeficientes de fourier - NDBC 96_01 e Steele (1992)
    c = snx[:,1] + sny[:,1]
    cc = np.sqrt(sn[:,1] * c)

    a1 = snnx[:,3] / cc
    b1 = snny[:,3] / cc

    a2 = (snx[:,1] - sny[:,1]) / c
    b2 = 2 * snxny[:,2] / c

    #calcula direcao de onda
    #mean direction
    dire1 = np.array([np.angle(np.complex(b1[i],a1[i]),deg=True) for i in range(len(a1))])

    #principal direction
    dire2 = 0.5 * np.array([np.angle(np.complex(b2[i],a2[i]),deg=True) for i in range(len(a2))])
    
    #condicao para valores maiores que 360 e menores que 0
    dire1[np.where(dire1 < 0)] = dire1[np.where(dire1 < 0)] + 360
    dire1[np.where(dire1 > 360)] = dire1[np.where(dire1 > 360)] - 360
    dire2[np.where(dire2 < 0)] = dire2[np.where(dire2 < 0)] + 360
    dire2[np.where(dire2 > 360)] = dire2[np.where(dire2 > 360)] - 360

    #acha o indice da frequencia de pico
    ind = np.where(sn[:,1] == np.max(sn[:,1]))[0]

    ii_swell = np.arange(0,18)
    ii_sea = np.arange(18, len(sn))

    # indice da frequencia de pico do swell
    ind_swell = np.where(sn[ii_swell,1] == np.max(sn[ii_swell,1]))[0]

    # indice da frequencia de pico do sea
    ind_sea = np.where(sn[ii_sea,1] == np.max(sn[ii_sea,1]))[0] + ii_sea[0]

    #periodo de pico
    tp = (1. / f[ind])[0]
    tp_swell = (1. / f[ind_swell])[0]
    tp_sea = (1. / f[ind_sea])[0]

    # calcula os momentos espectrais
    m0 = np.sum(f**0 * sn[:,1]) * df
    m1 = np.sum(f**1 * sn[:,1]) * df
    m2 = np.sum(f**2 * sn[:,1]) * df
    m3 = np.sum(f**3 * sn[:,1]) * df
    m4 = np.sum(f**4 * sn[:,1]) * df
    m0_swell = np.sum(f[ii_swell]**0 * sn[ii_swell,1]) * df
    m0_sea = np.sum(f[ii_sea]**0 * sn[ii_sea,1]) * df

    #calculo da altura significativa
    hm0 = 4.01 * np.sqrt(m0)
    hm0_swell = 4.01 * np.sqrt(m0_swell)
    hm0_sea = 4.01 * np.sqrt(m0_sea)

    #direcao do periodo de pico
    dp = dire1[ind][0]
    dp_swell = dire1[ind_swell][0]
    dp_sea = dire1[ind_sea][0]

    # mean spectral frequency
    mean_spec_freq = m1 / m0

    # mean spectral period
    mean_spec_period = 1.0 / mean_spec_freq

    # average zero-up-crossing frequency
    mean_zup_freq = np.sqrt(m2 / m0)

    # average zero-up-crossing period
    mean_zup_period = 1.0 / mean_zup_freq

    # spectral bandwidth (Cartwright & Longuet-Higgins, 1956)
    e = np.sqrt(1 - (m2 ** 2 / (m0 * m2)))

    # spectral bandwidth (Longuet-Higgins, 1975)
    v = np.sqrt((m0 * m2) / (m1 ** 2) - 1)

    #Espalhamento direcional

    #formula do sigma1 do livro Tucker&Pitt(2001) "Waves in Ocean Engineering" pags 196-198
    c1 = np.sqrt(a1 ** 2 + b1 ** 2)
    c2 = np.sqrt(a2 ** 2 + b2 ** 2)
    
    s1 = c1 / (1 - c1)
    s2 = (1 + 3 * c2 + np.sqrt(1 + 14 * c2 + c2 ** 2)) / (2 * (1 - c2))
    
    sigma1 = np.sqrt(2 - 2 * c1) * 180 / np.pi
    sigma2 = np.sqrt((1 - c2) / 2) * 180 / np.pi

    # acha o espalhamento angular da frequencia de pico
    sigma1p = np.real(sigma1[ind])[0]
    sigma2p = np.real(sigma2[ind])[0]

    # pondaf = np.array([hm0, tp, dp, sigma1p, sigma2p])

    # aa = np.array([hm0, tp, dp, sigma1, sigma2, sigma1p, sigma2p, f, df, k, sn, snx, sny, snn, snnx, snny, snxny, snxnx, snyny, a1, b1, a2, b2, dire1, dire2])

    waveout = {'hm0': hm0, 'tp': tp, 'dp': dp, 'f': f, 'sn': sn, 'dire1':dire1, 
               'mean_spec_freq': mean_spec_freq, 'mean_spec_period': mean_spec_period,
               'mean_zup_freq': mean_zup_freq, 'mean_zup_period': mean_zup_period,
               'e': e, 'v': v, 'hm0_sea': hm0_sea, 'hm0_swell': hm0_swell, 'tp_sea': tp_sea,
               'tp_swell': tp_swell, 'dp_sea': dp_sea, 'dp_swell': dp_swell}
            
    return waveout

def ondap(hm0, tp, dp, sn, dire1):
    '''
    Programa para calcular parametros
    de onda nas particoes de sea e swell
    
    desenvolvido para 32 gl
    
    divide o espectro em 2 partes: 
    parte 1 - 8.33 a 50 seg
    parte 2 - 1.56 a 7.14 seg
    
    calcula o periodo de pico de cada particao, e despreza o
    pico 2 (menos energetico) se a energia for inferior a 15% da
    energia do pico 1 (mais energetico)
    '''

    #vetor de frequencia e energia
    f,s = sn[:,[0,1]].T

    df = f[4] - f[3]

    # seleciona os picos espectrais - considera somente 2 picos
    g1=np.diff(s)
    g1=np.sign(g1)
    g1=np.diff(g1)
    g1=np.concatenate(([0],g1))
    g2=np.where(g1==-2)[0]
    picos=1 # a principio e unimodal
    l=np.size(g2)

    # inicializar considerando ser unimodal
    hm02 = np.nan #9999
    tp2 = np.nan #9999
    dp2 = np.nan #9999
    hm01 = hm0
    tp1 = tp 
    dp1 = dp 

    # se tiver picos
    if l > 1: #verificando espacamento entre picos (espacamento maior que 4 df)
        fr=np.argsort(s[g2])[::-1] #frequencia decrescente
        er=np.sort(s[g2])[::-1] # energia decrescente

        if (f[g2[fr[1]]]-f[g2[fr[0]]]) > 4*(f[1]-f[0]) and (er[1]/er[0] >= 0.15): #adota criterio de 4*deltaf
            picos=2
        
        # calcular o Hs dos picos pegando a cava e dividindo em pico 1 e pico 2
        if picos == 2:
            n1=g2[0] #pico mais energetico
            n2=g2[1] #pico menos energetico
            nc=np.where(g1[n1:n2]==2)[0] #indice da cava

            #particao do swell e sea
            swell = np.arange(n1+nc+1)
            sea = np.arange(n1+nc+1,len(s))
            #maxima energia do swell
            esw = max(s[swell])
            #maxima energia do sea
            ese = max(s[sea])
            #indice do pico do swell
            isw = np.where(s==esw)[0][0]
            #indice do pico do sea
            ise = np.where(s==ese)[0][0]
            #altura sig. do swell
            hm0sw = 4.01 * np.sqrt(sum(s[swell]) * df)
            #altura sig. do sea
            hm0se = 4.01 * np.sqrt(sum(s[sea]) * df)
            #periodo de pico do swell
            tpsw = 1./f[isw]
            #periodo de pico do sea
            tpse = 1./f[ise]
            #direcao do swell
            dpsw = dire1[isw]
            #direcao do sea
            dpse = dire1[ise]

            #deixa o pico 1 como swell e pico 2 como sea
            en1 = esw ; en2 = ese
            hm01 = hm0sw ; hm02 = hm0se
            tp1 = tpsw ; tp2 = tpse
            dp1 = dpsw ; dp2 = dpse

    if tp2 == tp1:
        tp2 = np.nan
        hm01 = np.nan
        dp2 = np.nan
        
            #seleciona pico 1 como mais energetico
            # e pico 2 com o menos energetico
            # if esw > ese:
            #   en1 = esw ; en2 = ese
            #   hm01 = hm0sw ; hm02 = hm0se
            #   tp1 = tpsw ; tp2 = tpse
            #   dp1 = dpsw ; dp2 = dpse
            # else:
            #   en1 = ese ; en2 = esw
            #   hm01 = hm0se ; hm02 = hm0sw
            #   tp1 = tpse ; tp2 = tpsw
            #   dp1 = dpse ; dp2 = dpsw

    # pondaf1 = np.array([hm01, tp1, dp1, hm02, tp2, dp2])

    return hm01, tp1, dp1, hm02, tp2, dp2 #pondaf1

def spread(en,a1,b1):
    '''
    Programa para calcular o espalhamento angular
    Kuik et al 1988
    Entrada:
    en - espectro 1D
    a1 - coef de fourier de 1 ordem
    b1 - conef de fourir de 1 ordem 
    Saida:
    spr - valor do espalhamento angular para cada
    frequencia
    '''
    #esplhamento com vetor complexo - radianos?
    sprc = (2 * (1 - ( (a1**2 + b1**2) / (en**2) ) **0.5) **0.5)
    #soma a parte real e imag e coloca em graus
    spr = (np.real(sprc) + np.imag(sprc)) * 180 / np.pi
    #parece que aparece na parte real onde tem energia no espectro
    sprr = np.real(sprc)
    #diminui 360 nos maiores que 360
    # spr[np.where(spr>360)[0]] = spr[np.where(spr>360)[0]] - 360
    #coloca zeros nos valores muito altos
    # spr[np.where(spr>360)[0]] = 0
    return sprc, spr, sprr

def jonswap(f, Hm0, Tp, gamma=3.3, sigma_low=.07, sigma_high=.09, g=9.81):
    """
    Jonswap spectrum
    """
    # Pierson-Moskowitz
    alpha = 1. / (.23 + .03 * gamma - .185 / (1.9 + gamma)) / 16.
    E_pm = alpha * Hm0**2 * Tp**-4 * f**-5 * np.exp(-1.25 * (Tp * f)**-4)
    # JONSWAP
    sigma = np.ones(f.shape) * sigma_low
    sigma[f > 1./Tp] = sigma_high
    E_js = E_pm * gamma**np.exp(-0.5 * (Tp * f - 1)**2. / sigma**2.)
    return E_js

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

## Processamento em batelada

if __name__ == '__main__':

    # pathname dos dados
    ppath = os.environ['HOME']
    path_hne = ppath + '/gdrive/salinopolis/dados/pnboia/HNE_fortaleza_201612/'
    path_out = ppath + '/gdrive/salinopolis/parametros_onda_fortaleza.csv'

    # lista hne
    lista_hne = np.sort(glob(path_hne + '*.HNE'))

    # vetor de tempo em segundos
    t = np.arange(0, 1930, 0.78)

    # profundidade
    h = 200.0

    wparam = []
    for filename in lista_hne:
        # condition of numer of lines (quality control)
        num_lines = sum(1 for line in open(filename, encoding='utf-8', errors='ignore'))

        # condicao de arquivo com todas as linhas
        if num_lines == 1324:
            print (filename)

            # leitura dos dados
            df = read_axys_hne(filename)

            # processamento dos dados no dominio do tempo
            H, T, hs, h10, hmax, tz, thmax = ondat(t, df.hv.values, h)

            # processamento dos dados no dominio da frequencia
            waveout = ondaf(eta=df.hv.values,
                            etax=df.dn.values,
                            etay=df.de.values,
                            h=h, nfft=int(len(df) / 16),
                            fs=1.28)

            # processamento de onda particionado
            hm01, tp1, dp1, hm02, tp2, dp2 = ondap(waveout['hm0'], waveout['tp'],
                                                   waveout['dp'], waveout['sn'],
                                                   waveout['dire1'])

            datastr = str(pd.to_datetime(filename.split('/')[-1].split('.')[0]))
            wparam.append([datastr, hs, h10, hmax, tz, thmax, waveout['hm0'], waveout['tp'],
                waveout['dp'], waveout['mean_spec_period'], waveout['mean_zup_period']])

            # plot_spec_dire1(df, waveout, filename)            
        else:
            print ('dado {} com erro'.format(filename.split('/')[-1]))

    # cria dataframe
    wparam = np.array(wparam)
    df_param = pd.DataFrame(wparam[:,1:], index=wparam[:,0],
                            columns = ['hs', 'h10', 'hmax', 'tz', 'thmax', 'hm0', 'tp', 'dp',
                            'mean_spec_period', 'mean_zup_period']).astype(float)

    # salva arquivo csv
    df_param.index.name = 'date'
    df_param.to_csv(path_out, float_format='%.2f')
