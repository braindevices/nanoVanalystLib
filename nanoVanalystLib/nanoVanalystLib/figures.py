# -*- coding: UTF-8 -*-
'''
Created on Mar 12, 2015-2:06:34 PM

@author: Ling Wang<LingWangNeuralEng@gmail.com>
'''

import matplotlib.pyplot as plt
from figUtils import getLim
import numpy as np
from statPlot import plotDists, plotGeoStatisticSum, plotStatisticSum



def genFigure_HbinDist_1Stat_1GeoStat(xs, dataList, binN=50, geoDist = True, plotDistsKwds = {}, plotErrbarKwds = {}, plotGeoErrbarKwds = {}, ylim=None, xlim=None, autoXYLimSepRatio= [0.01, 0.01], figDir = '/tmp', sbplotArgs = (1,1,1), sbplotKwds = {}):
    fig = plt.figure()
    ax = fig.add_subplot(*sbplotArgs,**sbplotKwds)
#     scatterKwds = dict(marker='x', s=1, c = 'k', lw= 0)
#     scale =0.01
    _binsList = []
    if geoDist:
        
        for _data in dataList:
            _bins = np.histogram(np.log10(_data), bins = binN)
    #         print _bins
            _vals = 10**_bins[1]
            _binsList.append((_bins[0], _vals))
    else:
        for _data in dataList:
            _bins = np.histogram(_data, bins = binN)

            _binsList.append((_bins))
    
    _outbox = []
    _outbox.append(plotDists(ax, xs, _binsList, binned = True, **plotDistsKwds))
    
        
#     errorBarKwds = dict(fmt='-r', linewidth = 1, markersize=5, ecolor='r',capsize=4, elinewidth=1)
    _outbox.append(plotStatisticSum(ax, xs, dataList, **plotErrbarKwds))
    _outbox.append(plotGeoStatisticSum(ax, xs, dataList, **plotGeoErrbarKwds))
    _outbox =np.array(_outbox)
    xl = _outbox[:, 0].min()
    xh = _outbox[:, 1].max()
    yl = _outbox[:, 2].min()
    yh = _outbox[:, 3].max()
    if ylim == None:
        
        ylim = getLim(yl, yh, autoXYLimSepRatio[1])
#         print yl, yh, ylim
    if xlim == None:
        xlim = getLim(xl, xh, autoXYLimSepRatio[0])
#         print xl, xh, xlim
    
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    return fig



