#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Created on Mar 12, 2015-2:06:34 PM

@author: Ling Wang<LingWangNeuralEng@gmail.com>
'''

import matplotlib.pyplot as plt
from figUtils import getLim, config1Axes, saveFig
import numpy as np
from statPlot import plotDists, plotGeoStatisticSum, plotStatisticSum, loadCSV
from Constants_and_Parameters import TMPDIR


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

def runGenFigs_Hbd1S1G(xs, dataList=[], csvFilename ="", kwargs = dict(delimiter =',', dtype =float, missing_values ={0:''}, filling_values={0:np.nan}, skip_header=1), logY = False):
    
    if not dataList:
        dataList = loadCSV(csvFilename, kwargs, logVal =False)
    if logY:
        sbplotKwds = dict(yscale = 'log')
    else:
        sbplotKwds = {}
    
    scatterKwds = dict(marker='x', s=2, c = 'k', lw= 0.1, label = "Dia. frequency")
    plotDistsKwds = dict (scatter_dX_Scale = "individual", scatterKwds =scatterKwds)
    

    errorBarKwds = dict(fmt='-g', linewidth = 1, markersize=5, ecolor='g',capsize=4, elinewidth=1, label = "Mean & St. Dev.")
    plotErrbarKwds = dict(errorBarKwds =errorBarKwds, yerrType= 'std', useMean =True)
    geoErrorBarKwds = dict(fmt='-r', linewidth = 1, markersize=5, ecolor='r',capsize=4, elinewidth=1, label = "Geo. Mean & Geo. St. Dev.")
    plotGeoErrbarKwds = dict(errorBarKwds =geoErrorBarKwds, yerrType= 'std', useMean =True)

    fig = genFigure_HbinDist_1Stat_1GeoStat(xs, dataList, plotDistsKwds=plotDistsKwds, plotErrbarKwds=plotErrbarKwds, plotGeoErrbarKwds=plotGeoErrbarKwds, sbplotKwds =sbplotKwds )
    config1Axes(fig.axes[0], "Laser power", "Pore diameter", "W", "nm", legendOn=True)
    
    saveFig(fig, figDir=TMPDIR, key1="Laser power", key2="Pore diameter")
    
    errorBarKwds = dict(fmt='-g', linewidth = 1, markersize=5, ecolor='g',capsize=4, elinewidth=1, label = "Median&St. Err.")
    plotErrbarKwds = dict(errorBarKwds =errorBarKwds, yerrType= 'ste', useMean =True)
    geoErrorBarKwds = dict(fmt='-r', linewidth = 1, markersize=5, ecolor='r',capsize=4, elinewidth=1, label = "Geo. Mean & Geo. St. Err.")
    plotGeoErrbarKwds = dict(errorBarKwds =geoErrorBarKwds, yerrType= 'ste', useMean =True)
    
    fig = genFigure_HbinDist_1Stat_1GeoStat(xs, dataList, plotDistsKwds=plotDistsKwds, plotErrbarKwds=plotErrbarKwds, plotGeoErrbarKwds=plotGeoErrbarKwds, sbplotKwds =sbplotKwds)
    
    config1Axes(fig.axes[0], "Laser power", "Pore diameter", "W", "nm", legendOn=True)
    
    saveFig(fig, figDir=TMPDIR, key1="Laser power", key2="Pore diameter")
    
    errorBarKwds = dict(fmt='-g', linewidth = 1, markersize=5, ecolor='g',capsize=4, elinewidth=1, label = r"Median&Confidence interval (95%)")
    plotErrbarKwds = dict(errorBarKwds =errorBarKwds, yerrType= 'ci', useMean =True)
    geoErrorBarKwds = dict(fmt='-r', linewidth = 1, markersize=5, ecolor='r',capsize=4, elinewidth=1, label = "Geo. Mean & Confidence interval (95%)")
    plotGeoErrbarKwds = dict(errorBarKwds =geoErrorBarKwds, yerrType= 'ci', useMean =True)
    
    fig = genFigure_HbinDist_1Stat_1GeoStat(xs, dataList, plotDistsKwds=plotDistsKwds, plotErrbarKwds=plotErrbarKwds, plotGeoErrbarKwds=plotGeoErrbarKwds, sbplotKwds =sbplotKwds)
    config1Axes(fig.axes[0], "Laser power", "Pore diameter", "W", "nm", legendOn=True)
    
    saveFig(fig, figDir=TMPDIR, key1="Laser power", key2="Pore diameter")

if __name__ == '__main__':
    pass