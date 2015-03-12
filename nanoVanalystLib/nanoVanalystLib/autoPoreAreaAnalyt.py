#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Created on Mar 12, 2015-2:21:06 PM

@author: Ling Wang<LingWangNeuralEng@gmail.com>
'''
import os
from imageProc import analyzePoresInDir
from figUtils import config1Axes, saveFig
import numpy as np
from statPlot import loadCSV
from figures import genFigure_HbinDist_1Stat_1GeoStat
from Constants_and_Parameters import TMPDIR

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
    
def autoPoreAreaAnalyt(xs, imgDirs, imgPatterns, resultDir, saveCSV = False, runPoreDetection1ImgKwds = dict(dirForEachFile = True)):
    if not os.path.isdir(resultDir):
        raise IOError("%s does not exist"%(resultDir))
    
    _parentDir = os.path.commonprefix(imgDirs)
    dataList = []
    for _imgDir, _imgPattern in zip(imgDirs, imgPatterns):
        _subResultDir = os.path.relpath(_imgDir, _parentDir)
        _subResultDir = os.path.join(resultDir, _subResultDir)
        if not os.path.isdir(_subResultDir):
            os.makedirs(_subResultDir)
        _data = analyzePoresInDir(_imgDir, _imgPattern, _subResultDir, saveCSV, runPoreDetection1ImgKwds)
        dataList.append(_data)
    runGenFigs_Hbd1S1G(xs, dataList, logY = True)
    
if __name__ == '__main__':
    imgDirs=[]
    resultDir = os.path.join(TMPDIR, "results")
    autoPoreAreaAnalyt(np.array([0,2.25,2.5,2.75,3,3.25,3.5,3.75,4]), imgDirs, "*.tif", resultDir, saveCSV =True)