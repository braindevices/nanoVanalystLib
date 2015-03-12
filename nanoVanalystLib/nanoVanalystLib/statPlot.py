# -*- coding: UTF-8 -*-
'''
Created on Mar 12, 2015-1:24:32 PM

@author: Ling Wang<LingWangNeuralEng@gmail.com>
'''
import numpy as np
from scipy import stats
# import matplotlib.pyplot as plt
# import os

from LWpyUtils import npyUtils
from LWpyUtils.npyUtils import getGeoMeanMeidanStdMad, getMeanMeidanStdMad

def loadCSV(csvFilename, kwargs, logVal = False):
    a = np.genfromtxt(csvFilename, **kwargs)
    data = []
    for _i in range(0, a.shape[1]):
        _a = a[:,_i]
        _a = _a[~np.isnan(_a)]
        if logVal:
            _a = np.log10(_a)
        data.append(_a)
    return data

def gendXYfor1Group(ys):
    N = ys.shape[0]
    ys.sort()
    dx = np.zeros((N,))
    _uni, unidx = np.unique(ys, return_index = True)
    unicounts = stats.itemfreq(ys)[:,1]
    maxCount = max(unicounts)
    for _i,_n in zip(unidx, unicounts):
            dx[_i:_i+_n]=np.arange(0.,_n)
    
        
    return dx, ys, maxCount

def gendXYbinedFor1Group(bins):
    '''
    bins are tuple or list [counts, bin_edge_values]
    the bin_edge_values has 1 more element than counts
    '''
#     print bins
    unicounts, _binEdges = bins
    _binMids = (_binEdges[0:-1] + _binEdges[1:])/2
    N = np.sum(unicounts)
    dx = np.zeros((N,))
    ys = np.zeros((N,))
    maxCount = unicounts.max()
    _i = 0
    for _n, _y in zip(unicounts, _binMids):
        dx[_i:_i+_n] = np.arange(0.,_n)
        ys[_i:_i+_n] = _y
        _i += _n
    return dx, ys, maxCount


def genXYScatterforAllGroup(xArray, dataList, scale, binned = False):
    dxsList =[]
    ysList =[]
    _idxs=xArray.argsort()
    _xdiff = np.diff(xArray[_idxs])
    _minXsep = _xdiff.min()
    maxCount = 0
    counts = []
    for y0 in dataList:
        if binned:
            _dx,_y, _maxCount = gendXYbinedFor1Group(y0)
#             print "_dx, _y shape", _dx.shape, _y.shape
            
        else:
            _dx,_y, _maxCount = gendXYfor1Group(y0)
        if maxCount < _maxCount:
            maxCount = _maxCount
        dxsList.append(_dx)
        ysList.append(_y)
        counts.append(_maxCount)
    
    xsList = []
    if scale == 'individual':
        scale = float('Inf')
        #automatically determine dx scale based on each group
        for _x0, _dx, _count in zip(xArray, dxsList, counts):
            _scale = _minXsep/(_count+4)
            print _scale
            _x = _x0+_dx*_scale
            xsList.append(_x)
            if scale > _scale:
                scale = _scale
    else:
        if scale == None:
            #automatically determine, uniform dx scale
        
            scale = _minXsep/(maxCount+4)
        
        
        for _x0, _dx in zip(xArray, dxsList, scale):
            _x = _x0+_dx*scale
            xsList.append(_x)
        
    xsAll = np.concatenate(tuple(xsList))
    ysAll = np.concatenate(tuple(ysList))
    return xsAll, ysAll, scale

def plotDists(ax, xs, dataList, scatter_dX_Scale =None, xoffset=None, xoffsetRatio = 2, scatterArgs=(), scatterKwds={}, binned = False, *args,**kwargs):
    
    xsAll, ysAll, scatter_dX_Scale = genXYScatterforAllGroup(xs, dataList, scatter_dX_Scale, binned)
#     if scatter_dX_Scale == None:
#         scatter_dX_Scale = _scale
    print scatter_dX_Scale
    if xoffset == None:
        #auto offset
        xoffset = scatter_dX_Scale * xoffsetRatio
    ax.scatter(xsAll+xoffset, ysAll, *scatterArgs, **scatterKwds)
    return xsAll.min(),xsAll.max(), ysAll.min(),ysAll.max()
    

def plotStatisticSum(ax, xs, dataList, useMean = False, yerrType= 'std', useMadBasedStd = False, errobarArgs=(), errorBarKwds={}, confidence=0.95, *args, **kwargs):
    _yMean, _yMed, _yStd, _yMad, _yN = getMeanMeidanStdMad(dataList)
    if useMean:
        _y = _yMean
        
    else:
        _y = _yMed
        
    if useMadBasedStd:
            _yStd = 1.4826 *_yMad
    
    if yerrType == 'std':
        _yerror = _yStd
        
    elif yerrType == 'mad':
        _yerror = _yMad
    elif yerrType == 'ste':
        
        _yerror = _yStd/np.sqrt(_yN)
    elif yerrType == 'ci':
        
        _yerror = npyUtils.halfConfidenceIntervalToMeanStudentT(_yStd/np.sqrt(_yN), _yN-1, confidence)
#         print _yStd, _yN, _yerror
    ax.errorbar(xs, _y, _yerror, *errobarArgs, **errorBarKwds)
    _ylow = (_y + _yerror).min()
    _yhigh = (_y - _yerror).min()
    return xs.min(), xs.max(), _ylow, _yhigh

def plotGeoStatisticSum(ax, xs, dataList, useMean = False, yerrType= 'std', useMadBasedStd = False, errobarArgs=(), errorBarKwds={}, confidence=0.95, *args, **kwargs):
    _yMean, _yMed, _yStd, _yMad, _yN = getGeoMeanMeidanStdMad(dataList)
    if useMean:
        _y = _yMean
        
    else:
        _y = _yMed
        
    if useMadBasedStd:
            _yStd = 1.4826 *_yMad
    
    if yerrType == 'std':
        _yerror = _yStd
        
    elif yerrType == 'mad':
        _yerror = _yMad
    elif yerrType == 'ste':
        
        _yerror = _yStd/np.sqrt(_yN)
    elif yerrType == 'ci':
        
        _yerror = npyUtils.halfConfidenceIntervalToMeanStudentT(_yStd/np.sqrt(_yN), _yN-1, confidence)
#         print _yStd, _yN, _yerror
    ax.errorbar(xs, _y, _yerror, *errobarArgs, **errorBarKwds)
    _ylow = (_y + _yerror).min()
    _yhigh = (_y - _yerror).min()
    return xs.min(), xs.max(), _ylow, _yhigh


