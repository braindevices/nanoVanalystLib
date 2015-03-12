# -*- coding: UTF-8 -*-
'''
Created on Mar 12, 2015-1:50:06 PM

@author: Ling Wang<LingWangNeuralEng@gmail.com>
'''
import os
from LWpyUtils.generalUtils import getNowUTCtimestamp

def getLim(low, high, ratio):
    _dy = high-low
    return [low- _dy*ratio, high + _dy*ratio]

def config1Axes(axes, key1, key2, key1unit, key2unit, xlabelTemp="{key1:} ({key1unit})", ylabelTemp="{key2:} ({key2unit})", lableOn=True, legendOn = False, legendloc = 2):
    
    if legendOn:
        axes.legend(loc=legendloc)
    else:
        legendloc = '-'
    axes.set_xlabel(xlabelTemp.format(key1=key1, key1unit=key1unit, key2=key2, key2unit=key2unit))
    axes.set_ylabel(ylabelTemp.format(key1=key1, key1unit=key1unit, key2=key2, key2unit=key2unit))
    

def saveFig(fig, figDir, key1, key2, prefix='', postfix = 'pdf'):
    fig.tight_layout()
    axesList = fig.axes
    _logscale=''
    _legendloc = ''
    #当有多个axes的时候这里就显得不方便了，应该用一个专门的genFig/adjustFig函数对所有axes同时调整。
    _i = 0
    for axes in axesList:
        _logscale += "%d"%(_i)
        _yscale=axes.get_yscale()
        if _yscale == 'log':
            _logscale=_logscale+'y'
        if _yscale == 'symlog':
            _logscale=_logscale+'sy'
        
        _xscale=axes.get_xscale()
        if _xscale == 'log':
            _logscale=_logscale+'x'
        if _xscale == 'symlog':
            _logscale=_logscale+'sx'
        _legendloc += "%d"%(_i)
        _legend = axes.get_legend()
        if _legend:
            _legendloc +="%d"%(_legend._loc)
        else:
            _legendloc +='-'
    
    _figfile="{prefix:}({key2:})_vs_({key1:})_lg({lgloc:})log({lgscale:})-{timestamp}.{postfix:s}".format(key1=key1, key2=key2, timestamp=getNowUTCtimestamp("0x%x", withMicroSec=True, microSecPatStr= '_%5x'), postfix=postfix, lgloc=_legendloc, lgscale=_logscale, prefix = prefix)
    _figfile=os.path.join(figDir, _figfile)
    print "save file: %s"%(_figfile)
    fig.savefig(_figfile, transparent=True)