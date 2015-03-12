# -*- coding: UTF-8 -*-
'''
Created on Mar 12, 2015-1:20:50 PM

@author: Ling Wang<LingWangNeuralEng@gmail.com>
'''
import os
from Constants_and_Parameters import *

def activeLogFile(logFilePrefix):
    _logFile = os.path.expandvars(os.path.join("${PoreAnalyzer_TMP}",logFilePrefix+"log.txt"))
    os.environ[K_PorAnalyzer_LogFile] = _logFile
    print "set PorAnalyzer_LogFile to ", os.environ[K_PorAnalyzer_LogFile]

def appendLogText(texts, logTextList = None, write = True):
    print texts
    if not logTextList == None:
        logTextList.append(texts)
    if write:
        _logfile = os.environ[K_PorAnalyzer_LogFile]
        with open(_logfile, 'ab') as hLog:
            hLog.write(texts)
            hLog.write('\n')
        
def currentLogFileName():
    if os.environ.has_key(K_PorAnalyzer_LogFile):
        return os.environ[K_PorAnalyzer_LogFile]
    else:
        return None
