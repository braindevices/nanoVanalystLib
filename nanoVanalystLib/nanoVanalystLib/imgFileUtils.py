# -*- coding: UTF-8 -*-
'''
Created on Mar 12, 2015-1:16:18 PM

@author: Ling Wang<LingWangNeuralEng@gmail.com>
'''

import cv2, os

from Constants_and_Parameters import *


def loadAsGray(imgFile, cropY=[0,880]):
    img = cv2.imread(imgFile)
    img = img[cropY[0]:cropY[1],:,:]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def showImg(img, windName, write = False, outDir = None, prefix = None, waitTime = None, flagShowImg = None):
    if outDir == None:
        
        outDir = os.environ[K_PoreAnalyzer_TMP]
    if waitTime == None:
        waitTime = int(os.environ[K_PoreAnalyzer_IMshowWait])
    if prefix == None:
        prefix = os.environ[K_PoreAnalyzer_IMprefix]
    if flagShowImg == None:
        flagSowImg = bool(os.environ[K_FLAG_SHOW_IMG])
    if flagShowImg:
        cv2.imshow(windName, img)
        cv2.waitKey(waitTime)
        cv2.destroyWindow(windName)
    cv2.imwrite(os.path.join(outDir, prefix+ windName+".png"), img)
