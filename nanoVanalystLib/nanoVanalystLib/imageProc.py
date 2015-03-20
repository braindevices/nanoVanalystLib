# -*- coding: UTF-8 -*-
'''
Created on Mar 12, 2015-1:08:22 PM

@author: Ling Wang<LingWangNeuralEng@gmail.com>
'''
import cv2, os
import numpy as np
from LWpyUtils.generalUtils import getNowUTCtimestamp, setEnviron
from scipy import stats
from Constants_and_Parameters import *
from imgFileUtils import loadAsGray, showImg, saveNPZ
from logUtils import activeLogFile, appendLogText, currentLogFileName

def composeMaskedGray(gray, mask, flagInv = True, maskCh = 2, maskFactor = 0.7, grayFactor = 1.):
    
    _maskRGB = np.zeros(mask.shape[0:2]+(3,), dtype = mask.dtype)
    if flagInv:
        
        _maskRGB [:,:,maskCh] = 255 - mask
    else:
        _maskRGB [:,:,maskCh] = mask
    
    _merge = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    _merge= cv2.addWeighted(_maskRGB, maskFactor, _merge, grayFactor, 0)
    return _merge

def detectPore1(imgFile, cropY=[0,880], CLAHEkwds = dict(clipLimit=2., tileGridSize = (4, 4)), gsBlurBlockSize = 11, binTh = 35, MorphOpenItN = 1, kernelSize=3, poreAsFG = True, autoBinTh_percent = 5, autoBinTh_mode_ratio = 0.35, binThType = 'modeRatio', seedWithBlur = False, shedOnWhat = 'gray_blur'):
    '''
    gsBlurBlockSize, the higher the blur size, the higher possibility to connect parts of big pores. Too high is prone to get false positives.
    autoBinTh_mode_ratio, low is prone to get false negative, high is prone to get false positive.
    binThType:
        'percentile', not good idea in normal condition
        'manual'
        'modeRatio', default
    seedWithBlur:
        except the image is too noisy, turn this off
        
    
    '''
    
    
    _prefix = getNowUTCtimestamp("%x_")
    setEnviron(PoreAnalyzer_IMprefix = _prefix)
    activeLogFile(_prefix
                  )
    _logTexts = []
    appendLogText("{0:}".format(locals()), _logTexts)
        
    gray = loadAsGray(imgFile, cropY)
    showImg(gray, "gray")
#     print gray.shape, gray.dtype
    clahe = cv2.createCLAHE(**CLAHEkwds)
    cl1 = clahe.apply(gray)
    _logText = "np.median(cl1) = {0:}".format(np.median(cl1))
    
    appendLogText(_logText, _logTexts)
    
    
    #find the peak of histogram
    _mode, _count = stats.mode(cl1.flat)
    _mode =_mode[0]
    #the better way is to smooth the histogram and find the peak
    
    _logText = "hist peak at: {0:}".format( _mode)
    appendLogText(_logText, _logTexts)
    
    showImg(cl1, "cl1(clip%d_tilegrid%d)_comp"%(CLAHEkwds['clipLimit'], CLAHEkwds['tileGridSize'][0]), True)
    cl1_blur = cv2.GaussianBlur(cl1, (gsBlurBlockSize, gsBlurBlockSize), 0)
    showImg(cl1_blur, "cl1(clip%d_tilegrid%d)_blur(%d)_comp"%(CLAHEkwds['clipLimit'], CLAHEkwds['tileGridSize'][0], gsBlurBlockSize ), True)
    
    if binThType == 'modeRatio':
        _autoBinTh = autoBinTh_mode_ratio*_mode
        
    elif binThType == 'percentile':
        _autoBinTh = np.percentile(cl1, autoBinTh_percent)
        
    elif binThType == "manual":
        _autoBinTh = binTh
    _logText = "seed threshold: {0:}".format( _autoBinTh)
    appendLogText(_logText, _logTexts)
    
    if seedWithBlur:
        _seed0 = cl1_blur
    else:
        _seed0 = cl1
    
    

    _retval, _seed=cv2.threshold(_seed0, _autoBinTh, 255, cv2.THRESH_BINARY)
    _seed = 255 - _seed # invert the seeds
    kernel_c = np.ones((kernelSize,kernelSize),np.uint8)
    _seed = cv2.morphologyEx(_seed, cv2.MORPH_CLOSE, kernel_c, iterations = MorphOpenItN)
    _comp = composeMaskedGray(gray, _seed, flagInv=False)
    showImg(_comp, "seed_MORPH_CLOSE(k%dx%d_it%d)_comp"%(kernel_c.shape[0],kernel_c.shape[1], MorphOpenItN), True)

    kernel_o = np.ones((5,5),np.uint8)
    
    _retval, _otsu = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _comp = composeMaskedGray(gray, _otsu, flagInv=False)
    showImg(_comp, "cl1_blur_otsu_comp", True)
    _result = cv2.morphologyEx(_otsu, cv2.MORPH_OPEN, kernel_o, iterations = MorphOpenItN)
    _comp = composeMaskedGray(gray, _result, flagInv=False)
    showImg(_comp, "cl1_blur_otsu_MORPH_OPEN(k%dx%d_it%d)_comp"%( kernel_o.shape[0],kernel_o.shape[1], MorphOpenItN), True)

    _sure_bg = _result
    _markerImg=np.zeros(_sure_bg.shape, dtype=np.int32)
    
    if poreAsFG:
        _markerImg = _markerImg + _sure_bg/255
        _markerImg = _markerImg + _seed/255*2
        _postfix = '_poreFG'
    else:
        _markerImg = _markerImg + _sure_bg/255*2
        _markerImg = _markerImg + _seed/255
        _postfix = '_poreBG'
        
    if shedOnWhat == "cl1_blur":
    
        _watershedInput = cl1_blur
    elif shedOnWhat == "cl1":
        _watershedInput = cl1
    elif shedOnWhat == "gray":
        _watershedInput = gray
    elif shedOnWhat == "gray_blur":
        gray_blur = cv2.GaussianBlur(gray, (gsBlurBlockSize, gsBlurBlockSize), 0)
        _watershedInput = gray_blur
    elif shedOnWhat == "otsu":
        #directly use binary image as watershed input
        #apparently this would not work, due to false positive
        _result = cv2.morphologyEx(_otsu, cv2.MORPH_CLOSE, kernel_c, iterations = MorphOpenItN)
        _result = cv2.morphologyEx(_result, cv2.MORPH_OPEN, kernel_o, iterations = MorphOpenItN)
        _watershedInput = _result
    
    _watershedInput = cv2.cvtColor(_watershedInput, cv2.COLOR_GRAY2RGB)
    
    
    print "watershed start"
    cv2.watershed(_watershedInput, _markerImg)
    print "watershed end"
    
    
    _GrayMarked = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    _GrayMarked[_markerImg == -1, :] = [0, 0, 255]
    showImg(_GrayMarked, "marked_all_FG(%s_withSureBG)_as_1Obj"%(shedOnWhat) +_postfix, True)
    return gray, _markerImg

def countPores(markerImg):
    _binImg = np.zeros(markerImg.shape, dtype = np.uint8)
    _binImg[markerImg == 2] =255
    contours, hierarchy = cv2.findContours(_binImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     print contours
#     print hierarchy
    _areas = np.zeros((len(contours),), dtype = float)
    for _i in xrange(0,_areas.size):
        _areas[_i] = cv2.contourArea(contours[_i])
    return _areas, contours, hierarchy
        
def runPoreDetection1Img(imgFile, inDir, outDir, detectPore1Kwds = {}, dirForEachFile = False, show0AreaObj = False):
    outDir = os.path.expanduser(outDir)
    if not os.path.exists(outDir):
        raise IOError("%s does not exist"%(outDir))
    if not os.path.isdir(outDir):
        raise IOError("%s does not exist"%(outDir))
    
    _subDir = os.path.dirname(imgFile)
    imgFile = os.path.join(inDir, imgFile)
    
    if dirForEachFile:
        _subsubDir = os.path.splitext(os.path.basename(imgFile))[0]
        PoreAnalyzer_TMP = os.path.join(outDir, _subDir, _subsubDir)
    else:
        PoreAnalyzer_TMP = os.path.join(outDir, _subDir)
    
    if not os.path.isdir(PoreAnalyzer_TMP):
        os.makedirs(PoreAnalyzer_TMP)
    
    setEnviron(PoreAnalyzer_TMP = PoreAnalyzer_TMP, PoreAnalyzer_IMshowWait = 10, PoreAnalyzer_IMprefix = "", FLAG_SHOW_IMG = False)
    
    
    
    _gray, _markerImg = detectPore1(imgFile, **detectPore1Kwds)
    _areas, cnts, hierarchys = countPores(_markerImg)
    _idxs0a = np.nonzero(_areas == 0)[0]
    
    if _idxs0a.size >0:
        appendLogText( "some contour(s) has(have) 0 area: {0:}".format(_idxs0a) )
        
        if show0AreaObj:
            _imgRGB = cv2.cvtColor(_gray, cv2.COLOR_GRAY2RGB)
    #         print [cnts[_i] for _i in _idxs0a]
            cv2.drawContours(_imgRGB, [cnts[_i] for _i in _idxs0a], -1, (0,0,255), 2)
            showImg(_imgRGB, "0Area contours", flagShowImg= True, waitTime=0)
        np.sort(_idxs0a)
        
        for _i in _idxs0a[::-1]:
            cnts.pop(_i)
        _areas = _areas[_areas >0]
    _logAreas = np.log10(_areas)
    
    appendLogText("{0:} Area: mean = {1:}, median = {2:}, max = {3:}, min = {4:}, geo_mean = {5:} ".format(imgFile, _areas.mean(), np.median(_areas), _areas.max(), _areas.min(), 10**_logAreas.mean()))
    #save the contours and hierarchy for future analysis
    structKwds = dict(areas = _areas, contours = cnts, hierarchy = hierarchys)
    _npzFile = saveNPZ("detectionResult", structKwds)
    return _areas, _npzFile

#set the default environ during module init
setEnviron(PoreAnalyzer_TMP = TMPDIR, PoreAnalyzer_IMshowWait = 0, PoreAnalyzer_IMprefix = "", FLAG_SHOW_IMG = True)

def analyzePoresInDir(imgDir, imgPattern = "*.tif", resultDir = os.path.join(TMPDIR, "results"), saveCSV = False, runPoreDetection1ImgKwds = dict(dirForEachFile = True)):
    """
    return array of areas, All images must belong to a single category 
    imgDir: the absolute path contain images
    imgPattern: the wildcard pattern of the image file. It can have subDir, for example: "*/*.tif", which will search exactly 1-level down to the imgDir for tif files. The dir structure will be maintained in resultDir
    resultDir: the parent dir for all results, resulting images of each image will be put into a separated subDir.
    saveCSV: if True, the area will be automatically saved into resultDir as <UTCstamp>_areas.
    
    """
    _start = getNowUTCtimestamp("%x")
    areasList = []
    logFiles = []
    npzFiles = []
    parameters = []
    import glob
    for _f in glob.glob(os.path.join(imgDir, imgPattern)):
        print "processing file:", _f
        _relativeF =  os.path.relpath(_f, imgDir)
        _areas, _npzFile = runPoreDetection1Img(_relativeF, imgDir, resultDir, **runPoreDetection1ImgKwds)
        areasList.append(_areas)
        logFiles.append(currentLogFileName())

        npzFiles.append(os.path.relpath(_npzFile, resultDir))
        
    #concatenate all 
    areas = np.concatenate(tuple(areasList))
    print "areas.shape = ", areas.shape
    _end = getNowUTCtimestamp("%x")
    csvFile = "%s-%s_areas.csv"%(_start, _end)
    csvFile = os.path.join(resultDir,csvFile)
    if saveCSV:
        
        for _f in logFiles:
            _t = ""
            with open(_f, 'rb') as _hf:
                _t = _hf.read()
#                 print _f, _t
            
            parameters.append("[%s]\n"%(_f)+_t)
        
        parameterStr = "\n\n".join(parameters)
        np.savetxt(csvFile, areas, delimiter = ",", header = "Area in pixels", footer = parameterStr, comments = '# ')
    npzlistFile = os.path.splitext(csvFile)[0] + '.npzlist'
    
    import codecs
    with codecs.open(npzlistFile, 'wb', 'utf-8') as _hf:
        _hf.write('\n'.join(npzFiles))
    
    return areas
