import pandas as pd
import numpy as np
import cv2
import os
from kpAnno import KpAnno

def normalize_image(cvmat):
    assert (cvmat.dtype == np.uint8) , " only support normalize np.uint8 to float -0.5 ~ 0.5'"
    cvmat = cvmat.astype(np.float)
    cvmat = (cvmat - 128.0) / 256.0
    return cvmat

def resize_image(cvmat, targetWidth, targetHeight):

    assert (cvmat.dtype == np.uint8) , " only support normalize np.uint8  in  resize_image'"

    # get scale
    srcHeight, srcWidth, channles = cvmat.shape
    minScale = min( targetHeight*1.0/srcHeight,  targetWidth*1.0/srcWidth)

    # resize
    resizedMat = cv2.resize(cvmat, None, fx=minScale, fy=minScale)
    reHeight, reWidth, channles = resizedMat.shape

    # pad to targetWidth or targetHeight
    outmat = np.zeros((targetHeight, targetWidth, 3), dtype=cvmat.dtype) + 128

    if targetHeight == reHeight and targetWidth == reWidth:
        outmat = resizedMat
    elif targetWidth != reWidth and targetHeight == reHeight:
        # add pad to width
        outmat[:, 0:reWidth, :] = resizedMat
    elif targetHeight != reHeight and targetWidth == reWidth:
        # add padding to height
        outmat[0:reHeight, :, :] = resizedMat
    else:
        assert(0), "after resize either width or height same as target width or target height"
    return (outmat, minScale)

def pad_image(cvmat, kpAnno, targetWidth, targetHeight):
    '''

    :param cvmat: input mat
    :param targetWidth:  width to pad
    :param targetHeight: height to pad
    :return:
    '''
    assert (cvmat.dtype == np.uint8) , " only support normalize np.uint8  in pad_image'" + str(cvmat.dtype)

    srcHeight, srcWidth, channles = cvmat.shape
    outmat = np.zeros((targetHeight, targetWidth, 3), dtype=cvmat.dtype) + 128

    if targetHeight == srcHeight and targetWidth == srcWidth:
        outmat =  cvmat
        outkpAnno = kpAnno
    elif targetWidth != srcWidth and targetHeight == srcHeight:
        # add pad to width
        outmat[:, 0:srcWidth, :] = cvmat
        outkpAnno = kpAnno
    elif targetHeight != srcHeight and targetWidth == srcWidth:
        # add padding to height
        outmat[0:srcHeight, :, :] = cvmat
        outkpAnno = kpAnno
    else:
        # resize at first, then pad
        outmat, scale = resize_image(cvmat, targetWidth, targetHeight)
        outkpAnno = list()
        for _kpAnno in kpAnno:
            _nkp = KpAnno.applyScale(_kpAnno, scale)
            outkpAnno.append(_nkp)
    return (outmat, outkpAnno)


def pad_image_inference(cvmat, targetWidth, targetHeight):
    '''

    :param cvmat: input mat
    :param targetWidth:  width to pad
    :param targetHeight: height to pad
    :return:
    '''
    assert (cvmat.dtype == np.uint8), " only support normalize np.uint8  in pad_image'" + str(cvmat.dtype)

    srcHeight, srcWidth, channles = cvmat.shape
    outmat = np.zeros((targetHeight, targetWidth, 3), dtype=cvmat.dtype) + 128

    if targetHeight == srcHeight and targetWidth == srcWidth:
        outmat = cvmat
        scale = 1.0
    elif targetWidth > srcWidth and targetHeight == srcHeight:
        # add pad to width
        outmat[:, 0:srcWidth, :] = cvmat
        scale = 1.0
    elif targetHeight > srcHeight and targetWidth == srcWidth:
        # add padding to height
        outmat[0:srcHeight, :, :] = cvmat
        scale = 1.0
    else:
        # resize at first, then pad
        outmat, scale = resize_image(cvmat, targetWidth, targetHeight)

    return (outmat, scale)

def rotate_image(cvmat, kpAnnLst, rotateAngle):

    assert (cvmat.dtype == np.uint8) , " only support normalize np.uint8  in rotate_image'"

    ##Make sure cvmat is square?
    height, width, channel = cvmat.shape

    center = ( width//2, height//2)
    rotateMatrix = cv2.getRotationMatrix2D(center, rotateAngle, 1.0)

    cos, sin = np.abs(rotateMatrix[0,0]), np.abs(rotateMatrix[0, 1])
    newH = int((height*sin)+(width*cos))
    newW = int((height*cos)+(width*sin))

    rotateMatrix[0,2] += (newW/2) - center[0] #x
    rotateMatrix[1,2] += (newH/2) - center[1] #y

    # rotate image
    outMat = cv2.warpAffine(cvmat, rotateMatrix, (newH, newW), borderValue=(128, 128, 128))

    # rotate annotations
    nKpLst = list()
    for _kp in kpAnnLst:
        _newkp = KpAnno.applyRotate(_kp, rotateMatrix)
        nKpLst.append(_newkp)

    return (outMat, nKpLst)


def rotate_image_with_invrmat(cvmat, rotateAngle):

    assert (cvmat.dtype == np.uint8) , " only support normalize np.uint  in rotate_image_with_invrmat'"

    ##Make sure cvmat is square?
    height, width, channel = cvmat.shape

    center = ( width//2, height//2)
    rotateMatrix = cv2.getRotationMatrix2D(center, rotateAngle, 1.0)

    cos, sin = np.abs(rotateMatrix[0,0]), np.abs(rotateMatrix[0, 1])
    newH = int((height*sin)+(width*cos))
    newW = int((height*cos)+(width*sin))

    rotateMatrix[0,2] += (newW/2) - center[0] #x
    rotateMatrix[1,2] += (newH/2) - center[1] #y

    # rotate image
    outMat = cv2.warpAffine(cvmat, rotateMatrix, (newH, newW), borderValue=(128, 128, 128))

    # generate inv rotate matrix
    invRotateMatrix = cv2.invertAffineTransform(rotateMatrix)

    return (outMat, invRotateMatrix, (width, height))

def rotate_mask(mask, rotateAngle):

    outmask = rotate_image_float(mask, rotateAngle)

    return outmask

def rotate_image_float(cvmat, rotateAngle, borderValue=(0.0, 0.0, 0.0)):

    assert (cvmat.dtype == np.float) , " only support normalize np.float  in rotate_image_float'"

    ##Make sure cvmat is square?
    height, width, channels = cvmat.shape

    center = ( width//2, height//2)
    rotateMatrix = cv2.getRotationMatrix2D(center, rotateAngle, 1.0)

    cos, sin = np.abs(rotateMatrix[0,0]), np.abs(rotateMatrix[0, 1])
    newH = int((height*sin)+(width*cos))
    newW = int((height*cos)+(width*sin))

    rotateMatrix[0,2] += (newW/2) - center[0] #x
    rotateMatrix[1,2] += (newH/2) - center[1] #y

    # rotate image
    outMat = cv2.warpAffine(cvmat, rotateMatrix, (newH, newW), borderValue=borderValue)

    return outMat


def crop_image(cvmat, kpAnnLst, lowLimitRatio, upLimitRatio):
    import random

    assert(lowLimitRatio < 1.0), 'lowLimitRatio should be less than 1.0'
    assert(upLimitRatio < 1.0), 'upLimitRatio should be less than 1.0'

    height, width, channels = cvmat.shape

    cropHeight = random.randrange(int(lowLimitRatio*height),  int(upLimitRatio*height))
    cropWidth  = random.randrange(int(lowLimitRatio*width),  int(upLimitRatio*width))

    top_x = random.randrange(0,  width - cropWidth)
    top_y = random.randrange(0,  height - cropHeight)

    # apply offset for keypoints
    nKpLst = list()
    for _kp in kpAnnLst:
        if _kp.visibility == -1:
            _newkp = _kp
        else:
            _newkp = KpAnno.applyOffset(_kp, (top_x, top_y))
            if _newkp.x <=0 or _newkp.y <=0:
                # negative location, return original image
                return cvmat, kpAnnLst
            if _newkp.x >= cropWidth or _newkp.y >= cropHeight:
                # keypoints are cropped out
                return cvmat, kpAnnLst
        nKpLst.append(_newkp)

    return cvmat[top_y:top_y+cropHeight,  top_x:top_x+cropWidth], nKpLst

if __name__ == "__main__":
    pass