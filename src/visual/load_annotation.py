import pandas as pd
import numpy as np
import cv2
import os
from data_process import pad_image, rotate_image
from kpAnno import KpAnno

SKIRT_KP_KEYS=['image_id', 'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right']

def make_gaussian(width, height, sigma=3, center=None):
    '''
        generate 2d guassion heatmap
    :return:
    '''

    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]

    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp( -4*np.log(2)*((x-x0)**2 + (y-y0)**2)/sigma**2)

def load_train_annotation(annfile, category):
    xpd = pd.read_csv(annfile)

    xpd = xpd[ xpd['image_category'] == category ]
    print xpd.shape, 'for ', category
    return xpd

def visualize_df(dfrow,  keys):
    datapath = "/mnt/sh_flex_storage/yli150/ai_fashion/train/"
    mlist   =  dfrow[keys]
    imgName = mlist[0]
    kplst   = mlist[1:]

    draw_gaussion = True

    #load cvmat
    xcvmat = cv2.imread(os.path.join(datapath, imgName))

    #draw kp point on image
    for _kpstr in kplst:
        _kpAnn = KpAnno(_kpstr)
        if _kpAnn.visibility == 1:
            cv2.circle(xcvmat, (_kpAnn.x, _kpAnn.y), radius=7, color=(0,255,255), thickness=2)

    #show cvmat with kp circles
    cv2.imshow(imgName, xcvmat)
    if draw_gaussion:
        mask = generate_mask(dfrow, keys, (xcvmat.shape[0], xcvmat.shape[1]))
        cv2.imshow(imgName+"mask", mask)

    cv2.waitKey()

def visualize_pad_df(dfrow,  keys):
    datapath = "/mnt/sh_flex_storage/yli150/ai_fashion/train/"
    mlist   =  dfrow[keys]
    imgName = mlist[0]
    kplst   = mlist[1:]

    draw_gaussion = True

    #load cvmat
    xcvmat = cv2.imread(os.path.join(datapath, imgName))

    #draw kp point on image
    kpAnnLst = list()
    for _kpstr in kplst:
        _kpAnn = KpAnno.readFromStr(_kpstr)
        kpAnnLst.append(_kpAnn)

    imgPadded, nKpAnnlst = pad_image(xcvmat, kpAnnLst, 512, 512)

    #for _kpAnn in nKpAnnlst:
    #    cv2.circle(imgPadded, (_kpAnn.x, _kpAnn.y), radius=7, color=(0,255,255), thickness=2)

    print xcvmat.shape, "padded to", imgPadded.shape

    outmat, nKpAnnlst = rotate_image(imgPadded, nKpAnnlst, 30)

    for _kpAnn in nKpAnnlst:
        cv2.circle(outmat, (_kpAnn.x, _kpAnn.y), radius=7, color=(0,255,255), thickness=2)

    cv2.imshow("padded", outmat)

    cv2.waitKey()


def generate_mask(dfrow, keys, matShape):
    mlist   =  dfrow[keys]
    imgName = mlist[0]
    kplst   = mlist[1:]

    mask = np.zeros(matShape)

    for _kpstr in kplst:
        _kpAnn = KpAnno(_kpstr)
        if _kpAnn.visibility == 1:
            radius = 100
            gaussMask = make_gaussian(radius, radius , 20, None)
            mask[ _kpAnn.y-radius/2: _kpAnn.y+radius/2, _kpAnn.x-radius/2: _kpAnn.x+radius/2] = gaussMask

    return mask

def main():
    datapath = "/mnt/sh_flex_storage/yli150/ai_fashion/train/"
    annfile  = os.path.join(datapath, "Annotations/train.csv")
    xpd = load_train_annotation(annfile, 'skirt')
    print xpd.keys()

    count = 0
    for _index, _row in xpd.iterrows():
        count +=1
        if count > 10:
            break
        visualize_pad_df(_row, SKIRT_KP_KEYS)

if __name__ == "__main__":
    main()