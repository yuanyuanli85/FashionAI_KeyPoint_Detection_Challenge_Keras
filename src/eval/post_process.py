import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from keras.layers import *
from kpAnno import KpAnno

def post_process_heatmap(heatMap, kpConfidenceTh=0.2):
    kplst = list()
    for i in range(heatMap.shape[-1]):
        # ignore last channel, background channel
        _map = heatMap[0, :, :, i]
        _map = gaussian_filter(_map, sigma=0.5)
        _nmsPeaks = non_max_supression(_map, windowSize=3, threshold=1e-6)

        y, x = np.where(_nmsPeaks == _nmsPeaks.max())
        confidence = np.amax(_nmsPeaks)
        if confidence > kpConfidenceTh:
            kplst.append(KpAnno(x[0], y[0], 1))
        else:
            kplst.append(KpAnno(x[0], y[0], -1))
    return kplst

def non_max_supression(plain, windowSize=3, threshold=1e-6):
    # clear value less than threshold
    under_th_indices = plain < threshold
    plain[under_th_indices] = 0
    return plain* (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))
