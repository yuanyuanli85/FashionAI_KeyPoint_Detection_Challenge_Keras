
import numpy as np
import pandas as pd
import os

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


def split_csv_train_val(allcsv, traincsv, valcsv, ratio=0.8):
    xdf = pd.read_csv(allcsv)
    # random shuffle
    xdf = xdf.sample(frac=1)

    # random sampling
    msk = np.random.rand(len(xdf)) < ratio
    trainDf= xdf[msk]
    valDf= xdf[~msk]
    print "total", len(xdf), "split into train ", len(trainDf), '  val', len(valDf)

    #save to file
    trainDf.to_csv(traincsv, index=False)
    valDf.to_csv(valcsv, index=False)


def np_euclidean_l2(x, y):
    assert (x.shape == y.shape), "shape mismatched " + x.shape +" :  " + y.shape
    loss = np.sum((x - y)**2)
    loss = np.sqrt(loss)
    return loss


def load_annotation_from_df(df, category):
    if category == 'all':
        return df
    else:
        return df[df['image_category'] == category]


