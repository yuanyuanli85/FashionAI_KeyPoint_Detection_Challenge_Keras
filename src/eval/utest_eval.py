import os
from evaluation import Evaluation
from kpAnno import KpAnno
import time
import pickle
import cv2
from fashion_net import FashionNet
from dataset import getKpNum
from post_process import post_process_heatmap
from time import time

def main_test():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    category = 'trousers'
    modelFile = "../../trained_models/trousers/2018_03_28_12_51_59/trousers_weights_21.hdf5"

    '''
    xnet = FashionNet(512, 512, getKpNum(category)+1)
    xnet.load_model(modelFile)
    '''

    image_id = "Images/trousers/73a0aad03b7fcf50bc1ee088a089ce82.jpg"
    image_file = os.path.join("../../data/train", image_id)

    xeval = Evaluation(category, modelFile)
    kplst = xeval.predict_kp(image_file, multiOutput=True)

    cvmat = cv2.imread(image_file)
    visualize_predicted_keypoint(cvmat, kplst, (255, 0, 0))
    cv2.imshow('image', cvmat)
    cv2.waitKey()

    '''
    predmap = xnet.predict_image(image_file)
    predmap = predmap[-1]

    cvmat = cv2.imread(image_file)

    print cvmat.shape

    for i in range(getKpNum(category)+1):
        premask =  predmap[0,:, :, i]

        cv2.imshow('image', cvmat)
        cv2.imshow('premask'+str(i), premask)
        cv2.waitKey()

    kplst = post_process_heatmap(premask)
    '''


def main_val():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    modelFile = "../../trained_models/skirt/2018_03_27_08_55_56/skirt_weights_26.hdf5"
    xEstimator = Evaluation('skirt', modelFile)

    start = time.time()
    valScore = xEstimator.eval(multiOut=True)
    end = time.time()

    print valScore, end - start


def main_val_scan():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    xpath = "../../trained_models/skirt/2018_03_27_08_55_56"
    for xfile in os.listdir(xpath):
        if xfile.endswith(".hdf5"):
            modelFile = os.path.join(xpath, xfile)
            xEstimator = Evaluation('skirt', modelFile)

            start = time.time()
            valScore = xEstimator.eval(multiOut=True)
            end = time.time()
            print valScore, end - start, xfile

def read_kp_from_str(xstr):
    kplst = list()
    for kpstr in xstr.split(','):
        kplst.append(KpAnno.readFromStr(kpstr))
    return kplst

def utest_trousers():
    gtkpStr = ["430_284_0,713_303_0,560_537_1,560_626_1,361_588_1,573_622_1,-1_-1_-1",
                "359_301_1,464_297_1,417_403_1,340_669_1,308_658_1,456_713_1,491_714_1"]
    dtkpStr = ["430_294_0,713_323_0,560_567_1,560_666_1,361_638_1,573_682_1,123_345_1",
               "359_311_1,464_317_1,417_433_1,340_709_1,308_708_1,456_773_1,491_784_1"]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    modelFile = os.path.join("../../trained_models/2018_03_20_21_43",
                             "skirt_adam_weights_45_39.33.hdf5")

    xEstimator = Evaluation('trousers', modelFile)

    mlist = list()
    for i in range(2):
        gtA = read_kp_from_str(gtkpStr[i])
        dtA = read_kp_from_str(dtkpStr[i])
        score = xEstimator._calc_ne_score(dtA, gtA)
        mlist.extend(score)

    print sum(mlist)/len(mlist)

def main_val_debug():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    modelFile = "../../trained_models/trousers/2018_03_28_12_51_59/trousers_weights_21.hdf5"
    xEstimator = Evaluation('trousers', modelFile)

    start = time.time()
    scoreDict = xEstimator.eval_debug(multiOut=True)
    end = time.time()

    print "inference done", end - start

    with open("../../eval_temp/trousers_temp.pkl", 'w') as outfile:
        pickle.dump(scoreDict, outfile )

def visualize_predicted_keypoint(cvmat, kpoints, color, radius=7, all=True):
    for _kpAnn in kpoints:
        if _kpAnn.visibility == 1 or all == True:
            cv2.circle(cvmat, (_kpAnn.x, _kpAnn.y), radius=radius, color=color, thickness=2)
    return cvmat

def ananlyze_skirt_prediction(evalpkl):
    with open(evalpkl) as xfile:
        scores, evaldict   = pickle.load(xfile)

    scorelst = list()
    for key, value in evaldict.items():
        _imageName = key
        gtKpoints, dtKpoints, singleImgScore = value
        if len(singleImgScore) != 0:
            scorelst.append( ( key, sum(singleImgScore)/len(singleImgScore) ))

    def get_key(item):
        return item[1]

    topN = sorted(scorelst, key=get_key)[-10:]

    print topN

    for imageid, score in topN:
        _cvmat = cv2.imread(os.path.join("../../data/train", imageid))
        gtKpoints, dtKpoints, singleImgScore = evaldict[imageid]

        for i in range(len(gtKpoints)):
            dtkp = dtKpoints[i]
            gtkp = gtKpoints[i]
            print (gtkp.x,  dtkp.y), (dtkp.x, gtkp.y), gtkp.visibility, singleImgScore[i]

        _cvmat = visualize_predicted_keypoint(_cvmat, gtKpoints, color=(0, 0, 255), radius=7, all=False)
        _cvmat = visualize_predicted_keypoint(_cvmat, dtKpoints, color=(255, 0, 0), radius=9)
        cv2.imshow(imageid, _cvmat)
        cv2.waitKey()

def analyze_prediction_error(evalpkl):
    with open(evalpkl) as xfile:
        scores, evaldict = pickle.load(xfile)

    scorelst = list()
    for key, value in evaldict.items():
        _imageName = key
        gtKpoints, dtKpoints, singleImgScore = value
        scorelst.append(singleImgScore)

    count = [0 for i in range(len(scorelst[0]))]
    for _score in scorelst:
        for partindex, partscore in enumerate(_score):
            count[partindex] += partscore

    print count


def debug():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    category='blouse'
    xEval = Evaluation(category, "../../trained_models/blouse/2018_03_28_16_53_58/blouse_weights_23.hdf5")
    score = xEval.eval(multiOut=True)
    print score

def visual_gt_kp():
    import pandas as pd
    OUTWEAR_KP_KEYS = ['image_id', 'neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right',
                       'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                       'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right']


    with open("../../eval_temp/outwear_temp.pkl") as xfile:
        scores, evaldict   = pickle.load(xfile)

    image_id = "Images/outwear/1f0e67c84a5cf10715b4622d591d3438.jpg"
    annfile = os.path.join("../../data/train/Annotations", "val_split.csv")

    xpd = pd.read_csv(annfile)
    xpd = xpd[xpd['image_category'] == 'outwear']
    xpd = xpd[xpd['image_id'] == image_id]

    _cvmat = cv2.imread(os.path.join("../../data/train", image_id))

    gtKpoints, dtKpoints, singleImgScore = evaldict[image_id]

    _cvmat = visualize_predicted_keypoint(_cvmat, gtKpoints, color=(0, 0, 255), all=False, radius=5)
    _cvmat = visualize_predicted_keypoint(_cvmat, dtKpoints, color=(255, 0, 0), radius=8)

    for i in range(len(gtKpoints)):
        dtkp = dtKpoints[i]
        gtkp = gtKpoints[i]
        print gtkp.x , dtkp.x, gtkp.y, dtkp.y ,gtkp.visibility

    print Evaluation.calc_ne_score('outwear',dtKpoints, gtKpoints)

    cv2.imshow(image_id, _cvmat)
    cv2.waitKey()

def scan_zero_all(xpath):

    for xfile in os.listdir(xpath):
        if xfile.endswith('pkl'):
            zerocount, totallen = scan_zero_kp(os.path.join(xpath, xfile))
            print zerocount, totallen, xfile


def scan_zero_kp(mfile):
    with open(mfile) as xfile:
        scores, evaldict   = pickle.load(xfile)

    zero_count = 0
    for key, value in evaldict.items():
        _imageName = key
        gtKpoints, dtKpoints, singleImgScore = value
        for _kp, _gtkp  in zip(dtKpoints, gtKpoints):
            if _gtkp.visibility == 1:
                if _kp.x == 0 and _kp.y == 0 and _kp.visibility == -1:
                    zero_count += 1

    return zero_count, len(evaldict)


def rescan_model_eval():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    from test import BestModels
    print BestModels

    for key, value in BestModels.items():
        category = key
        modelfile = value[0]
        preScore = value[1]

        if category == 'outwear':
            xeval = Evaluation(category, modelfile)
            refinedScore, xdict = xeval.eval_debug(multiOut=True)

            print modelfile, preScore, sum(refinedScore)/len(refinedScore)


def eval_scan(category, modelPath):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    valLog = os.path.join(modelPath, "val.txt")
    for modelfile in os.listdir(modelPath):
        if modelfile.endswith('.hdf5'):
            xmodelfile = os.path.join(modelPath, modelfile)
            xEval = Evaluation(category, xmodelfile)

            start = time()
            neScore = xEval.eval(multiOut=True)
            end = time()
            print xmodelfile, " Evaluation Done", str(neScore), " cost ", end - start, " seconds!"

            with open(valLog, 'a+') as xfile:
                xfile.write(xmodelfile + ", Socre " + str(neScore) + "\n")
            xfile.close()


if __name__ == "__main__":
    rescan_model_eval()
    #eval_scan('skirt',    "../../trained_models/skirt/2018_03_30_16_35_29/")
    #eval_scan('blouse',   "../../trained_models/blouse/2018_03_31_08_41_44/")
    #eval_scan('trousers', "../../trained_models/trousers/2018_03_30_22_19_12/")
    #eval_scan('dress',    "../../trained_models/dress/2018_03_30_16_40_11/")
    #eval_scan('outwear', "../../trained_models/outwear/2018_03_31_03_29_02/")

    #rescan_model_eval()
    #main_val_debug()
    #main_test()
    #scan_zero_all("../../")
    #analyze_prediction_error("../../eval_temp/outwear_temp.pkl")
    #main_val_debug()
