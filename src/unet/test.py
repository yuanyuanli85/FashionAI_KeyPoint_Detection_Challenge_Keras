import argparse
import os
from fashion_net import FashionNet
from dataset import getKpNum
import pandas as pd
from evaluation import Evaluation
import pickle

'''
#Best Model for 2018_06_26 Submission. VGG Net. score 0.33
BestModels={
    'skirt':    ("../../trained_models/skirt/2018_03_25_22_20_03/skirt_weights_23.hdf5",  0.1402114940594051),
    'trousers': ("../../trained_models/trousers/2018_03_25_14_15_21/trousers_weights_12.hdf5", 0.11658021921491965),
    'dress':    ("../../trained_models/dress/2018_03_25_14_15_02/dress_weights_28.hdf5", 0.14163446602933985),
    'blouse':   ("../../trained_models/blouse/2018_03_22_23_13_19/blouse_weights_11.hdf5", 0.0786875168025268),
    'outwear':  ("../../trained_models/outwear/2018_03_22_23_03_32/outwear_weights_29.hdf5", 0.08480500074609196)
}
'''
'''
#Best Model for 2018_03_27 Submission. VGG Net. 0.26
BestModels={
    'skirt':    ("../../trained_models/skirt/2018_03_27_13_21_29/skirt_weights_17.hdf5",  0.11524739140686265),
    'trousers': ("../../trained_models/trousers/2018_03_27_17_26_05/trousers_weights_13.hdf5", 0.09115299117353642),
    'dress':    ("../../trained_models/dress/2018_03_27_15_02_30/dress_weights_21.hdf5",  0.09793824499191259),
    'blouse':   ("../../trained_models/blouse/2018_03_27_15_02_04/blouse_weights_15.hdf5", 0.06314956156765669),
    'outwear':  ("../../trained_models/outwear/2018_03_27_20_08_47/outwear_weights_24.hdf5", 0.08677644593823085)
}
'''
'''
#Best Model for 2018_03_28 Submission. Resnet50 Net. 0.16
BestModels={
    'skirt':    ("../../trained_models/skirt/2018_03_28_14_42_35/skirt_weights_14.hdf5",  0.0698002982048666),
    'trousers': ("../../trained_models/trousers/2018_03_28_12_51_59/trousers_weights_21.hdf5", 0.06290934801519793),
    'dress':    ("../../trained_models/dress/2018_03_28_16_19_59/dress_weights_25.hdf5", 0.06735959685128637),
    'blouse':   ("../../trained_models/blouse/2018_03_28_16_53_58/blouse_weights_23.hdf5", 0.0362057181574364),
    'outwear':  ("../../trained_models/outwear/2018_03_28_16_55_47/outwear_weights_25.hdf5",  0.06221689915623416)
}
'''


#Best Model for 2018_03_29 Submission. Resnet50 Net. Use eval with fix to re-eval the submission 0.159
BestModels_0329={
    'skirt':    ("../../trained_models/skirt/2018_03_28_14_42_35/skirt_weights_14.hdf5",  0.0588873056834),
    'trousers': ("../../trained_models/trousers/2018_03_28_12_51_59/trousers_weights_21.hdf5", 0.06290934801519793),
    'dress':    ("../../trained_models/dress/2018_03_28_16_19_59/dress_weights_25.hdf5", 0.0652740429229),
    'blouse':   ("../../trained_models/blouse/2018_03_28_16_53_58/blouse_weights_23.hdf5", 0.0351579070828),
    'outwear':  ("../../trained_models/outwear/2018_03_28_16_55_47/outwear_weights_25.hdf5",  0.0492414384113)
}


#Best Model for 2018_03_30 Submission. Resnet50 Net. Use predict_kp_mask to solve the false detections at (0,0) corner in outwear
#Slight improvement from 0.159 to 0.152
BestModels_0330={
    'skirt':    ("../../trained_models/skirt/2018_03_28_14_42_35/skirt_weights_14.hdf5",  0.0588873056834),
    'trousers': ("../../trained_models/trousers/2018_03_28_12_51_59/trousers_weights_21.hdf5", 0.06290934801519793),
    'dress':    ("../../trained_models/dress/2018_03_28_16_19_59/dress_weights_25.hdf5", 0.0652740429229),
    'blouse':   ("../../trained_models/blouse/2018_03_28_16_53_58/blouse_weights_23.hdf5", 0.0351579070828),
    'outwear':  ("../../trained_models/outwear/2018_03_28_16_55_47/outwear_weights_25.hdf5",  0.0354971702938)
}

#Best Model for 2018_04_02 Submission. Resnet101. 0.152 -> 0.111
BestModels_0402={
    'skirt':    ("../../trained_models/skirt/2018_03_30_16_35_29/skirt_weights_8.hdf5",  0.0446304929077803),
    'trousers': ("../../trained_models/trousers/2018_03_30_22_19_12/trousers_weights_25.hdf5", 0.039162348555361666 ),
    'dress':    ("../../trained_models/dress/2018_03_30_16_40_11/dress_weights_20.hdf5",     0.05182003532558566),
    'blouse':   ("../../trained_models/blouse/2018_03_31_08_41_44/blouse_weights_20.hdf5",   0.027468196108032836),
    'outwear':  ("../../trained_models/outwear/2018_03_31_03_29_02/outwear_weights_24.hdf5", 0.0291968529842)
}

#Best Model for 2018_04_04 Submission. Resnet101 with flip in data augmentation. Blouse model not changed. 0.111 -> 0.1042
BestModels_0404={
    'skirt':    ("../../trained_models/skirt/2018_04_03_20_19_10/skirt_weights_15.hdf5", 0.0412277380944543),
    'trousers': ("../../trained_models/trousers/2018_04_03_16_12_29/trousers_weights_26.hdf5", 0.036052273520051706),
    'dress':    ("../../trained_models/dress/2018_04_03_12_21_01/dress_weights_29.hdf5",     0.04125825446962443),
    'blouse':   ("../../trained_models/blouse/2018_04_04_06_30_26/blouse_weights_24.hdf5",   0.022291405592591337),
    'outwear':  ("../../trained_models/outwear/2018_04_03_21_53_27/outwear_weights_23.hdf5", 0.02498769022)
}

#Best Model for 2018_04_06 Submission. v8 network, with 2 stacked refined net. 0.1005 -> 0.0994
BestModels_0406={
    'skirt':    ("../../trained_models/skirt/2018_04_05_18_38_15/skirt_weights_24.hdf5", [0.03643287886445002, 0.0528]),
    'trousers': ("../../trained_models/trousers/2018_04_05_09_06_00/trousers_weights_27.hdf5", [0.032952730245957254, 0.0533]),
    'dress':    ("../../trained_models/dress/2018_04_04_20_00_30/dress_weights_23.hdf5",     [0.038578562185730685, 0.10]),
    'blouse':   ("../../trained_models/blouse/2018_04_05_09_06_56/blouse_weights_24.hdf5",  [0.02200512919488186, 0.076]),
    'outwear':  ("../../trained_models/outwear/2018_04_05_15_12_30/outwear_weights_22.hdf5", [0.026, 0.11])
}

#Best Model for 0413. V9 network. finetune from all-in-one network. 0.0994 -> 0.0753
BestModels={
    'skirt':    ("../../trained_models/skirt/2018_04_11_16_56_21/skirt_weights_17.hdf5", 0.0469),
    'trousers': ("../../trained_models/trousers/2018_04_12_02_45_06/trousers_weights_19.hdf5", 0.0517),
    'blouse':   ("../../trained_models/blouse/2018_04_12_21_37_47/blouse_weights_12.hdf5", 0.0605),
    'all_in_one': ("../../trained_models/all/2018_04_09_10_46_41_x2/all_weights_11.hdf5", 'outwear_0.072'+'dress_0.071'),
}

def load_image_names(annfile, category):
    # read into dataframe
    xdf = pd.read_csv(annfile)
    xdf = xdf[xdf['image_category'] == category]
    return xdf

def run_test(category, savefile):
    testfile = os.path.join("../../data/test/", 'test.csv')

    #init network
    bstModel = BestModels[category]
    print "Run best model", bstModel[0], " val score , ",bstModel[1]
    xEval = Evaluation(category, bstModel[0])

    #load images and run prediction
    xdict = dict()
    xdf = load_image_names(testfile, category)
    print len(xdf), " images to process"
    for _index, _row in xdf.iterrows():
        _image_id = _row['image_id']
        imageName = os.path.join("../../data/test", _image_id)
        if category == 'outwear':
            dtkp = xEval.predict_kp_with_mask(imageName, multiOutput=True)
        else:
            dtkp = xEval.predict_kp(imageName, multiOutput=True)
        xdict[_image_id] = dtkp

    with open(savefile, 'wb') as xfile:
        pickle.dump(xdict, xfile)

    print "prediction save to ", savefile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--category", help="specify cloth category")
    parser.add_argument("--outpath", help="path to save predicted keypoints")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    if args.category == 'all':
        for _category in ['skirt', 'blouse', 'outwear', 'trousers', 'dress']:
            run_test(_category, os.path.join(args.outpath, _category+".pkl"))
    else:
        run_test(args.category, os.path.join(args.outpath, args.category+".pkl"))
