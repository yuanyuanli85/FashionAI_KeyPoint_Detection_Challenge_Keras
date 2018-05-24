import sys
sys.path.insert(0, "../unet/")

import os
from test import BestModels
from evaluation import Evaluation

def rescan_outwear_with_mask():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    from test import BestModels
    print BestModels

    for key, value in BestModels.items():
        category = key
        modelfile = value[0]
        preScore = value[1]

        if category == 'outwear':

            xeval = Evaluation(category, modelfile)
            scores, xdict = xeval.eval_mask()

            print modelfile, ' original ', preScore,  ' vs mask ', sum(scores)/len(scores)

if __name__ == "__main__":
    rescan_outwear_with_mask()