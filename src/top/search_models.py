
from collections import defaultdict

def scan_valfile(valfile):
    with open(valfile) as xval:
        lines = xval.readlines()

    xlist = list()
    for linenum, xline in enumerate(lines):
        if 'hdf5' in xline and 'Socre' in xline:
            modelname = xline.strip().split(',')[0]
            overallscore = xline.strip().split(',')[1]
            xlist.append((modelname, overallscore, linenum))
    #print xlist

    xdict = defaultdict(list)
    for xitem in xlist:
        modelname, score, linenum = xitem
        for xline in lines[linenum+1:linenum+6]:
            category, categoryscore = xline.strip().split(':')
            xdict[category].append((modelname, float(categoryscore)))
    return xdict

def get_best_item(scorelst):

    def get_key(item):
        return item[1]

    bestmodel = sorted(scorelst, key=get_key)[0]

    return bestmodel


def get_best_models(valfile):
    modeldict = scan_valfile(valfile)

    mdict = dict()
    for key in modeldict.keys():
        bestmodel = get_best_item(modeldict[key])
        mdict[key] = bestmodel

    return mdict


def get_best_single_model(valfile):
    def get_key(item):
        return item[1]

    with open(valfile) as xval:
        lines = xval.readlines()

    xlist = list()
    for linenum, xline in enumerate(lines):
        if 'hdf5' in xline and 'Socre' in xline:
            modelname = xline.strip().split(',')[0]
            overallscore = xline.strip().split(',')[1]
            xlist.append((modelname, overallscore))

    bestmodel = sorted(xlist, key=get_key)[0]

    return bestmodel

if __name__ == "__main__":
    print get_best_single_model('../../trained_models/all/2018_04_28_09_32_57/val_flip.txt')