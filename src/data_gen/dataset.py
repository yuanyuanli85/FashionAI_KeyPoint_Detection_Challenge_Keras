

def getKpNum(category):
    # remove one column 'image_id'
    return len(getKpKeys(category)) - 1

TROUSERS_PART_KYES=['waistband_left', 'waistband_right', 'crotch', 'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
TROUSERS_PART_FLIP_KYES=['waistband_right', 'waistband_left', 'crotch', 'bottom_right_in', 'bottom_right_out', 'bottom_left_in', 'bottom_left_out']

SKIRT_PART_KEYS=['waistband_left', 'waistband_right', 'hemline_left', 'hemline_right']
SKIRT_PART_FLIP_KEYS=['waistband_right', 'waistband_left', 'hemline_right', 'hemline_left']


DRESS_PART_KEYS= ['neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right', 'center_front',
              'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
              'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'hemline_left', 'hemline_right']
DRESS_PART_FLIP_KEYS=['neckline_right', 'neckline_left', 'shoulder_right', 'shoulder_left', 'center_front',
               'armpit_right', 'armpit_left', 'waistline_right', 'waistline_left', 'cuff_right_in',
               'cuff_right_out', 'cuff_left_in', 'cuff_left_out', 'hemline_right', 'hemline_left']

BLOUSE_PART_KEYS=['neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right',
           'center_front', 'armpit_left', 'armpit_right', 'top_hem_left', 'top_hem_right',
           'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out']

BLOUSE_PART_FLIP_KEYS=['neckline_right', 'neckline_left', 'shoulder_right', 'shoulder_left',
           'center_front', 'armpit_right', 'armpit_left', 'top_hem_right', 'top_hem_left',
           'cuff_right_in', 'cuff_right_out', 'cuff_left_in', 'cuff_left_out']

OUTWEAR_PART_KEYS=['neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right',
            'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
            'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right']

OUTWEAR_PART_FLIP_KEYS = ['neckline_right', 'neckline_left', 'shoulder_right', 'shoulder_left',
           'armpit_right', 'armpit_left', 'waistline_right', 'waistline_left', 'cuff_right_in',
           'cuff_right_out', 'cuff_left_in', 'cuff_left_out', 'top_hem_right', 'top_hem_left']

ALL_PART_KEYS = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
               'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out',
               'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left', 'waistband_right',
               'hemline_left', 'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out',
               'bottom_right_in', 'bottom_right_out']

ALL_PART_FLIP_KEYS = [  'neckline_right', 'neckline_left', 'center_front', 'shoulder_right', 'shoulder_left',
                        'armpit_right', 'armpit_left',   'waistline_right', 'waistline_left', 'cuff_right_in', 'cuff_right_out',
                        'cuff_left_in', 'cuff_left_out', 'top_hem_right', 'top_hem_left',  'waistband_right','waistband_left',
                        'hemline_right', 'hemline_left',  'crotch',  'bottom_right_in', 'bottom_right_out',
                        'bottom_left_in', 'bottom_left_out']

def getFlipKeys(category):
    if category == 'skirt':
        keys, mapkeys = SKIRT_PART_KEYS, SKIRT_PART_FLIP_KEYS
    elif category == 'dress':
        keys, mapkeys = DRESS_PART_KEYS, DRESS_PART_FLIP_KEYS
    elif category == 'trousers':
        keys, mapkeys = TROUSERS_PART_KYES, TROUSERS_PART_FLIP_KYES
    elif category == 'blouse':
        keys, mapkeys = BLOUSE_PART_KEYS, BLOUSE_PART_FLIP_KEYS
    elif category == 'outwear':
        keys, mapkeys = OUTWEAR_PART_KEYS, OUTWEAR_PART_FLIP_KEYS
    elif category == 'all':
        keys, mapkeys = ALL_PART_KEYS, ALL_PART_FLIP_KEYS
    else:
        assert (0), category + " not supported"

    xdict = dict()
    for i in range(len(keys)):
        xdict[keys[i]] = mapkeys[i]
    return keys, xdict

def getFlipMapID(category, partid):
    keys, mapDict = getFlipKeys(category)
    mapKey = mapDict[keys[partid]]
    mapID  = keys.index(mapKey)
    return mapID

def getKpKeys(category):
    '''

    :param category:
    :return: get the keypoint keys in annotation csv
    '''
    SKIRT_KP_KEYS = ['image_id', 'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right']
    DRESS_KP_KEYS = ['image_id', 'neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right', 'center_front',
                     'armpit_left',  'armpit_right' ,  'waistline_left' , 'waistline_right', 'cuff_left_in',
                     'cuff_left_out', 'cuff_right_in',  'cuff_right_out',  'hemline_left',  'hemline_right']
    TROUSERS_KP_KEYS=['image_id',  'waistband_left', 'waistband_right', 'crotch',  'bottom_left_in',
                      'bottom_left_out', 'bottom_right_in', 'bottom_right_out']
    BLOUSE_KP_KEYS = [ 'image_id', 'neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right',
                       'center_front', 'armpit_left', 'armpit_right', 'top_hem_left', 'top_hem_right',
                       'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out']
    OUTWEAR_KP_KEYS= ['image_id', 'neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right',
                      'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',
                      'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right']

    ALL_KP_KESY = ['image_id','neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                 'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in',
                 'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right' ,
                 'crotch', 'bottom_left_in' , 'bottom_left_out', 'bottom_right_in' ,'bottom_right_out']

    if category == 'skirt':
        return SKIRT_KP_KEYS
    elif category == 'dress':
        return DRESS_KP_KEYS
    elif category == 'trousers':
        return TROUSERS_KP_KEYS
    elif category == 'blouse':
        return BLOUSE_KP_KEYS
    elif category == 'outwear':
        return OUTWEAR_KP_KEYS
    elif category == 'all':
        return ALL_KP_KESY
    else:
        assert(0), category + ' not supported'


def fill_dataframe(kplst, category, dfrow):
    keys = getKpKeys(category)[1:]

    # fill category
    dfrow['image_category'] = category

    assert (len(keys) == len(kplst)), str(len(kplst)) + ' must be the same as ' + str(len(keys))
    for i, _key in enumerate(keys):
        kpann = kplst[i]
        outstr = str(int(kpann.x))+"_"+str(int(kpann.y))+"_"+str(1)
        dfrow[_key] = outstr


def get_kp_index_from_allkeys(kpname):
    ALL_KP_KEYS = ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right',
                   'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out',
                   'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left', 'waistband_right',
                   'hemline_left', 'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']

    return ALL_KP_KEYS.index(kpname)


def generate_input_mask(image_category, shape, nobgFlag=True):
    import numpy as np
    # 0.0 for invalid key points for each category
    # 1.0 for valid key points for each category
    h, w, c = shape
    mask = np.zeros((h // 2, w // 2, c), dtype=np.float)

    for key in getKpKeys(image_category)[1:]:
        index = get_kp_index_from_allkeys(key)
        mask[:, :, index] = 1.0

    # for last channel, background
    if nobgFlag:     mask[:, :, -1] = 0.0
    else:   mask[:, :, -1] = 1.0

    return mask