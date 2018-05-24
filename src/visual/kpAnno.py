import numpy as np


class KpAnno(object):
    '''
        Convert string to x, y, visibility
    '''
    def __init__(self, x, y, visibility):
        self.x = int(x)
        self.y = int(y)
        self.visibility = visibility

    @classmethod
    def readFromStr(cls, xstr):
        xarray = xstr.split('_')
        x = int(xarray[0])
        y = int(xarray[1])
        visibility = int(xarray[2])
        return cls(x,y, visibility)

    @classmethod
    def applyScale(cls, kpAnno, scale):
        x = int(kpAnno.x*scale)
        y = int(kpAnno.y*scale)
        v = kpAnno.visibility
        return cls(x, y, v)

    @classmethod
    def applyRotate(cls, kpAnno, rotateMatrix):
        vector = [kpAnno.x, kpAnno.y, 1]
        rotatedV = np.dot(rotateMatrix, vector)
        return cls( int(rotatedV[0]), int(rotatedV[1]), kpAnno.visibility)

    @classmethod
    def applyOffset(cls, kpAnno, offset):
        x = kpAnno.x - offset[0]
        y = kpAnno.y - offset[1]
        v = kpAnno.visibility
        return cls(x, y, v)

    @staticmethod
    def calcDistance(kpA, kpB):
        distance = (kpA.x - kpB.x)**2 + (kpA.y - kpB.y)**2
        return np.sqrt(distance)
