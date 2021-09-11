import cv2
import numpy as np
import math
from functools import reduce
from operator import iconcat

'''
pip install opencv-python opencv-contrib-python

'''


f = lambda x,y: 2*x*y
g = lambda x,y: x**2 - y**2

class IGFTT:
    def __init__(self, _nfeatures=200, _scaleFactor=1.2, _nlevels=8,
        _firstLevel=0, _qualityLevel=0.01, _blockSize=31,_minDistance=10):

        self.detector = cv2.GFTTDetector_create( _nfeatures, 
                                                 _qualityLevel, _minDistance, 
                                                 _blockSize)
        self.descriptor = cv2.xfeatures2d.FREAK_create()
        self.scaleFactor = _scaleFactor
        self.nlevels = _nlevels
        self.firstLevel = _firstLevel
        self.blockSize = _blockSize

    def computePyramid(self, image):
        self.imagePyramid = [image]

        if len(image.shape)==2:
            rows, cols = image.shape
        else:
            rows, cols,_ = image.shape

        self.scales = [1]

        for level in range(1, self.nlevels):
            scale = 1/(self.scaleFactor**(level-self.firstLevel))
            self.scales.append(scale)
            self.imagePyramid.append(cv2.resize(self.imagePyramid[level-1], (round(cols*scale), round(rows*scale))))

    def computeOrientation(self, image, keypoints, smooth=False):
        # make a reflect border frame to simplify kernel operation on borders
        borderedImg = cv2.copyMakeBorder(   image, 
                                            self.blockSize,
                                            self.blockSize,
                                            self.blockSize,
                                            self.blockSize, 
                                            cv2.BORDER_DEFAULT)

        # apply a gradient in both axis
        sobelx = cv2.Sobel(borderedImg, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(borderedImg, cv2.CV_64F, 0, 1, ksize=3)
        angles = []
        h,w = image.shape

        for point in keypoints:
            nominator = 0.
            denominator = 0.
            i,j = point.pt
            for k in range(self.blockSize):
                for l in range(self.blockSize):
                    posX = round(self.blockSize-1 + (i*self.blockSize) + k)
                    posY = round(self.blockSize-1 + (j*self.blockSize) + l)

                    if posX < 0:
                        posX = 0
                    elif posX > w:
                        posX = w
                    if posY < 0:
                        posY = 0
                    elif posY > h:
                        posY = h

                    valX = sobelx.item(posY, posX)
                    valY = sobely.item(posY, posX)

                    nominator += f(valX, valY)
                    denominator += g(valX, valY)

            # if the strength (norm) of the vector 
            # is not greater than a threshold
            if math.sqrt(nominator**2 + denominator**2) < 1000000:
                angle = 0.
            else:
                if denominator >= 0:
                    angle = cv2.fastAtan2(nominator, denominator)
                elif denominator < 0 and nominator >= 0:
                    angle = cv2.fastAtan2(nominator, denominator) + math.pi
                else:
                    angle = cv2.fastAtan2(nominator, denominator) - math.pi
                angle /= float(2)

            angles.append(angle)

        if smooth:
            angles = np.array(angles)
            angles = cv2.GaussianBlur(angles, (3,3), 0, 0)
            angles = angles.reshape(-1,)

        return angles

    def compute(self, _, k):
        self.descriptor_list = []
        for level in range(self.nlevels):
            img = self.imagePyramid[level]
            if self.keypoints_list[level] != []:
                k,d = self.descriptor.compute(img, self.keypoints_list[level])
                if d is not None:
                    self.descriptor_list.append(d)
                    self.keypoints_list[level] = k
                else:
                    self.keypoints_list[level] = []
        return reduce(iconcat, self.keypoints_list, []), np.array(reduce(iconcat, self.descriptor_list, []))

    def detect(self, image, mask):
        self.keypoints_list = []
        self.computePyramid(image)
        for level in range(self.nlevels):
            img = self.imagePyramid[level]
            keypoints = self.detector.detect(img, mask)
            angles = self.computeOrientation(self.imagePyramid[level], keypoints)
            if (level != self.firstLevel):
                for k in range(len(keypoints)):
                    keypoints[k].octave = level
                    keypoints[k].size = self.blockSize * self.scales[level]
                    keypoints[k].pt = (keypoints[k].pt[0]*self.scales[level],keypoints[k].pt[1]*self.scales[level])
                    keypoints[k].angle = angles[k]
            self.keypoints_list.append(keypoints)
        return reduce(iconcat, self.keypoints_list, [])

    def detectAndCompute(self, image, mask):
        keypoints = self.detect(image, mask)
        keypoints, descriptor = self.compute(image, keypoints)
        return keypoints, descriptor