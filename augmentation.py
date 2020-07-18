from __future__ import division
import cv2
import math
from PIL import Image
import numpy as np
import random

def generatePre(length, angle):
    ''' generate convolution kernel and anchor
    '''
    EPS = np.finfo(float).eps
    half = length/2 
    alpha = (angle - math.floor(angle/180)*180)/180*math.pi
    cosalpha = math.cos(alpha)
    sinalpha = math.sin(alpha)
    if cosalpha < 0:
        xsign = -1
    elif angle == 90:
        xsign = 0
    else:
        xsign = 1
    psfwdt = 1
    ## blur mask size 
    sx = int(math.fabs(length*cosalpha + psfwdt*xsign - length*EPS))
    sy = int(math.fabs(length*sinalpha + psfwdt - length*EPS))
    psf1 = np.zeros((sy,sx))
    
    ## psf1
    for i in range(0, sy):
        for j in range(0, sx):
            psf1[i][j] = i*math.fabs(cosalpha) - j*sinalpha
            rad = math.sqrt(i*i + j*j)
            if rad >= half and math.fabs(psf1[i][j]) <= psfwdt:
                temp = half - math.fabs((j + psf1[i][j]*sinalpha)/cosalpha)
                psf1[i][j] = math.sqrt(psf1[i][j]*psf1[i][j]+temp*temp)
            psf1[i][j] = psfwdt + EPS - math.fabs(psf1[i][j])
            if psf1[i][j] < 0:
                psf1[i][j] = 0
    ## 
    anchor = (0,0)
    if angle <90 and angle >0:
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1]-1, 0)
    elif angle >-90 and angle <0:
        psf1 = np.flipud(psf1)
        psf1 = np.fliplr(psf1)
        anchor = (psf1.shape[1]-1, psf1.shape[0]-1)
    elif angle <-90:
        psf1 = np.flipud(psf1)
        anchor = (0, psf1.shape[0] - 1)
    psf1 = psf1/psf1.sum()
    return psf1, anchor
    



class guassian_blur(object):
    ''' blur image 
    '''
    def __init__(self, mask_size):
        assert isinstance(mask_size, int)
        self.mask_size = mask_size

    def __call__(self,image):
        if random.random()>0.5:
            cv_img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            dst = cv2.GaussianBlur(cv_img,(self.mask_size,self.mask_size),0)
            pil_img = Image.fromarray(cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))
            return pil_img
        else:
            return image

class median_blur(object):
    ''' median blur image
    '''
    def __init__(self, mask_size):
        assert isinstance(mask_size, int)
        self.mask_size = mask_size
        
    def __call__(self,image):
        if random.random()>0.5:
            cv_img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            dst = cv2.medianBlur(cv_img, self.mask_size)
            pil_img = Image.fromarray(cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))
            return pil_img
        else:
            return image

class mean_bur(object):
    ''' mean blur
    '''
    def __init__(self, mask_size):
        assert isinstance(mask_size, int)
        self.mask_size = mask_size
    def __call__(self, image):
        if random.random()>0.5:
            cv_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            dst = cv2.blur(cv_img, (self.mask_size, self.mask_size))
            pil_img = Image.fromarray(cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))
            return pil_img
        else:
            return image

class motion_blur(object):
    ''' motion blur image 
    '''
    def __init__(self, length, angle):
        self.k , self.a = generatePre(length, angle)
    
    def __call__(self,image):
        if random.random()>0.5:
            cv_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            dst = cv2.filter2D(cv_img, -1, self.k, anchor = self.a)
            pil_img = Image.fromarray(cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))
            return pil_img
        else:
            return image


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.3, sh = 0.6, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    def __call__(self, img):
        #if random.uniform(0, 1) > self.probability:
            #return img
        for attempt in range(100):
            img_w,img_h,c = img.shape
            area = img_w * img_h
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_w and h < img_h:
                x1 = random.randint(0, img_w - w)
                y1 = random.randint(0, img_h - h)
                if len(img.shape) == 3:
                    img[x1:x1+w, y1:y1+h, 0] = self.mean[0]
                    img[x1:x1+w, y1:y1+h, 1] = self.mean[1]
                    img[x1:x1+w, y1:y1+h, 2] = self.mean[2]
                else:
                    img[x1:x1+w, y1:y1+h, 0] = self.mean[0]
                return img
        return img
