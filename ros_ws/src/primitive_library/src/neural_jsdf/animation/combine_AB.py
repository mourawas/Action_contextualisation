"""
@Author: Yiting CHEN
@Time: 2023/8/2 下午6:13
@Email: chenyiting@whu.edu.cn
version: python 3.9
Created by PyCharm
"""
import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool

if __name__ == "__main__":

    for i in range(120):
        im_A = cv2.imread("gt/{}.jpg".format(i),  1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
        im_B = cv2.imread("pred/{}.jpg".format(i), 1)  # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
        im_A = im_A[100:800, 200:1000, :]
        im_B = im_B[100:800, 200:1000, :]

        print(im_B.shape)
        im_AB = np.concatenate([im_A, im_B], 1)
        cv2.imwrite("states/{}.jpg".format(i), im_AB)
