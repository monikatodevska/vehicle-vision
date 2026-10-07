import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from joblib import Parallel, delayed
import multiprocessing
import pickle
import h5py
# custom package imports
#from SingleShotDetector.Helpers import helper_model, helper_data, helper_stats, losses
import sys
#sys.path.insert(0, 'E:\Science\Monika\SingleShotDetector\Helpers' )
import cv2
import random
#from tqdm import tqdm

import xml.etree.ElementTree as ET
import helper_model, helper_stats, helper_losses
import helper_data

srcImagesPath = r'E:\Science\Monika1\proba'
# srcAnnotationsPath = r'E:\Science\Monika\GRAM-RTMv4\Annotations\site2'
srcAnnotationsPathTrain = r'E:\Science\Monika\GRAM-RTMv4\Annotations\baseline_blurred'

for im_ind, im_name in enumerate(os.listdir(srcImagesPath)):
    image=cv2.imread(os.path.join(srcImagesPath, im_name))

    annot_name = str(int(im_name[5:11].lstrip('0')) - 1) + '.xml'
    # print(annot_name)

    root = ET.parse(os.path.join(srcAnnotationsPathTrain, annot_name)).getroot()

    objects = []  # list of all objects in the image
    # cv2.imwrite(os.path.join(im_path_resized, im_name), image)
    for object in root.findall('object'):
        cl = object.find('class').text

        bb_xml = object.find('bndbox')
        bb = [np.int(bb_xml.find('xmin').text),  # min_col
              np.int(bb_xml.find('xmax').text),  # max_col
              np.int(bb_xml.find('ymin').text),  # min_row
              np.int(bb_xml.find('ymax').text),  # max_row
              ]
        # if flag1 == 1:
        # print(1)
        annot = [int(np.round(bb[2])), int(np.round(bb[0])), int(np.round(bb[3])), int(np.round(bb[1]))]
        cv2.rectangle(image, (annot[1],annot[0]), (annot[3], annot[2]), color=(0, 0, 0), thickness=1)
    cv2.imshow("slika", image)
    cv2.waitKey(0)
