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
import shutil
import xml.etree.ElementTree as ET
import helper_model, helper_stats, helper_losses
import helper_data

srcImagesPath = r'D:\KlasifikacijaVozila\M-30\Sliki\M-30'
dstImages=r'D:\KlasifikacijaVozila\M-30\Sliki_Test'
# srcAnnotationsPath = r'E:\Science\Monika\GRAM-RTMv4\Annotations\site2'
# srcAnnotationsPathTrain = r'E:\Science\Monika\GRAM-RTMv4\Annotations\baseline_blurred'
GT=r'D:\KlasifikacijaVozila\M-30\GroundTruthPairwise'
test=r'D:\KlasifikacijaVozila\M-30\GT_test'
# val=r'E:\Science\Monika\val'
anchorless_test=r'E:\Science\Monika\anchorless_test1'
# anchorless_val=r'E:\Science\Monika\anchorless_val1'

# list=os.listdir(test)

# for im_name in lista_odzeme:
#     for k in list:
#         if (im_name==k):
#             shutil.move(os.path.join(GT, im_name),os.path.join(anchorless_test, im_name))
def zemi_val_test(srcImagesPath, dstImages, GT, test, anchorless_test):
    m=1
    lista_odzeme = os.listdir(GT)
    lista_val = os.listdir(anchorless_test)
    for gt in lista_odzeme:
        for k in lista_val:
            im_name=k[:-4]+'.jpg'
            if (gt==k):
                # im_name_new='image'+str(m)+'.jpg'
                # gt_new='image'+str(m)+'.txt'
                shutil.move(os.path.join(GT, gt),os.path.join(test, gt))
                shutil.move(os.path.join(srcImagesPath, im_name),os.path.join(dstImages, im_name))
                # m+=1
            else:
                continue

zemi_val_test(srcImagesPath, dstImages, GT, test, anchorless_test)

srcImagesPath1 = r'D:\KlasifikacijaVozila\M-30\Sliki\M-30'
dstImages1=r'D:\KlasifikacijaVozila\M-30\Sliki_Val'
# srcAnnotationsPath = r'E:\Science\Monika\GRAM-RTMv4\Annotations\site2'
# srcAnnotationsPathTrain = r'E:\Science\Monika\GRAM-RTMv4\Annotations\baseline_blurred'
GT1=r'D:\KlasifikacijaVozila\M-30\GroundTruthPairwise'
test1=r'D:\KlasifikacijaVozila\M-30\GT_val'
# val=r'E:\Science\Monika\val'
anchorless_test1=r'E:\Science\Monika\anchorless_val1'

zemi_val_test(srcImagesPath1, dstImages1, GT1, test1, anchorless_test1)
