import os
import numpy as np
import cv2
import random
import copy
import pickle
# from tqdm import tqdm

import xml.etree.ElementTree as ET

# custom imports
import sys

sys.path.insert(0, 'E:\Science\Monika\SingleShotDetector\Helpers')
import helper_postprocessing
# from tqdm import tqdm

from joblib import Parallel, delayed
import math

def normalize_output_by_region(regions, norm_coefs, destination_matrix, out_reg_norm, out_reg_val_norm, anchor_stride):
    """

    :param regions:
    :param norm_coefs:
    :param out_reg_norm:
    :param out_reg_val_norm:
    :param destination_matrix:
    :return:
    """

    regions = [int(round(x / anchor_stride)) for x in regions]
    # print(out_reg_norm.shape)
    # print(regions)

    for i in range(len(regions) - 1):
        for redica in range(regions[i], regions[i + 1]):
            if redica >= out_reg_norm.shape[1]:
                break
            out_reg_norm[:, redica, :, destination_matrix] = out_reg_norm[:, redica, :, destination_matrix] / norm_coefs[i]
            out_reg_val_norm[:, redica, :, destination_matrix] = out_reg_val_norm[:, redica, :, destination_matrix] / norm_coefs[i]

    return out_reg_norm, out_reg_val_norm


def normalize_output_by_region_optimized(regions, norm_coefs, destination_matrix, out_reg_norm, out_reg_val_norm, anchor_stride):
    """

    :param regions:
    :param norm_coefs:
    :param out_reg_norm:
    :param out_reg_val_norm:
    :param destination_matrix:
    :return:
    """

    regions = [int(round(x / anchor_stride)) for x in regions]

    for i in range(len(regions) - 1):

        out_reg_norm[:, regions[i]:min(regions[i+1], out_reg_norm.shape[1]), :, destination_matrix] = \
            out_reg_norm[:, regions[i]:min(regions[i+1], out_reg_norm.shape[1]), :, destination_matrix] / norm_coefs[i]

        out_reg_val_norm[:, regions[i]:min(regions[i+1], out_reg_norm.shape[1]), :, destination_matrix] = \
            out_reg_val_norm[:, regions[i]:min(regions[i+1], out_reg_norm.shape[1]), :, destination_matrix] / norm_coefs[i]

    return out_reg_norm, out_reg_val_norm


def reverse_normalization(regions, norm_coefs, destination_matrix, out_reg_norm, anchor_stride):
    """

    :param regions:
    :param norm_coefs:
    :param out_reg_norm:
    :param out_reg_val_norm:
    :param destination_matrix:
    :return:
    """

    regions = [int(round(x / anchor_stride)) for x in regions]
    # print(out_reg_norm.shape)
    # print(regions)

    for i in range(len(regions) - 1):
        for redica in range(regions[i], regions[i + 1]):
            if redica >= out_reg_norm.shape[1]:
                break
            out_reg_norm[:, redica, :, destination_matrix] = out_reg_norm[:, redica, :, destination_matrix] * norm_coefs[i]
            # out_reg_val_norm[:, redica, :, destination_matrix] = out_reg_val_norm[:, redica, :, destination_matrix] * norm_coefs[i]

    return out_reg_norm

def get_image_gt(groundtruthfiles_root, im_path_root):
    out_class_list = []
    out_reg_list = []
    image_list = []
    dirnames=[dirname for dirname in os.listdir(groundtruthfiles_root)]

    for dir_name in dirnames:

        im_path=os.path.join(im_path_root,dir_name)
        filenames = [x for x in os.listdir(dir_name)]

        for filename in tqdm(filenames):



            # print(filename)
            # im_name=filename[:-4] +'.jpg'
            # print(br)
            im_filename = filename[:-4] + '.jpg'

            fid1 = open(os.path.join(dir_name, filename), 'rb')
            out_class_dims = pickle.load(fid1)
            out_class_back = pickle.load(fid1)
            out_reg_dims = pickle.load(fid1)
            out_reg_back = pickle.load(fid1)
            fid1.close()
            out_class = np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
            # out_reg= np.reshape(out_reg_back, (out_reg_dims[0], out_reg_dims[1], out_reg_dims[2]))

            out_class_list.append(out_class)
            out_reg = np.reshape(out_reg_back, (out_reg_dims[0], out_reg_dims[1], out_reg_dims[2]))
            out_reg_list.append(out_reg)
            image = cv2.imread(os.path.join(im_path, im_filename), 0)
            image = image.reshape(image.shape[0], image.shape[1], 1)
            image_list.append(image)

        out_class_list = np.array(out_class_list)
        out_reg_list = np.array(out_reg_list)
        image_list = np.array(image_list)
        return image_list, out_class_list, out_reg_list


def load_image_and_gt_centerness(filename, im_path, groundtruthfiles):

    im_filename = filename[:-4] + '.jpg'

    fid1 = open(os.path.join(groundtruthfiles, filename), 'rb')
    out_class_dims = pickle.load(fid1)
    out_class_back = pickle.load(fid1)

    out_reg_dims = pickle.load(fid1)
    out_reg_back = pickle.load(fid1)
    out_centerness_dims = pickle.load(fid1)
    out_centerness_back = pickle.load(fid1)
    fid1.close()

    out_class = np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
    out_reg = np.reshape(out_reg_back, (out_reg_dims[0], out_reg_dims[1], out_reg_dims[2]))
    out_centerness = np.reshape(out_centerness_back, (out_centerness_dims[0], out_centerness_dims[1], out_centerness_dims[2]))


    image = cv2.imread(os.path.join(im_path, im_filename), 0)
    # print(os.path.join(im_path, im_filename)
    image = image.reshape(image.shape[0], image.shape[1], 1)

    return image, out_class,out_centerness, out_reg

# import pywt

def load_image_and_gt(filename, im_path, groundtruthfiles):

    im_filename = filename[:-4] + '.jpg'

    fid1 = open(os.path.join(groundtruthfiles, filename), 'rb')
    out_class_dims = pickle.load(fid1)
    out_class_back = pickle.load(fid1)

    out_reg_dims = pickle.load(fid1)
    out_reg_back = pickle.load(fid1)

    fid1.close()

    out_class = np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
    out_reg = np.reshape(out_reg_back, (out_reg_dims[0], out_reg_dims[1], out_reg_dims[2]))


    image = cv2.imread(os.path.join(im_path, im_filename), 0)

    # image=image/255.0
    # print(os.path.join(im_path, im_filename)
    image = image.reshape(image.shape[0], image.shape[1], 1)

    return image, out_class,out_reg
def load_image_and_gt_parts_wavelet(filename, im_path, groundtruthfiles):

    im_filename = filename[:-4] + '.bmp'

    fid1 = open(os.path.join(groundtruthfiles, filename), 'rb')
    out_class_dims = pickle.load(fid1)
    out_class_back = pickle.load(fid1)

    out_reg_dims = pickle.load(fid1)
    out_reg_back = pickle.load(fid1)

    fid1.close()

    out_class = np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
    out_reg = np.reshape(out_reg_back, (out_reg_dims[0], out_reg_dims[1], out_reg_dims[2]))


    image = cv2.imread(os.path.join(im_path, im_filename), 0)
    coeffs_db1 = pywt.dwt2(image, 'db1')
    coeffs_db2 = pywt.dwt2(image, 'db2')
    coeffs_db3 = pywt.dwt2(image, 'db3')
    coeffs_db4 = pywt.dwt2(image, 'db4')

    LL_db1, (LH_db1, HL_db1, HH_db1) = coeffs_db1
    LL_db2, (LH_db2, HL_db2, HH_db2) = coeffs_db2
    LL_db3, (LH_db3, HL_db3, HH_db3) = coeffs_db3
    LL_db4, (LH_db4, HL_db4, HH_db4) = coeffs_db4

    LL_db2_resized = cv2.resize(LL_db2, (LH_db1.shape[1], LH_db1.shape[0]))
    LH_db2_resized = cv2.resize(LH_db2, (LH_db1.shape[1], LH_db1.shape[0]))
    HL_db2_resized = cv2.resize(HL_db2, (HL_db1.shape[1], HL_db1.shape[0]))
    HH_db2_resized = cv2.resize(HH_db2, (HH_db1.shape[1], HH_db1.shape[0]))

    LL_db3_resized = cv2.resize(LL_db3, (LH_db1.shape[1], LH_db1.shape[0]))
    LH_db3_resized = cv2.resize(LH_db3, (LH_db1.shape[1], LH_db1.shape[0]))
    HL_db3_resized = cv2.resize(HL_db3, (HL_db1.shape[1], HL_db1.shape[0]))
    HH_db3_resized = cv2.resize(HH_db3, (HH_db1.shape[1], HH_db1.shape[0]))

    LL_db4_resized = cv2.resize(LL_db4, (LH_db1.shape[1], LH_db1.shape[0]))
    LH_db4_resized = cv2.resize(LH_db4, (LH_db1.shape[1], LH_db1.shape[0]))
    HL_db4_resized = cv2.resize(HL_db4, (HL_db1.shape[1], HL_db1.shape[0]))
    HH_db4_resized = cv2.resize(HH_db4, (HH_db1.shape[1], HH_db1.shape[0]))

    stacked = np.dstack((LL_db1, LH_db1, HL_db1, HH_db1, LL_db2_resized, LH_db2_resized, HL_db2_resized, HH_db2_resized, LL_db3_resized, LH_db3_resized, HL_db3_resized,
                         HH_db3_resized, LL_db4_resized, LH_db4_resized,
                         HL_db4_resized, HH_db4_resized))
    # image=image/255.0
    # print(os.path.join(im_path, im_filename)
    # image = image.reshape(image.shape[0], image.shape[1], 1)
    # images_list.append(stacked)
    return stacked, out_class, out_reg
def load_image_and_gt_parts_wavelet_not_parallel( groundtruthfiles,filenames,im_path,finetune):
    images_list=[]
    out_class_list=[]
    out_reg_list=[]
    for filename in tqdm(filenames):
        im_filename = filename[:-4] + '.bmp'

        fid1 = open(os.path.join(groundtruthfiles, filename), 'rb')
        out_class_dims = pickle.load(fid1)
        out_class_back = pickle.load(fid1)

        out_reg_dims = pickle.load(fid1)
        out_reg_back = pickle.load(fid1)

        fid1.close()

        out_class = np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
        out_reg = np.reshape(out_reg_back, (out_reg_dims[0], out_reg_dims[1], out_reg_dims[2]))


        image = cv2.imread(os.path.join(im_path, im_filename), 0)
        coeffs_db1 = pywt.dwt2(image, 'db1')
        coeffs_db2 = pywt.dwt2(image, 'db2')
        coeffs_db3 = pywt.dwt2(image, 'db3')
        coeffs_db4 = pywt.dwt2(image, 'db4')

        LL_db1, (LH_db1, HL_db1, HH_db1) = coeffs_db1
        LL_db2, (LH_db2, HL_db2, HH_db2) = coeffs_db2
        LL_db3, (LH_db3, HL_db3, HH_db3) = coeffs_db3
        LL_db4, (LH_db4, HL_db4, HH_db4) = coeffs_db4

        LL_db2_resized = cv2.resize(LL_db2, (LH_db1.shape[1], LH_db1.shape[0]))
        LH_db2_resized = cv2.resize(LH_db2, (LH_db1.shape[1], LH_db1.shape[0]))
        HL_db2_resized = cv2.resize(HL_db2, (HL_db1.shape[1], HL_db1.shape[0]))
        HH_db2_resized = cv2.resize(HH_db2, (HH_db1.shape[1], HH_db1.shape[0]))

        LL_db3_resized = cv2.resize(LL_db3, (LH_db1.shape[1], LH_db1.shape[0]))
        LH_db3_resized = cv2.resize(LH_db3, (LH_db1.shape[1], LH_db1.shape[0]))
        HL_db3_resized = cv2.resize(HL_db3, (HL_db1.shape[1], HL_db1.shape[0]))
        HH_db3_resized = cv2.resize(HH_db3, (HH_db1.shape[1], HH_db1.shape[0]))

        LL_db4_resized = cv2.resize(LL_db4, (LH_db1.shape[1], LH_db1.shape[0]))
        LH_db4_resized = cv2.resize(LH_db4, (LH_db1.shape[1], LH_db1.shape[0]))
        HL_db4_resized = cv2.resize(HL_db4, (HL_db1.shape[1], HL_db1.shape[0]))
        HH_db4_resized = cv2.resize(HH_db4, (HH_db1.shape[1], HH_db1.shape[0]))

        stacked = np.dstack((LL_db1, LH_db1, HL_db1, HH_db1, LL_db2_resized, LH_db2_resized, HL_db2_resized, HH_db2_resized, LL_db3_resized, LH_db3_resized, HL_db3_resized,
                             HH_db3_resized, LL_db4_resized, LH_db4_resized,
                             HL_db4_resized, HH_db4_resized))
        # image=image/255.0
        # print(os.path.join(im_path, im_filename)
        # image = image.reshape(image.shape[0], image.shape[1], 1)
        images_list.append(stacked)
        out_class_list.append(out_class)
        out_reg_list.append(out_reg)

    images_list=np.array(images_list)
    out_class_list=np.array(out_class_list)
    out_reg_list=np.array(out_reg_list)
    return images_list, out_class_list, out_reg_list

def load_image_and_gt_parts(filename, im_path, groundtruthfiles):

    im_filename = filename[:-4] + '.jpg'

    fid1 = open(os.path.join(groundtruthfiles, filename), 'rb')
    out_class_dims = pickle.load(fid1)
    out_class_back = pickle.load(fid1)

    out_reg_dims = pickle.load(fid1)
    out_reg_back = pickle.load(fid1)

    fid1.close()

    out_class = np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
    out_reg = np.reshape(out_reg_back, (out_reg_dims[0], out_reg_dims[1], out_reg_dims[2]))


    image = cv2.imread(os.path.join(im_path, im_filename), 0)
    # image=image/255.0
    # print(os.path.join(im_path, im_filename)
    image = image.reshape(image.shape[0], image.shape[1], 1)

    return image, out_class,out_reg

def load_image_and_gt_cls_parallel(filename, im_path, groundtruthfiles):

    im_filename = filename[:-4] + '.jpg'

    fid1 = open(os.path.join(groundtruthfiles, filename), 'rb')
    out_class_dims = pickle.load(fid1)
    out_class_back = pickle.load(fid1)
    fid1.close()
    out_class = np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))

    image = cv2.imread(os.path.join(im_path, im_filename), 0)
    image = image.reshape(image.shape[0], image.shape[1], 1)

    return image, out_class


def parse_anchor_data_results(result):
    """

    :param result:
    :return:
    """

    images = []
    y_class = []
    y_reg = []

    for ind, img_data in enumerate(result):

        images.append(result[ind][0])
        y_class.append(result[ind][1])
        y_reg.append(result[ind][2])

    images = np.array(images)
    y_class = np.array(y_class)
    y_reg = np.array(y_reg)

    return images, y_class, y_reg


def parse_anchor_data_results_lists(result):
    """

    :param result:
    :return:
    """

    images = []
    y_class = []
    y_reg = []

    for ind, img_data in enumerate(result):

        images.append(result[ind][0])
        y_class.append(result[ind][1])
        y_reg.append(result[ind][2])

    return images, y_class, y_reg


def parse_anchor_data_results_lists_centerness(result):
    """

    :param result:
    :return:
    """

    images = []
    y_class = []
    y_centerness=[]
    y_reg = []

    for ind, img_data in enumerate(result):

        images.append(result[ind][0])
        y_class.append(result[ind][1])
        y_centerness.append(result[ind][2])
        y_reg.append(result[ind][3])

    return images, y_class, y_centerness, y_reg


def parse_anchor_data_results_cls(result):
    """

    :param result:
    :return:
    """

    images = []
    y_class = []

    for ind, img_data in enumerate(result):

        images.append(result[ind][0])
        y_class.append(result[ind][1])

    images = np.array(images)
    y_class = np.array(y_class)

    return images, y_class


def get_image_gt_parallel(groundtruthfiles, im_path,finetune):
    """

    :param groundtruthfiles:
    :param im_path:
    :return:
    """
    cnt=0
    dirnames = [dirname for dirname in os.listdir(groundtruthfiles)]
    print(dirnames)
    image_list = []
    out_class_list = []
    out_reg_list = []

    for dir_name in dirnames:
        # if not dir_name=='kam40_soncevo':
        #     continue
        if 'Kineski' in dir_name or 'tmp' in dir_name:
            continue
        if finetune:
            if 'DGood' in dir_name or 'M-30' in dir_name or 'seq' in dir_name or 'sunny' in dir_name or 'lp' in dir_name or 'dir' in dir_name or 'image_2' in dir_name:
                continue


        cnt+=1
        print(cnt)
        dir_path_im = os.path.join(im_path, dir_name)

        filenames = [x for x in os.listdir(os.path.join(groundtruthfiles,dir_name))]

        result = Parallel(n_jobs=-1, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing") \
            (delayed(load_image_and_gt)(filenames[i], dir_path_im, os.path.join(groundtruthfiles, dir_name))
             for i in tqdm(range(len(filenames)), desc='Loading images and ground truth data...'))

        image_list_1, out_class_list_1, out_reg_list_1 = parse_anchor_data_results_lists(result)

        image_list = image_list + image_list_1
        out_class_list = out_class_list + out_class_list_1
        out_reg_list = out_reg_list + out_reg_list_1

    image_list = np.array(image_list)
    out_class_list = np.array(out_class_list)
    out_reg_list = np.array(out_reg_list)

    return image_list, out_class_list, out_reg_list




def get_image_gt_parallel_parts(GT_root, dirnames_path, im_path_root,finetune):
    """

    :param groundtruthfiles:
    :param im_path:
    :return:
    """
    cnt=0
    # dirnames = [dirname for dirname in os.listdir(groundtruthfiles)]
    # print(dirnames)
    image_list = []
    out_class_list = []
    out_reg_list = []

    # for path_file in dirnames_path:
    #     # if not dir_name=='kam40_soncevo':
    #     #     continue
    #     dir_name=path_file.split(r"'\'")
    #     print(dir_name[0])
    #     if 'Kineski' in dir_name[0]:
    #         continue
    #     if finetune:
    #         if 'DGood' in dir_name[0] or 'M-30' in dir_name[0] or 'seq' in dir_name[0] or 'sunny' in dir_name[0] or 'lp' in dir_name[0] or 'dir' in dir_name[0] or 'image_2' in dir_name[0]:
    #             continue


        # cnt+=1
        # print(cnt)
        # dir_path_im = os.path.join(im_path, dir_name)
        # filenames_all.extend(os.listdir(os.path.join(groundtruthfiles,dir_name)))


    # filenames = [x for x in os.listdir(os.path.join(groundtruthfiles,dir_name))]
    if finetune:
        dirnames_path[:]=[x for x in dirnames_path if not 'Kineski' in x.split('\\')[0] or not 'DGood' in x.split('\\')[0] ]
    # for i in range (len(dirnames_path)):
        # print(dirnames_path[i].split('\\')[0])
        # novo=dirnames_path[i].split(r'\\')
        # print(dirnames_path[i].split(r'\\'))
        # print(os.path.join(im_path_root,dirnames_path[i].split(r'"\"')[0]))
        # print(os.path.join(GT_root, dirnames_path.split(r'"\"')[0]))
    result = Parallel(n_jobs=-1, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing") \
        (delayed(load_image_and_gt)(dirnames_path[i].split('\\')[1], os.path.join(im_path_root,dirnames_path[i].split('\\')[0]), os.path.join(GT_root, dirnames_path[i].split('\\')[0]))
         for i in tqdm(range(len(dirnames_path)), desc='Loading images and ground truth data...'))

    image_list, out_class_list, out_reg_list = parse_anchor_data_results_lists(result)

    # image_list = image_list + image_list_1
    # out_class_list = out_class_list + out_class_list_1
    # out_reg_list = out_reg_list + out_reg_list_1

    image_list = np.array(image_list)
    out_class_list = np.array(out_class_list)
    out_reg_list = np.array(out_reg_list)

    return image_list, out_class_list, out_reg_list
def get_image_gt_parallel_parts_renamed(GT_root, filenames, im_path_root,finetune):
    """

    :param groundtruthfiles:
    :param im_path:
    :return:
    """
    cnt=0
    # dirnames = [dirname for dirname in os.listdir(groundtruthfiles)]
    # print(dirnames)
    image_list = []
    out_class_list = []
    out_reg_list = []

    # for path_file in dirnames_path:
    #     # if not dir_name=='kam40_soncevo':
    #     #     continue
    #     dir_name=path_file.split(r"'\'")
    #     print(dir_name[0])
    #     if 'Kineski' in dir_name[0]:
    #         continue
    #     if finetune:
    #         if 'DGood' in dir_name[0] or 'M-30' in dir_name[0] or 'seq' in dir_name[0] or 'sunny' in dir_name[0] or 'lp' in dir_name[0] or 'dir' in dir_name[0] or 'image_2' in dir_name[0]:
    #             continue


        # cnt+=1
        # print(cnt)
        # dir_path_im = os.path.join(im_path, dir_name)
        # filenames_all.extend(os.listdir(os.path.join(groundtruthfiles,dir_name)))


    # filenames = [x for x in os.listdir(os.path.join(groundtruthfiles,dir_name))]
    # if finetune:
    #     filenames[:]=[x for x in dirnames_path if not 'Kineski' in x.split('\\')[0] or not 'DGood' in x.split('\\')[0] ]
    # for i in range (len(dirnames_path)):
        # print(dirnames_path[i].split('\\')[0])
        # novo=dirnames_path[i].split(r'\\')
        # print(dirnames_path[i].split(r'\\'))
        # print(os.path.join(im_path_root,dirnames_path[i].split(r'"\"')[0]))
        # print(os.path.join(GT_root, dirnames_path.split(r'"\"')[0]))
    result = Parallel(n_jobs=-1, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing") \
        (delayed(load_image_and_gt_parts_wavelet)(filenames[i], im_path_root, GT_root)
         for i in tqdm(range(len(filenames)), desc='Loading images and ground truth data...'))

    image_list, out_class_list, out_reg_list = parse_anchor_data_results_lists(result)

    # image_list = image_list + image_list_1
    # out_class_list = out_class_list + out_class_list_1
    # out_reg_list = out_reg_list + out_reg_list_1

    image_list = np.array(image_list)
    out_class_list = np.array(out_class_list)
    out_reg_list = np.array(out_reg_list)

    return image_list, out_class_list, out_reg_list




def get_image_gt_parallel_centerness(groundtruthfiles, im_path,finetune):
    """

    :param groundtruthfiles:
    :param im_path:
    :return:
    """
    cnt=0
    dirnames = [dirname for dirname in os.listdir(groundtruthfiles)]
    print(dirnames)
    image_list = []
    out_class_list = []
    out_centerness_list=[]
    out_reg_list = []

    for dir_name in dirnames:
        # if not dir_name=='kam44_soncevo':
        #     continue

        if 'Kineski' in dir_name or 'tmp' in dir_name:
            continue
        if finetune:
            if 'DGood' in dir_name or 'M-30' in dir_name:
                continue


        cnt+=1
        print(cnt)
        dir_path_im = os.path.join(im_path, dir_name)

        filenames = [x for x in os.listdir(os.path.join(groundtruthfiles,dir_name))]

        result = Parallel(n_jobs=-1, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing") \
            (delayed(load_image_and_gt_centerness)(filenames[i], dir_path_im, os.path.join(groundtruthfiles, dir_name))
             for i in tqdm(range(len(filenames)), desc='Loading images and ground truth data...'))

        image_list_1, out_class_list_1, out_centerness_list_1, out_reg_list_1 = parse_anchor_data_results_lists_centerness(result)

        image_list = image_list + image_list_1
        out_class_list = out_class_list + out_class_list_1
        out_centerness_list=out_centerness_list+out_centerness_list_1
        out_reg_list = out_reg_list + out_reg_list_1

    image_list = np.array(image_list)
    out_class_list = np.array(out_class_list)
    out_centerness_list=np.array(out_centerness_list)
    out_reg_list = np.array(out_reg_list)

    return image_list, out_class_list,out_centerness_list, out_reg_list


def get_image_gt_cls(groundtruthfiles, im_path):
    """

    :param groundtruthfiles:
    :param im_path:
    :return:
    """

    out_class_list = []
    image_list = []

    filenames = [x for x in os.listdir(groundtruthfiles)]

    for filename in tqdm(filenames):

        im_filename = filename[:-4] + '.jpg'

        fid1 = open(os.path.join(groundtruthfiles, filename), 'rb')
        out_class_dims = pickle.load(fid1)
        out_class_back = pickle.load(fid1)
        fid1.close()
        out_class = np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))

        out_class_list.append(out_class)
        image = cv2.imread(os.path.join(im_path, im_filename), 0)
        image = image.reshape(image.shape[0], image.shape[1], 1)
        image_list.append(image)
    out_class_list = np.array(out_class_list)
    image_list = np.array(image_list)
    return image_list, out_class_list


def get_image_gt_cls_parallel(groundtruthfiles, im_path):
    """

    :param groundtruthfiles:
    :param im_path:
    :return:
    """

    filenames = [x for x in os.listdir(groundtruthfiles)]

    result = Parallel(n_jobs=-1, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing") \
        (delayed(load_image_and_gt_cls_parallel)(filenames[i], im_path, groundtruthfiles)
         for i in tqdm(range(len(filenames)), desc='Loading images and ground truth data...'))

    image_list, out_class_list = parse_anchor_data_results_cls(result)

    return image_list, out_class_list


def read_data_ssd(im_path, im_size, im_depth):
    """
    load, resize, and normalize image data
    loads images and annotations from one folder
    :param im_path: global path of folder containing images of a data subset [string]
    :param im_size: output dimensions of the images (cols, rows) [tuple]
    :param im_depth: required depth of the loaded images (value: 1 or 3) [int]
    :param shuffle: whether to shuffle input data order [bool]
    :return: images_list - array of normalized depth maps [ndarray]
             object_annotations_list - annotated bounding boxes min_row, min_col, max_row, max_col [list]
    """
    images_list = []

    for im_name1 in (os.listdir(im_path)):
        im_name = im_name1
        # flag1 = 0
        # print(im_name)
        # cv2.waitKey(0)
        # --- load image ---
        # if not im_name[-4:] != '.bmp':  # exclude system files
        #     continue
        if (im_depth == 1):
            image = cv2.imread(os.path.join(im_path, im_name), 0)

        images_list.append(image)
    images_list = np.array(images_list)

    if len(images_list) == 0:
        print("No images were read.")
        exit(100)
    return images_list


def read_data_rpn(gt_path, im_path, im_size, im_depth, annot_path, exclude_empty, shuffle):
    """
    load, resize, and normalize image data
    loads images and annotations from one folder
    :param im_path: global path of folder containing images of a data subset [string]
    :param im_size: output dimensions of the images (cols, rows) [tuple]
    :param im_depth: required depth of the loaded images (value: 1 or 3) [int]
    :param shuffle: whether to shuffle input data order [bool]
    :return: images_list - array of normalized depth maps [ndarray]
             object_annotations_list - annotated bounding boxes min_row, min_col, max_row, max_col [list]
    """
    # im_path_resized=r'E:\Science\Monika\M-30HD_resized'
    images_list = []  # array of normalized images
    object_annotations_list = []  # array of array of bounding boxes for each image
    # WScale=800/1200
    # HScale=480/720
    # list images in source folder

    for im_name1 in (os.listdir(gt_path)):
        im_name = str(im_name1[0:-4]) + '.jpg'
        # flag1 = 0
        # print(im_name)
        # cv2.waitKey(0)
        # --- load image ---
        if not im_name[-4:] != '.bmp':  # exclude system files
            continue

        if im_depth == 3:
            image = cv2.imread(os.path.join(im_path, im_name))
        else:
            image = cv2.imread(os.path.join(im_path, im_name), 0)
        rows, col = image.shape[:2]
        if im_size != (col, rows):
            image = cv2.resize(image, im_size, interpolation=cv2.INTER_AREA)
        # WScale = 512 / col
        # print(WScale)
        # HScale = 341 / rows
        image = image.reshape(image.shape[0], image.shape[1], im_depth)

        # --- load annotations ---
        annot_name = str(int(im_name[5:11].lstrip('0')) - 1) + '.xml'
        # print(annot_name)

        root = ET.parse(os.path.join(annot_path, annot_name)).getroot()

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

            # print(1)
            annot = [int(np.round(bb[2])), int(np.round(bb[0])), int(np.round(bb[3])), int(np.round(bb[1]))]
            # annot_n=[bb[2], bb[0], bb[3], bb[1]]
            # min_row, min_col, max_row, max_col
            # select positive car samples, height > 25px
            if cl == 'car' and annot[2] - annot[0] + 1 > 20:
                # print("zemen")
                objects.append(annot)
            # else:
            #     annot=[bb[2], bb[0], bb[3], bb[1]]
            #     if cl == 'car' and bb[3] - bb[2] + 1 > 20:
            # if flag1==1:
            #     annot[0]=int(np.round(annot[0]*WScale))
            #     annot[1]=int(np.round(annot[1]*HScale))
            #     annot[2] =int( np.round(annot[2] * WScale))
            #     annot[3] =int(np.round(annot[3] * HScale))
            # cv2.rectangle(imgs[img_ind], (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0, 0, 0), thickness=1)

            # cv2.imshow("slika", imgs[img_ind])
            # cv2.waitKey(0)

            # objects.append(annot)
            # cv2.rectangle(image, (annot[1], annot[0]), (annot[3], annot[2]), color=(0, 0, 0), thickness=1)
        # cv2.rectangle(image, (annot_n[1], annot_n[0]), (annot_n[3], annot_n[2]), color=(255, 0, 0), thickness=1)
        #  cv2.imshow("slika", image)
        #  cv2.waitKey(0)

        # if flag1==1:
        #     for i, anotacija in enumerate(objects):
        #
        if exclude_empty:
            if len(objects) > 0:
                images_list.append(image)
                object_annotations_list.append(objects)

        else:
            images_list.append(image)
            object_annotations_list.append(objects)

    if len(images_list) == 0:
        print("No images were read.")
        exit(100)

    # if shuffle:
    #     data = list(zip(images_list, object_annotations_list))
    #     random.shuffle(data)
    #     images_list, object_annotations_list = zip(*data)

    images_list = np.array(images_list).astype(np.uint8)

    return images_list, object_annotations_list


def generate_anchor_level_object_masks(bboxes, img_dims, anchor_stride):
    """
    generate masks of bounding boxes in the output-level matrices
    :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]  NOTE: per image !!!!!!
    :param img_dims: (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :return: object_masks - anchor-level masks of object locations [ndarray]
    """

    object_masks_dims = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride))
    object_masks = np.zeros(object_masks_dims).astype(np.int)

    # iterate over ground truth files for each image

    for bbox in bboxes:  # iterate over annotated bounding boxes row1 col1 row2 col2

        # calculate bbox coordinates in output matrix
        # NOTE: floor & ceil?
        bbox_out = [np.int(np.round(bbox[0] / anchor_stride)),  # min_row
                    np.int(np.round(bbox[1] / anchor_stride)),  # min_col
                    np.int(np.round(bbox[2] / anchor_stride)),  # max_row
                    np.int(np.round(bbox[3] / anchor_stride))]  # max_col

        # fill object mask
        object_masks[bbox_out[0]:bbox_out[2], bbox_out[1]:bbox_out[3]] = 1

    return object_masks


def get_anchorless_ground_truth_data_parallel_optimized(bboxes, obj_mask, img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio):
    """
    generate ground truth output for fully convolutional network for object detection
    multi-output, classifier and regressor branch
    :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]
    :param obj_mask: mask of possible object locations (bounding box centers which lie within an object bounding box) [ndarray]
    :param img_dims: dimensions of input images (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param iou_low: samples with lower iou with all objects are declared negative (range: 0 to 1) [float]
    :param iou_high: samples with higher iou with an object are declared negative (range: 0 to 1) [float]
    :param num_negs_ratio: select negative samples num_negs_ratio times more than positive samples [int]
    :return: output_cls_arr - ground truth classes [ndarray]
             output_cls_arr - ground truth regression (normalized to [-1, 1]) [ndarray]
             valid_inds - indices of images containing at least one object [list]
             reg_norm_coef - normalization coefficient for ground truth regression data [float]
    """

    output_dims_class = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), 2)
    output_dims_reg = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), 4)

    output_class = np.zeros(output_dims_class).astype(np.int)
    output_reg = np.zeros(output_dims_reg).astype(np.float64)

    # first position of an anchor center
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    [r, c] = np.where(obj_mask == 1)  # possible object locations

    # ctr = 0
    # for output_row, center_row in enumerate(range(start_r, img_dims[0], anchor_stride)):  # iterate through rows of anchor centers
    #     for output_col, center_col in enumerate(range(start_c, img_dims[1], anchor_stride)):  # iterate through columns of anchor centers
    #         ctr += 1
    #
    # print(f'Iterations: {len(r) ** 2} {ctr}')

    for object_num in range(len(r)):

        output_row = r[object_num]
        output_col = c[object_num]

        for bbox in bboxes:  # iterate through annotated bounding boxes

            # --- assign classes: calculate IOU, place 1 or 0 at the required position ---
            bbox_h = bbox[2] - bbox[0]
            bbox_w = bbox[3] - bbox[1]

            if bbox_h <= 1 or bbox_w <= 1:
                continue

            half_bbox_dim_h = np.int(np.round(bbox_h / 2))
            half_bbox_dim_w = np.int(np.round(bbox_w / 2))

            center_row = output_row * anchor_stride + start_r
            center_col = output_col * anchor_stride + start_c

            anchor = [max(0, center_row - half_bbox_dim_h),
                      max(0, center_col - half_bbox_dim_w),
                      min(center_row + half_bbox_dim_h, img_dims[0]),
                      min(center_col + half_bbox_dim_w, img_dims[1])]
            # min_row, min_col, max_row, max_col

            iou = helper_postprocessing.calc_iou(bbox, anchor)

            if iou >= iou_high:
                # positive sample: set class, calculate deltas
                output_class[output_row, output_col, 0] = 1

                # set deltas - current location minus correct location
                delta_r = bbox[0] - anchor[0]
                delta_c = bbox[1] - anchor[1]

                h_percent = bbox_h / img_dims[0]
                w_percent = bbox_w / img_dims[1]

                output_reg[output_row, output_col, 0] = delta_r
                output_reg[output_row, output_col, 1] = delta_c
                output_reg[output_row, output_col, 2] = h_percent
                output_reg[output_row, output_col, 3] = w_percent

            if (iou < iou_high) and (iou > iou_low):
                # IOU between iou_min and iou_max
                output_class[output_row, output_col, 0] = 2  # temporarily mark class with 2

                # set deltas
                # current location minus correct location
                delta_r = bbox[0] - anchor[0]
                delta_c = bbox[1] - anchor[1]

                h_percent = bbox_h / img_dims[0]
                w_percent = bbox_w / img_dims[1]

                output_reg[output_row, output_col, 0] = delta_r
                output_reg[output_row, output_col, 1] = delta_c
                output_reg[output_row, output_col, 2] = h_percent
                output_reg[output_row, output_col, 3] = w_percent

    # mark negative samples
    for out_row in range(output_class.shape[0]):  # iterate through rows of output
        for out_col in range(output_class.shape[1]):  # iterate through columns of output

            if sum(output_class[out_row, out_col, :]) == 0:  # if no anchors at the specified center is marked with 1 (positive) or 2 (in-between)
                output_class[out_row, out_col, 1] = 1

    # replace 2s with 0s
    output_class = np.where(output_class == 2, 0, output_class)

    # --- select negative samples ---
    # count positives and negatives
    num_positives = np.sum(output_class[:, :, 0])

    # find negatives
    negs = output_class[:, :, 1]
    [r, c] = np.where(negs == 1)

    # select negatives to remove
    ind_to_remove = np.arange(len(r))
    np.random.shuffle(ind_to_remove)

    num_neg = min(len(r), num_positives * num_negs_ratio)  # number of positive to negative samples ratio: 1 to 10
    num_to_remove = len(r) - num_neg
    ind_to_remove = ind_to_remove[:num_to_remove]

    # remove negatives that were not selected
    for ind in ind_to_remove:
        output_class[r[ind], c[ind], :] = 0

    for ind in ind_to_remove:
        output_reg[r[ind], c[ind], :] = 0

    if num_positives > 0:
        # out_class=copy.deepcopy(output_class)
        # out_c_rows, out_c_cols, depth = out_class.shape
        # out_class_dims = [out_c_rows, out_c_cols, depth]
        # out_class_flat = np.ndarray.flatten(out_class)
        # filename = str(im_name) + '_' + str(iou_low) + '_' + str(iou_high) + '_' + str(anchor_stride) + '.txt'
        # fid = open(os.path.join(GroundTruthFiles, filename), 'wb+')
        # # pickle.dump(out_class_dims,fid)
        # pickle.dump(out_class_flat, fid)
        # fid.close()
        #
        # im_name_b = int(im_name[5:11])
        # im_name_b_1 = im_name_b + 16910
        # im_name_new = 'image' + str(im_name_b_1).zfill(6) + '.jpg'
        # filename = str(im_name_new) + '_' + '.txt'
        # fid_b = open(os.path.join(GroundTruthFiles, filename), 'wb+')
        # # pickle.dump(out_class_dims, fid_b)
        # pickle.dump(out_class_flat, fid_b)
        # fid_b.close()
        #
        # im_name_f = im_name_b_1 + 16910
        # im_name_new_f = 'image' + str(im_name_f).zfill(6) + '.jpg'
        # filename = str(im_name_new_f) + '_' + '.txt'
        # fid_f = open(os.path.join(GroundTruthFiles, filename), 'wb+')
        # out_class_f = np.fliplr(out_class)
        # # pickle.dump(out_class_dims, fid_f)
        # pickle.dump(out_class_f, fid_f)
        # fid_f.close()
        #
        # im_name_f1 = im_name_f + 16910
        # im_name_new_f1 = 'image' + str(im_name_f1).zfill(6) + '.jpg'
        # filename = str(im_name_new_f1) + '_' + '.txt'
        # fid_f1 = open(os.path.join(GroundTruthFiles, filename), 'wb+')
        # # out_class_f = np.fliplr(out_class)
        # # pickle.dump(out_class_dims, fid_f1)
        # pickle.dump(out_class_f, fid_f1)
        # fid_f1.close()
        return output_class, output_reg
    else:
        return None, None
