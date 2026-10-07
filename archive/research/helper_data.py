"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 09.08.2021

Description: function library
             data operations: load, save, process
Python version: 3.6
"""

# python imports
#from tqdm import tqdm
from collections import Counter
import os
import numpy as np
import cv2
import random
#from tqdm import tqdm

import xml.etree.ElementTree as ET

# custom imports
import sys
sys.path.insert(0, 'E:\Science\Monika\SingleShotDetector\Helpers' )
import helper_postprocessing
import zipfile
# from tqdm import tqdm

def read_zippedData_ssd(dataset, im_size, scalefactor, im_depth, annot_path, exclude_empty, shuffle):
    """
    load and normalize image data
    load annotations in XML format (GRAM dataset)
    :param im_path: global path of folder containing images of a data subset [string]
    :param im_size: output dimensions of the images (cols, rows) [tuple]
    :param im_depth: required depth of the loaded images (value: 1 or 3) [int]
    :param annot_path: global path of XM files containing ground truth annotations [string]
    :param exclude_empty: flag marking whether to include images that don't contain objects in the output, or ignore them [bool]
    :param shuffle: whether to shuffle input data order [bool]
    :return: images_list - array of normalized depth maps [ndarray]
             object_annotations_list - annotated bounding boxes min_row, min_col, max_row, max_col [list]
    """

    images_list = []  # array of normalized images
    object_annotations_list = []  # array of array of bounding boxes for each image
    object_annotations_namelist = []
    objects = []
    # list images in source folder
    # for im_name in tqdm(os.listdir(os.path.join(im_path))):
    im_dirnames = os.listdir(r"D:\Tatjana\NewBeginning\Dataset\ReadyDataset")
    # print("im_dirnames",im_dirnames)
    # scalefactor=0.2
    cnt = 0
    for dirname in im_dirnames:
        # print(dirname)
        if "Data" in dirname:
            # print("Data")
            im_path = r"D:\Tatjana\NewBeginning\Dataset\\ReadyDataset\Data\odbrani"
            annot_path = r'D:\Tatjana\NewBeginning\Dataset\ReadyDataset\Data\anotirani'
            dirpath = r"D:\Tatjana\NewBeginning\Dataset\ReadyDataset"
            for datadir in os.listdir(
                    os.path.join(r"D:\Tatjana\NewBeginning\Dataset\ReadyDataset", dirname)):
                if 'odbrani' in datadir:
                    # print(datadir)
                    for imgfile in os.listdir(os.path.join(dirpath, dirname, datadir, dataset)):

                        imgpath = os.path.join(dirpath, dirname, datadir, dataset, imgfile)
                        # print(imgpath)
                        if im_depth == 3:
                            img = cv2.imread(imgpath)
                        else:
                            img = cv2.imread(imgpath, 0)
                        if img is None:
                            continue
                        # cv2.imshow("slika", img)
                        # cv2.waitKey(0)
                        img = np.asarray(img)
                        tmpfile = imgfile[0:11] + "_"
                        # #print(tmpfile)
                        # #print(dataset)
                        objects_list = []
                        for txtfile in os.listdir(os.path.join(dirpath, dirname, 'anotirani', dataset)):
                            if tmpfile in txtfile:
                                object = np.loadtxt(os.path.join(dirpath, dirname, 'anotirani', dataset, txtfile), delimiter=",")
                                object = [int(el) for el in object]

                                # #print(object)
                                objects_list.append(object)
                        if len(objects_list) > 0:
                            # for obj in objects_list:
                            #     cv2.rectangle(img, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=2)
                            object_annotations_list.append(objects_list)
                            images_list.append(img)
                        # cv2.imshow("slika", img)
                        # cv2.waitKey(0)

            # #print("here",object_annotations_list)

        if "Pat" in dirname:
            scalerow = 0.57
            scalecol = 0.82
        else:
            scalerow = scalefactor
            scalecol = scalefactor

        if dataset in dirname:
            # print(dirname)
            zippedImgs = zipfile.ZipFile(r"D:\Tatjana\NewBeginning\Dataset\ReadyDataset/" + dirname)
            temp = zippedImgs.namelist()[2][:-4]
            # #print("temp",temp)
            for i in range(len(zippedImgs.namelist())):
                file_in_zip = zippedImgs.namelist()[i]
                # #print(file_in_zip)
                # --- load image ---

                if "odbrani" in file_in_zip:
                    if not '.jpg' in file_in_zip:
                        if not '.bmp' in file_in_zip:  # exclude system files and other file types in the folder
                            continue

                    if im_depth == 3:
                        # image = cv2.imread(os.path.join(im_path, im_name))
                        # #print("Found image: ", file_in_zip, " -- ")
                        data = zippedImgs.read(file_in_zip)
                        # #print(type(data))
                        nparr = np.fromstring(data, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        # cv2.imshow("image",image)
                        # cv2.waitKey(0)

                    else:
                        # image = cv2.imread(os.path.join(im_path, im_name))
                        # #print("Found image: ", file_in_zip, " -- ")
                        data = zippedImgs.read(file_in_zip)
                        # #print(type(data))
                        nparr = np.fromstring(data, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

                    # image=cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    if im_size != (image.shape[1], image.shape[0]):  # resize image if needed
                        # #print(np.asarray(image).shape)
                        image = cv2.resize(image, im_size, interpolation=cv2.INTER_AREA)
                        # cv2.imshow("resized",image)
                        # cv2.waitKey(0)
                        # #print(np.asarray(image).shape)

                    # if "Ses" in dirname:
                    cv2.imshow("image", image)
                    cv2.waitKey(0)
                    # print(np.asarray(image).shape)
                    # print(object_annotations_list[cnt])
                    for obj in object_annotations_list[cnt]:
                        cv2.rectangle(image, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=2)
                    cv2.imshow("image2", image)
                    cv2.waitKey(0)
                    cnt += 1
                    # cv2.imwrite(r"C:\Users\tatja\Desktop\img.bmp",image)
                    # #print(np.asarray(image).shape)
                    images_list.append(image)
                else:

                    if ".txt" in file_in_zip:
                        # #print(object_annotations_list)
                        objects.clear()
                        for txtfile in zippedImgs.namelist():
                            if "Feit" in dirname:
                                if ".txt" in txtfile:
                                    data = zippedImgs.read(txtfile)
                                    # #print(data)
                                    # strdata=data.split(sep=",")
                                    strdata = data[:-1].decode("utf-8")
                                    # #print(strdata,type(strdata))
                                    coord = strdata.split(",")

                                    coordint = []
                                    for c in coord:
                                        coordint.append(int(c))

                                    # #print(type(coordint))

                                    # coordint = scalefactor * np.asarray(coordint)
                                    coordint = np.asarray(coordint)
                                    coordint[0] *= scalerow
                                    coordint[1] *= scalecol
                                    coordint[2] *= scalerow
                                    coordint[3] *= scalecol
                                    # coordint[2] =20*round(coordint[2]/20)
                                    # coordint[3] = 20 * round(coordint[3] / 20)
                                    coordint = [int(el) for el in coordint]

                                    if "Ses" not in dirname:
                                        coordint[2] += coordint[0]

                                        coordint[3] += coordint[1]
                                        # #print("check if same:",file_in_zip,"\t",temp)
                                    finalcoordint = [coordint[0], coordint[2], coordint[1], coordint[3]]
                                    objects.append(coordint)
                                    object_annotations_namelist.append(txtfile)

                            elif file_in_zip[:-4] in txtfile and ".txt" in txtfile and txtfile not in object_annotations_namelist:
                                # #print(file_in_zip,txtfile)
                                # #print(object_annotations_namelist)
                                if "Pat" in dirname and not file_in_zip[:-3] in txtfile:
                                    continue

                                # #print(file_in_zip,txtfile)
                                data = zippedImgs.read(txtfile)
                                # #print(data)
                                # strdata=data.split(sep=",")
                                if "Ses" not in dirname:
                                    strdata = data[:-1].decode("utf-8")
                                else:
                                    strdata = data.decode("utf-8")

                                # #print(strdata,type(strdata))
                                coord = strdata.split(",")

                                coordint = []
                                # #print(coord,len(coord))
                                if (len(coord) == 1) and coord[0] == '':
                                    object_annotations_list.append(coordint)
                                    continue

                                for c in coord:
                                    coordint.append(int(c))

                                # #print(type(coordint))
                                if "Pat" in dirname:
                                    coordint = np.asarray(coordint)
                                    coordint[0] *= scalerow
                                    coordint[1] *= scalecol
                                    coordint[2] *= scalerow
                                    coordint[3] *= scalecol
                                    # coordint[2] = round(coordint[2] / 20)
                                    # coordint[3] = 20 * round(coordint[3] / 20)
                                elif "Ses" in dirname:
                                    # #print(dirname)
                                    # #print(coordint)
                                    # #print(strdata)
                                    coordint[0] *= 0.75
                                    coordint[2] *= 0.75
                                    coordint[1] *= 0.75
                                    coordint[3] *= 0.75
                                    temp = coordint.copy()
                                    coordint = [temp[1], temp[0], temp[3], temp[2]]
                                else:

                                    coordint = scalefactor * np.asarray(coordint)
                                    # coordint[2] = 20 * round(coordint[2] / 20)
                                    # coordint[3] = 20 * round(coordint[3] / 20)
                                coordint = [int(el) for el in coordint]
                                # #print(coordint)

                                if "Ses" not in dirname:
                                    coordint[2] += coordint[0]

                                    coordint[3] += coordint[1]
                                # #print("check if same:",file_in_zip,"\t",temp)
                                # if "D2" in dirname:
                                # finalcoordint = coordint
                                # else:
                                #     finalcoordint=[coordint[1], coordint[0], coordint[2], coordint[3]]
                                objects.append(coordint)
                                object_annotations_namelist.append(txtfile)


                            elif file_in_zip[:-7] in txtfile and ".txt" in txtfile and txtfile not in object_annotations_namelist and "Ses" in dirname:
                                # #print(file_in_zip,txtfile)
                                # #print(object_annotations_namelist)
                                if "Pat" in dirname and not file_in_zip[:-3] in txtfile:
                                    continue

                                # #print(file_in_zip,txtfile)
                                data = zippedImgs.read(txtfile)
                                # #print(data)
                                # strdata=data.split(sep=",")
                                if "Ses" not in dirname:
                                    strdata = data[:-1].decode("utf-8")
                                else:
                                    strdata = data.decode("utf-8")

                                # #print(strdata,type(strdata))
                                coord = strdata.split(",")

                                coordint = []
                                # #print(coord,len(coord))
                                if (len(coord) == 1) and coord[0] == '':
                                    object_annotations_list.append(coordint)
                                    continue

                                for c in coord:
                                    coordint.append(int(c))

                                # #print(type(coordint))
                                if "Pat" in dirname:
                                    coordint = np.asarray(coordint)
                                    coordint[0] *= scalerow
                                    coordint[1] *= scalecol
                                    coordint[2] *= scalerow
                                    coordint[3] *= scalecol
                                    # coordint[2] = round(coordint[2] / 20)
                                    # coordint[3] = 20 * round(coordint[3] / 20)
                                elif "Ses" in dirname:
                                    # #print(dirname)
                                    # #print(coordint)
                                    # #print(strdata)
                                    coordint[0] *= 0.75
                                    coordint[2] *= 0.75
                                    coordint[1] *= 0.75
                                    coordint[3] *= 0.75
                                    temp = coordint.copy()
                                    coordint = [temp[1], temp[0], temp[3], temp[2]]
                                else:

                                    coordint = scalefactor * np.asarray(coordint)
                                    # coordint[2] = 20 * round(coordint[2] / 20)
                                    # coordint[3] = 20 * round(coordint[3] / 20)
                                coordint = [int(el) for el in coordint]
                                # #print(coordint)

                                if "Ses" not in dirname:
                                    coordint[2] += coordint[0]

                                    coordint[3] += coordint[1]
                                # #print("check if same:",file_in_zip,"\t",temp)
                                # if "D2" in dirname:
                                # finalcoordint = coordint
                                # else:
                                #     finalcoordint=[coordint[1], coordint[0], coordint[2], coordint[3]]
                                objects.append(coordint)
                                object_annotations_namelist.append(txtfile)
                        # #print(file_in_zip)
                        # #print("len of objects",len(objects))
                        # #print(file_in_zip)
                        # #print("len of objects",len(objects))
                        if len(objects) >= 1:
                            # #print(objects)
                            # #print("inside:", object_annotations_list)
                            object_annotations_list.append(objects.copy())
                            # #print("after:", object_annotations_list)
                        ##print(objects)

        ##print("namelist",object_annotations_namelist)
    if len(images_list) == 0:
        # print("No images were read.")
        exit(100)

    if shuffle:
        data = list(zip(images_list, object_annotations_list))  # shuffle data and annotations together
        random.shuffle(data)
        images_list, object_annotations_list = zip(*data)

    images_list = np.array(images_list).astype(np.uint8)

    # images_list = np.array(images_list)

    return images_list, object_annotations_list

from copy import deepcopy
def read_data_test(im_path, im_size, im_depth,dir_name, flag_same_dims):
    """
    load, resize, and normalize image data
    loads images and annotations from one folder
    :param im_path: global path of folder containing images of a data subset [string]
    :param im_size: output dimensions of the images (cols, rows) [tuple]
    :param im_depth: required depth of the loaded images (value: 1 or 3) [int]
    :param shuffle: whether to shuffle input data order [bool]
    dir_name: string, name of directory
    :return: images_list - array of normalized depth maps [ndarray]
             object_annotations_list - annotated bounding boxes min_row, min_col, max_row, max_col [list]
    """
    images_list=[]
    images_list_color=[]
    crop_left_list=[]
    crop_width_list=[]
    full_hd_list=[]
    # dirnames=os.listdir(im_root)
    # print(dirnames)

    images_names=[]
    # for dir_name in dirnames:
    # # br=0
    # if dir_name[-3:]=='.db' or dir_name=='drugo_renamed':
    #     continue
    # print(dir_name)
    # im_path = os.path.join(im_root, dir_name)
    # print (im_path)
    filenames=[x for x in os.listdir(im_path) if x.endswith('.bmp')]
    # filenames=[(x).zfill(5) for x in filenames1]
    # print(filenames)
    filenames=sorted(filenames)
    # first=cv2.imread(os.path.join(im_path,filenames[0]),0)
    # rows, col = first.shape[:2]
    # cropped_image1 = first[120:rows, 0:1500]
    # cv2.imshow("sli",cropped_image1)
    # cv2.waitKey(0)
    # print(filenames)
    # exit(1)
    # print(filenames)

    for im_name in filenames:

        # image = cv2.imread(os.path.join(im_path, im_name),0)
        # rows, col = image.shape[:2]
        # cropped_image2 = image[120:rows, 0:1500]


        # image_check_empty=cv2.subtract(cropped_image1,cropped_image2)
        # print(np.max(image_check_empty))
        # [r,c]=np.where(image_check_empty>80)
        # print(len(r))
        # cv2.imshow("odzemena", image_check_empty)
        # # br+=1
        # # br1=br
        # if(br>10000):
        #     break
        # if(br>1000):
        #     break
        # flag=0
        # if im_name[-4:] != '.bmp' or im_name[-4:] != '.jpg':
        #     continue
        if im_name[-3:]=='.db':
            continue


        image = cv2.imread(os.path.join(im_path, im_name))
        full_hd_list.append(image.copy())
        # image_color = cv2.imread(os.path.join(im_path, im_name))
        if image is None:
            print(os.path.join(im_path, im_name))
            continue
        rows, cols = image.shape[:2]

        # if flag_same_dims:
        #
        #     # image = image.reshape(image.shape[0], image.shape[1], 1)
        #
        #     # images_list.append(image)
        #     # images_names.append(im_name)
        #     resized=cv2.resize(image, (cols, rows), interpolation=cv2.INTER_AREA)
        # else:
        # if 'videoframes' in dir_name:
        #     # print('videoframes')
        #
        #     cropped_image = image[0:rows, 0:1141]
        #     # resulting_image = cv2.filter2D(src_image, -1, kernel)
        #     # cv2.imwrite(os.path.join(trainBlur, str(im_name[:-4]) + '_b' + '.jpg'), resulting_image)
        #
        # elif 'miladinovci' in dir_name:
        #     cropped_image = image[0:rows, 0:1621]
        #
        # elif 'kamera2' in dir_name or 'kam23' in dir_name:
        #     cropped_image = image[0:rows, 160:1782]
        #
        # elif 'DGood' in dir_name:
        #     cropped_image = image[0:rows, 149:cols]
        #
        # elif 'MVI' in dir_name:
        #
        #     cropped_image = image[0:rows, 149:cols]
        #
        # elif ('M-30' in dir_name) or ('drugo' in dir_name):
        #     # break
        #     # print('m30')
        #     pass
        #
        # elif 'mil_res' in dir_name:
        #    pass
        #
        # elif 'kam25' in dir_name:
        #     cropped_image=image[0:rows, 0:1621]
        #
        # elif 'kam28' in dir_name:
        #     cropped_image=image[0:rows, 0:1621]
        # elif 'kam30' in dir_name:
        #     cropped_image=image[0:rows, 0:1621]
        # elif 'kam32' in dir_name or 'kam38' in dir_name:
        #     cropped_image=image[0:rows,299:cols]
        # elif 'kam33' in dir_name:
        #     cropped_image = image[0:rows, 99:1720]
        # elif 'kam36' in dir_name:
        #     cropped_image = image[0:rows, 189:1810]
        # elif 'kam40' in dir_name:
        #     cropped_image = image[0:rows, 255:255+1620]
        # elif 'kam42' in dir_name:
        #     cropped_image = image[0:rows, 299:cols]
        # elif 'kam44' in dir_name:
        #     cropped_image = image[0:rows, 299:cols]
        # elif 'kam46' in dir_name:
        #     cropped_image = image[0:rows, 100:1721]
        # elif 'kam48' in dir_name:
        #     cropped_image = image[0:rows, 299:cols]
        # # elif 'kamera30' in dir_name:
        # #     image = cv2.imread(os.path.join(im_path, im_name))
        # #     rows, col = image.shape[:2]
        # #     cropped_image = image[140:rows, 0:1411]
        # #     # rows_cr, col_cr = cropped_image.shape[:2]
        # #     resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)
        # # elif 'kamera32' in dir_name:
        # #     image = cv2.imread(os.path.join(im_path, im_name))
        # #     rows, col = image.shape[:2]
        # #     cropped_image = image[0:rows, 260:1880]
        # #     resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)
        # elif 'proba_secena' in dir_name:
        #     cropped_image = image[0:785, 0:1178]


        if 'videoframes' in dir_name:
            cropped_image = image[:, 0:1141]
            crop_left_px = 0
            crop_width_px = 1141

        elif 'miladinovci' in dir_name:
            cropped_image = image[:, 0:1621]
            crop_left_px = 0
            crop_width_px = 1621

        elif 'kamera2' in dir_name or 'kam23' in dir_name:
            cropped_image = image[:, 160:1782]
            crop_left_px = 160
            crop_width_px = 1782 - 160

        elif 'DGood' in dir_name:
            cropped_image = image[:, 149:cols]
            crop_left_px = 149
            crop_width_px = cols - 149

        elif 'MVI' in dir_name:
            cropped_image = image[:, 149:cols]
            crop_left_px = 149
            crop_width_px = cols - 149

        elif ('M-30' in dir_name) or ('drugo' in dir_name):
            # no cropping done
            cropped_image = image
            crop_left_px = 0
            crop_width_px = cols

        elif 'mil_res' in dir_name:
            cropped_image = image
            crop_left_px = 0
            crop_width_px = cols

        elif 'kam25' in dir_name:
            cropped_image = image[:, 0:1621]
            crop_left_px = 0
            crop_width_px = 1621

        elif 'kam28' in dir_name:
            cropped_image = image[:, 0:1621]
            crop_left_px = 0
            crop_width_px = 1621

        elif 'kam30' in dir_name:
            cropped_image = image[:, 0:1621]
            crop_left_px = 0
            crop_width_px = 1621

        elif 'kam32' in dir_name or 'kam38' in dir_name:
            cropped_image = image[:, 299:cols]
            crop_left_px = 299
            crop_width_px = cols - 299

        elif 'kam33' in dir_name:
            cropped_image = image[:, 99:1720]
            crop_left_px = 99
            crop_width_px = 1720 - 99

        elif 'kam36' in dir_name:
            cropped_image = image[:, 189:1810]
            crop_left_px = 189
            crop_width_px = 1810 - 189

        elif 'kam40' in dir_name:
            cropped_image = image[:, 255:255 + 1620]
            crop_left_px = 255
            crop_width_px = 1620

        elif 'kam42' in dir_name:
            cropped_image = image[:, 299:cols]
            crop_left_px = 299
            crop_width_px = cols - 299

        elif 'kam44' in dir_name:
            cropped_image = image[:, 299:cols]
            crop_left_px = 299
            crop_width_px = cols - 299

        elif 'kam46' in dir_name:
            cropped_image = image[:, 100:1721]
            crop_left_px = 100
            crop_width_px = 1721 - 100

        elif 'kam48' in dir_name:
            cropped_image = image[:, 299:cols]
            crop_left_px = 299
            crop_width_px = cols - 299

        elif 'proba_secena' in dir_name:
            cropped_image = image[0:785, 0:1178]
            crop_left_px = 0
            crop_width_px = 1178
        elif 'cut' in dir_name:
            cropped_image=image
            crop_left_px = 0
            crop_width_px = 512
        if not flag_same_dims:
            resized_color = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)
            resized=cv2.cvtColor(resized_color,cv2.COLOR_BGR2GRAY)
        else:
            resized=deepcopy(image)
            # pass
            # resized=cv2.resize(cropped_image, (cols, rows), interpolation=cv2.INTER_AREA)

        # else:
        #     break
        # if len(objects)<0:
        #     continue
        #   &NOTE: otkoga ke se popravat kamionite da se vrati ovoj del
        # if(br1>12150*4):
        #     break

        resized=resized[96:,:]
        resized_color=resized_color[96:,:]
        images_list.append(resized)
        images_list_color.append(resized_color)

        images_names.append(im_name[:-4]+'_'+dir_name+'.bmp')
        crop_left_list.append(crop_left_px)
        crop_width_list.append(crop_width_px)

    # images_list=np.array(images_list)
    return images_list, images_list_color,images_names,crop_left_list,crop_width_list,full_hd_list



def custom_sort_key(file_name):
    parts = file_name.split("_")
    channel = parts[1]
    start_date_time = parts[3]
    end_date_time = parts[4]
    frame_index = int(parts[-1].split(".")[0])
    return (channel, start_date_time, end_date_time, frame_index)


def read_data_test_pedestrians(im_path,img_dims,flag_nvr,flag_same_dim):
    im_files_unsorted = [x for x in os.listdir(im_path)]

    if flag_nvr:
        im_files = sorted(im_files_unsorted, key=custom_sort_key)
    else:
        im_files=im_files_unsorted
    # print(im_files)
    images = []
    images_names=[]
    aspect_ratio=img_dims[0]/img_dims[1]
    print(aspect_ratio)
    for im_name in im_files:
        image = cv2.imread(os.path.join(im_path, im_name), 0)
        if image is None:
            continue
        if flag_same_dim:
            image = image.reshape(image.shape[0], image.shape[1], 1)
            images.append(image)
            images_names.append(im_name)
        # 0 for reading the image in grayscale
        else:
            new_width=int(image.shape[0]/aspect_ratio)
            print(new_width)
            if new_width>image.shape[1]:
                print('upsampling na slika-izlez')
                cropped_image=image[48:image.shape[0]-48, 0:image.shape[1]]
                # cropped_image = image[0:image.shape[0], 0:1621]
            else:
                cropped_image=image[0:image.shape[0], 0:new_width]
            print(image.shape)
            image=cv2.resize(cropped_image,(img_dims[1],img_dims[0]), interpolation=cv2.INTER_CUBIC)
            image = image.reshape(image.shape[0], image.shape[1], 1)
            images.append(image)
            images_names.append(im_name)
    images = np.array(images)
    images_names=np.array((images_names))
    return images,images_names


def read_data_test_pedestrians_fp(im_root,img_dims):

    dataset_names=os.listdir(im_root)
    images=[]
    images_paths=[]
    for dataset in dataset_names:

        if not "FP" in dataset:
            continue

        print(dataset)

        if dataset[-3:] == '.gz' or dataset[-4:] == '.rar' or dataset[-4:] == '.zip':
             continue

        dir_images_folders = os.listdir(os.path.join(im_root, dataset, 'Images'))  # folderite vo sliki od datasetot
        print(dir_images_folders)
        # if 'Kitti' in dataset or 'CVC' in dataset:
        #     continue

        for dir in dir_images_folders:

            if dir[-3:] == '.gz' or dir[-4:] == '.rar':
                continue


            im_path = os.path.join(im_root, dataset, 'Images', dir)
            images_filenames = os.listdir(im_path)
            for im_name in images_filenames:
                image = cv2.imread(os.path.join(im_path, im_name), 0)

                resized = cv2.resize(image, (img_dims[1], img_dims[0]), interpolation=cv2.INTER_AREA)
                images.append(resized)
                images_paths.append(os.path.join(im_path,im_name))

    return images,images_paths
def read_data_rpn(im_path, im_size, im_depth, exclude_empty, shuffle):
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
    #im_path_resized=r'E:\Science\Monika\M-30HD_resized'
    images_list = []       # array of normalized images
    object_annotations_list = []       # array of array of bounding boxes for each image
    # WScale=800/1200
    # HScale=480/720
    # list images in source folder

    for im_name1 in (os.listdir(im_path)):

        im_name = im_name1
        # flag1 = 0
        #print(im_name)
        #cv2.waitKey(0)
        # --- load image ---
        # if not im_name[-4:] != '.bmp':  # exclude system files
        #     continue
        image = cv2.imread(os.path.join(im_path, im_name))
        rows, col = image.shape[:2]
        cropped_image = image[0:rows, 0:1141]
        rows_cr, col_cr = cropped_image.shape[:2]
        resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)

        images_list.append(resized)


    if len(images_list) == 0:
        print("No images were read.")
        exit(100)

    # if shuffle:
    #     data = list(zip(images_list, object_annotations_list))
    #     random.shuffle(data)
    #     images_list, object_annotations_list = zip(*data)

    images_list = np.array(images_list).astype(np.uint8)

    return images_list


def get_anchor_data_ssd(bboxes, anchor_dims, img_dims, anchor_stride, iou_low, iou_high):
    """

    :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]
                   # NOTE: ensure the coordinates are integers
    :param anchor_dims: tuple of anchor dimensions - (height, width) [tuple]
    :param img_dims: (rows, cols, depth) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param iou_low: [int]
    :param iou_high: [int]
    :return:
    """

    num_anchors = len(anchor_dims)
    # print(anchor_dims)
    output_dims_class = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), num_anchors + 1)
    output_dims_reg = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), num_anchors * 4)
    # print(output_dims_class)
    output_class = np.zeros(output_dims_class).astype(np.int)
    output_reg = np.zeros(output_dims_reg).astype(np.int)

    # first position of an anchor center
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))
    # print (img_dims[0])
    for output_row, center_row in enumerate(range(start_r, img_dims[0]-start_r+1, anchor_stride)):  # iterate through rows of centers
        for output_col, center_col in enumerate(range(start_c, img_dims[1]-start_c+1, anchor_stride)):  # iterate through columns of centers
            # print(output_row)
            # print(output_col)
            for anchor_ind, anchor_dim in enumerate(anchor_dims):  # iterate through different anchor dimensions
                # print(anchor_dim)

                half_anchor_dim_h = np.int(np.round(anchor_dim[0] / 2))
                half_anchor_dim_w = np.int(np.round(anchor_dim[1] / 2))

                for bbox in bboxes:  # iterate through annotated bounding boxes
                    # print(bbox)
                    # --- assign classes: calculate IOU, place 1 or 0 at the required position ---
                    anchor = [max(0, center_row - half_anchor_dim_h),
                              max(0, center_col - half_anchor_dim_w),
                              min(center_row - half_anchor_dim_h + anchor_dim[0], img_dims[0]),
                              min(center_col - half_anchor_dim_w + anchor_dim[1], img_dims[1])]
                    # min_row, min_col, max_row, max_col
                    # print(bbox)
                    # print(anchor)
                    iou = helper_postprocessing.calc_iou(bbox, anchor)

                    if iou >= iou_high:
                        # print(output_row)
                        # positive sample: set class, calculate deltas
                        output_class[output_row, output_col, anchor_ind] = 1

                        # --- set deltas ---
                        # current location minus correct location
                        # delta_r = bbox[0] - anchor[0]
                        # delta_c = bbox[1] - anchor[1]
                        # delta_h = bbox[2] - bbox[0] - anchor_dim[0]
                        # delta_w = bbox[3] - bbox[1] - anchor_dim[1]
                        #
                        # output_reg[output_row, output_col, anchor_ind * 4 + 0] = delta_r
                        # output_reg[output_row, output_col, anchor_ind * 4 + 1] = delta_c
                        # output_reg[output_row, output_col, anchor_ind * 4 + 2] = delta_h
                        # output_reg[output_row, output_col, anchor_ind * 4 + 3] = delta_w

                    if (iou < iou_high) and (iou > iou_low):
                        # IOU between iou_min and iou_max
                        # class - marked 2, deltas - 0
                        output_class[output_row, output_col, anchor_ind] = 2

    # assign background
    for out_row in range(output_class.shape[0]):  # iterate through rows of output
        for out_col in range(output_class.shape[1]):  # iterate through columns of output

            if sum(output_class[out_row, out_col, :]) == 0:
                # print(out_row, out_col)
                output_class[out_row, out_col, num_anchors] = 1

    # replace 2s with 0s
    output_class = np.where(output_class == 2, 0, output_class)

    # remove border pixels
    for ind_a, a_dim in enumerate(anchor_dims):
        border_padding = np.int((anchor_dims[ind_a][0] / anchor_stride) / 2) + 1

        output_class[0:border_padding, :, ind_a] = 0
        output_class[output_class.shape[0] - border_padding:, :, ind_a] = 0
        output_class[:, 0:border_padding, ind_a] = 0
        output_class[:, output_class.shape[1] - border_padding:, ind_a] = 0

        # output_reg[0:border_padding, :, ind_a] = 0
        # output_reg[output_class.shape[0] - border_padding:, :, ind_a] = 0
        # output_reg[:, 0:border_padding, ind_a] = 0
        # output_reg[:, output_class.shape[1] - border_padding:, ind_a] = 0


    # --- select negative samples ---

    # count positives and negatives
    num_positives = np.sum(output_class[:, :, 0:num_anchors])

    # find negatives
    negs = output_class[:, :, num_anchors]
    [r, c] = np.where(negs == 1)


    # select negatives to remove
    ind_to_remove = np.arange(len(r))
    np.random.shuffle(ind_to_remove)

    num_neg = min(len(r), num_positives * 10)   # number of positive to negative samples ratio: 1 to 3
    num_to_remove = len(r) - num_neg
    ind_to_remove = ind_to_remove[:num_to_remove]


    # remove negatives
    for ind in ind_to_remove:
        output_class[r[ind], c[ind], :] = 0

    if num_positives > 0:
        # valid.append(i)
        # i=+1
        # return output_class, output_reg / 100
        return output_class
    else:
        return None


def plot_gt_annotations(images, bboxes, dst_path):
    """
    plot annotated ground truth rectangles onto photos
    :param images: grayscale photos [ndarray]
    :param dst_path: path of the destination folder of the annotated images [string]
    :return: None
    """

    for ind, img in enumerate(images):
        coords = bboxes[ind]  # min_row, min_col, max_row, max_col

        for rect in coords:
            cv2.rectangle(img, (rect[1], rect[0]), (rect[3], rect[2]), color=(0, 0, 0), thickness=1)

        cv2.imwrite(os.path.join(dst_path, str(ind).zfill(4) + '.bmp'), img)


def save_results(results_path, images, plot_color, output_cls, output_reg, anchor_dims, anchor_stride, prob_thr, norm_coef, output_branch):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder [str]
    :param images: test images [ndarray]
    :param plot_color: BGR values of the color of the annotations (tuple)
    :param output_cls: output of the classifier [ndarray]
    :param output_reg: output of the regressor [ndarray]
    :param anchor_dims: tuple of tuples of anchor dimensions (height, width) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef: coefficient to reverse the range normalization of regressor ground truth applied before training [int]
    :param output_branch: specifies the output branch results to be saved, accepted values are 'classifier' and 'regressor' [string]
    :return: None
    """

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    output_reg = np.round(output_reg * norm_coef).astype(np.int)  # regressor output, shape = (num_images, 30, 50, 12)

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    for im_ind, image in (enumerate(images)):

        res = output_cls[im_ind, :, :, 0:len(anchor_dims)]     # classifier output

        [r, c, d] = np.where(res > 0.5)     # get coordinate of positive anchors
                                            # d contains the indices of anchor size

        for pred_ind in range(len(r)):      # iterate over positive predictions

            anchor_dim = anchor_dims[d[pred_ind]]
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 1]
            delta_h = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 2]
            delta_w = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 3]

            min_row = np.int(center_row - np.round(anchor_dim[0] / 2))
            min_col = np.int(center_col - np.round(anchor_dim[1] / 2))

            # adjust position and size with regressor predictions
            min_row_adj = np.int(min_row + delta_r)
            min_col_adj = np.int(min_col + delta_c)
            h_adj = np.int(anchor_dim[0] + delta_h)
            w_adj = np.int(anchor_dim[1] + delta_w)

            max_row_adj = min_row_adj + h_adj
            max_col_adj = min_col_adj + w_adj

            # plot bounding box onto image
            if output_branch == 'classifier':
                cv2.rectangle(image, (min_col, min_row), (min_col + anchor_dim[1], min_row + anchor_dim[0]), color=plot_color, thickness=1)

            elif output_branch == 'regressor':
                cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=plot_color, thickness=1)

            else:
                print("Invalid network output branch specified in function parameters.")
                exit(1)

        # save test image with bounding boxes of detected objects
        cv2.imwrite(os.path.join(results_path, str(im_ind).zfill(5) + '.bmp'), image)

def save_results_cls(results_path, images, plot_color, output_cls, anchor_dims, anchor_stride, prob_thr, norm_coef, output_branch):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder [str]
    :param images: test images [ndarray]
    :param plot_color: BGR values of the color of the annotations (tuple)
    :param output_cls: output of the classifier [ndarray]
    :param output_reg: output of the regressor [ndarray]
    :param anchor_dims: tuple of tuples of anchor dimensions (height, width) [tuple]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef: coefficient to reverse the range normalization of regressor ground truth applied before training [int]
    :param output_branch: specifies the output branch results to be saved, accepted values are 'classifier' and 'regressor' [string]
    :return: None
    """


    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    # output_reg = np.round(output_reg * norm_coef).astype(np.int)  # regressor output, shape = (num_images, 30, 50, 12)

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    for im_ind, image in (enumerate(images)):
        lista = []
        res = output_cls[im_ind, :, :, 0:len(anchor_dims)]     # classifier output

        [r, c, d] = np.where(res > 0.5)     # get coordinate of positive anchors
                                            # d contains the indices of anchor size

        for pred_ind in range(len(r)):      # iterate over positive predictions

            anchor_dim = anchor_dims[d[pred_ind]]
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            # delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 0]
            # delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 1]
            # delta_h = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 2]
            # delta_w = output_reg[im_ind, r[pred_ind], c[pred_ind], d[pred_ind] * 4 + 3]

            min_row = np.int(center_row - np.round(anchor_dim[0] / 2))
            min_col = np.int(center_col - np.round(anchor_dim[1] / 2))
            max_row=min_row + anchor_dim[0]
            max_col=min_col + anchor_dim[1]

            lista.append([min_row,min_col, max_row, max_col])
            # adjust position and size with regressor predictions
            # min_row_adj = np.int(min_row + delta_r)
            # min_col_adj = np.int(min_col + delta_c)
            # h_adj = np.int(anchor_dim[0] + delta_h)
            # w_adj = np.int(anchor_dim[1] + delta_w)

            # max_row_adj = min_row_adj + h_adj
            # max_col_adj = min_col_adj + w_adj

            # plot bounding box onto image
            if output_branch == 'classifier':
                print("0")
                # cv2.rectangle(image, (min_col, min_row), (min_col + anchor_dim[1], min_row + anchor_dim[0]), color=plot_color, thickness=1)

            # elif output_branch == 'regressor':
                #cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=plot_color, thickness=1)

            else:
                print("Invalid network output branch specified in function parameters.")
                exit(1)

        # save test image with bounding boxes of detected objects
        # cv2.imwrite(os.path.join(results_path, str(im_ind).zfill(4) + '.bmp'), image)
        img_dest=os.path.join(results_path, str(im_ind).zfill(4) + '.bmp')
        thr=0.2
        # # print(lista)
        finalw=helper_postprocessing.nms_tanja(image, img_dest, lista, thr)
#
# def save_results_new_anchorless(results_path, images, plot_color, output_cls, output_reg, anchor_dims, anchor_stride, prob_thr,
#                                 norm_coef, output_branch):
#     """
#     plot bounding boxes of detected objects onto test images and save as images
#     :param results_path: path of destination folder [str]
#     :param images: test images [ndarray]
#     :param plot_color: BGR values of the color of the annotations (tuple)
#     :param output_cls: output of the classifier [ndarray]
#     :param output_reg: output of the regressor [ndarray]
#     :param anchor_dims: tuple of tuples of anchor dimensions (height, width) [tuple]
#     :param anchor_stride: stride along rows and columns [int]
#     :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
#     :param norm_coef: coefficient to reverse the range normalization of regressor ground truth applied before training [int]
#     :param output_branch: specifies the output branch results to be saved, accepted values are 'classifier' and 'regressor' [string]
#     :return: None
#     """
#
#     img_dims = [images[0].shape[0], images[0].shape[1]]
#
#     list_of_windows = []
#     # binarize classifier output probabilities
#     output_cls[output_cls >= prob_thr] = 1
#     output_cls[output_cls < prob_thr] = 0
#
#     # round regressor output and cast to integer pixel values
#     output_reg = output_reg * norm_coef  # regressor output, shape = (num_images, 30, 50, 12)
#
#     # calculate location of first (top left) anchor center - start at half of stride size
#     start_r = np.int(np.round(anchor_stride / 2))
#     start_c = np.int(np.round(anchor_stride / 2))
#
#     for im_ind, image in enumerate(images):
#         img_reg = image.copy()
#         reg_list = []
#         res = output_cls[im_ind, :, :, 0]  # classifier output
#
#         [r, c] = np.where(res > 0.5)  # get coordinate of positive anchors
#         # d contains the indices of anchor size
#
#         for pred_ind in range(len(r)):  # iterate over positive predictions
#
#             center_row = r[pred_ind] * anchor_stride + start_r
#             center_col = c[pred_ind] * anchor_stride + start_c
#
#             # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
#             delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
#             delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
#             h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
#             w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]
#
#             h = h_percent * img_dims[0]
#             w = w_percent * img_dims[1]
#
#             # bbox top left point
#             min_row = np.int(center_row - np.round(h / 2))
#             min_col = np.int(center_col - np.round(w / 2))
#
#             # adjust position and size with regressor predictions
#             min_row_adj = np.int(min_row + delta_r)
#             min_col_adj = np.int(min_col + delta_c)
#
#             max_row_adj = min_row_adj + h
#             max_col_adj = min_col_adj + w
#
#             # plot bounding box onto image
#             reg_list.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj])
#
#             min_col_adj = int(min_col_adj)
#             min_row_adj = int(min_row_adj)
#             max_col_adj = int(max_col_adj)
#             max_row_adj = int(max_row_adj)
#
#             cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=(0, 255, 0), thickness=1)
#             # cv2.circle(image, (center_col, center_row), 3, color=(0, 0, 255), thickness=3)
#
#         # save test image with bounding boxes of detected objects
#         # cv2.imshow("a",image)
#         # cv2.waitKey(0)
#         cv2.imwrite(os.path.join(results_path, str(im_ind).zfill(4) + '.bmp'), image)
#
#         # break
def save_results_anchorless_limits(results_path, images, plot_color, output_cls, output_reg, anchor_stride, prob_thr, norm_coef):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder [str]
    :param images: test images [ndarray]
    :param plot_color: BGR values of the color of the annotations (tuple)
    :param output_cls: output of the classifier [ndarray]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef: normalization coefficient for regression data [float]
    :return: None
    """

    img_dims = (images[0].shape[0], images[0].shape[1])     # height, width

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    output_reg = output_reg * norm_coef     # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    for im_ind, image in enumerate(images):
        lista = []

        res = output_cls[im_ind, :, :, 0]  # classifier output, probability maps for positive objects only

        [r, c] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]

            # bbox top left point
            min_row = np.int(center_row - np.round(h / 2))
            min_col = np.int(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = np.int(min_row + delta_r)
            min_col_adj = np.int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, np.int(min_col_adj))
            min_row_adj = max(0, np.int(min_row_adj))
            max_col_adj = min(np.int(max_col_adj), img_dims[1])
            max_row_adj = min(np.int(max_row_adj), img_dims[0])

            lista.append([min_row_adj,min_col_adj, max_row_adj, max_col_adj])

            # cv2.circle(image, (center_col, center_row), 3, color=plot_color, thickness=3)     # plot object centers
            # cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=plot_color, thickness=1)

        # save test image with bounding boxes of detected objects
        # cv2.imwrite(os.path.join(results_path, str(im_ind).zfill(4) + '.bmp'), image)
        img_dest = os.path.join(results_path, str(im_ind).zfill(5) + '.bmp')
        thr = 0.2
        # # print(lista)
        finalw = helper_postprocessing.nms_tanja(image, img_dest, lista, thr)

# za debugging
def save_results_anchorless_limits_1(results_path, images, plot_color, output_cls, output_reg, anchor_stride, prob_thr, norm_coef):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder [str]
    :param images: test images [ndarray]
    :param plot_color: BGR values of the color of the annotations (tuple)
    :param output_cls: output of the classifier [ndarray]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef: normalization coefficient for regression data [float]
    :return: None
    """

    img_dims = (images[0].shape[0], images[0].shape[1])     # height, width

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    output_reg = output_reg * norm_coef     # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    for im_ind, image in enumerate(images):
        lista = []

        res = output_cls[im_ind, :, :, ]  # classifier output, probability maps for positive objects only

        [r, c] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]

            # bbox top left point
            min_row = np.int(center_row - np.round(h / 2))
            min_col = np.int(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = np.int(min_row + delta_r)
            min_col_adj = np.int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, np.int(min_col_adj))
            min_row_adj = max(0, np.int(min_row_adj))
            max_col_adj = min(np.int(max_col_adj), img_dims[1])
            max_row_adj = min(np.int(max_row_adj), img_dims[0])

            lista.append([min_row_adj,min_col_adj, max_row_adj, max_col_adj])

            # cv2.circle(image, (center_col, center_row), 3, color=plot_color, thickness=3)     # plot object centers
            # cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=plot_color, thickness=1)

        # save test image with bounding boxes of detected objects
        # cv2.imwrite(os.path.join(results_path, str(im_ind).zfill(4) + '.bmp'), image)
        img_dest = os.path.join(results_path, str(im_ind).zfill(5) + '.bmp')
        thr = 0.2
        # # print(lista)
        finalw = helper_postprocessing.nms_tanja(image, img_dest, lista, thr)

def save_results_anchorless_limits_cls(results_path, images, plot_color, output_cls, output_reg, anchor_stride, prob_thr, norm_coef):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder [str]
    :param images: test images [ndarray]
    :param plot_color: BGR values of the color of the annotations (tuple)
    :param output_cls: output of the classifier [ndarray]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef: normalization coefficient for regression data [float]
    :return: None
    """

    img_dims = (images[0].shape[0], images[0].shape[1])     # height, width

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    output_reg = output_reg * norm_coef     # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))

    for im_ind, image in enumerate(images):
        lista = []
        res = output_cls[im_ind, :, :, 0]  # classifier output, probability maps for positive objects only

        [r, c] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]

            # bbox top left point
            min_row = np.int(center_row - np.round(h / 2))
            min_col = np.int(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = np.int(min_row + delta_r)
            min_col_adj = np.int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, np.int(min_col_adj))
            min_row_adj = max(0, np.int(min_row_adj))
            max_col_adj = min(np.int(max_col_adj), img_dims[1])
            max_row_adj = min(np.int(max_row_adj), img_dims[0])

            max_col = np.int(min(min_col + w, img_dims[1]))
            max_row = np.int(min(min_row + h, img_dims[0]))
            min_col = np.int(max(0, min_col))
            min_row = np.int(max(0, min_row))
            lista.append([min_row, min_col, max_row, max_col])
            # cv2.circle(image, (center_col, center_row), 3, color=plot_color, thickness=3)     # plot object centers
            # cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=plot_color, thickness=1)
            cv2.rectangle(image, (min_col, min_row), (max_col, max_row), color=plot_color, thickness=1)

        # save test image with bounding boxes of detected objects
        # cv2.imwrite(os.path.join(results_path, str(im_ind).zfill(4) + '.bmp'), image)
        img_dest = os.path.join(results_path, str(im_ind).zfill(5) + '.bmp')
        thr = 0.2
        # # print(lista)
        finalw = helper_postprocessing.nms_tanja(image, img_dest, lista, thr)
