"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 19.04.2022

Description: apply a fully convolutional SSD architecture for object classification and localization to test data
Python version: 3.6
"""

# python imports
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
from helpers import homography
# custom package imports
import helper_model, helper_data, helper_losses
import anchorless_genertor_plot
import helper_postprocessing
import statistics

def average_window(maxwindows):
    # print(maxwindows)
    maxcoord = maxwindows[0][0]
    for window in maxwindows:
        for x in window:
            if maxcoord>x:
                maxcoord=x
    # print(maxcoord)
    suma = [sum(x) for x in zip(*maxwindows)]
    avg = [x / len(maxwindows) for x in suma]

    # print(suma,avg)
    return avg

def calc_iou(box1, box2):
    """

    :param box1: list of coordinates: row1, col1, row2, col2 [list]
    :param box2: list of coordinates: row1, col1, row2, col2 [list]
    :return: iou value
    """

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # respective area of the two boxes
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # overlap area
    interArea = max(xB - xA, 0) * max(yB - yA, 0)

    # IOU

    if (boxAArea+boxBArea-interArea) > 0:
        # print(box1, box2)
        iou = interArea / (boxAArea + boxBArea - interArea)
        return iou

    print("union < 0")
    # return
    exit(1)
def getclusterforanchor(anchor, lista, thr):
    cluster = []
    for anch in lista:
        # print(anch)
        iou = calc_iou(anchor, anch)
        # print(round(iou,1))
        if iou > thr:
            cluster.append(anch)
    # print("length clys", len(cluster))
    return cluster


def getclustersforanchors(lista, thr):
    clusters = []
    res = []
    finalclusters = []
    countconnections = []
    # print(lista)

    for anchor in lista:


        # print(anchor)
        cluster = getclusterforanchor(anchor, lista, thr)
        # print(cluster)
        countconnections.append(len(cluster))
        clusters.append(cluster)
    maxnum = countconnections[0]
    cnt = -1
    indexestoremove = []
    # print("countconnections")
    # print(len(countconnections))
    # print(countconnections)
    if (len(clusters) == 1):
        return clusters
    for anchor in lista:
        clusters_idx = []
        # connection_sublist = []
        # print("anchor", anchor)
        # print("clusters", clusters)
        for cluster in clusters:
            # print(cluster)
            if anchor in cluster:
                clusters_idx.append(1)
                # connection_sublist.append(countconnections[cnt])
            else:
                clusters_idx.append(0)

        indices = [i for i, x in enumerate(clusters_idx) if x == 1]
        # print(len(clusters_idx))
        # print(len(clusters))
        # print(indices)
        maxcluster = clusters[indices[0]]
        # print(maxcluster)
        for indx in indices:

            # print(indx,clusters[indx])
            # print(maxcluster,len(maxcluster))
            if (len(clusters[indx]) > len(maxcluster)):
                maxcluster = clusters[indx]
        res.append(maxcluster)

    # print("res", res)
    finalclusters = []
    for i in res:
        if i not in finalclusters:
            finalclusters.append(i)

    # print("finalclusters", finalclusters)

    return finalclusters

def nms_new(lista, thr, num_classes):
    finalwindows = []

    if len(lista) == 0:
        return []

    all_clusters = getclustersforanchors(lista, thr)

    for cluster in all_clusters:

        maxwindows = []
        maxwindows_new = []

        if len(cluster) == 0:
            continue

        elements = [item[2] - item[0] for item in cluster]

        if len(elements) <= 2:
            first2max = sorted(elements)
        else:

            first2max = sorted(elements)[:]

        first2max_noduplicates = []
        for i in first2max:
            if i not in first2max_noduplicates:
                first2max_noduplicates.append(i)

        for maxnum in first2max_noduplicates:
            indices = [i for i, x in enumerate(elements) if x == maxnum]

            for idx in indices:
                maxwindows.append(cluster[idx])

        bboxes_per_class = [[] for i in range(num_classes)]  # bounding boxes grouped by class

        for maxwindow in maxwindows:
            bboxes_per_class[maxwindow[-1]].append(maxwindow)

        max_class = max((len(l), i) for i, l in enumerate(bboxes_per_class))[1]

        for maxwindow in maxwindows:
            if maxwindow[-1] == max_class:
                maxwindows_new.append(maxwindow)

        finalwindow = average_window(maxwindows_new)
        finalwindows.append(finalwindow)

    return finalwindows
def save_results_anchorless_limits(number_of_images, output_cls, output_reg, anchor_stride, prob_thr,
                                   thr_clustering):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param results_path: path of destination folder for all CNN detections [string]
    :param results_path_nms: path of destination folder of prostprocessed data [string]
    :param images: test images [ndarray]
    :param output_cls: output of the classifier [ndarray]
    :param output_reg: output of the regressor [ndarray]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef_position: normalization coefficient for position regression data [float]
    :param norm_coef_size: normalization coefficient for image dimensions regression data [float]
    :param thr_clustering: iou threshold for combining detections into one cluster [float]
    :param colors_list: list of colors to plot each class [tuple]
    :return: None
    """
    img_dims = (341, 512)  # height, width
    # print(images_s[0].shape[0])
    num_classes = output_cls.shape[3] - 1

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = np.int(np.round(anchor_stride / 2))
    start_c = np.int(np.round(anchor_stride / 2))
    finalwindows_batch = []

    for im_ind in range(number_of_images):

        # image_za_nms = deepcopy(image)
        valid_bboxes = []  # list of bounding boxes with positive height and width

        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions
            # ako e klasa nevozilo ne crtaj i ne zacucuvaj koordinati
            if d[pred_ind] == 2:
                continue
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]
            # print(h)
            # print(w)
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

            if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:
                valid_bboxes.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])

        if len(r) != 0:

            finalwindows = nms_new(valid_bboxes, thr_clustering, num_classes)

            finalwindows_batch.append(finalwindows)

        else:

            finalwindows_batch.append([])

    return finalwindows_batch


# NOTE da se menuva flagot normalizeAll vo zavisnost od normalizacijata iskoristena pri dobivanje na modelot
x_test_all = []
images_names_all = []
# --- flags ---

flag_save_coords = True
combined_output = True
flag_nms = True
normalize = False
# --- paths ---

# NOTE: specify destination paths
# srcImagesPath = r'D:\Monika\VideosTest\All_Cameras_annnotated\Images\Renamed'


# annot_path = r'D:\Monika\VideosTest\All_Cameras_annnotated\Annotations\Renamed'
srcModelPath=r'D:\Monika\Models\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva'
config_directory=r'M:\Homography\Configs'
src_root_images=r'D:\Monika\VideosTest\All_Cameras_annnotated\Images\proba'

dir_names = os.listdir(src_root_images)


# --- variables ---
imgDims = {'rows': 341, 'cols': 512}
num_classes = 4
img_depth = 3
img_dims = (imgDims['rows'], imgDims['cols'], img_depth)
orig_img_height=1080
orig_img_width=1920

# x_test_all = np.array(x_test_all)
#
# print(x_test_all.shape)
# x_test_1 = [x for x in x_test_all]
# x_test_1 = np.array(x_test_1)
# x_test_1 = x_test_1.reshape(x_test_1.shape + (1,))


model = helper_model.load_model(model_path=os.path.join(srcModelPath, 'model.json'),
                                weights_path=os.path.join(srcModelPath, 'model.h5'))  # build model architecture

reg_norm_coef_position_rows, reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width = 1, 1, 1, 1


# --- plot ground truth network output ---
prob_thr = 0.85
thr_clustering = 0.3
color_small = (0, 255, 0)
color_large = (0, 0, 255)
colors_list = (color_small, color_large)

anchor_stride=8
max_prob_thr = 0.95
best_f1 = 0
best_thr = 0
scale=1.5
t=200/1000
def preprocess_image(img,config_directory,camera_name,orig_img_height,orig_img_width):
    config=homography.load_config(config_directory,camera_name)
    crop_rect = list(map(int, config["CropRect"].split(",")))

    crop_top = crop_rect[1]
    crop_left = crop_rect[0]
    crop_bottom = orig_img_height - crop_rect[3] - crop_rect[1]
    crop_right = orig_img_width - crop_rect[2] - crop_rect[0]
    cropped_image=img[crop_top:orig_img_height-crop_bottom, crop_left:orig_img_width-crop_right]
    resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)
    return resized
for dir_name in dir_names:
    camera_name=dir_name[0:5]

    h = homography.homography_matrix_create(config_directory, camera_name, orig_img_height, orig_img_width, 512, 768)
    vehicle_dir_names=os.listdir(os.path.join(src_root_images,dir_name))
    for vehicle_name in vehicle_dir_names:
        im_names_vehicle=os.listdir(os.path.join(src_root_images,dir_name,vehicle_name))
        speeds=[]
        for i in range(len(im_names_vehicle)-1):
            x_test=[]
            print(im_names_vehicle[i])
            im1=cv2.imread(os.path.join(src_root_images,dir_name,vehicle_name,im_names_vehicle[i]),0)

            im2=cv2.imread(os.path.join(src_root_images,dir_name,vehicle_name,im_names_vehicle[i+1]),0)

            im1_preprocessed=preprocess_image(im1,config_directory, camera_name, orig_img_height, orig_img_width)
            im2_preprocessed=preprocess_image(im2,config_directory, camera_name, orig_img_height, orig_img_width)
            im1_reshaped = im1_preprocessed.reshape(im2_preprocessed.shape + (1,))
            im2_reshaped = im2_preprocessed.reshape(im2_preprocessed.shape + (1,))
            x_test.append(im1_reshaped)
            x_test.append(im2_reshaped)

            x_test=np.array(x_test)
            comb_output = model.predict(x_test, verbose=1)

            output_cls = comb_output[:, :, :, :num_classes]
            output_reg = comb_output[:, :, :, num_classes:]

            batch_finalwindows = save_results_anchorless_limits(len(x_test), output_cls,
                                                                                   output_reg,
                                                                                   anchor_stride, prob_thr,
                                                                                   thr_clustering)

            point1_x=((batch_finalwindows[0][0][1]+batch_finalwindows[0][0][3])/2)*scale
            point1_x=int(point1_x)
            point1_y=int(batch_finalwindows[0][0][2]*scale)
            proj_point1=homography.image2ProjPoint([point1_x,point1_y],h)

            point2_x = ((batch_finalwindows[1][0][1] + batch_finalwindows[1][0][3]) / 2) * scale
            point2_x = int(point2_x)
            point2_y = int(batch_finalwindows[1][0][2] * scale)
            proj_point2 = homography.image2ProjPoint([point2_x, point2_y], h)

            speed=(proj_point1[1]-proj_point2[1])/t
            speeds.append(speed)
            # bboxes=anchorless_genertor_plot.save_results_anchorless_limits_cls1(resultsPath, x_test_all, images_names_all,
            #                                                          output_cls, output_reg, anchor_stride,
            #                                                          prob_thr,
            #                                                          reg_norm_coef_position, reg_norm_coef_size,
            #                                                          flag_normalizeAll, flag_save_coords,
            #                                                          flag_after_regressor)

        avg_speed=statistics.mean(speeds)



print(best_f1)
print(best_thr)
