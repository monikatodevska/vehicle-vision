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

from cv2 import cvtColor

from helpers import homography
# custom package imports
import helper_model, helper_data, helper_losses
import anchorless_genertor_plot
import helper_postprocessing
import statistics
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda,Input, Flatten, Dense, Layer, Conv2D, MaxPool2D, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.models import model_from_json
def bbox_resized_to_original(bbox_resized, crop_top, crop_bottom, crop_left, crop_right,
                             original_size=(1920, 1080), resized_size=(512, 341)):
    """
    Converts bounding box coordinates from resized image back to original image coordinates.

    Parameters:
    - bbox_resized: (ymin, xmin, ymax, xmax) in the resized image
    - crop_top, crop_bottom, crop_left, crop_right: Cropping parameters
    - original_size: (width, height) of the original image (width=1920, height=1080)
    - resized_size: (width, height) of the resized image (width=512, height=341)

    Returns:
    - bbox_original: (ymin, xmin, ymax, xmax) in the original image
    """

    orig_w, orig_h = original_size
    res_w, res_h = resized_size

    # Compute cropped image size
    cropped_w = orig_w - crop_left - crop_right
    cropped_h = orig_h - crop_top - crop_bottom

    # Compute scale factors
    scale_x = cropped_w / res_w
    scale_y = cropped_h / res_h

    # Unpack resized bbox
    ymin_resized, xmin_resized, ymax_resized, xmax_resized = bbox_resized

    # Convert to cropped image coordinates
    ymin_cropped = ymin_resized * scale_y
    xmin_cropped = xmin_resized * scale_x
    ymax_cropped = ymax_resized * scale_y
    xmax_cropped = xmax_resized * scale_x

    # Convert to original image coordinates
    ymin_original = round(ymin_cropped + crop_top)
    xmin_original = round(xmin_cropped + crop_left)
    ymax_original = round(ymax_cropped + crop_top)
    xmax_original = round(xmax_cropped + crop_left)

    return (ymin_original, xmin_original, ymax_original, xmax_original)



# # Example usage
# bbox_resized = (50, 30, 100, 60)  # Example bbox in resized image
# crop_top, crop_bottom, crop_left, crop_right = (100, 80, 50, 70)
#
# bbox_original = bbox_resized_to_original(bbox_resized, crop_top, crop_bottom, crop_left, crop_right)
# print("Bounding box in original image:", bbox_original)

def adjust_bboxes(bboxes, orig_img_size):
    """
    Adjusts bounding boxes after cropping and two resizing steps.

    Args:
        bboxes (np.array): Array of bounding boxes in (ymin, xmin, height, width).
        crop_rect (list): Crop rectangle in (xmin, ymin, width, height).
        orig_img_size (tuple): Original image size (width, height).
        resized_size1 (tuple): First resized image size (width, height).
        resized_size2 (tuple): Final resized image size (width, height).

    Returns:
        np.array: Adjusted bounding boxes in (ymin, xmin, height, width).
    """
    config = homography.load_config(config_directory, camera_name)
    crop_rect = list(map(int, config["CropRect"].split(",")))
    orig_width, orig_height = orig_img_size

    # Compute crop offsets
    crop_left = crop_rect[0]
    crop_top = crop_rect[1]
    crop_right = orig_width - (crop_rect[0] + crop_rect[2])
    crop_bottom = orig_height - (crop_rect[1] + crop_rect[3])

    # Cropped image dimensions
    cropped_width = orig_width - crop_left - crop_right
    cropped_height = orig_height - crop_top - crop_bottom

    bboxes = np.array(bboxes, dtype=np.float32)

    # # Step 1: Adjust for cropping
    # bboxes[:, 0] -= crop_top  # Adjust ymin
    # bboxes[:, 1] -= crop_left  # Adjust xmin

    # # Step 2: Resize from cropped size to resized_size1 (512x341)
    # scale_x1 = resized_size1[0] / cropped_width
    # scale_y1 = resized_size1[1] / cropped_height
    # bboxes[:, [0, 2]] *= scale_y1  # Scale ymin and height
    # bboxes[:, [1, 3]] *= scale_x1  # Scale xmin and width
    #
    # # Step 3: Upscale from resized_size1 to resized_size2 (768x512)
    # scale_x2 = resized_size2[0] / resized_size1[0]
    # scale_y2 = resized_size2[1] / resized_size1[1]
    # bboxes[:, [0, 2]] *= scale_y2  # Scale ymin and height
    # bboxes[:, [1, 3]] *= scale_x2  # Scale xmin and width
    #
    # # Convert to (ymin, xmin, ymax, xmax) format
    bboxes_yxyx = np.copy(bboxes)
    bboxes_yxyx[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # ymax = ymin + height
    bboxes_yxyx[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # xmax = xmin + width
    return bboxes_yxyx
def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
        keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def network_reg(encoded_reg_a, encoded_reg_i):
    c = Concatenate(axis=-1)([encoded_reg_a, encoded_reg_i])
    #reg_dist = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='linear', kernel_initializer='he_uniform',name="reg_out")(
    #    c)
    flat_c = Flatten()(c)
    reg_dist = Dense(1, activation='linear', kernel_initializer='he_normal', name="reg_out")(flat_c)

    return reg_dist

def siamese_both_build(input_shape, network):

    """
    Define the Keras Model for training
        Input :
            input_shape : shape of input images
            network : Neural network to train outputing embeddings
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)

    """

    # Define the tensors for the three input images
    anchor_input = Input(input_shape, name="anchor_input")
    input2 = Input(input_shape, name="input2")

    # Generate the encodings (feature vectors) for the three images
    # encoded_a, encoded_reg_a = network(anchor_input)
    # encoded_i, encoded_reg_i = network(input2)
    encoded_reg_a = network(anchor_input)
    encoded_reg_i = network(input2)

    reg_dist = network_reg(encoded_reg_a, encoded_reg_i)
    # distance = Lambda(euclidean_distance, name="out_class")([encoded_a, encoded_i])

    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input, input2], outputs=[reg_dist]) # distance,

    # return the model
    return network_train


def load_branch_structure(model_path):
    """
    loads a pre-trained model configuration and calculated weights
    :param model_path: path of the serialized model configuration file (.json) [string]
    :return: model - keras model object
    """

    # --- load model configuration ---
    json_file = open(model_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)     # load model architecture

    return model

def make_square_bbox(bbox, image_shape, new_dims, target_dim):
    """
    Convert the bounding box into a square, based on the longer side.

    Parameters:
        bbox (tuple): (x_min, y_min, x_max, y_max)
        image_shape (tuple): (height, width) of the image

    Returns:
        tuple: Square bounding box (x_min, y_min, x_max, y_max)
    """
    y_min, x_min, y_max, x_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    # size = max(width, height)  # Make it square based on the longer side
    diff_h = abs(target_dim - height)
    diff_w = abs(target_dim - width)
    # diff=abs(height-width)
    # if width>height:
    y_min=max(0,int(y_min-diff_h)) # //2
    y_max=min(image_shape[0],int(y_max)) # //2+diff_h
# elif height>=width:
    x_min=max(0,int(x_min-diff_w/2))
    x_max=min(image_shape[1],int(x_max+diff_w/2))
    # # Compute the new center
    # center_x = (x_min + x_max) // 2
    # center_y = (y_min + y_max) // 2
    #
    # # Compute new square bbox
    # x_min_new = max(0, center_x - size // 2)
    # y_min_new = max(0, center_y - size // 2)
    # x_max_new = min(image_shape[1], x_min_new + size)
    # y_max_new = min(image_shape[0], y_min_new + size)

    return y_min, x_min, y_max, x_max, target_dim/new_dims[0]


def crop_bbox(image, bbox):
    """
    Crop a bounding box from an image.

    Parameters:
        image (numpy array): The input image.
        bbox (tuple): (x_min, y_min, x_max, y_max)

    Returns:
        numpy array: Cropped image.
    """
    y_min, x_min, y_max, x_max = bbox
    return image[y_min:y_max, x_min:x_max]
def scale_bbox(bbox, orig_size=(341, 512), new_size=(512, 768)):
    """
    Scale bounding box coordinates to match the new image size.

    Parameters:
        bbox (tuple): (x_min, y_min, x_max, y_max) coordinates.
        orig_size (tuple): (width, height) of the original image.
        new_size (tuple): (width, height) of the new image.

    Returns:
        tuple: Scaled bounding box (x_min', y_min', x_max', y_max')
    """
    scale_x = new_size[0] / orig_size[0]
    scale_y = new_size[1] / orig_size[1]

    y_min, x_min, y_max, x_max = bbox
    x_min_new = x_min * scale_x
    y_min_new = y_min * scale_y
    x_max_new = x_max * scale_x
    y_max_new = y_max * scale_y

    return round(y_min_new), round(x_min_new), round(y_max_new), round(x_max_new)


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

def preprocess_image(img,config_directory,camera_name,orig_img_height,orig_img_width):
    config=homography.load_config(config_directory,camera_name)
    crop_rect = list(map(int, config["CropRect"].split(",")))

    crop_top = crop_rect[1]
    crop_left = crop_rect[0]
    crop_bottom = orig_img_height - crop_rect[3] - crop_rect[1]
    crop_right = orig_img_width - crop_rect[2] - crop_rect[0]
    cropped_image=img[crop_top:orig_img_height-crop_bottom, crop_left:orig_img_width-crop_right]

    return cropped_image
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
src_root_images=r'D:\Monika\VideosTest\Vozila\brzini_test_ramki'
src_root_annots=r'D:\Monika\VideosTest\Vozila\brzini_test_bbox'
src_branch_structure_path=r'M:\Siamese_DS\JK_Models\JK_reg_160v10_nodense_resnetlike_v1_wd0_same_32bsize_SIAMESE_v2\branch.json'
src_weights_path1=r'M:\Siamese_DS\JK_Models\JK_reg_160v10_nodense_resnetlike_v1_wd0_same_32bsize_SIAMESE_v2\model.h5'
src_model_structure_path = r'M:\Siamese_DS\JK_Models\JK_reg_160v10_nodense_resnetlike_v1_wd0_same_32bsize_SIAMESE_v2\model.json'
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


model_vehicle = helper_model.load_model(model_path=os.path.join(srcModelPath, 'model.json'),
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
t=120/1000
input_dims=(160,160,1)


for dir_name in dir_names:
    camera_name=dir_name[0:5]
    if not ('Cam46') in dir_name:
        continue
    config = homography.load_config(config_directory, camera_name)
    crop_rect = list(map(int, config["CropRect"].split(",")))

    crop_top = crop_rect[1]
    crop_left = crop_rect[0]
    crop_bottom = orig_img_height - crop_rect[3] - crop_rect[1]
    crop_right = orig_img_width - crop_rect[2] - crop_rect[0]
    h = homography.homography_matrix_create(config_directory, camera_name, orig_img_height, orig_img_width, 1080, 1920)
    vehicle_dir_names=os.listdir(os.path.join(src_root_images,dir_name))
    for vehicle_name in vehicle_dir_names:
        print(vehicle_name)
        # if 'kola_1206' not in vehicle_name:
        #     continue
        im_names_vehicle=os.listdir(os.path.join(src_root_images,dir_name,vehicle_name))

        speeds=[]
        speeds_corrected=[]
        speeds_gt=[]
        for i in range(len(im_names_vehicle)-1):
            annot_path_im1=os.path.join(src_root_annots,dir_name,vehicle_name,im_names_vehicle[i][:-4]+'.txt')
            annot_path_im2=os.path.join(src_root_annots,dir_name,vehicle_name,im_names_vehicle[i+1][:-4]+'.txt')
            bboxes1=np.loadtxt(annot_path_im1,delimiter=',',dtype=int,ndmin=2)
            bboxes2=np.loadtxt(annot_path_im2,delimiter=',',dtype=int,ndmin=2)
            # y_max_center1=bboxes1[0]+bboxes1[2]
            # x_max_center1=(bboxes1[1]+bboxes1[3]+bboxes1[1])/2
            # y_max_center2=bboxes2[0]+bboxes2[2]
            # x_max_center2=(bboxes2[1]+bboxes2[3]+bboxes2[1])/2
            adjusted_bboxes1=adjust_bboxes(bboxes1,(1920,1080))
            adjusted_bboxes2=adjust_bboxes(bboxes2,(1920,1080))



            x_max_center_gt1=(adjusted_bboxes1[0][1]+adjusted_bboxes1[0][3])/2
            x_max_center_gt2=(adjusted_bboxes2[0][1]+adjusted_bboxes2[0][3])/2
            y_max_center_gt1=adjusted_bboxes1[0][2]
            y_max_center_gt2=adjusted_bboxes2[0][2]

            x_test=[]
            print(im_names_vehicle[i])

            #read pair of images
            im1=cv2.imread(os.path.join(src_root_images,dir_name,vehicle_name,im_names_vehicle[i]))
            im2=cv2.imread(os.path.join(src_root_images,dir_name,vehicle_name,im_names_vehicle[i+1]))
            # cv2.rectangle(im2,(adjusted_bboxes2[0][1],adjusted_bboxes2[0][0]),(adjusted_bboxes2[0][3],adjusted_bboxes2[0][2]), color=(255,0,0),thickness=1)
            # cv2.imwrite(os.path.join(r'C:\Users\User\Desktop\sliki', str(vehicle_name)+str(i).zfill(6) + ".bmp"), im2)
            #preprocess image(crop basd on config)
            im1_preprocessed=preprocess_image(im1,config_directory, camera_name, orig_img_height, orig_img_width)
            im2_preprocessed=preprocess_image(im2,config_directory, camera_name, orig_img_height, orig_img_width)
            #resize for cnn input, convert to grey and reshape for prediction
            im1_resized = cv2.resize(im1_preprocessed, (512, 341), interpolation=cv2.INTER_AREA)
            im2_resized = cv2.resize(im2_preprocessed, (512, 341), interpolation=cv2.INTER_AREA)

            im1_resized = cv2.cvtColor(im1_resized, cv2.COLOR_BGR2GRAY)
            im2_resized = cv2.cvtColor(im2_resized, cv2.COLOR_BGR2GRAY)

            im1_reshaped = im1_resized.reshape(im1_resized.shape + (1,))
            im2_reshaped = im2_resized.reshape(im2_resized.shape + (1,))
            #prepare array for prediction
            x_test.append(im1_reshaped)
            x_test.append(im2_reshaped)
            x_test=np.array(x_test)

            comb_output = model_vehicle.predict(x_test, verbose=1)

            output_cls = comb_output[:, :, :, :num_classes]
            output_reg = comb_output[:, :, :, num_classes:]

            batch_finalwindows = save_results_anchorless_limits(len(x_test), output_cls,
                                                                                   output_reg,
                                                                                   anchor_stride, prob_thr,
                                                                                   thr_clustering)
            if len(batch_finalwindows[0])==0 or len(batch_finalwindows[1])==0:
                continue
            bbox1=batch_finalwindows[0][0][:4]
            bbox2=batch_finalwindows[1][0][:4]
            #scale predicted bboxes to 512x768 for speed calculations
            bbox1_original=bbox_resized_to_original(batch_finalwindows[0][0][:4],crop_top,crop_bottom,crop_left,crop_right)
            bbox2_original=bbox_resized_to_original(batch_finalwindows[1][0][:4],crop_top,crop_bottom,crop_left,crop_right)

            target_dim = max((bbox1_original[2] - bbox1_original[0]), (bbox1_original[3] - bbox1_original[1]), (bbox2_original[2] - bbox2_original[0]), (bbox2_original[3] - bbox2_original[1]))

            # cv2.rectangle(im1,(int(bbox1_original[1]),int(bbox1_original[0])),(int(bbox1_original[3]),int(bbox1_original[2])),color=(255,0,0),thickness=2)
            # cv2.imshow('original',im1)
            # cv2.waitKey(0)
            # #resize full hd image to 512x768 (for drawing and debugging)
            # # im1_resized_tracking = cv2.resize(im1_preprocessed, (768, 512), interpolation=cv2.INTER_AREA)
            # # im2_resized_tracking = cv2.resize(im2_preprocessed, (768, 512), interpolation=cv2.INTER_AREA)
            #
            # # cv2.rectangle(im1_resized_tracking,(adjusted_bboxes1[0][1],adjusted_bboxes1[0][0]),(adjusted_bboxes1[0][3],adjusted_bboxes1[0][2]),color=(255,0,0),thickness=2)
            # # cv2.imshow('trackingg_slika',im1_resized_tracking)
            # # cv2.waitKey(0)
            # # Make the bbox square
            # #make predicted bboxes square for the siamese network
            square_bbox1, scale1 = make_square_bbox(bbox1_original, im1.shape, input_dims, target_dim)[:4], make_square_bbox(bbox1_original, im1.shape, input_dims, target_dim)[4]
            square_bbox2, scale2 = make_square_bbox(bbox2_original, im2.shape, input_dims, target_dim)[:4], make_square_bbox(bbox2_original, im2.shape, input_dims, target_dim)[4]
            # cv2.rectangle(im1, (square_bbox1[1], square_bbox1[0]), (square_bbox1[3], square_bbox1[2]),
            #               color=(255, 0, 0), thickness=2)
            # cv2.imshow('trackingg_slika_squared', im1)
            # cv2.waitKey(0)
            # print(square_bbox1)
            print(square_bbox2)
            # square_bbox1 = make_square_bbox(bbox1_scaled, im1_resized_tracking.shape)
            # square_bbox2 = make_square_bbox(bbox2_scaled, im2_resized_tracking.shape)

            # Crop the square bbox from the image
            #crop images from the squared bboxes
            cropped_image_siamese1 = crop_bbox(im1, square_bbox1)
            cropped_image_siamese2 = crop_bbox(im2, square_bbox2)
            print(cropped_image_siamese2.shape)
            # cv2.imshow('siamese_cropped', cropped_image_siamese1)
            # cv2.waitKey(0)
            #resize cropped images to 48x48 for siamese cnn input
            if max(cropped_image_siamese1.shape) > input_dims[0]:
                interp1=cv2.INTER_AREA
            else:
                interp1=cv2.INTER_CUBIC
            if max(cropped_image_siamese2.shape) > input_dims[0]:
                interp2=cv2.INTER_AREA
            else:
                interp2=cv2.INTER_CUBIC
            cropped_image_siamese1_resized = cv2.resize(cropped_image_siamese1, (input_dims[0], input_dims[1]), interpolation=interp1)
            cropped_image_siamese2_resized = cv2.resize(cropped_image_siamese2, (input_dims[0], input_dims[1]), interpolation=interp2)
            # cv2.imwrite(r'D:\Monika\VideosTest\Vozila\brzini_test_ramki\results\\' + im_names_vehicle[i], cropped_image_siamese1_resized)
            # cv2.imwrite(r'D:\Monika\VideosTest\Vozila\brzini_test_ramki\results\\' + im_names_vehicle[i+1],cropped_image_siamese2_resized)

            # cv2.imshow('siamese_48', cropped_image_siamese2_resized)
            # cv2.waitKey(0)
            #siamese predict here
            #normalize siamese input pair of images
            cropped_image_siamese1_resized = cv2.cvtColor(cropped_image_siamese1_resized, cv2.COLOR_BGR2GRAY)
            cropped_image_siamese2_resized = cv2.cvtColor(cropped_image_siamese2_resized, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("a",cropped_image_siamese1_resized)
            # cv2.imshow('prva',cropped_image_siamese1_resized)
            # cv2.waitKey(0)
            # cv2.imshow('vtora', cropped_image_siamese2_resized)
            # cv2.waitKey(0)

            cropped_image_siamese1_resized = cropped_image_siamese1_resized / 255.0
            cropped_image_siamese2_resized = cropped_image_siamese2_resized / 255.0


            cropped_image_siamese1_reshaped = np.expand_dims(cropped_image_siamese1_resized, axis=0)
            cropped_image_siamese2_reshaped = np.expand_dims(cropped_image_siamese2_resized, axis=0)


            network = load_branch_structure(src_branch_structure_path)
            model = load_branch_structure(src_model_structure_path)
            # model = siamese_both_build(input_dims, network)

            model.load_weights(src_weights_path1)  # load weights

            delta = model.predict([cropped_image_siamese1_reshaped, cropped_image_siamese2_reshaped]) # preds_cls,


            if scale2 < 0.8:
                y_max_new_c=bbox2_original[2]-(scale2*delta-2)
            elif scale2 < 0.9:
                y_max_new_c = bbox2_original[2] - (scale2 * delta - 1)
            elif scale2 < 1:
                y_max_new_c=bbox2_original[2]-(scale2*delta-0.5)
            else:
                y_max_new_c = bbox2_original[2] - (scale2 * delta)
            #

            print(f'delta:{delta}')
            print(f'scale2:{scale2}')

            print(f'ymaxold:{bbox2_original[2]}')
            print(f'ymaxnew:{y_max_new_c}')

            point1_x_corrected = int((bbox1_original[1] + bbox1_original[3]) / 2)

            # if i != 0:
            #     point1_y_corrected = corrected_ymax
            # else:
            point1_y_corrected = bbox1_original[2] #square_bbox1[2]

            proj_point1_corrected=homography.image2ProjPoint([point1_x_corrected,point1_y_corrected],h)

            point2_x_corrected = int((bbox2_original[1]+bbox2_original[3])/2)
            proj_point2_corrected=homography.image2ProjPoint([point2_x_corrected,y_max_new_c],h)

            corrected_ymax=y_max_new_c
            speed_corrected = (((proj_point1_corrected[1] - proj_point2_corrected[1])) / t)*3.6
            # if scale2 >= 1:
            speeds_corrected.append(speed_corrected)

            point1_x = ((bbox1_original[1] + bbox1_original[3]) / 2)
            point1_x = round(point1_x)
            point1_y = round(bbox1_original[2])
            proj_point1 = homography.image2ProjPoint([point1_x, point1_y], h)

            point2_x = ((bbox2_original[1] + bbox2_original[3]) / 2)
            point2_x = round(point2_x)
            point2_y = round(bbox2_original[2])
            proj_point2 = homography.image2ProjPoint([point2_x, point2_y], h)

            speed = ((proj_point1[1] - proj_point2[1]) / t) * 3.6
            speeds.append(speed)


            cv2.circle(im2, (point2_x, point2_y), color=(0, 0, 255), thickness=1,radius=3)
            cv2.circle(im2, (point2_x_corrected, y_max_new_c), color=(0, 255, 0), thickness=1,radius=3)
            cv2.circle(im2, (int(x_max_center_gt2), int(y_max_center_gt2)), color=(0, 255, 255), thickness=1,radius=3)
            cv2.rectangle(im2, (bbox2_original[1], bbox2_original[0]), (bbox2_original[3],bbox2_original[2]), color=(0, 0, 255), thickness=1)
            print("error gt - corrected: ", int(y_max_center_gt2) - y_max_new_c)
            cv2.imwrite(r'D:\Monika\VideosTest\Vozila\brzini_test_ramki\results\\' + im_names_vehicle[i+1],im2)

            # cv2.imshow('old_new', im2)
            # cv2.waitKey(0)
            #


            #calculate gt speeds
            proj_point1_gt = homography.image2ProjPoint([x_max_center_gt1, y_max_center_gt1], h)
            proj_point2_gt = homography.image2ProjPoint([x_max_center_gt2, y_max_center_gt2], h)
            speed_gt=(((proj_point1_gt[1]-proj_point2_gt[1])) /t)*3.6
            speeds_gt.append(speed_gt)
            # bboxes=anchorless_genertor_plot.save_results_anchorless_limits_cls1(resultsPath, x_test_all, images_names_all,
            #                                                          output_cls, output_reg, anchor_stride,
            #                                                          prob_thr,
            #                                                          reg_norm_coef_position, reg_norm_coef_size,
            #                                                          flag_normalizeAll, flag_save_coords,
            #                                                          flag_after_regressor)

        avg_speed=statistics.mean(speeds)
        avg_speed_corrected=statistics.mean(speeds_corrected)
        avg_speed_gt=statistics.mean(speeds_gt)
        print("Average speed detections: ",avg_speed)
        print("Average speed regressor corrected: ",avg_speed_corrected)
        print("average speed ground truth: ", avg_speed_gt)


