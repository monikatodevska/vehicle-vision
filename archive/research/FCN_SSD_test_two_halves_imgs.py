"""
Course: Mashinski vid, FEEIT, Spring 2021
Date: 19.04.2022

Description: apply a fully convolutional SSD architecture for object classification and localization to test data
Python version: 3.6
"""
import os

# from form_gt_proba import im_size

# from FCN_SSD_test_video import dir_name

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# python imports

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from keras.optimizers import Adam

# custom package imports
import helper_model, helper_data, helper_losses
# from DataAnalysis.generate_plot_gt_novo_statsOnly import img_dims

#import anchorless_genertor_plot
# import pywt
#import helper_model1

# NOTE da se menuva flagot normalizeAll vo zavisnost od normalizacijata iskoristena pri dobivanje na modelot

# --- flags ---
flag_save_intermediate_output = False
flag_normalizeAll = True

# --- paths ---
# version = 'vinf_400_cls_reg_finetune_cel_so_nokni_samrak_kamera2_vtor_del'
version = r'guzva_test_pola_slika_kam33_merged_filtered_overlap_3vid_189_277_correct_filtered_w_nms'
# NOTE: specify destination paths
# srcImagesPath = r'D:\Monika\VideosTest\Frames_kam23_FP'
# srcImagesPath = r'D:\Monika\VideosTest\Frames_morning_5fps'
# srcImagesPath = r'D:\Monika\VideosTest\Vozila\test_guzva_fp'
# srcImagesPath = r'D:\Monika\VideosTest\Sijamski'
# srcImagesPath = r'D:\Monika\VideosTest\Vozila\obratni_loso_spojuvanje'
# srcImagesPath = r'D:\Monika\Results\za_sijamka_loso_spojuvanje_new_full_hd'
srcImagesPath = r'D:\Monika\Results\guzva_test_pola_slika_kam33_3vid'

src_model_version = r'Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva\new_189'  # NOTE: da se pishe
src_model_version_bottom = r'Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva\new_277'  # NOTE: da se pishe
# srcModelPath = os.path.join(r'C:\Users\User\Desktop\FolderForSharing\MonikaDatasetVehicles\Models',src_model_version)

srcModelPath = os.path.join(r'D:\Monika\Models', src_model_version)
srcModelPath_bottom = os.path.join(r'D:\Monika\Models', src_model_version_bottom)

dstResultsPath = r'D:\Monika\Results'
file_path_reg_coef = os.path.join(r'D:\Monika\Models', src_model_version, 'reg_coef')

dir_names = os.listdir(srcImagesPath)
# napraveni=os.listdir(os.path.join(dstResultsPath,version))
# parameters

# imgDims = {'rows': 171, 'cols': 256}
# imgDims = {'cols': 536, 'rows': 176}
# imgDims = {'cols': 536, 'rows': 216}
imgDims = {'cols': 512, 'rows': 189}
imgDims_bottom = {'cols': 512, 'rows': 277}
# imgDims = {'cols': 536, 'rows': 268}
# upper_roi_orig = (160, 650, 390, 1400)
# upper_roi_orig=(160,640, 360,1550)
upper_roi_orig=(94,500, 400,1500)
overlap=296 #px
bottom_cut_min_r=upper_roi_orig[1]-overlap
bottom_cut_height=1080-bottom_cut_min_r

# imgDims = {'rows': 1080, 'cols': 1920}
img_depth = 1
anchor_stride = 8
img_dims = (imgDims['cols'], imgDims['rows'], img_depth)
img_dims_bottom = (imgDims_bottom['cols'], imgDims_bottom['rows'], img_depth)
prob_thr = 0.85
thr_clustering = 0.3
color_small = (0, 255, 0)
color_large = (0, 0, 255)
color_nevozilo = (255, 0, 0)
colors_list = (color_small, color_large, color_nevozilo)
flag_same_dims = False
wavelet = False
num_classes = 4
coef_bottom=341/img_dims_bottom[1]
coef_upper=341/img_dims[1]
def read_images(im_path, im_size,im_size_bottom,dir_name,upper_roi_orig,bottom_cut_min_r):
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
    # images_list=[]
    # images_list_color=[]
    crop_left_list=[]
    crop_width_list=[]
    full_hd_list=[]

    images_names=[]

    filenames=[x for x in os.listdir(im_path) if x.endswith('.bmp')]

    filenames=sorted(filenames)

    images_upper_color = []  # NEW
    images_upper_gray = []  # NEW
    images_base_gray = []
    images_base_color = []
    images_orig_color=[]

    for im_name in filenames:

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
        else:
            cropped_image = image
            crop_left_px = 0
            crop_width_px = cols
        images_orig_color.append(cropped_image)
        base_cropped=cropped_image[bottom_cut_min_r:,:]
        resized_base_color = cv2.resize(
            base_cropped, (im_size_bottom[0], im_size_bottom[1]), interpolation=cv2.INTER_AREA
        )
        resized_base_gray = cv2.cvtColor(
            resized_base_color, cv2.COLOR_BGR2GRAY
        )
        # cv2.imshow('croppbase', base_cropped)
        # cv2.waitKey(0)
        # cv2.imshow('resbase', resized_base_color)
        # cv2.waitKey(0)
        images_base_color.append(resized_base_color)
        images_base_gray.append(resized_base_gray)
        # --------------------------------
        # UPPER CUT image (NEW)
        # --------------------------------
        roi_h, roi_w = cropped_image.shape[:2]

        if roi_w >= im_size[0] and roi_h >= im_size[1]:
            upper_cut = cropped_image[upper_roi_orig[0]:upper_roi_orig[1],upper_roi_orig[2]:upper_roi_orig[3]]
            upper_cut = cv2.resize(
                upper_cut, (im_size[0], im_size[1]), interpolation=cv2.INTER_AREA
            )

            upper_gray = cv2.cvtColor(upper_cut, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('sl_up',upper_cut)
            # cv2.waitKey(0)

            images_upper_color.append(upper_cut)
            images_upper_gray.append(upper_gray)
        else:
            images_upper_color.append(None)
            images_upper_gray.append(None)

        # --------------------------------
        # Metadata
        # --------------------------------
        images_names.append(im_name[:-4] + '_' + dir_name + '.bmp')
        crop_left_list.append(crop_left_px)
        crop_width_list.append(crop_width_px)

    return (
        images_base_gray,
        images_base_color,
        images_upper_gray,
        images_upper_color,
        images_orig_color,
        images_names,
        crop_left_list,
        crop_width_list,
        full_hd_list
    )
#
# def decode_anchorless(
#     output_cls, output_reg, im_ind,
#     img_dims, anchor_stride, prob_thr,
#     reg_norm_coef_position_rows,
#     reg_norm_coef_position_cols,
#     reg_norm_coef_size_height,
#     reg_norm_coef_size_width,
#     skip_class=2
# ):
#     H, W = img_dims
#     start_r = int(anchor_stride / 2)
#     start_c = int(anchor_stride / 2)
#
#     dets = []
#
#     cls_map = output_cls[im_ind, :, :, :-1]
#     cls_map = (cls_map >= prob_thr).astype(np.uint8)
#
#     r_idx, c_idx, d_idx = np.where(cls_map > 0)
#
#     for i in range(len(r_idx)):
#         cls = d_idx[i]
#         if cls == skip_class:
#             continue
#
#         r, c = r_idx[i], c_idx[i]
#
#         center_r = r * anchor_stride + start_r
#         center_c = c * anchor_stride + start_c
#
#         delta_r = output_reg[im_ind, r, c, 0] * reg_norm_coef_position_rows
#         delta_c = output_reg[im_ind, r, c, 1] * reg_norm_coef_position_cols
#         h = output_reg[im_ind, r, c, 2] * H
#         w = output_reg[im_ind, r, c, 3] * reg_norm_coef_size_width * W
#
#         min_r = int(center_r - h / 2 + delta_r)
#         min_c = int(center_c - w / 2 + delta_c)
#         max_r = int(min_r + h)
#         max_c = int(min_c + w)
#
#         min_r = max(0, min_r)
#         min_c = max(0, min_c)
#         max_r = min(H, max_r)
#         max_c = min(W, max_c)
#
#         if max_r > min_r and max_c > min_c:
#             dets.append([min_r, min_c, max_r, max_c, cls])
#
#     return dets
# def save_results_dual_pipeline(
#     results_path,
#     images_base, images_upper,
#     images_color_fullhd,
#     output_cls_base, output_reg_base,
#     output_cls_upper, output_reg_upper,
#     crop_left_list, crop_width_list,
#     upper_roi_base,  # (r1, r2, c1, c2) in BASE resized coords
#     images_names,
#     anchor_stride, prob_thr,
#     reg_norm_coef_position_rows,
#     reg_norm_coef_position_cols,
#     reg_norm_coef_size_height,
#     reg_norm_coef_size_width,
#     colors_list,
#     flag_save_coords_full_hd=True
# ):
#     """
#     Dual pipeline:
#     - BASE detections: whole cropped image resized to 341x512
#     - UPPER detections: upper ROI resized to 341x512
#     """
#
#     resized_h, resized_w = images_base[0].shape[:2]
#     orig_h, orig_w = 1080, 1920
#
#     r1_u, r2_u, c1_u, c2_u = upper_roi_base
#
#     for im_ind in range(len(images_base)):
#
#         # ---------- scales ----------
#         scale_x = crop_width_list[im_ind] / resized_w
#         scale_y = orig_h / resized_h
#
#         # ---------- decode ----------
#         dets_base = decode_anchorless(
#             output_cls_base, output_reg_base, im_ind,
#             (resized_h, resized_w),
#             anchor_stride, prob_thr,
#             reg_norm_coef_position_rows,
#             reg_norm_coef_position_cols,
#             reg_norm_coef_size_height,
#             reg_norm_coef_size_width
#         )
#
#         dets_upper = decode_anchorless(
#             output_cls_upper, output_reg_upper, im_ind,
#             (resized_h, resized_w),
#             anchor_stride, prob_thr,
#             reg_norm_coef_position_rows,
#             reg_norm_coef_position_cols,
#             reg_norm_coef_size_height,
#             reg_norm_coef_size_width
#         )
#
#         # ---------- filter BASE: remove boxes inside upper ROI ----------
#         dets_base_filtered = []
#         for b in dets_base:
#             br1, bc1, br2, bc2, cls = b
#             if not (br2 <= r2_u and br1 >= r1_u and
#                     bc2 <= c2_u and bc1 >= c1_u):
#                 dets_base_filtered.append(b)
#
#         # ---------- map UPPER detections into BASE coords ----------
#         dets_upper_mapped = []
#         for b in dets_upper:
#             ur1, uc1, ur2, uc2, cls = b
#             dets_upper_mapped.append([
#                 ur1 + r1_u,
#                 uc1 + c1_u,
#                 ur2 + r1_u,
#                 uc2 + c1_u,
#                 cls
#             ])
#
#         # ---------- merge ----------
#         dets_merged = dets_base_filtered + dets_upper_mapped
#
#         # ---------- draw + map to Full HD ----------
#         img_draw = images_base[im_ind].copy()
#         coords_fullhd = []
#
#         for r1, c1, r2, c2, cls in dets_merged:
#             cv2.rectangle(
#                 img_draw,
#                 (c1, r1),
#                 (c2, r2),
#                 colors_list[cls],
#                 1
#             )
#
#             full_r1 = int(r1 * scale_y)
#             full_r2 = int(r2 * scale_y)
#             full_c1 = int(c1 * scale_x + crop_left_list[im_ind])
#             full_c2 = int(c2 * scale_x + crop_left_list[im_ind])
#
#             coords_fullhd.append([full_r1, full_c1, full_r2, full_c2, cls])
#
#         # ---------- save ----------
#         name = images_names[im_ind][:-4]
#         cv2.imwrite(f"{results_path}/{name}_merged.png", img_draw)
#
#         if flag_save_coords_full_hd and len(coords_fullhd):
#             np.savetxt(
#                 f"{results_path}/{name}.txt",
#                 np.array(coords_fullhd),
#                 fmt="%i",
#                 delimiter=","
#             )
def box_center(box):
    min_r, min_c, max_r, max_c = box[:4]
    return 0.5 * (min_r + max_r), 0.5 * (min_c + max_c)

def compute_iou(box1, box2):
    min_r1, min_c1, max_r1, max_c1 = box1[:4]
    min_r2, min_c2, max_r2, max_c2 = box2[:4]

    inter_min_r = max(min_r1, min_r2)
    inter_min_c = max(min_c1, min_c2)
    inter_max_r = min(max_r1, max_r2)
    inter_max_c = min(max_c1, max_c2)

    inter_h = max(0, inter_max_r - inter_min_r)
    inter_w = max(0, inter_max_c - inter_min_c)
    inter_area = inter_h * inter_w

    area1 = (max_r1 - min_r1) * (max_c1 - min_c1)
    area2 = (max_r2 - min_r2) * (max_c2 - min_c2)

    union = area1 + area2 - inter_area

    if union == 0:
        return 0

    return inter_area / union

def cluster_boxes_iou(boxes, iou_thr=0.3):
    clusters = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        cluster = [boxes[i]]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue

            if compute_iou(boxes[i], boxes[j]) >= iou_thr:
                cluster.append(boxes[j])
                used[j] = True

        clusters.append(cluster)

    return clusters
def filter_overlap_boxes_clustered(
    boxes,
    overlap_min_r,
    overlap_max_r,
    source,      # "upper" or "bottom"
    iou_thr=0.3
):
    clusters = cluster_boxes_iou(boxes, iou_thr)

    final_boxes = []

    for cluster in clusters:
        discard_cluster = False

        for box in cluster:
            min_r, min_c, max_r, max_c = box[:4]

            if source == "upper":
                if max_r >= overlap_max_r:
                    discard_cluster = True
                    break

            elif source == "bottom":
                if min_r <= overlap_min_r:
                    discard_cluster = True
                    break

        if not discard_cluster:
            final_boxes.extend(cluster)

    return final_boxes

def filter_overlap_boxes(
    boxes,
    overlap_min_r,
    overlap_max_r,
    source,      # "upper" or "bottom"
):
    touching_overlap = False

    for box in boxes:
        min_r, min_c, max_r, max_c = box[:4]

        if source == "upper":
            # touches bottom of overlap
            if max_r >= overlap_max_r:
                touching_overlap = True
                break

        elif source == "bottom":
            # touches top of overlap
            if min_r <= overlap_min_r:
                touching_overlap = True
                break

    # If at least one box touches → ignore ALL
    if touching_overlap:
        return []

    # If none touches → keep ALL
    return boxes

def draw_bboxes(
    coords_base,
    coords_upper,
    img_orig_color,
    results_path,
    im_name,
    colors_list=None,
    thickness=2
):
    """
    Draw base and upper bounding boxes on original image and save it.

    coords_base / coords_upper:
        [min_r, min_c, max_r, max_c, class_id]

    img_orig_color:
        original BGR image (OpenCV)

    colors_list:
        list or dict mapping class_id -> (B, G, R)
    """

    img = img_orig_color.copy()

    # ---- draw BASE boxes ----
    for box in coords_base:
        min_r, min_c, max_r, max_c, cls = box
        color = colors_list[cls] if colors_list is not None else (0, 255, 0)

        cv2.rectangle(
            img,
            (int(min_c), int(min_r)),
            (int(max_c), int(max_r)),
            color,
            thickness
        )

    # ---- draw UPPER boxes ----
    for box in coords_upper:
        min_r, min_c, max_r, max_c, cls = box
        color = colors_list[cls] if colors_list is not None else (0, 0, 255)

        cv2.rectangle(
            img,
            (int(min_c), int(min_r)),
            (int(max_c), int(max_r)),
            color,
            thickness
        )

    # ---- save ----
    os.makedirs(results_path, exist_ok=True)
    out_path = os.path.join(results_path, f"{im_name}_merged.png")
    cv2.imwrite(out_path, img)

    return out_path

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


def findelement(maxnum, cluster):
    output = []
    for item in cluster:
        if item[0] == maxnum:
            output.append(item)

    if len(output) == 0:
        print("Imas bug vo baranjeto maximum")
        exit(1)
    return output


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
def nms_new(image, img_dest, lista, thr, colors_list, num_classes, flag_save_coords):
    # print(img_dest)
    coords_list = []

    finalwindows = []
    # image_orig=deepcopy(image)
    if len(lista) == 0:
        cv2.imwrite(os.path.join(img_dest), image)
        return

    all_clusters = getclustersforanchors(lista, thr)

    for cluster in all_clusters:
        # print(cluster)
        maxwindows = []
        maxwindows_new=[]

        if len(cluster) == 0:
            continue

        elements = [item[2] - item[0] for item in cluster]

        if len(elements) <= 2:
            first2max = sorted(elements)
        else:
            # first2max = sorted(elements)[len(elements) - 2:len(elements)]
            first2max = sorted(elements)[:]

        first2max_noduplicates = []
        for i in first2max:
            if i not in first2max_noduplicates:
                first2max_noduplicates.append(i)

        for maxnum in first2max_noduplicates:
            indices = [i for i, x in enumerate(elements) if x == maxnum]

            for idx in indices:
                maxwindows.append(cluster[idx])


        bboxes_per_class = [[] for i in range(num_classes)]     # bounding boxes grouped by class

        for maxwindow in maxwindows:
            bboxes_per_class[maxwindow[-1]].append(maxwindow)

        max_class = max((len(l), i) for i, l in enumerate(bboxes_per_class))[1]

        for maxwindow in maxwindows:
            if maxwindow[-1]==max_class:
                maxwindows_new.append(maxwindow)
        color = colors_list[max_class]

        finalwindow = average_window(maxwindows_new)
        finalwindows.append(finalwindow)
        # if nacrtaj:
        #     cv2.rectangle(image, (int(finalwindow[1]*(1620/512))+left_crop, int(finalwindow[0]*(1080/341))), (int(finalwindow[3]*(1620/512))+left_crop, int(finalwindow[2]*(1080/341))), color, thickness=1)
        cv2.rectangle(image, (int(finalwindow[1]), int(finalwindow[0])), (int(finalwindow[3]), int(finalwindow[2])), color, thickness=1)

        coords_list.append([int(finalwindow[0]), int(finalwindow[1]), int(finalwindow[2]), int(finalwindow[3]), max_class])
        # coords_list.append([int(finalwindow[0]*(1080/341)), int(finalwindow[1]*(1620/512))+left_crop, int(finalwindow[2]*(1080/341)), int(finalwindow[3]*(1620/512))+left_crop, max_class])
    # if not flag_choose:
    # cv2.imshow('sl',image)
    # cv2.waitKey(0)
    cv2.imwrite(img_dest, image)
    # cv2.imwrite(os.path.join(img_dest), image_orig[:-4]+'_ORIG.bmp')

    if flag_save_coords:
        np.savetxt(img_dest[:-4] + '.txt', coords_list, delimiter=',', fmt='%i')



    # return
def save_results_dual_debug(results_path,
                            images_base_color,
                            images_upper_color,
                            images_orig_color,
                            im_size_resized, im_size_resized_bottom,
                            output_base_cls, output_base_reg,
                            output_upper_cls, output_upper_reg,
                            bottom_cut_height,
                            bottom_cut_min_r,
                            upper_roi_orig,  # (min_r, max_r, min_c, max_c) in CROP original image
                            crop_width_list,
                            images_names,
                            anchor_stride,
                            prob_thr,
                            colors_list,
                            flag_save_coords=True):
    """
    Debug saving for dual pipeline:
    - upper ROI detections separately
    - bottom/base detections separately
    upper_roi_orig: tuple in original cropped image coordinates (not resized)
    """
    resized_h, resized_w = im_size_resized[1],im_size_resized[0]  # 341x512
    resized_h_bottom, resized_w_bottom = im_size_resized_bottom[1],im_size_resized_bottom[0]  # 341x512
    orig_h, orig_w = 1080, crop_width_list[0]  # height of cropped image, width of cropped image
    overlap_min_r=bottom_cut_min_r
    overlap_max_r=upper_roi_orig[1]
    # scale upper ROI to detection image
    r1_u, r2_u, c1_u, c2_u = upper_roi_orig

    #print(f"Upper ROI scaled to detection image: {r1_u_scaled},{r2_u_scaled},{c1_u_scaled},{c2_u_scaled}")

    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))
    scale_y=bottom_cut_height/resized_h_bottom
    scale_x=crop_width_list[0]/resized_w_bottom
    for idx, im_name in enumerate(images_names):
        # img_base_only = images_base_color[idx].copy()
        # img_upper_only = images_base_color[idx].copy()  # draw upper ROI detections on base image for visualization
        img_orig_color=images_orig_color[idx].copy()
        img_orig_color_tmp=images_orig_color[idx].copy()
        img_orig_color_nms=images_orig_color[idx].copy()
        img_upper_color=images_upper_color[idx].copy()
        img_base_color= images_base_color[idx].copy()
        # ---- BASE detections outside upper ROI ----
        coords_base = []
        res_base = output_base_cls[idx, :, :, :-1]
        r_base, c_base, d_base = np.where(res_base >= prob_thr)
        for i in range(len(r_base)):
            # skip if inside scaled upper ROI
            # if r1_u_scaled <= r_base[i] < r2_u_scaled and c1_u_scaled <= c_base[i] < c2_u_scaled:
            #     continue
            center_row = r_base[i] * anchor_stride + start_r
            center_col = c_base[i] * anchor_stride + start_c

            delta_r = output_base_reg[idx, r_base[i], c_base[i], 0]
            delta_c = output_base_reg[idx, r_base[i], c_base[i], 1]
            h = output_base_reg[idx, r_base[i], c_base[i], 2]*resized_h_bottom*coef_bottom
            w = output_base_reg[idx, r_base[i], c_base[i], 3]*resized_w_bottom

            min_r = int(center_row - h / 2 + delta_r)
            min_c = int(center_col - w / 2 + delta_c)
            max_r = int(min_r + h)
            max_c = int(min_c + w)

            min_r = int((min_r *scale_y)+bottom_cut_min_r)
            max_r = int((max_r *scale_y)+bottom_cut_min_r)
            min_c = int((min_c *scale_x))
            max_c = int((max_c *scale_x))
            coords_base.append([min_r, min_c, max_r, max_c, d_base[i]])
            cv2.rectangle(img_orig_color_tmp, (min_c, min_r), (max_c, max_r),(0,255,0), 1)
        # cv2.imshow('sl',img_base_color)
        # cv2.waitKey(0)
        # ---- UPPER ROI detections ----
        coords_upper = []
        if images_upper_color:
            res_upper = output_upper_cls[idx, :, :, :-1]
            r_upper, c_upper, d_upper = np.where(res_upper >= prob_thr)
            img_upper_h=r2_u-r1_u
            img_upper_w = c2_u-c1_u
            scale_y_u=img_upper_h/resized_h
            scale_x_u=img_upper_w/resized_w

            for i in range(len(r_upper)):
                center_row = r_upper[i] * anchor_stride + start_r
                center_col = c_upper[i] * anchor_stride + start_c

                delta_r = output_upper_reg[idx, r_upper[i], c_upper[i], 0]
                delta_c = output_upper_reg[idx, r_upper[i], c_upper[i], 1]
                h = output_upper_reg[idx, r_upper[i], c_upper[i], 2] *resized_h*coef_upper
                w = output_upper_reg[idx, r_upper[i], c_upper[i], 3] * resized_w

                min_r = int(center_row - h / 2 + delta_r)
                min_c = int(center_col - w / 2 + delta_c)
                max_r = int(min_r + h)
                max_c = int(min_c + w)

                # map to original cropped image coords
                min_r = int((min_r*scale_y_u)+r1_u)
                max_r = int((max_r *scale_y_u)+r1_u)
                min_c = int((min_c *scale_x_u)+c1_u)
                max_c = int((max_c *scale_x_u)+c1_u)

                coords_upper.append([min_r, min_c, max_r, max_c, d_upper[i]])
                cv2.rectangle(img_orig_color_tmp, (min_c, min_r), (max_c, max_r), (255,0,0), 1)
            # cv2.imshow('s', img_upper_color)
            # cv2.waitKey(0)
        # ---- Save for debug ----
        # cv2.imwrite(os.path.join(results_path, im_name + '_base_only.png'), img_orig_color)
        cv2.imwrite(os.path.join(results_path, im_name + '_bmerged.png'), img_orig_color_tmp)
        #FILTER DETECTIONS
        coords_bottom=filter_overlap_boxes_clustered(coords_base,overlap_min_r,overlap_max_r,source='bottom')
        coords_upper=filter_overlap_boxes_clustered(coords_upper,overlap_min_r,overlap_max_r,source='upper')
        all_dets=coords_upper+coords_bottom
        # coords_base = filter_overlap_boxes(
        #     coords_base,
        #     overlap_min_r,
        #     overlap_max_r,
        #     source="bottom",
        #     margin=50
        # )
        # coords_upper = filter_overlap_boxes(
        #     coords_upper,
        #     overlap_min_r,
        #     overlap_max_r,
        #     source="upper",
        #     margin=50
        # )

        draw_bboxes(coords_bottom,coords_upper,img_orig_color,results_path+r'\filter',im_name)
        if len(all_dets)>0:
            nms_new(img_orig_color_nms, os.path.join(results_path, im_name + '_nms.png'),
                                               all_dets, thr_clustering, colors_list, num_classes,
                                                flag_save_coords=False)
        # if flag_save_coords:
        #     # np.savetxt(os.path.join(results_path, im_name + '_base_only.txt'), np.array(coords_base), fmt='%i', delimiter=',')
        #     np.savetxt(os.path.join(results_path, im_name + '_upper_only.txt'), np.array(coords_upper), fmt='%i', delimiter=',')



for dir_name in dir_names:

    if 'kamion' in dir_name or 'tmp' in dir_name:
        continue
    print(dir_name)

    if not os.path.exists(os.path.join(dstResultsPath, version, dir_name)):
        os.makedirs(os.path.join(dstResultsPath, version, dir_name))
    resultsPath = os.path.join(dstResultsPath, version, dir_name)

    results_path_nms = os.path.join(dstResultsPath, version, dir_name + '_postprocessing')
    if not os.path.exists(results_path_nms):
        os.mkdir(results_path_nms)

    os.makedirs(resultsPath + r'\filter', exist_ok=True)

    # --- load and format data ---
    # load full dataset into memory - image data and labels
    srcImagesPathNew = os.path.join(srcImagesPath, dir_name)
    # x_test, x_test_color, images_names, crop_left_list, crop_width_list, full_hd_list = helper_data.read_data_test(
    #     srcImagesPathNew, (imgDims['cols'], imgDims['rows']), img_depth, dir_name, flag_same_dims)

    images_base_gray,images_base_color,images_upper_gray,images_upper_color,images_orig_color,images_names,crop_left_list,crop_width_list,full_hd_list=read_images(srcImagesPathNew, (imgDims['cols'], imgDims['rows']),(imgDims_bottom['cols'], imgDims_bottom['rows']),dir_name,upper_roi_orig,bottom_cut_min_r )

    x_test_new = []

    x_test_nn = []


    model_bottom = helper_model.load_model(model_path=os.path.join(srcModelPath_bottom,  'model.json'),
                                    weights_path=os.path.join(srcModelPath_bottom,
                                                              'model.h5'))  # build model architecture

    # print(model.summary())
    def prepare_input(x_gray):
        x = np.array(x_gray, dtype=np.float64)
        x = x.reshape(x.shape + (1,))
        return x


    # colors_list = [
    #     (0, 255, 0),  # class 0: green
    #     (0, 0, 255),  # class 1: red
    #     (255, 0, 0),  # class 2: blue (if you have a third class)
    # ]
    #BASE
    x_base = prepare_input(images_base_gray)
    output_base = model_bottom.predict(x_base, verbose=1)

    output_base_cls = output_base[:, :, :, :num_classes]
    output_base_cls_pos = output_base_cls[:, :, :, :-1]
    output_base_reg = output_base[:, :, :, num_classes:]

    #UPPER
    valid_upper_idx = [i for i, img in enumerate(images_upper_gray) if img is not None]
    images_upper_gray_valid = [images_upper_gray[i] for i in valid_upper_idx]
    images_upper_color_valid = [images_upper_color[i] for i in valid_upper_idx]
    images_names_upper_valid = [images_names[i] for i in valid_upper_idx]
    crop_left_list_upper_valid = [crop_left_list[i] for i in valid_upper_idx]
    crop_width_list_upper_valid = [crop_width_list[i] for i in valid_upper_idx]
    full_hd_list_upper_valid = [full_hd_list[i] for i in valid_upper_idx]
    x_upper_valid = [images_upper_gray[i] for i in valid_upper_idx]
    x_upper = prepare_input(x_upper_valid)

    model = helper_model.load_model(model_path=os.path.join(srcModelPath,  'model.json'),
                                    weights_path=os.path.join(srcModelPath,
                                                              'model.h5'))  # build model architecture
    output_upper = model.predict(x_upper, verbose=1)
    reg_norm_coef_position_rows,reg_norm_coef_position_cols,reg_norm_coef_size_height,reg_norm_coef_size_width,=1,1,1,1
    output_upper_cls = output_upper[:, :, :, :num_classes]
    output_upper_cls_pos = output_upper_cls[:, :, :, :-1]
    output_upper_reg = output_upper[:, :, :, num_classes:]
    # upper_roi_orig=(160,540, 360,1550)
    save_results_dual_debug(resultsPath,
                            images_base_color,
                            images_upper_color,images_orig_color, img_dims,img_dims_bottom,
                            output_base_cls, output_base_reg,
                            output_upper_cls, output_upper_reg,
                            bottom_cut_height,bottom_cut_min_r,
                            upper_roi_orig,  # (min_r, max_r, min_c, max_c) in CROP original image
                            crop_width_list,
                            images_names,
                            anchor_stride,
                            prob_thr,
                            colors_list,
                            flag_save_coords=True)


    #
    # # --- plot ground truth network output ---
    # results_kamioni = os.path.join(dstResultsPath, version)
    # if not os.path.exists(os.path.join(results_kamioni, 'kamioni')):
    #     os.mkdir(os.path.join(results_kamioni, 'kamioni'))
    # results_kamioni1 = os.path.join(results_kamioni, 'kamioni')
    #
    # reg_norm_coef_position_rows, reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width = 1, 1, 1, 1
    #
    #
    # anchorless_genertor_plot.save_results_anchorless_limits(resultsPath, results_path_nms, results_kamioni1, x_test,
    #                                                         x_test_color, crop_left_list, crop_width_list,
    #                                                         full_hd_list, images_names, output_cls, output_reg,
    #                                                         anchor_stride, prob_thr, reg_norm_coef_position_rows,
    #                                                         reg_norm_coef_position_cols, reg_norm_coef_size_height,
    #                                                         reg_norm_coef_size_width, thr_clustering,
    #                                                         colors_list, left_crop=0, nacrtaj=False,
    #                                                         flag_save_coords=False, flag_save_coords_full_hd=True,
    #                                                         flag_choose=False)
    #
