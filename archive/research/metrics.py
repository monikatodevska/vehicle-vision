# NOTE Metrics for regressor output before nms, inside an interval (points, different for each camera) where speed is measured

import os

from tensorflow.python.ops.metrics_impl import precision

# import numpy as np
# import shutil
import helper_postprocessing
import cv2

# from FCN_SSD_testingFrames import flag_nms

# import matplotlib.pyplot as plt

# from collections import Counter
# import plotly.express as px
# import pandas as pd
#
# from copy import deepcopy

# results_path_reg = r'D:\Monika\Results\TEST_ALL_CAMERAS2\reg'
# results_path_class = r'D:\Monika\Results\TEST_ALL_CAMERAS2\class'
# file_path_r = r'D:\Monika\Results\stats'
# annot_path = r'D:\Monika\VideosTest\All_Cameras_annnotated\Annotations\Renamed'
# predictions=r'D:\Monika\Results\vinf_1000_SO_DVA_INCEPTION_SKIP_48_normalizationWithMax_0.45_0.7_prodolzenie_finetune_morning_10fps_anotirani_postprocessing'
# predictions=r'D:\Monika\Results\v1300_SO_DVA_INCEPTION_SKIP_48_0.6_0.7_normall_NOVO_ANOTIRANI_posle_reg1'
# predictions_class=r'D:\Monika\Results\v1300_SO_DVA_INCEPTION_SKIP_48_0.6_0.7_normall_NOVO_ANOTIRANI_posle_class1'
# predictions=r'D:\Monika\Results\anotirani\v1700_SO_DVA_INCEPTION_SKIP_48_0.6_0.7_normall_NOVO1-so_inceptions_bez_skip_v2_finetune_ANOTIRANI_posle_reg'
# predictions_class=r'D:\Monika\Results\anotirani\v1700_SO_DVA_INCEPTION_SKIP_48_0.6_0.7_normall_NOVO1-so_inceptions_bez_skip_v2_finetune_ANOTIRANI_posle_class'

# annotations=[x for x in os.listdir(annot_path) if x[-4:]=='.txt']
# print(annotations)


#
#
# HScale=(341/1080)
# WScale=(512/1621)
#
# dirs=os.listdir(results_path_reg)
# # row = 0
# column = 1

region = 9

regions_list = [1, 2, 9]
debug=False


def calculate_stats(results_path_reg, results_path_class, annot_path, file_path_r,region,model_name,best_thr,best_accuracy_flag):
    # dirs = os.listdir(results_path_reg)
    import xlsxwriter
    # from openpyxl import load_workbook
    annots_dirs = os.listdir(annot_path)

    column = 0
    cnt1 = 0
    average_rows = 0
    average_cols = 0
    mae_rows_tmp = []
    mae_cols_tmp = []
    if best_accuracy_flag:

        workbook = xlsxwriter.Workbook(os.path.join(file_path_r, model_name+'_postproc'+'.xlsx'))
        worksheet = workbook.add_worksheet()
    # elif best_accuracy_flag:
    #
    #
    #     # Load the existing workbook
    #     # workbook = load_workbook(os.path.join(file_path_r, 'proba2.xlsx'))
    #
    #     # Select the active sheet (or specify the sheet name)
    #     sheet = workbook.active  # Or workbook["SheetName"]

    HScale_Config = 341 / 512


    # if 'NOVO1' not in dir:
    #     continue
    num_detections = 0
    predictions = os.path.join(results_path_reg)
    predictions_class = os.path.join(results_path_class)
    # print(predictions_class)
    false_negatives = 0
    false_negatives_by_class = []
    false_positives = 0
    true_positives=0
    false_positives_obratni = 0
    # col_diff_all = []
    suma_cols_reg = 0
    suma_rows_reg = 0
    suma_cols_class = 0
    suma_rows_class = 0
    suma_cols_all = 0
    suma_rows_reg_norm_all = 0
    suma_cols_reg_norm_all = 0

    suma_rows_all = 0
    num_sum = 0
    cnt = 0
    golemi = 0
    pairs_all_class = []  # useless??
    pairs_all_general = []
    dict_num_det_per_image = {}
    suma_detekcii_all = 0
    img_height = 341

    HScale = 341 / 1080
    WScale = 512 / 1621

    for annot_dir in annots_dirs:
        annot_path1 = os.path.join(annot_path, annot_dir)
        annotations = [x for x in os.listdir(annot_path1) if x[-4:] == '.txt']

        if 'miladinovci' in annot_dir:
            MaxRowSpeed = 425 * HScale_Config
            MinRowSpeed = 125 * HScale_Config

        elif 'kam32' in annot_dir:
            MinRowSpeed = 90 * HScale_Config
            MaxRowSpeed = 425 * HScale_Config
        elif 'kam40' in annot_dir:
            MinRowSpeed = 95 * HScale_Config
            MaxRowSpeed = 425 * HScale_Config
        elif 'kam42' in annot_dir:
            MinRowSpeed = 60 * HScale_Config
            MaxRowSpeed = 425 * HScale_Config
        elif 'kam44' in annot_dir:
            MinRowSpeed = 60 * HScale_Config
            MaxRowSpeed = 425 * HScale_Config
        elif 'kam48' in annot_dir:
            MinRowSpeed = 75 * HScale_Config
            MaxRowSpeed = 425 * HScale_Config
        elif 'kam33' in annot_dir:
            MinRowSpeed = 95 * HScale_Config
            MaxRowSpeed = 425 * HScale_Config

        elif 'kam46' in annot_dir:
            MinRowSpeed = 90 * HScale_Config
            MaxRowSpeed = 425 * HScale_Config

        elif 'kam36' in annot_dir:
            MinRowSpeed = 185 * HScale_Config
            MaxRowSpeed = 425 * HScale_Config

        if region == 1:  # stats for upper half of ROI
            MaxRowSpeed = int(((MaxRowSpeed - MinRowSpeed) / 2) + MinRowSpeed)
            MinRowSpeed = int(MinRowSpeed)
        elif region == 2:  # stats for bottom half of ROI
            MinRowSpeed = int(((MaxRowSpeed - MinRowSpeed) / 2) + MinRowSpeed)
            MaxRowSpeed = int(MaxRowSpeed)
        elif region == 3:
            MaxRowSpeed = int(((MaxRowSpeed - MinRowSpeed) / 3) + MinRowSpeed)
            MinRowSpeed = int(MinRowSpeed)
        else:
            MinRowSpeed = int(MinRowSpeed)
            MaxRowSpeed = int(MaxRowSpeed)

        for annot_file in annotations:
            # print(annot_file)
            f = open(os.path.join(annot_path1, annot_file), 'r')
            bbox_annots = f.readlines()
            pairs_per_image = []
            bboxes_annots_formatted = []
            bbox_annots = [[int(str(j)) for j in i.split(',')] for i in bbox_annots]
            img = cv2.imread(os.path.join(predictions, annot_file[:-4] + '_' + annot_dir + '_reg.bmp'))
            # print(img.shape)
            bgr_image = img

            for bbox_annot in bbox_annots:
                bbox_annot[2] = bbox_annot[0] + bbox_annot[2]
                bbox_annot[3] = bbox_annot[1] + bbox_annot[3]

                if bbox_annot[4] == 6:
                    continue

                if 'miladinovci' in annot_dir:

                    bbox_annot[0] = int(bbox_annot[0] * HScale)
                    bbox_annot[1] = int(bbox_annot[1] * WScale)
                    bbox_annot[2] = int(bbox_annot[2] * HScale)
                    bbox_annot[3] = int(bbox_annot[3] * WScale)


                elif 'kam32' in annot_dir or 'kam40' in annot_dir or 'kam42' in annot_dir or 'kam44' in annot_dir or 'kam48' in annot_dir:

                    bbox_annot[0] = int(bbox_annot[0] * HScale)
                    bbox_annot[1] = int((bbox_annot[1] - 299) * WScale)
                    bbox_annot[2] = int(bbox_annot[2] * HScale)
                    bbox_annot[3] = int((bbox_annot[3] - 299) * WScale)

                elif 'kam33' in annot_dir:

                    bbox_annot[0] = int(bbox_annot[0] * HScale)
                    bbox_annot[1] = int((bbox_annot[1] - 99) * WScale)
                    bbox_annot[2] = int(bbox_annot[2] * HScale)
                    bbox_annot[3] = int((bbox_annot[3] - 99) * WScale)



                elif 'kam46' in annot_dir:

                    bbox_annot[0] = int(bbox_annot[0] * HScale)
                    bbox_annot[1] = int((bbox_annot[1] - 100) * WScale)
                    bbox_annot[2] = int(bbox_annot[2] * HScale)
                    bbox_annot[3] = int((bbox_annot[3] - 100) * WScale)


                elif 'kam36' in annot_dir:

                    bbox_annot[0] = int(bbox_annot[0] * HScale)
                    bbox_annot[1] = int((bbox_annot[1] - 189) * WScale)
                    bbox_annot[2] = int(bbox_annot[2] * HScale)
                    bbox_annot[3] = int((bbox_annot[3] - 189) * WScale)

                cv2.circle(img, (350, int(MinRowSpeed)), 2, color=(255, 255, 0), thickness=3)

                cv2.circle(img, (350, int(MaxRowSpeed)), 2, color=(255, 255, 0), thickness=3)

                # MinRowSpeed=int(MinRowSpeed)

                bbox_formatted = [bbox_annot[0], bbox_annot[1], bbox_annot[2], bbox_annot[3], bbox_annot[4]]
                bboxes_annots_formatted.append(bbox_formatted)
            # print(os.path.join(predictions,annot_file[:-4]+'_class.txt'))
            # num_lines = sum(1 for line in f if line.rstrip())
            # print('Total lines:', num_lines)
            # print(MaxRowSpeed)
            # print(MinRowSpeed)

            if os.path.exists(os.path.join(predictions, annot_file[:-4] + '_' + annot_dir + '_reg.txt')):
                # print('postopiiiii')
                fp = open(os.path.join(predictions, annot_file[:-4] + '_' + annot_dir + '_reg.txt'))
                fc = open(os.path.join(predictions_class, annot_file[:-4] + '_' + annot_dir + '_class.txt'))

            else:
                cnt += 1
                print('nema')
                for bbox_annot in bboxes_annots_formatted:
                    if bbox_annot[2] > MinRowSpeed and bbox_annot[2] < MaxRowSpeed:
                        false_negatives += 1

                    # if bbox_annot[2]<341/2:
                    #     false_negatives+=1
                # false_negatives+=len(bbox_annots)
                continue

            # num_lines_pred = sum(1 for line in fp if line.rstrip())
            # print('Total lines:', num_lines)
            # if num_lines>=num_lines_pred:

            iou_max = 0

            bbox_preds = fp.readlines()
            bbox_preds_class = fc.readlines()
            # bbox_annots = [[int(str(j)) for j in i.split(',')] for i in bbox_annots]
            bbox_preds = [[int(str(j)) for j in i.split(',')] for i in bbox_preds]
            bbox_preds_class = [[int(str(j)) for j in i.split(',')] for i in bbox_preds_class]
            # num_detections=len(bbox_preds)

            # dict_num_det_per_image[annot_file]=num_detections
            # print(bbox_preds)
            # print(len(bbox_preds))
            # print(len(bbox_preds_class))
            # NOTE ova ako generiranite bboxes od predikciite se min_r, min_c, h,w

            for bbox_pred in bbox_preds:
                # if int((bbox_pred[2]+bbox_pred[0])/2) < int(341/2):

                if bbox_pred[2] > MinRowSpeed and bbox_pred[2] <= MaxRowSpeed:
                    num_detections += 1

            #     bbox_pred[2] = bbox_pred[0] + bbox_pred[2]
            #     bbox_pred[3] = bbox_pred[1] + bbox_pred[3]
            cv2.circle(img, (350, MinRowSpeed), 2, color=(0, 255, 255), thickness=3)
            cv2.circle(img, (250, MaxRowSpeed), 2, color=(0, 255, 255), thickness=3)
            # cv2.imshow('slika',img)
            # cv2.waitKey(0)
            # continue

            for bbox_annot in bboxes_annots_formatted:

                used_preds = []
                used_pred_class = []
                pairs_all = []  # pairs per annotation
                flag_imadet = False
                used_preds_obratni = []
                broi = 0

                iou_max = 0
                pair = []
                # bbox_annot[2] = bbox_annot[0] + bbox_annot[2]
                # bbox_annot[3] = bbox_annot[1] + bbox_annot[3]

                # if annotation is in bootom half of image-continue
                # if int((bbox_annot[2]+bbox_annot[0])/2) > int(341 / 2):
                #     continue

                # print(bbox_annot)
                for ind, bbox_pred in enumerate(bbox_preds):
                    # print(bbox_pred)

                    # if bbox_pred in used_preds:
                    #     continue
                    if bbox_pred[2] < bbox_pred[0] or bbox_pred[3] < bbox_pred[1]:
                        if bbox_pred[2] > MinRowSpeed and bbox_pred[2] < MaxRowSpeed:
                            false_positives_obratni += 1
                            used_preds_obratni.append(bbox_pred)
                        continue

                    iou = helper_postprocessing.calc_iou(bbox_annot[:-1], bbox_pred[:-1])
                    if iou > 0.3:
                        broi += 1
                        pair = [(bbox_annot), (bbox_pred)]
                        pairs_all.append(pair)
                        pairs_per_image.append(pair)
                        pairs_all_general.append(pair)

                        pairs_all_class.append([(bbox_annot), bbox_preds_class[ind]])
                        used_preds.append(bbox_pred)
                        used_pred_class.append(bbox_preds_class[ind])
                        # print('imaaa')
                        # bbox_preds.remove(bbox_pred)
                    elif iou > 0.0 and iou < 0.3:
                        flag_imadet = True


                if broi == 0 and not flag_imadet:
                    # ako centarot po redici e vo gorniot del od slikata i anotacijata nema nitu eden par so predvidenite bboxes=FN
                    # if int((bbox_annot[2]+bbox_annot[0])/2)<int(341/2):
                    if bbox_annot[2] > MinRowSpeed and bbox_annot[2] < MaxRowSpeed:
                        false_negatives_by_class.append(bbox_annot[-1])
                        false_negatives += 1
                        # print('FN')
                    continue

                elif broi == 0 and flag_imadet:
                    print('continue')
                    continue

                # ---
                # ako bile iskoristeni predvidenite bboxes za edna anotacija (spoeni so iou vo pairs_all) togas ne treba da postojat za slednata anotacija vo slikata
                for element in used_preds:
                    # if element in bbox_preds:
                    # print(element)
                    bbox_preds.remove(element)
                for element in used_pred_class:
                    bbox_preds_class.remove(element)

                # ------

                # -----
                # ne presmetuvaj greska ako dolnata tocka od bbox na anotacijata izleguva od regionot od interes
                indexes_to_remove_per_annot = []
                for ind, el in enumerate(pairs_all):
                    if el[0][2] < MinRowSpeed or el[0][2] > MaxRowSpeed:
                        # if int((el[0][2] + el[0][0]) / 2) > int(341 / 2):
                        #     print(el[0])
                        indexes_to_remove_per_annot.append(ind)

                for ind_to_remove in indexes_to_remove_per_annot[::-1]:
                    pairs_all.pop(ind_to_remove)
                # ------
                for element in used_preds_obratni:
                    # if element in bbox_preds:
                    # print(element)
                    bbox_preds.remove(element)
                # print(false_positives)
                # false_positives_by_class.append(bbo)
                # print(f'cntttt:{cnt}')
                if len(pairs_all) == 0:
                    continue
                # for pair in pairs_all:
                #     # print(pair)
                #     cv2.rectangle(img, (pair[1][1], pair[1][0]), (pair[1][3], pair[1][2]), color=(0, 255, 255))
                # cv2.imshow('slika', img)
                # cv2.waitKey(0)
                list_sum_cols = []
                list_sum_rows = []

                list_cols_class = []
                list_rows_class = []
                suma_cols_reg = 0
                suma_rows_reg = 0
                suma_cols_reg_norm = 0
                suma_rows_reg_norm = 0
                visini = []
                shirini = []

                for ind, pair in enumerate(pairs_all):
                    # print(pair[0][3])
                    # za centrite kolku setaat posle regresor

                    half_w_b1 = pair[0][3] - pair[0][1]
                    half_w_b1 = int(round(half_w_b1 / 2))

                    half_h_b1 = pair[0][2] - pair[0][0]
                    half_h_b1 = int(round(half_h_b1 / 2))

                    center_row_b1 = pair[0][2] - half_h_b1
                    center_col_b1 = pair[0][1] + half_w_b1  # b1 e bbox na anotacija

                    half_h_b2 = pair[0][2] - pair[0][0]
                    half_h_b2 = int(round(half_h_b2 / 2))
                    half_w_b2 = pair[1][3] - pair[1][1]
                    half_w_b2 = int(round(half_w_b2 / 2))

                    center_row_b2 = pair[1][2] - half_h_b2
                    center_col_b2 = pair[1][1] + half_w_b2  # b2 e predvideniot bbox posle regresor

                    suma_cols_reg += abs(center_col_b2 - center_col_b1)
                    suma_rows_reg += abs(center_row_b2 - center_row_b1)

                    # Note: centri na izlez od klasifikatorot
                    # pairs_all_class e par (bbox_anotacija, bbox_klasifikator)

                    half_h_class = pairs_all_class[ind][1][2] - pairs_all_class[ind][1][0]
                    half_h_class = int(round(half_h_class / 2))
                    center_row_class = pairs_all_class[ind][1][2] - half_h_class

                    half_w_class = pairs_all_class[ind][1][3] - pairs_all_class[ind][1][1]
                    half_w_class = int(round(half_w_class / 2))
                    center_col_class = pairs_all_class[ind][1][3] - half_w_class

                    list_cols_class.append(
                        abs(center_col_b1 - center_col_class))  # razlika megju centarot na anotacija i centarot na edna bbox od klasifikator
                    list_sum_cols.append(
                        abs(center_col_b2 - center_col_b1))  # lista so pomestuvanja(centar na predvidena bbox minus centar na anotacija) na centarot po koloni posle regresor
                    suma_cols_class += abs(center_col_b1 - center_col_class)

                    list_rows_class.append(abs(center_row_b1 - center_row_class))
                    list_sum_rows.append(
                        abs(center_row_b2 - center_row_b1))  # lista so pomestuvanja na centarot po redici posle regresor
                    suma_rows_class += abs(center_row_b1 - center_row_class)

                    visini.append(int(round(pairs_all_class[ind][1][2] - pairs_all_class[ind][1][0])))
                    shirini.append(int(round(pairs_all_class[ind][1][3] - pairs_all_class[ind][1][1])))

                    # cv2.rectangle(img, (pair[1][1],pair[1][0]), (pair[1][3],pair[1][2]),color=(0,255,0), thickness=1 )

                suma_cols_reg_norm = suma_cols_reg / (bbox_annot[3] - bbox_annot[1])
                suma_rows_reg_norm = suma_rows_reg / (bbox_annot[2] - bbox_annot[0])

                suma_cols_reg_norm /= len(pairs_all)
                suma_rows_reg_norm /= len(pairs_all)

                suma_cols_reg /= len(pairs_all)
                suma_rows_reg /= len(pairs_all)

                # suma_cols_class/=len
                suma_cols_reg_norm_all += suma_cols_reg_norm
                suma_rows_reg_norm_all += suma_rows_reg_norm

                suma_cols_all += suma_cols_reg
                num_sum += 1
                suma_rows_all += suma_rows_reg

                for pair in pairs_all:
                    if pair[0][2] > MinRowSpeed and pair[0][2] < MaxRowSpeed:
                        # if int((pair[0][2] + pair[0][0]) / 2) < int(img_height / 2):
                        if (pair[1][2] - pair[1][0]) > int(
                                (int((pair[0][2] - pair[0][0]) * 0.3) + (pair[0][2] - pair[0][0]))):
                            golemi += 1
                true_positives+=len(pairs_all)
            # NOTE DEBUG !!
            # cv2.imshow('slika', img)
            # cv2.waitKey(0)
            indexes_to_remove = []

            # dodadeno
            # for ind, el in enumerate(pairs_per_image):
            #     if int((el[0][2]+el[0][1])/2)>int(341/2):
            #         indexes_to_remove.append(ind)

            # for ind_to_remove in indexes_to_remove[::-1]:
            #     #pairs_all.pop(ind_to_remove)
            #     bbox_preds.remove(pairs_per_image[ind_to_remove][1])
            #     # pairs_per_image.remove(pairs_per_image[ind_to_remove])

            bbox_preds_to_remove = []
            for ind_pred, bbox_pred in enumerate(bbox_preds):

                if bbox_pred[2] < MinRowSpeed or bbox_pred[2] > MaxRowSpeed:

                    bbox_preds_to_remove.append(ind_pred)  # note: removes by element but only first occurance

            for ind_pred in bbox_preds_to_remove[::-1]:
                bbox_preds.pop(ind_pred)
                # bbox_preds.remove(bbox_to_remove)

            # print(bbox_preds)

            false_positives += len(bbox_preds)
            print(false_positives)
            if debug:
                for bbox in bbox_preds:
                    cv2.rectangle(bgr_image, (bbox[1],bbox[0]), (bbox[3],bbox[2]), color=(255,0,0), thickness=3)
                cv2.imshow('sl',bgr_image)
                cv2.waitKey(0)
            # print(false_positives)

            # if false_positives>0:
            #     for bbox_pred in bbox_preds:
            #         cv2.rectangle(img, (bbox_pred[1],bbox_pred[0]), (bbox_pred[3], bbox_pred[2]), color=(0,255,255), thickness=1)
            #
            # cv2.imshow('slika', img)
            # cv2.waitKey(0)

            # print(bbox_preds)
            # print(os.path.join(predictions,annot_file))
            #
            # print('nnnnn')

    mae_cols = suma_cols_all / num_sum
    mae_rows = suma_rows_all / num_sum

    mae_cols_norm = suma_cols_reg_norm_all / num_sum
    mae_rows_norm = suma_rows_reg_norm_all / num_sum
    print(false_negatives)
    # print(dir)
    print(false_positives)

    import xlsxwriter

    # Start from the first cell.
    # Rows and columns are zero indexed.

    cnt1 += 1
    mae_rows_tmp.append(mae_rows_norm)
    mae_cols_tmp.append(mae_cols_norm)

    if cnt1 % 3 == 0:
        average_rows = sum(mae_rows_tmp) / len(mae_rows_tmp)
        average_cols = sum(mae_cols_tmp) / len(mae_cols_tmp)
        cnt1 = 0
        mae_rows_tmp = []
        mae_cols_tmp = []



    precision=true_positives/(true_positives+false_positives)
    recall=true_positives/(true_positives+false_negatives)

    f1_score=2*precision*recall/(precision+recall)
    content = [mae_rows, mae_cols, false_negatives, false_positives, false_positives_obratni, num_detections,
               golemi, mae_rows_norm, mae_cols_norm, precision,recall,f1_score,best_thr]
    if best_accuracy_flag:

        # if 'FINETUNE' in dir or 'finetune' in dir:
        #     row = 14
        #     column -= 1
        # else:
        row = 1
        # iterating through content list
        worksheet.write(row - 1, column, results_path_reg)

        for item in content:
            # write operation perform

            worksheet.set_column(row, column, 30)
            worksheet.write(row, column, item)

            # incrementing the value of row by one
            # with each iterations.
            row += 1

        if average_rows != 0:
            # print(row)
            # print(column)
            worksheet.write(row + 1, column - 1, average_rows)
            worksheet.write(row + 2, column - 1, average_cols)

            average_rows = 0

    if best_accuracy_flag:
        workbook.close()

    return f1_score
