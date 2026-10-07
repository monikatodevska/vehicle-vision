import os
import numpy as np
import shutil
import helper_postprocessing
from matplotlib import pyplot as plt
annot_path=r'D:\Monika\VideosTest\Frames_morning_10fps\anotacii'
predictions=r'D:\Monika\Results\vinf_900_SO_DVA_INCEPTION_SKIP_48_finetune_morning_10fps_ANOTIRANI_postprocessing'
false_positives_ious_p=r'D:\Monika\Results\vinf_900_SO_DVA_INCEPTION_SKIP_48_finetune_morning_10fps_ANOTIRANI_postprocessing'
annotations=[x for x in os.listdir(annot_path) if x[-4:]=='.txt']
col_diff_all=[]
suma_cols=0
suma_rows=0
false_negatives = 0
false_positives=0
pairs_all=[]
HScale=(341/1080)
WScale=(512/1621)

for annot_file in annotations:

    f=open(os.path.join(annot_path,annot_file),'r')
    bbox_annots=f.readlines()

    # num_lines = sum(1 for line in f if line.rstrip())
    # print('Total lines:', num_lines)
    if os.path.exists(os.path.join(predictions,annot_file)):

        fp=open(os.path.join(predictions,annot_file))
    else:
        false_negatives+=len(bbox_annots)
        continue

    # num_lines_pred = sum(1 for line in fp if line.rstrip())
    # print('Total lines:', num_lines)
    # if num_lines>=num_lines_pred:


    iou_max=0
    bbox_preds=fp.readlines()
    bbox_annots = [[int(str(j)) for j in i.split(',')] for i in bbox_annots]
    bbox_preds = [[int(str(j)) for j in i.split(',')] for i in bbox_preds]
    for bbox_pred in bbox_preds:
        bbox_pred[2] = bbox_pred[0] + bbox_pred[2]
        bbox_pred[3] = bbox_pred[1] + bbox_pred[3]

    for bbox_annot in bbox_annots:
        iou_max=0
        pair=[]
        bbox_annot[2] = bbox_annot[0] + bbox_annot[2]
        bbox_annot[3] = bbox_annot[1] + bbox_annot[3]
        bbox_annot[0]=int(bbox_annot[0]*HScale)
        bbox_annot[1]=int(bbox_annot[1]*WScale)
        bbox_annot[2]=int(bbox_annot[2]*HScale)
        bbox_annot[3]=int(bbox_annot[3]*WScale)

        # print(bbox_annot)
        for bbox_pred in bbox_preds:
            # print(bbox_pred)

            iou=helper_postprocessing.calc_iou(bbox_annot[:-1],bbox_pred[:-1])
            if iou>0.0:
                if iou>iou_max:
                    iou_max=iou
                    pair=[(bbox_annot),(bbox_pred)]



        if len(pair)==0:
            false_negatives+=1

            continue



        if(pair[1] in bbox_preds):
            # print(pair[1])
            bbox_preds.remove(pair[1])
            pairs_all.append(pair)

    false_positives += len(bbox_preds)


    false_positives_ious = []

    for ind, bbox_pred_unused in enumerate(bbox_preds):

        iou_max_2 = 0

        for bbox_annot in bbox_annots:

            # if ind == 0:
            #     bbox_annot[2] = bbox_annot[0] + bbox_annot[2]
            #     bbox_annot[3] = bbox_annot[1] + bbox_annot[3]
            #     bbox_annot[0] = int(bbox_annot[0] * HScale)
            #     bbox_annot[1] = int(bbox_annot[1] * WScale)
            #     bbox_annot[2] = int(bbox_annot[2] * HScale)
            #     bbox_annot[3] = int(bbox_annot[3] * WScale)

            print(bbox_annot)

            iou = helper_postprocessing.calc_iou(bbox_annot[:-1], bbox_pred_unused[:-1])
            if iou > iou_max_2:
                iou_max_2 = iou

        false_positives_ious.append(iou_max_2)

        print('\n\n')


np.savetxt(os.path.join(false_positives_ious_p,'FP_ious.txt'), np.array(false_positives_ious), delimiter=',', fmt="%f")





width=[]
height=[]

width_errors=[]
height_errors=[]

print("")
# print(pairs_all)
# if len(pairs_all)==0:
#     continue
for pair in pairs_all:
    # print(pair[0][3])
    half_w_b1=pair[0][3]-pair[0][1]
    half_w_b1=int(round(half_w_b1/2))

    center_row_b1=pair[0][2]
    center_col_b1=pair[0][1]+half_w_b1



    half_w_b2 = pair[1][3] - pair[1][1]
    half_w_b2=int(round(half_w_b2/2))

    center_row_b2 = pair[1][2]
    center_col_b2 = pair[1][1] + half_w_b2

    col_diff=abs(center_col_b2-center_col_b1)
    col_diff_norm=col_diff/(pair[0][3]-pair[0][1])

    row_diff=abs(center_row_b2-center_row_b1)
    row_diff_norm=row_diff/(pair[0][2]-pair[0][0])

    width.append(pair[0][3]-pair[0][1])
    width_errors.append(col_diff_norm)
    height.append(pair[0][2]-pair[0][0])
    height_errors.append(row_diff_norm)

    suma_cols+=abs(center_col_b2-center_col_b1)
    suma_rows+=abs(center_row_b2-center_row_b1)

    # print(len(pairs_all))
mae_cols=suma_cols/len(pairs_all)
mae_rows=suma_rows/len(pairs_all)


for i in range(len(width_errors)):
    width_errors[i]*=100
    height_errors[i]*=100

plt.scatter(width,width_errors,color='blue')
plt.xlabel('Shirina')
plt.ylabel('Greska po koloni(%)')
plt.show()

plt.scatter(height,height_errors,color='red')
plt.xlabel('Visina')
plt.ylabel('Greska po redici(%)')
plt.show()


# num_bins=np.arange(min(width_errors), max(width_errors)+1,1)
plt.hist(width_errors, bins=np.unique(width_errors))
plt.xlabel("Greska po koloni(%)")
plt.show()

plt.hist(height_errors,bins=np.unique(height_errors))
plt.xlabel('Greska po redici(%)')
plt.show()

print(f'MAE columns: {mae_cols}')
print(f'MAE rows: {mae_rows}')
print('kraj')
print(f'False Negatives: {false_negatives}')
print(f'False Positives: {false_positives}')
        # col_diff=abs(center_col_b1-center_col_b2)
        # row_diff=abs(center_row_b1-center_row_b2)
        # iou = helper_postprocessing.calc_iou(pair[0][:-1], pair[1][:-1])
        # col_diff_all.append(col_diff)
# suma=0
# for col_diff in col_diff_all:
#     suma+=col_diff*col_diff
# mse=(1/len(col_diff_all))*suma





