import numpy as np
import os
import cv2
import shutil
# from numpy import dtype

import metrics
im_width_new=512
im_height_new=341
# im_path=r'D:\Monika\yolov5\runs\detect\exp11\exp11_renamed'
yolo_path=r'D:\Monika\yolov5\runs\detect\exp11\exp11_renamed'
annot_path=r'D:\Monika\VideosTest\All_Cameras_annnotated\Annotations\Renamed'
reg_path=r'D:\Monika\yolov5\runs\detect\exp11\exp11_renamed\reformatted_labels'
class_path=r'D:\Monika\Results\TEST_ALL_CAMERAS2\class\YOLO'
file_path_r=r'D:\Monika\yolov5\runs\detect\exp11\exp11_renamed'
region=9
best_thr=0.45
# for prob_thr in folders:
model_name='YOLO'
def convert_yolo_format(yolo_path,reg_path):
    files = [x for x in os.listdir(yolo_path) if x.endswith('.txt')]

    for annot_file in files:
        objects=[]
        bboxes = np.loadtxt(os.path.join(yolo_path, annot_file), delimiter=' ', ndmin=2).astype(np.float)

        im_name = annot_file[:-4] + '.bmp'
        if not os.path.exists(os.path.join(yolo_path, im_name)):
            continue
        image = cv2.imread(os.path.join(yolo_path, im_name), 0)

        resized = cv2.resize(image, (im_width_new, im_height_new), interpolation=cv2.INTER_AREA)
        rows, cols = image.shape[:2]
        HScale = im_height_new / rows
        WScale = im_width_new / cols

        for bbox in bboxes:
            c_c = int(bbox[1] * cols)
            c_r = int(bbox[2] * rows)
            w = int(bbox[3] * cols)
            h = int(bbox[4] * rows)
            cls=bbox[0]
            xmin = int(c_c - (w / 2))
            ymin = int(c_r - (h / 2))
            xmax = int(xmin + w)
            ymax = int(ymin + h)

            # xmin_res = int(xmin * WScale)
            # ymin_res = int(ymin * HScale)
            # xmax_res = int(xmax * WScale)
            # ymax_res = int(ymax * HScale)
            if bbox[0]==2:
                cls=0

            object = [min(ymin, image.shape[0]), min(xmin, image.shape[1]),
                      min(ymax, image.shape[0]), min(xmax, image.shape[1]), cls]
            objects.append(object)

        # for bbox in objects:
        #     cv2.rectangle(image,(bbox[1],bbox[0]),(bbox[3],bbox[2]),color=(255,0,0),thickness=1)
        # cv2.imshow('s',image)
        # cv2.waitKey(0)
        # objects=objects.astype(int)
        np.savetxt(os.path.join(reg_path,annot_file),objects,delimiter=',',fmt='%d')
        cv2.imwrite(os.path.join(reg_path,im_name[:-4]+'.bmp'),image)

def copy_reg_to_class(source_folder,destination_folder):


    # Paths


    # Ensure destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Iterate through all files in the source folder
    for file_name in os.listdir(source_folder):
        # Check if the file matches the required pattern
        if '_reg' in file_name:
            # Construct the source and destination file paths
            source_path = os.path.join(source_folder, file_name)
            new_file_name = file_name.replace('_reg', '_class')
            destination_path = os.path.join(destination_folder, new_file_name)

            # Copy the file
            shutil.copy(source_path, destination_path)
            print(f"Copied and renamed: {source_path} -> {destination_path}")

convert_yolo_format(yolo_path, reg_path)
# f1=metrics.calculate_stats(reg_path, class_path, annot_path, file_path_r,region,model_name,best_thr,True)
# print(f1)
# source_folder = r"D:\Monika\Results\TEST_ALL_CAMERAS2\reg\YOLO"  # Replace with your source folder path
# destination_folder = r"D:\Monika\Results\TEST_ALL_CAMERAS2\class\YOLO"  # Replace with your destination folder path
# copy_reg_to_class(source_folder,destination_folder)