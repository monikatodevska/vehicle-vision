import os
import cv2

annot_root_path=r''
image_folder=r''


for ground_truth_folder in os.listdir(annot_root_path):
    # if not 'kam33_1' in ground_truth_folder:
    #     continue
    for gt_file in os.listdir(os.path.join(annot_root_path, ground_truth_folder)):
        if gt_file.endswith(".txt"):
            image_name = gt_file.replace(".txt", ".bmp")
            image_path = os.path.join(image_folder, ground_truth_folder, image_name)
            im_orig = cv2.imread(image_path, 1)
            image = cv2.imread(image_path, 0)
