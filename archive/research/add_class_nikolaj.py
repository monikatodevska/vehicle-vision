
# python imports
import os
import numpy as np
import cv2

# from tqdm import tqdm

# custom imports


# --- paths ---
src_path_videos = r'\\HP-1060\ForSharingD\VideoFrames\video2_frames'  # f_name: 00254.bmp
src_annotations_path = r'\\HP-1060\ForSharingD\AnotaciiKoliDenes\Annotations\Video2'  # fname: 00254.txt
src_annotations_path_new=r'E:\Science\Monika1\Video2_annot'
# dst_path = r'\\HP-1060\ForSharingD\AnnotationsVisualized\Video4_plus'

# --- load and visualize annotations ---
annotations_names = [x for x in os.listdir(src_annotations_path) if x[-4:] == '.txt']
# print(f'Number of annotated images: {len(annotations_names)}')

for annotations_name in annotations_names:

    annotations = np.loadtxt(os.path.join(src_annotations_path, annotations_name),
                            delimiter=',', ndmin=2).astype(np.int)
    # if(annotations_name=="01322.txt"):
    #     print(annotations)

    image_name = annotations_name[:-4] + '.bmp'
    image = cv2.imread(os.path.join(src_path_videos, image_name))
    fh=open(os.path.join(src_annotations_path_new, annotations_name), 'w+')
    # cv2.imshow("slika", image)
    for bbox in annotations:
        if(len(bbox)==4):
            cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[1] + bbox[3], bbox[0] + bbox[2]),
                          color=(0, 255, 0), thickness=1)
            cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(image_name, (1200, 700))
            cv2.imshow(image_name, image)
            cv2.waitKey(0)
            cl=input("Klasa:")
            if int(cl)==1 or int(cl)==2 or int(cl)==3 or int(cl)==4:
                bbox=np.append(bbox, cl)
                bbox_str=[str(bb) for bb in bbox]
                fh.write(','.join(bbox_str))
                fh.write("\n")

        else:
            continue

    fh.close()
    cv2.destroyWindow(image_name)
    # cv2.imwrite(os.path.join(dst_path, image_name), image)