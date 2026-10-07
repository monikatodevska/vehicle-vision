"""
CityScapes Dataset
Analysis and visualization of CityPersons annotations
"""

import os
import cv2
import numpy as np
import json

if __name__ == '__main__':

    # --- paths list ---
    src_images_path = r'\\192.168.1.148\Public\Datasets\Soobrakjaj\CityScapes\leftImg8bit_trainvaltest\leftImg8bit'
    src_annnotations_path = r'\\192.168.1.148\Public\Datasets\Soobrakjaj\CityScapes\gtFine_trainvaltest\gtFine'

    dst_vis_path = r'D:\Monika\Cityscapes_Visualize'

    # --- input variables list ---
    img_size = {'h': 1024, 'w': 2048}
    # colors = {'pedestrian': (0, 0, 255),            # ok
    #           'sitting person': (0, 255, 255),      # ok
    #           'rider': (255, 0, 0),                 # tocak i motor naedno
    #           'ignore': (0, 0, 0),                  # da se isfrlat
    #           'person (other)': (255, 255, 0),      # kaj se vakvite?!
    #           'person group': (0, 255, 0)}          # da se isfrlat

    # --- stats variables list ---
    objects_per_class = {}
    draw_class = {'person': (255, 0, 0), 'bicycle': (255, 255, 255), 'persongroup': (255, 255, 0), 'rider': (0, 255, 255),'pedestrian': (207, 243, 270), 'sitting person': (0,255,0) }
    # {'pedestrian': 19683, 'sitting person': 1217, 'rider': 2189, 'ignore': 8399, 'person (other)': 504,
    # 'person group': 1573}

    # --- process data ---
    subset_folders = [x for x in os.listdir(src_annnotations_path)
                      if os.path.isdir(os.path.join(src_annnotations_path, x))]
    print(f'Data subsets: {subset_folders}.')

    for subset_folder in subset_folders:

        cities = [x for x in os.listdir(os.path.join(src_annnotations_path, subset_folder))
                  if os.path.isdir(os.path.join(src_annnotations_path, subset_folder, x))]
        print(f'List of cities in the {subset_folder} subset: {cities}')

        for city in cities:
            if city != 'bochum':
                continue
            annot_folder_path = os.path.join(src_annnotations_path, subset_folder, city)
            print(f'Processing folder: {annot_folder_path}')

            file_names = sorted([x for x in os.listdir(annot_folder_path) if os.path.splitext(x)[1] == '.json'])

            for file_name in file_names:

                f = open(os.path.join(annot_folder_path, file_name))
                data = json.load(f)
                f.close()

                objects = data['objects']

                image_name_1 = os.path.splitext(file_name)[0].split('_')[:-2]
                image_name = '_'.join(image_name_1) + '_leftImg8bit.png'
                #
                image_path = os.path.join(src_images_path, subset_folder, city, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    print(f'Image {image_path} is None.')
                    exit(1)

                # count objects
                for obj in objects:
                    # obj: {'instanceId': 24000, 'bbox': [1922, 353, 63, 155], 'bboxVis': [1924, 353, 55, 102],
                    # 'label': 'pedestrian'}

                    if obj['label'] not in objects_per_class.keys():
                        objects_per_class[obj['label']] = 1
                    else:
                        objects_per_class[obj['label']] += 1

                    # bbox = [obj['bbox'][1], obj['bbox'][0], obj['bbox'][3], obj['bbox'][2]]   # min_r, min_c, h, w
                    # cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[1] + bbox[3], bbox[0] + bbox[2]),
                    #               color=colors[obj['label']], thickness=3)

                    # exit(1)
                    # draw polygon on image if class is in draw_class dict
                    if obj['label'] in draw_class.keys():
                        polygon = obj['polygon']
                        # draw polygon on image
                        polygon_arr = np.array(polygon, np.int32)
                        min_coords = np.amin(polygon_arr, axis=0)
                        min_c = min_coords[0]
                        min_r = min_coords[1]

                        max_coords = np.amax(polygon_arr, axis=0)
                        max_c = max_coords[0]
                        max_r = max_coords[1]
                        cv2.rectangle(image, (min_c, min_r), (max_c, max_r), draw_class[obj['label']], thickness=1)

                        # img_mod = cv2.polylines(image, [penta], True, draw_class[obj['label']], 1)

                cv2.imwrite(os.path.join(dst_vis_path,subset_folder, city, image_name))
                cv2.imshow('Shapes', image)
                cv2.waitKey(0)

                # cv2.imshow('slika', image)
                # cv2.waitKey(0)

                # save images visualizing annotations
                # dst_image_path = os.path.join(dst_vis_path, image_name)
                # cv2.imwrite(dst_image_path, image)
    print(objects_per_class.keys())
    print(f'Number of objects per class: {objects_per_class}')
