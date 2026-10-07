import os
import numpy as np
from copy import deepcopy

# custom package imports
import cv2

import xml.etree.ElementTree as ET
import helper_anchorless_mc

version = 'vinf_12'

# res=r'D:\KlasifikacijaVozila\Miladinovci\crtanje'

# modelsPath = os.path.join(dstModelsPath, version)
imgDims = {'rows': 341, 'cols': 512}
num_classes = 2
img_depth = 1

img_dims = (imgDims['rows'], imgDims['cols'], img_depth)

im_size = (img_dims[1], img_dims[0])

# anchor_dims = ((18,18),(23, 23), (32, 32), (45, 45), (65, 65))
anchor_stride = 8
# norm_coef = 100  # constant to normalize regression ground truth data

# IOU thresholds for selecting positive and negative anchors
iou_low = 0.6
iou_high = 0.7
# num_cores = multiprocessing.cpu_coun t()
kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9



def read_data_and_generate_gt(im_root, annot_root, im_size, im_depth, GroundTruthFiles_root, augmented_dst, m, resultspath, no_positives, x1, hard_negative, samo_nacrtaj):
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
    nemapoz = 0
    nemapoz_s = 0
    nemaanot = 0
    prazenfajl = 0
    nemaobjekt = 0
    frli = 0
    broj = 0
    cnt = 0
    nemaslika=0
    brojac=0
    nemadir=0
    anotacii_netxt=0
    dirnames = os.listdir(annot_root)
    print(dirnames)
    anotacija=0
    napraeni=os.listdir(r'\\192.168.1.133\Monika\Trening_ZaDemo\GT')
    klasi=[]

    site_bboxovi = []
    site_pomestuvanja= []

    # ext_list=['.bmp','.jpg']
    for dir_name in dirnames:
        # if dir_name in napraeni:
        #     # print(dir_name)
        #     continue

        if 'miladinovci' not in dir_name and 'drugo' not in dir_name:
            continue

        #if 'DGood' not in dir_name:
        #    continue
        print(dir_name)

        # if (dir_name[-4:]=='.zip') or (dir_name[-3:]=='.db') or ('miladinovci' in dir_name) or ('kamera2' in dir_name) or ('videoframes' in dir_name):
        #      continue
        # cnt=0
        # 'nvr11' in dir_name or 'nvr12' in dir_name or

        if dir_name[-3:] == '.db':
            continue
        if dir_name[-4:] == '.zip':
            continue
        # print(dir_name)
        im_path = os.path.join(im_root, dir_name)
        if not os.path.exists(im_path):
            print('nema dir za sliki')
            nemadir+=1
            continue
        # print(im_path)
        annot_path = os.path.join(annot_root, dir_name)
        hard_negatives_path = os.path.join(hard_negative, dir_name)

        if not os.path.exists(resultspath):
            os.mkdir(resultspath)

        folder_augmented_photos = os.path.join(augmented_dst, dir_name)
        if not os.path.exists(folder_augmented_photos):
            os.mkdir(folder_augmented_photos)

        GroundTruthFiles = os.path.join(GroundTruthFiles_root, dir_name)
        if not os.path.exists(GroundTruthFiles):
            os.mkdir(GroundTruthFiles)

        GroundTruthAnalysis = os.path.join(resultspath, dir_name)
        if not os.path.exists(GroundTruthAnalysis):
            os.mkdir(GroundTruthAnalysis)

        # GroundTruthAnalysis_s=os.path.join(resultspath,dir_name+'_shift')
        # if not os.path.exists(GroundTruthAnalysis_s):
        #     os.mkdir(GroundTruthAnalysis_s)
        if not os.path.exists(no_positives):
            os.mkdir(no_positives)
        no_positives1 = os.path.join(no_positives, dir_name)
        if not os.path.exists(no_positives1):
            os.mkdir(no_positives1)

        # no_positives1_s = os.path.join(no_positives, dir_name+'_shift')
        # if not os.path.exists(no_positives1_s):
        #     os.mkdir(no_positives1_s)

        filenames = os.listdir(annot_path)
        # anotacii_filenames=os.listdir(annot_path)
        # print(filenames)
        for annot_name in filenames:
            # print(os.path.join(annot_path, annot_name))   # NOTE: printanje ime na fajl
            anotacija+=1
            # print(annot_name[-4:])
            if annot_name[-4:]!='.txt' and annot_name[-4:]!='.xml':
                anotacii_netxt+=1
                # print('anotacii')
                continue


            flag_postoi = False
            # for debugging purposes
            # cnt+=1
            # if cnt>5:
            #     break
            objects = []
            objects_s = []
            # if im_name[-3:] == '.db':
            #     continue
            #NOTE: da se menuva ova!!!!!!!!!!!!!!!!!!!!!!!!!!!

            if dir_name=='kamera2_nvr5_oblacno' or dir_name=='kamera2_nvr6_oblacno' or dir_name=='kamera2_nvr7_oblacno' or 'DGood' in dir_name:
                im_name=annot_name[:-4]+'.jpg'
            elif 'M-30' in dir_name :
                im_name='image'+str(int(annot_name[:-4])+1).zfill(6)+'.jpg'
                # print(annot_name)
                # print(im_name)
            else:
                im_name=annot_name[:-4]+'.bmp'

            if os.path.exists(os.path.join(im_path,im_name)):
                # print('postoi')
                image=cv2.imread(os.path.join(im_path,im_name))
                # cv2.imshow("slika",image)
                # cv2.waitKey(0)
            else:
                # print(os.path.join(im_path,im_name))
                # print(os.path.join(annot_path,annot_name))
                nemaslika+=1
                # print(nemaslika)
                continue
            rows, col = image.shape[:2]
            flag = 0
            # if im_name[-4:] != '.bmp' or im_name[-4:] != '.jpg':
            #     continue

            if dir_name == 'M-30':
                cropped_image = image[0:rows, 0:721]
                # annot_name = str(int(im_name[5:11].lstrip('0')) - 1) + '.xml'
            elif dir_name == 'M-30-HD':
                cropped_image = image[0:rows, 0:1080]
                # annot_name = str(int(im_name[5:11].lstrip('0')) - 1) + '.xml'


            else:
                if 'videoframes' in dir_name:
                    cropped_image = image[0:rows, 0:1141]

                elif 'miladinovci' in dir_name:
                    cropped_image = image[0:rows, 0:1621]

                elif 'kamera2' in dir_name:
                    cropped_image = image[0:rows, 160:1782]

                elif 'DGood' in dir_name:
                    cropped_image = image[0:rows, 149:col]

                elif 'drugo' in dir_name:
                    cropped_image = image.copy()
                # annot_name = im_name[:-4] + '.txt'

            rows_cr, col_cr = cropped_image.shape[:2]
            resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)
            HScale = 341 / rows_cr
            WScale = 512 / col_cr

            # shifted=helper_anchorless_mc.shiftImageHorizontally(resized, x)

            if os.path.exists(os.path.join(annot_path, annot_name)):
                if os.stat(os.path.join(annot_path, annot_name)).st_size != 0:
                    if dir_name == 'M-30' or dir_name == 'M-30-HD':
                        root = ET.parse(os.path.join(annot_path, annot_name)).getroot()

                        objects = []  # list of all objects in the image
                        # cv2.imwrite(os.path.join(im_path_resized, im_name), image)
                        for object in root.findall('object'):

                            cl = object.find('class').text
                            if cl not in klasi:
                             klasi.append(cl)
                             # print(cl)

                            bb_xml = object.find('bndbox')
                            bb = [np.int(bb_xml.find('xmin').text),  # min_col
                                  np.int(bb_xml.find('xmax').text),  # max_col
                                  np.int(bb_xml.find('ymin').text),  # min_row
                                  np.int(bb_xml.find('ymax').text),  # max_row
                                  ]
                            if cl == 'car' and bb[3] - bb[2] + 1 > 13:
                                annot = [int(np.round(bb[2] * HScale)), int(np.round(bb[0] * WScale)), int(np.round(bb[3] * HScale)), int(np.round(bb[1] * WScale)), 1]
                                objects.append(annot)
                            elif cl == "van":
                                flag = 1
                                break
                                # and bb[3] - bb[2] + 1 > 15:
                                # annot = [int(np.round(bb[2])), int(np.round(bb[0])), int(np.round(bb[3])), int(np.round(bb[1])), 1]
                                # objects.append(annot)
                            elif cl == "truck" and bb[3] - bb[2] + 1 > 15:
                                annot = [int(np.round(bb[2] * HScale)), int(np.round(bb[0] * WScale)), int(np.round(bb[3] * HScale)), int(np.round(bb[1] * WScale)), 2]

                                objects.append(annot)
                            elif cl=="motorbike":
                                annot = [int(np.round(bb[2] * HScale)), int(np.round(bb[0] * WScale)), int(np.round(bb[3] * HScale)), int(np.round(bb[1] * WScale)), 3]
                                objects.append(annot)
                            else:
                                continue
                        if (flag == 1):
                            frli += 1
                            continue
                        # for obj in objects:
                        #     cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=1)
                        #     # print("crtam")
                        # cv2.imshow("slika", resized)
                        # # cv2.imshow("slika1", im_res)
                        # cv2.waitKey(0)
                    else:
                        # print(os.path.join(annot_path, annot_name))
                        objs = np.loadtxt(os.path.join(annot_path, annot_name), delimiter=',', ndmin=2).astype(np.int)
                        # print(objs)
                        if 'kamera2' in dir_name:
                            for bbox in objs:
                                # print(bbox)
                                r2 = bbox[0] + bbox[2]

                                c1 = bbox[1] - 160
                                c2 = c1 + bbox[3]
                                # bbox_c2 = bbox[1]+bbox[3] - 160
                                if (bbox[4] == 4):
                                    object = [int((bbox[0] - 1) * HScale), min(int((c1 - 1) * WScale), 512), int((bbox[0] - 1 + bbox[2]) * HScale), min(int((c2 - 1) * WScale), 512), 2]
                                elif (bbox[4] == 3):
                                    object = [int((bbox[0] - 1) * HScale), min(int((c1 - 1) * WScale), 512), int((bbox[0] - 1 + bbox[2]) * HScale), min(int((c2 - 1) * WScale), 512), 1]
                                elif (bbox[4] == 5):
                                    object = [int((bbox[0] - 1) * HScale), min(int((c1 - 1) * WScale), 512), int((bbox[0] - 1 + bbox[2]) * HScale), min(int((c2 - 1) * WScale), 512), 2]

                                else:
                                    object = [int((bbox[0] - 1) * HScale), min(int((c1 - 1) * WScale), 512), int((bbox[0] - 1 + bbox[2]) * HScale), min(int((c2 - 1) * WScale), 512), int(bbox[4])]

                                # object_s=[object[0],object[1]+x, object[2], object[3]+x, object[4]]
                                if len(object) != 0:
                                    objects.append(object)

                                # if len(object_s) != 0:
                                #     objects_s.append(object_s)
                                # print('appended')
                                # print(os.path.join(annot_path,annot_name))
                        elif ('miladinovci' in dir_name) or ('videoframes' in dir_name) or ('drugo' in dir_name):
                            for bbox in objs:
                                if (bbox[4] == 4):
                                    object = [int((bbox[0] - 1) * HScale), min(int((bbox[1] - 1) * WScale), 512), int((bbox[0] - 1 + bbox[2]) * HScale),
                                              min(int((bbox[1] - 1 + bbox[3]) * WScale), 512), 2]
                                elif (bbox[4] == 3):
                                    object = [int((bbox[0] - 1) * HScale), min(int((bbox[1] - 1) * WScale), 512), int((bbox[0] - 1 + bbox[2]) * HScale),
                                              min(int((bbox[1] - 1 + bbox[3]) * WScale), 512), 1]
                                elif (bbox[4] == 5):
                                    object = [int((bbox[0] - 1) * HScale), min(int((bbox[1] - 1) * WScale), 512), int((bbox[0] - 1 + bbox[2]) * HScale),
                                              min(int((bbox[1] - 1 + bbox[3]) * WScale), 512), 2]

                                else:
                                    object = [int((bbox[0] - 1) * HScale), min(int((bbox[1] - 1) * WScale), 512), int((bbox[0] - 1 + bbox[2]) * HScale),
                                              min(int((bbox[1] - 1 + bbox[3]) * WScale), 512),
                                              int(bbox[4])]
                                # object_s = [object[0], object[1] + x, object[2], object[3] + x, object[4]]
                                if len(object) != 0:
                                    objects.append(object)
                                # if len(object_s)!=0:
                                #     objects_s.append(object_s)
                        # for obj in objects:
                        #     cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=1)
                        #     print(os.path.join(annot_path,annot_name))
                        #     # print("crtam")
                        # cv2.imshow("slika", resized)
                        # cv2.waitKey(0)
                        # # cv2.imshow("slika1", im_res)
                        # cv2.waitKey(0)
                        elif 'DGood' in dir_name:
                            for bbox in objs:
                                # print(bbox)
                                c1 = bbox[1] - 149
                                c2 = bbox[3] - 149
                                if (bbox[4] == 4):
                                    object = [int(bbox[0] * HScale), int(c1 * WScale), int(bbox[2] * HScale), int(c2 * WScale), 2]
                                elif (bbox[4] == 3):
                                    object = [int(bbox[0] * HScale), int(c1 * WScale), int(bbox[2] * HScale), int(c2 * WScale), 1]
                                elif (bbox[4] == 5):
                                    continue
                                else:
                                    object = [int(bbox[0] * HScale), int(c1 * WScale), int(bbox[2] * HScale), int(c2 * WScale), int(bbox[4])]
                                objects.append(object)
                        # elif 'drugo' in dir_name:




                else:
                    prazenfajl += 1

                    continue
            else:
                nemaanot += 1
                continue

            if len(objects) == 0:
                nemaobjekt += 1
                # print('nema_objekt')
                # print(os.path.join(annot_path,annot_name))
                continue
            bboxes_train = deepcopy(objects)
            site_bboxovi.append(bboxes_train)   # NOTE: added

            # bboxes_train_s=deepcopy(objects_s)
            num_negs_ratio = 3
            # print(bboxes_train)
            # if 'nvr11' in dir_name or 'nvr12' in dir_name:
            #     coords_filename=int(annot_name[:-4])-1
            #     coord_filename1=str(coords_filename).zfill(6)+'.txt'
            # else:
            coord_filename1 = annot_name

            if os.path.exists(os.path.join(hard_negatives_path, coord_filename1)):
                flag_postoi = True
                # print(coord_filename1)
                coords = np.loadtxt(os.path.join(hard_negatives_path, coord_filename1), delimiter=',', ndmin=2).astype(int)
                negative_mask = helper_anchorless_mc.anchor_level_false_positives_masks(coords, (341, 512, 1), anchor_stride)  # samo tocki

            else:
                negative_mask = None
                # negative_mask_racni=None

            negative_mask_racni = helper_anchorless_mc.generate_anchor_level_hard_negatives_masks(bboxes_train, (341, 512, 1), anchor_stride, 0.1, 0.3)

            obj_masks_train = helper_anchorless_mc.generate_anchor_level_object_masks(bboxes_train, img_dims, anchor_stride)
            # bboxes_train - bboxes for a single image
            # obj_masks_train = helper_anchorless_mc.anchor_level_false_positives_masks_mindims2(bboxes_train, img_dims, anchor_stride)

            # NOTE: SAMO ZA EDEN PRIMEROK PO OBJEKT
            # iou_low, iou_high = 0.3, 0.3
            out_class, out_reg = helper_anchorless_mc.get_anchorless_ground_truth_data_parallel_optimized(bboxes_train, obj_masks_train[:, :], img_dims, anchor_stride, iou_low, iou_high, num_negs_ratio, num_classes, negative_mask,negative_mask_racni)

            site_pomestuvanja.append(out_reg)

            # obj_masks_train_s = helper_anchorless_mc.generate_anchor_level_object_masks(bboxes_train_s, img_dims, anchor_stride)
            # out_class_s, out_reg_s = helper_anchorless_mc.get_anchorless_ground_truth_data_parallel_optimized(bboxes_train_s, obj_masks_train[:, :], img_dims, anchor_stride, iou_low, iou_high,

            # out_class_arr_s=np.array(out_class_s)
            # out_reg_arr_s=np.array(out_reg_s)
            # resized_arr=np.array(resized)
            #NOTE: flag_crtanje kazuva na koja slika da se crta, na pomestena ili originalna. ako e true se crta GT na pomestena
            flag_crtanje = False
            #NOTE: flag_nemapoz e true ako nema pozitivni primeroci na kraj od site pomestuvanja na slikata.
            flag_nemapoz = False
            x = x1
            #NOTE: ako nema pozitivni shift na slikata i presmetaj novi GT. ako i na shift so 2 i 4 nema pozitivni se zacuvuva slikata vo nopositives pateka
            '''if out_class is None and out_reg is None:
                flag_crtanje = True
                shifted = helper_anchorless_mc.shiftImageHorizontally(resized, x)

                #NOTE: shift annotations
                for ob in objects:
                    object_s = [ob[0], ob[1] + x, ob[2], ob[3] + x, ob[4]]
                    objects_s.append(object_s)

                bboxes_train_s = deepcopy(objects_s)
                # print(bboxes_train_s)
                if flag_postoi:
                    coords = [coords[0], coords[1] + x, coords[2], coords[3]]
                    negative_mask = helper_anchorless_mc.anchor_level_false_positives_masks(coords, (341, 512, 1), anchor_stride)  # samo tocki

                negative_mask_racni = helper_anchorless_mc.generate_anchor_level_hard_negatives_masks(bboxes_train_s, (341, 512, 1), anchor_stride, 0.1, 0.3)

                obj_masks_train = helper_anchorless_mc.generate_anchor_level_object_masks(bboxes_train_s, img_dims, anchor_stride)
                out_class, out_reg = helper_anchorless_mc.get_anchorless_ground_truth_data_parallel_optimized(bboxes_train_s, obj_masks_train[:, :], img_dims, anchor_stride, iou_low, iou_high,
                                                                                                              num_negs_ratio,
                                                                                                              num_classes, negative_mask, negative_mask_racni)

                # NOTE: SAMO ZA EDEN PRIMEROK PO OBJEKT
                #iou_low, iou_high = 0.3, 0.3
                # out_class, out_reg = helper_anchorless_mc.get_anchorless_ground_truth_data_parallel_optimized_1persample(bboxes_train_s, obj_masks_train[:, :], img_dims, anchor_stride, iou_low, iou_high,
                #                                                                                               num_negs_ratio,



                while (out_class is None and out_reg is None):
                    objects_s = []
                    # nemapoz += 1
                    if (x == 4):
                        nemapoz += 1
                        flag_nemapoz = True
                        im_draw1 = resized.copy()
                        for obj in objects:
                            # print(obj[0], obj[1],obj[2],obj[3])
                            cv2.rectangle(im_draw1, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=1)
                        cv2.imwrite(os.path.join(no_positives1, im_name), resized)
                        break
                    # for obj in objects:
                    #     print(obj[0], obj[1],obj[2],obj[3])
                    #     cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=1)
                    # cv2.imwrite(os.path.join(no_positives1,im_name),resized)

                    # if no positives shift the image and generate new gt
                    x += 2
                    shifted = helper_anchorless_mc.shiftImageHorizontally(resized, x)
                    for ob in objects:
                        object_s = [ob[0], ob[1] + x, ob[2], ob[3] + x, ob[4]]
                        objects_s.append(object_s)
                    bboxes_train_s = deepcopy(objects_s)
                    # print(bboxes_train_s)

                    if flag_postoi:
                        coords = [coords[0], coords[1] + x, coords[2], coords[3]]
                        negative_mask = helper_anchorless_mc.anchor_level_false_positives_masks(coords, (341, 512, 1), anchor_stride)  # samo tocki

                    negative_mask_racni = helper_anchorless_mc.generate_anchor_level_hard_negatives_masks(bboxes_train_s, (341, 512, 1), anchor_stride, 0.1, 0.3)

                    obj_masks_train = helper_anchorless_mc.generate_anchor_level_object_masks(bboxes_train_s, img_dims, anchor_stride)
                    #obj_masks_train = helper_anchorless_mc.anchor_level_false_positives_masks_mindims2(bboxes_train_s, img_dims, anchor_stride)


                    out_class, out_reg = helper_anchorless_mc.get_anchorless_ground_truth_data_parallel_optimized(bboxes_train_s, obj_masks_train[:, :], img_dims, anchor_stride, iou_low, iou_high,
                                                                                                                  num_negs_ratio,
                                                                                                                  num_classes, negative_mask, negative_mask_racni)

                    # obj_masks_train = helper_anchorless_mc.generate_anchor_level_object_masks(bboxes_train_s, img_dims, anchor_stride)
                    # out_class, out_reg = helper_anchorless_mc.get_anchorless_ground_truth_data_parallel_optimized(bboxes_train_s, obj_masks_train[:, :], img_dims, anchor_stride, iou_low, iou_high,
                    #                                                                                                   num_negs_ratio,
                    #                                                                                                  num_classes)

            out_class_arr = np.array(out_class)
            out_reg_arr = np.array(out_reg)'''

            if out_class is None or out_reg is None:
                # nemapoz+=1
                # cv2.imwrite(os.path.join(no_positives1,im_name),resized)
                # if there are no positives
                continue

            # TODO: things
            # print(len(site_bboxovi))
            # print(len(site_pomestuvanja))

            # else:              #if there are positives

            '''if flag_crtanje:
                im_draw = shifted.copy()
                # print('crtam na pomestena')
                anchorless_genertor_plot.save_results_anchorless_limits_cls_one(GroundTruthAnalysis, im_draw, im_name[:-4] + '_s' + im_name[-4:], out_class_arr, out_reg_arr, anchor_stride, 0.5)
            else:
                im_draw = resized.copy()
                anchorless_genertor_plot.save_results_anchorless_limits_cls_one(GroundTruthAnalysis, im_draw, im_name, out_class_arr, out_reg_arr, anchor_stride, 0.5)

            if samo_nacrtaj:
                continue'''

            '''# get dimensions of gt output matrices
            out_c_rows, out_c_cols, depth = out_class.shape
            out_r_rows, out_r_cols, depth_r = out_reg.shape
            out_class_dims = [out_c_rows, out_c_cols, depth]
            out_reg_dims = [out_r_rows, out_r_cols, depth_r]

            # flatten classifier and regressor arrays
            out_class_flat = np.ndarray.flatten(out_class)
            out_reg_flat = np.ndarray.flatten(out_reg)

            # save resized image
            if flag_crtanje:
                resized = shifted.copy()
            # print("zacuvuvam gt")
            # save resized image
            im_name_new = 'image' + str(m).zfill(6) + '.jpg'
            cv2.imwrite(os.path.join(folder_augmented_photos, im_name_new), resized)

            # save gt of resized image
            filename = 'image' + str(m).zfill(6) + '.txt'

            fid = open(os.path.join(GroundTruthFiles, filename), 'wb+')
            pickle.dump(out_class_dims, fid)
            pickle.dump(out_class_flat, fid)
            pickle.dump(out_reg_dims, fid)
            pickle.dump(out_reg_flat, fid)
            fid.close()
            m += 1
            brojac+=1
            #
            # blur and save image
            im_name_new = 'image' + str(m).zfill(6) + '.jpg'
            resulting_image = cv2.filter2D(resized, -1, kernel)
            cv2.imwrite(os.path.join(folder_augmented_photos, im_name_new), resulting_image)

            # save gt of blurred resized image
            gt_blurred_filename = 'image' + str(m).zfill(6) + '.txt'
            fid_b = open(os.path.join(GroundTruthFiles, gt_blurred_filename), 'wb+')
            pickle.dump(out_class_dims, fid_b)
            pickle.dump(out_class_flat, fid_b)
            pickle.dump(out_reg_dims, fid_b)
            pickle.dump(out_reg_flat, fid_b)
            fid_b.close()
            m += 1

            # regular flip of resized image
            im_name_new = 'image' + str(m).zfill(6) + '.jpg'
            regular_flipped = cv2.flip(resized, 1)
            cv2.imwrite(os.path.join(folder_augmented_photos, im_name_new), regular_flipped)

            # flip and save gt of flipped resized image
            gt_flipped_filename = 'image' + str(m).zfill(6) + '.txt'
            fid_f = open(os.path.join(GroundTruthFiles, gt_flipped_filename), 'wb+')
            # flip gt

            out_class_f = np.fliplr(out_class)
            out_class_flat_f = np.ndarray.flatten(out_class_f)

            out_reg_f = np.fliplr(out_reg)
            out_reg_flip_f = np.ndarray.flatten(out_reg_f)

            pickle.dump(out_class_dims, fid_f)
            pickle.dump(out_class_flat_f, fid_f)
            pickle.dump(out_reg_dims, fid_f)
            pickle.dump(out_reg_flip_f, fid_f)
            fid_f.close()
            m += 1

            # blurred flip
            im_name_new = 'image' + str(m).zfill(6) + '.jpg'
            blured_flipped = cv2.flip(resulting_image, 1)
            cv2.imwrite(os.path.join(folder_augmented_photos, im_name_new), blured_flipped)

            # save gt of blurred flipped image
            gt_flipped_blurred_filename = 'image' + str(m).zfill(6) + '.txt'
            fid_f1 = open(os.path.join(GroundTruthFiles, gt_flipped_blurred_filename), 'wb+')
            # out_class_f = np.fliplr(out_class)
            pickle.dump(out_class_dims, fid_f1)
            pickle.dump(out_class_flat_f, fid_f1)
            pickle.dump(out_reg_dims, fid_f1)
            pickle.dump(out_reg_flip_f, fid_f1)
            fid_f1.close()
            m += 1

            # NOTE: augmentacii so brightness i kontrast
            if ('miladinovci' in dir_name) or ('kamera2' in dir_name) or ('videoframes' in dir_name) or ('drugo' in dir_name):
                # print('miladinovci')
                original_im = resized.copy()
                # print(dir_name)


            if 'soncevo' in dir_name:
                # print('soncevo')
                #NOTE: reduce brightness
                # im_name_new = 'image' + str(m).zfill(6) + '.jpg'
                # # print(im_name_new)
                # im_reduced_brightness=change_brightness.change_brightness(original_im,0.6)
                # cv2.imwrite(os.path.join(folder_augmented_photos, im_name_new), im_reduced_brightness)
                #
                # gt_reduced_brightness='image'+str(m).zfill(6) + '.txt'
                # fid_f1 = open(os.path.join(GroundTruthFiles, gt_reduced_brightness), 'wb+')
                # # out_class_f = np.fliplr(out_class)
                # pickle.dump(out_class_dims, fid_f1)
                # pickle.dump(out_class_flat, fid_f1)
                # pickle.dump(out_reg_dims, fid_f1)
                # pickle.dump(out_reg_flat, fid_f1)
                # fid_f1.close()
                # m += 1

                #NOTE: reduce contrast
                im_name_new = 'image' + str(m).zfill(6) + '.jpg'
                im_reduced_contrast = change_brightness.change_contrast(original_im, 0.6)
                cv2.imwrite(os.path.join(folder_augmented_photos, im_name_new), im_reduced_contrast)

                gt_reduced_contrast = 'image' + str(m).zfill(6) + '.txt'
                fid_f1 = open(os.path.join(GroundTruthFiles, gt_reduced_contrast), 'wb+')
                # out_class_f = np.fliplr(out_class)
                pickle.dump(out_class_dims, fid_f1)
                pickle.dump(out_class_flat, fid_f1)
                pickle.dump(out_reg_dims, fid_f1)
                pickle.dump(out_reg_flat, fid_f1)
                fid_f1.close()
                m += 1

                    # reduce brightness and contrast
                    # im_name_new = 'image' + str(m).zfill(6) + '.jpg'
                    # im_reduced_brightness_contrast = change_brightness.change_brightness(im_reduced_contrast, 0.8)
                    # cv2.imwrite(os.path.join(folder_augmented_photos, im_name_new), im_reduced_brightness_contrast)
                    # gt_reduced_brightness_contrast = 'image' + str(m).zfill(6) + '.txt'
                    # fid_f1 = open(os.path.join(GroundTruthFiles, gt_reduced_brightness_contrast), 'wb+')
                    # # out_class_f = np.fliplr(out_class)
                    # pickle.dump(out_class_dims, fid_f1)
                    # pickle.dump(out_class_flat, fid_f1)
                    # pickle.dump(out_reg_dims, fid_f1)
                    # pickle.dump(out_reg_flat, fid_f1)
                    # fid_f1.close()
                    # m += 1
            elif ('oblacno' in dir_name) or ('samrak' in dir_name):

                # #NOTE: increase brightness by 40%
                # im_name_new = 'image' + str(m).zfill(6) + '.jpg'
                # im_reduced_brightness = change_brightness.change_brightness(original_im, 1.4)
                # cv2.imwrite(os.path.join(folder_augmented_photos, im_name_new), im_reduced_brightness)
                #
                # gt_reduced_brightness = 'image' + str(m).zfill(6) + '.txt'
                # fid_f1 = open(os.path.join(GroundTruthFiles, gt_reduced_brightness), 'wb+')
                # # out_class_f = np.fliplr(out_class)
                # pickle.dump(out_class_dims, fid_f1)
                # pickle.dump(out_class_flat, fid_f1)
                # pickle.dump(out_reg_dims, fid_f1)
                # pickle.dump(out_reg_flat, fid_f1)
                # fid_f1.close()
                # m += 1
                # NOTE: increase contrast by 40%

                im_name_new = 'image' + str(m).zfill(6) + '.jpg'
                im_reduced_contrast = change_brightness.change_contrast(original_im, 1.4)
                cv2.imwrite(os.path.join(folder_augmented_photos, im_name_new), im_reduced_contrast)

                gt_reduced_contrast = 'image' + str(m).zfill(6) + '.txt'
                fid_f1 = open(os.path.join(GroundTruthFiles, gt_reduced_contrast), 'wb+')
                # out_class_f = np.fliplr(out_class)
                pickle.dump(out_class_dims, fid_f1)
                pickle.dump(out_class_flat, fid_f1)
                pickle.dump(out_reg_dims, fid_f1)
                pickle.dump(out_reg_flat, fid_f1)
                fid_f1.close()
                m += 1
            #
            #         # # reduce brightness and contrast
            #         # im_name_new = 'image' + str(m).zfill(6) + '.jpg'
            #         # im_reduced_brightness_contrast = change_brightness.change_brightness(im_reduced_contrast, 1.2)
            #         # cv2.imwrite(os.path.join(folder_augmented_photos, im_name_new), im_reduced_brightness_contrast)
            #         # gt_reduced_brightness_contrast = 'image' + str(m).zfill(6) + '.txt'
            #         # fid_f1 = open(os.path.join(GroundTruthFiles, gt_reduced_brightness_contrast), 'wb+')
            #         # # out_class_f = np.fliplr(out_class)
            #         # pickle.dump(out_class_dims, fid_f1)
            #         # pickle.dump(out_class_flat, fid_f1)
            #         # pickle.dump(out_reg_dims, fid_f1)
            #         # pickle.dump(out_reg_flat, fid_f1)
            #         # fid_f1.close()
            #         # m += 1'''

    print(nemapoz)
    print(nemaanot)
    print(nemaobjekt)
    print(prazenfajl)
    print(frli)
    print(anotacii_netxt)
    print(nemaslika)
    print(brojac)
    print(nemadir)
    print(anotacija)

    return site_bboxovi, site_pomestuvanja


GroundTruthFilesTrain1 = r'\\192.168.1.133\Monika\Trening_ZaDemo\GT_test'
os.makedirs(GroundTruthFilesTrain1, exist_ok=True)

# srcAnnotationsPathVal=r'E:\Science\Monika4\video4_anotacii'
# GroundTruthFilesVal=r'E:\Science\Monika4\GT_val'
# annot_video2=r'E:\Science\Monika4\video4_anotacii'
# dst=r'E:\Science\Monika4\empty_val'
# augmented_dst11 = r'\\192.168.1.133\Monika\Trening_ZaDemo\Images'
augmented_dst11=r'\\192.168.1.153\d\KlasifikacijaVozila\Miladinovci\IOUPairwise\augm-proba_test'
# if not os.path.exists(augmented_dst11):
os.makedirs(augmented_dst11, exist_ok=True)

root_im1 = r'\\192.168.1.153\d\KlasifikacijaVozila\Miladinovci1\ZaVoTrening'
root_annot1 = r'\\192.168.1.153\d\KlasifikacijaVozila\Miladinovci1\ZaVoTrening-Anotacii'

# path to save ground truth images with rectangles
resultspath = r'\\192.168.1.133\Monika\GTAnalysis\GT_Visualized_Original_Trening_ZaDemo_test'
os.makedirs(resultspath, exist_ok=True)

# path to save (original) resized images with annotations (rectangles) with no positives
no_positives = r'\\192.168.1.133\Monika\GTAnalysis\NoPositives_Original_Trening_ZaDemo_test'
os.makedirs(no_positives, exist_ok=True)

hard_negatives = r'\\192.168.1.153\d\KlasifikacijaVozila\Miladinovci1\Negatives_test'
m1 = 1
x = 2  # start number of pixels for image shifting to the right

site_bboxovi, site_pomestuvanja = read_data_and_generate_gt(root_im1, root_annot1, (imgDims['cols'], imgDims['rows']), img_depth, GroundTruthFilesTrain1, augmented_dst11, m1, resultspath, no_positives, x, hard_negatives,
                          samo_nacrtaj=False)

# -----------------------------------------------

# NOTE: TMP

site_bboxovi_2 = []
for bboxovi_slika in site_bboxovi:
    for bbox in bboxovi_slika:
        site_bboxovi_2.append(bbox)

site_dolu = []
for bbox in site_bboxovi_2:
    site_dolu.append(bbox[2])

site_dolu_2 = np.array(site_dolu)

# np.savetxt(r'C:\Users\User\Desktop\dolni_redici.txt', site_dolu_2, fmt='%d')

# ----------------------------------------

redici = []
pomestuvanja_red = []
pomestuvanja_kolona = []
dimenzija_visina = []
dimenzija_shirina = []

for x in site_pomestuvanja:

    if x is None:
        continue

    slika = x[:, :, :2]
    [r, c, d] = np.where(slika != 0)

    for i in range(len(r)):
        redici.append(r[i])
        pomestuvanja_red.append(abs(slika[r[i], c[i], 0]))
        pomestuvanja_kolona.append(abs(slika[r[i], c[i], 1]))

        dimenzija_visina.append(x[r[i], c[i], 2])
        dimenzija_shirina.append(x[r[i], c[i], 3])


redici_2 = np.array(redici)
# np.savetxt(r'C:\Users\User\Desktop\redici.txt', redici_2, fmt='%d')

pomestuvanja_red_2 = np.array(pomestuvanja_red)
# np.savetxt(r'C:\Users\User\Desktop\pomestuvanja_redici.txt', pomestuvanja_red, fmt='%d')

pomestuvanja_kolona_2 = np.array(pomestuvanja_kolona)
# np.savetxt(r'C:\Users\User\Desktop\pomestuvanja_koloni.txt', pomestuvanja_kolona, fmt='%d')

dimenzija_visina_2 = np.array(dimenzija_visina)
# np.savetxt(r'C:\Users\User\Desktop\dimenzija_visina.txt', dimenzija_visina_2, fmt='%f')

dimenzija_shirina_2 = np.array(dimenzija_shirina)
# np.savetxt(r'C:\Users\User\Desktop\dimenzija_shirina.txt', dimenzija_shirina_2, fmt='%f')

# --------------------------------------------------
# NOTE: normalizacija

# pozicija redici
regions = [0, 50, 100, 170, 341]    # regioni po redici (centroidi), vo resizuvana slika (ne vo kvantiziran prostor)
regions = [int(round(x / anchor_stride)) for x in regions]
regions = np.array(regions)

norm_coefs = [60, 100, 160, 190]   # vo vleznata slika
norm_coefs = [float(x)/img_dims[1] for x in norm_coefs]   # vo kvantiziraniot izlez

dimenzija_shirina_2_norm = deepcopy(dimenzija_shirina_2)

for i in range(len(redici_2)):

    redica = redici_2[i]
    regions_podeleni = regions / float(redica)
    inds = np.where(regions_podeleni < 1)
    region = np.max(inds)

    dimenzija_shirina_2_norm[i] = dimenzija_shirina_2_norm[i] / norm_coefs[region]

# --------------------------------------------------
# NOTE: histogram po regioni

import matplotlib.pyplot as plt

regions = [0, 50, 100, 170, 341]
regions = [int(round(x / anchor_stride)) for x in regions]
regions = np.array(regions).astype(np.float)

num_regions = len(regions) - 1

broj_po_regioni = [0] * num_regions

out = [[] for i in range(len(regions))]

for i in range(len(redici_2)):

    redica = redici_2[i]
    regions_podeleni = regions / redica
    inds = np.where(regions_podeleni < 1)
    region = np.max(inds)

    broj_po_regioni[region] += 1

    # out[region].append(pomestuvanja_red_2[i])
    out[region].append(np.int(np.round(dimenzija_shirina_2[i] * 512)))

print(broj_po_regioni)

plt.close()
reg = 3
plt.hist(out[reg], bins=len(np.unique(out[reg])))
plt.show()


# -------------------------------------------------
# NOTE: broj na primeroci po region

import matplotlib.pyplot as plt

regions = [0, 50, 120, 270, 341]
regions = [int(round(x / anchor_stride)) for x in regions]
regions = np.array(regions).astype(np.float)

num_regions = len(regions) - 1

broj_po_regioni = [0] * num_regions

for i in range(len(redici_2)):

    redica = redici_2[i]
    regions_podeleni = regions / redica

    inds = np.where(regions_podeleni < 1)

    region = np.max(inds)

    broj_po_regioni[region] += 1

print(broj_po_regioni)

fig, ax = plt.subplots()
plt.bar([x for x in range(len(regions) - 1)], broj_po_regioni)
for index, data in enumerate(broj_po_regioni):
    plt.text(x=index, y=data + 1, s=f"{data}", fontdict=dict(fontsize=10))
plt.show()


# --------------------------------------------------
# NOTE: grafik so site pomestuvanja po redica (prviot)

plt.scatter([x*8 for x in redici_2], pomestuvanja_red_2)
# plt.scatter([x*8 for x in redici_2], dimenzija_shirina_2_norm)
plt.axvline(x=50, color='r', linestyle='--')
plt.axvline(x=120, color='r', linestyle='--')
plt.axvline(x=270, color='r', linestyle='--')
plt.xlabel('redici')
plt.ylabel('pomestuvanja - redici')
# plt.ylabel('dimenzii - shirina (normalizirani)')
plt.show()

plt.scatter([x*8 for x in redici_2], pomestuvanja_kolona_2)
plt.axvline(x=50, color='r', linestyle='--')
plt.axvline(x=120, color='r',  linestyle='--')
plt.axvline(x=190, color='r', linestyle='--')
plt.xlabel('redici')
plt.ylabel('pomestuvanja - koloni')
plt.show()


plt.scatter([x*8 for x in redici_2], dimenzija_visina_2)
plt.xlabel('redici')
plt.ylabel('dimenzii - visina - normalizirani')
plt.show()

plt.scatter([x*8 for x in redici_2], dimenzija_visina_2 * 341)
plt.axvline(x=50, color='r',  linestyle='--')
plt.axvline(x=75, color='r', linestyle='--')
plt.axvline(x=100, color='r',  linestyle='--')
plt.axvline(x=150, color='r', linestyle='--')
plt.xlabel('redici')
plt.ylabel('dimenzii - visina - original resized')
plt.show()

plt.scatter([x*8 for x in redici_2], dimenzija_shirina_2)
plt.xlabel('redici')
plt.ylabel('dimenzii - shirina - normalizirani')
plt.show()

plt.scatter([x*8 for x in redici_2], dimenzija_shirina_2 * 512)
plt.axvline(x=50, color='r', linestyle='--')
plt.axvline(x=100, color='r',  linestyle='--')
plt.axvline(x=170, color='r', linestyle='--')
plt.xlabel('redici')
plt.ylabel('dimenzii - shirina - original resized')
plt.show()
