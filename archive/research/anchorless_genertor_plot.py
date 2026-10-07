from copy import deepcopy
import os
import numpy as np
import cv2
import pickle
import sys
import helper_postprocessing
# from tqdm import tqdm
import helper_anchorless
#from FCN_SSD_test_combined_output import full_hd_list, crop_left_list, crop_width_list

np.set_printoptions(threshold=sys.maxsize)


# # # za crtanje kade se pozitivnite prozorci
# # # iskomentirani se redovite za crtanje posle podesuvanjeto so regresor
# srcImages=r'D:\KlasifikacijaVozila\Miladinovci\augm-proba'
# groundtruthfiles=r'D:\KlasifikacijaVozila\Miladinovci\proba-gt'
# res=r'D:\KlasifikacijaVozila\Miladinovci\crtanje3'
# out_class_list=[]
# out_reg_list=[]
# image_list=[]
#
# filenames = [x for x in os.listdir(groundtruthfiles)]
# for filename in filenames:
#     print(filename)
#     br=int(filename[5:11])
#     if False: # (br<5158):
#         continue
#     else:
#         # print(br)
#         im_filename = filename[:-4]+'.jpg'
#         fid1 = open(os.path.join(groundtruthfiles, filename), 'rb')
#         out_class_dims = pickle.load(fid1)
#         out_class_back = pickle.load(fid1)
#         out_reg_dims = pickle.load(fid1)
#         out_reg_back = pickle.load(fid1)
#         fid1.close()
#         out_class = np.reshape(out_class_back, (out_class_dims[0], out_class_dims[1], out_class_dims[2]))
#         # print(out_class)
#         # reg_norm_coef = np.max(np.abs(out_reg))
#
#         [r, c, d] = np.where(out_class > 0.5)
#         # print(r, c)
#         maximum = np.max(out_class)
#         out_class_list.append(out_class)
#         out_reg = np.reshape(out_reg_back, (out_reg_dims[0], out_reg_dims[1], out_reg_dims[2]))
#         out_reg_list.append(out_reg)
#         image=cv2.imread(os.path.join(srcImages,im_filename))
#         image_list.append(image)
# out_class_list = np.array(out_class_list)
# out_reg_list = np.array(out_reg_list)


def save_results_anchorless_limits_pedestrians(results_path, results_path_nms, images, images_names, output_cls,output_reg, anchor_stride, prob_thr, reg_norm_coef_position_rows,
                                               reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width,
                                               thr_clustering, colors_list, flag_save_coords):
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
    regressor=True
    # print(len(images_names))
    # print(np.shape(images))
    img_dims = (images[0].shape[0], images[0].shape[1])     # height, width
    num_classes = output_cls.shape[3] - 1
    cls_orig=deepcopy(output_cls)
    # binarize classifier output probabilities

    # output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    if output_reg is not None:
        output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows     # regressor output
        output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_cols     # regressor output
        output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_size_height    # regressor output
        output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_size_width     # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))

    for im_ind, image in enumerate(images):
        lista_verojatnosti=[]
        image1 = image.copy()
        # coords_list = [0,0,280,468]
        broi=0
        image_za_nms = deepcopy(image)
        valid_bboxes = []  # list of bounding boxes with positive height and width

        res = output_cls[im_ind, :, :, 0]  # classifier output, probability maps for positive objects only

        [r, c] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)
        image_draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for pred_ind in range(len(r)):  # iterate over positive predictions
            # if d[pred_ind]==2:
            #     continue
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            if output_reg is not None:

                delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
                delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
                h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
                w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]
                #
                h = h_percent * img_dims[0]
                w = w_percent * img_dims[1]

            # bbox top left point
                min_row = int(center_row - np.round(h / 2))
                min_col = int(center_col - np.round(w / 2))

                 # adjust position and size with regressor predictions
                min_row_adj = int(min_row + delta_r)
                min_col_adj = int(min_col + delta_c)

                max_row_adj = min_row_adj + h
                max_col_adj = min_col_adj + w

                # plot bounding box onto image
                # clip bounding boxes falling out of the image borders
                min_col_adj = max(0, int(min_col_adj))
                min_row_adj = max(0, int(min_row_adj))
                max_col_adj = min(int(max_col_adj), img_dims[1])
                max_row_adj = min(int(max_row_adj), img_dims[0])

                cv2.rectangle(image_draw, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=(0, 255, 0), thickness=1)
            # valid_bboxes.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])
            # if d[pred_ind]==1:
            #     continue
            # else:
            #     broi+=1
            # print(d[pred_ind])
            # print(colors_list[d[pred_ind]])
            cv2.circle(image_draw, (center_col, center_row), 3, color=(255,0,0), thickness=3)  # plot object centers
            lista_verojatnosti.append(float(cls_orig[im_ind,r[pred_ind],c[pred_ind],0]))
            # cv2.circle((image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=colors_list[d[pred_ind]], thickness=1)
            # cv2.putText(image, str(d[pred_ind]), (center_col - 3, center_row - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors_list[d[pred_ind]],2)
            # coords_list.append([min_row_adj, min_col_adj, max_row_adj - min_row_adj, max_col_adj - min_col_adj, d[pred_ind]])
        # if d[pred_ind]==1:
        #     cv2.imwrite(os.path.join(results_path_kamioni, str(im_ind).zfill(5) + '.bmp'), image1) NOTE: ova koga sakam teski vozila da mi izvadi samo
        if len(r)==0:
            pass
            # cv2.imwrite(os.path.join(results_path, str(images_names)), image_draw)
        else:
            # if broi!=0:
            cv2.imwrite(os.path.join(results_path, str(images_names[:-4]) +'_PROB_'+str(max(lista_verojatnosti))+'_HUMAN.bmp'), image_draw)

        # if flag_save_coords:
        #     coords_list = np.array(coords_list)
        #     ime=str(images_names[im_ind][:-4])
        #     ime=ime+'.txt'
        #     print(ime)
        #     np.savetxt(os.path.join(results_path, ime), coords_list, delimiter=',', fmt='%i')

        # helper_postprocessing.nms_tanja(image_za_nms, os.path.join(results_path_nms, str(images_names[im_ind])), valid_bboxes, thr_clustering, colors_list, num_classes, flag_save_coords)



def save_results_anchorless_limits_pedestrians_original_matrices(images_paths, images, output_cls,prob_thr):
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

    img_dims = (images[0].shape[0], images[0].shape[1])     # height, width
    num_classes = output_cls.shape[3] - 1
    cls_orig=deepcopy(output_cls)
    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    nema_fp = 0
    for ind, image in enumerate(images):
        lista_verojatnosti=[]

        image1 = image.copy()

        res = output_cls[ind, :, :, :]  # classifier output, probability maps for positive objects only

        [r, c,d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)
        image_draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # out_c_rows, out_c_cols, depth = res.shape
        # parts_path=images_paths[ind].split("//")

        # annot_files = os.listdir(os.path.join(im_root, dataset, 'Annotations', dir))
        # save_path=os.path.join(results_path_root,,parts_path[-4],parts_path[-2],parts_path[-1])
        print(images_paths[ind])
        save_path = images_paths[ind].replace("Images", "Weights").replace(images_paths[ind].split("//")[-1][-4:],".txt")
        print(save_path)


        # for pred_ind in range(len(r)):  # iterate over positive predictions
        #     # if d[pred_ind]==2:
        #     #     continue
        #     center_row = r[pred_ind] * anchor_stride + start_r
        #     center_col = c[pred_ind] * anchor_stride + start_c
        #
        #     # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
        #     if output_reg is not None:
        #
        #         delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
        #         delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
        #         h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
        #         w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]
        #         #
        #         h = h_percent * img_dims[0]
        #         w = w_percent * img_dims[1]
        #
        #     # bbox top left point
        #         min_row = int(center_row - np.round(h / 2))
        #         min_col = int(center_col - np.round(w / 2))
        #
        #          # adjust position and size with regressor predictions
        #         min_row_adj = int(min_row + delta_r)
        #         min_col_adj = int(min_col + delta_c)
        #
        #         max_row_adj = min_row_adj + h
        #         max_col_adj = min_col_adj + w
        #
        #         # plot bounding box onto image
        #         # clip bounding boxes falling out of the image borders
        #         min_col_adj = max(0, int(min_col_adj))
        #         min_row_adj = max(0, int(min_row_adj))
        #         max_col_adj = min(int(max_col_adj), img_dims[1])
        #         max_row_adj = min(int(max_row_adj), img_dims[0])
        #
        #         cv2.rectangle(image_draw, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=(0, 255, 0), thickness=1)
            # valid_bboxes.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])
            # if d[pred_ind]==1:
            #     continue
            # else:
            #     broi+=1
            # print(d[pred_ind])
            # print(colors_list[d[pred_ind]])
            # cv2.circle(image_draw, (center_col, center_row), 3, color=(255,0,0), thickness=3)  # plot object centers
            # lista_verojatnosti.append(float(cls_orig[im_ind,r[pred_ind],c[pred_ind],0]))
            # cv2.circle((image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=colors_list[d[pred_ind]], thickness=1)
            # cv2.putText(image, str(d[pred_ind]), (center_col - 3, center_row - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors_list[d[pred_ind]],2)
            # coords_list.append([min_row_adj, min_col_adj, max_row_adj - min_row_adj, max_col_adj - min_col_adj, d[pred_ind]])
        # if d[pred_ind]==1:
        #     cv2.imwrite(os.path.join(results_path_kamioni, str(im_ind).zfill(5) + '.bmp'), image1) NOTE: ova koga sakam teski vozila da mi izvadi samo
        if len(r)==0:
            pass
            nema_fp+=1
            # cv2.imwrite(os.path.join(results_path, str(images_names)), image_draw)
        else:
            directory_path = os.path.dirname(save_path)
            os.makedirs(directory_path, exist_ok=True)
            fid_f = open(os.path.join(save_path), 'wb+')


            # Create the directory tree


            # fid_f1 = open(os.path.join(images_paths[ind],), 'wb+')
            out_c_rows, out_c_cols, depth = res.shape
            if depth>1:
                print(f'LOSO:{images_paths[ind]}')
            out_class_dims = [out_c_rows, out_c_cols, depth]

            # flatten classifier
            out_class_flat = np.ndarray.flatten(res)

            pickle.dump(out_class_dims, fid_f)
            pickle.dump(out_class_flat, fid_f)
            fid_f.close()
            # exit(1000000)

    print(nema_fp)
def save_results_anchorless_limits_save_original(results_path, results_path_nms, results_path_kamioni, images_s, image_orig, images_names, output_cls, output_reg, anchor_stride, prob_thr,
                                   reg_norm_coef_position_rows,
                                   reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width,
                                   thr_clustering, colors_list, left_crop, nacrtaj, flag_save_coords, flag_choose):
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
    # print(len(images_names))
    # print(np.shape(images))
    img_dims = (images_s[0].shape[0], images_s[0].shape[1])  # height, width
    print(images_s[0].shape[0])
    num_classes = output_cls.shape[3] - 1

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows  # regressor output
    output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_cols  # regressor output
    output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_size_height  # regressor output
    output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_size_width  # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))
    images = []
    # for im in images_s:
    #     slika=im[:,:,0]
    #     images.append(slika)

    for im_ind, image in tqdm(enumerate(images_s)):
        img_orig = deepcopy(image)
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # image=image[:,:,0]
        # image=image.reshape(image.shape+(1,))
        # image_draw=image
        # image_draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        coords_list = []
        # cv2.imshow('sl',image)
        image_za_nms = deepcopy(image_orig[0])
        valid_bboxes = []  # list of bounding boxes with positive height and width

        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions
            # ako e klasa nevozilo ne crtaj i ne zacucuvaj koordinati
            if d[pred_ind] == 2:
                continue
            # print(d[pred_ind])
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            # center_row_reg = int(center_row + delta_r)
            # center_col_reg = int(center_col + delta_c)

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]
            print(h)
            print(w)
            # bbox top left point
            min_row = int(center_row - np.round(h / 2))
            min_col = int(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = int(min_row + delta_r)
            min_col_adj = int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, int(min_col_adj))
            min_row_adj = max(0, int(min_row_adj))
            max_col_adj = min(int(max_col_adj), img_dims[1])
            max_row_adj = min(int(max_row_adj), img_dims[0])

            center_row_reg = int(min_row_adj + np.round(h / 2))
            center_col_reg = int(min_col_adj + np.round(w / 2))
            # print(center_row_reg)
            # print(center_col_reg)

            if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:
                valid_bboxes.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])

                cv2.circle(image, (center_col, center_row), 1, color=(255, 0, 0), thickness=2)  # plot object centers
                # cv2.circle(image, (center_col_reg, center_row_reg), 1, color=(255, 0, 0), thickness=1)     # plot object centers
                # image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
                cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=colors_list[d[pred_ind]], thickness=1)

                # coords_list.append([min_row_adj, min_col_adj, max_row_adj - min_row_adj, max_col_adj - min_col_adj, 6])
                coords_list.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj])
            # if d[pred_ind]==1:
        # cv2.imwrite(os.path.join(results_path, images_names[im_ind]), image) #NOTE: ova koga sakam teski vozila da mi izvadi samo


        if len(r) != 0:
            cv2.imwrite(os.path.join(results_path, str(images_names[im_ind])), image)
        #     # cv2.imwrite(os.path.join(results_path, str(images_names[im_ind])[:-4]+'_ORIG.bmp'),img_orig)
        #     # cv2.imwrite(os.path.join(results_path, images_names[im_ind]), image)
        #
        #     if flag_save_coords:
        #         coords_list = np.array(coords_list)
        #         ime = str(images_names[im_ind][:-4])
        #         ime = ime + '.txt'
        #         print(ime)
        #         np.savetxt(os.path.join(results_path, ime), coords_list, delimiter=',', fmt='%i')

            result = helper_postprocessing.nms_new_save_original(image_za_nms, os.path.join(results_path_nms, str(images_names[im_ind])), valid_bboxes, thr_clustering, colors_list, num_classes,
                                                   left_crop, nacrtaj, flag_save_coords, flag_choose)


def save_results_anchorless_limits(results_path, results_path_nms, results_path_kamioni, images_s, images_color,crop_left_list,crop_width_list,full_hd_list,images_names, output_cls, output_reg, anchor_stride, prob_thr,
                                   reg_norm_coef_position_rows,
                                   reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width,
                                   thr_clustering, colors_list,  left_crop, nacrtaj, flag_save_coords,flag_save_coords_full_hd,flag_choose):
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
    # print(len(images_names))
    # print(np.shape(images))
    img_dims = (images_s[0].shape[0], images_s[0].shape[1])  # height, width
    print(images_s[0].shape[0])
    num_classes = output_cls.shape[3] - 1

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows  # regressor output
    output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_cols  # regressor output
    output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_size_height  # regressor output
    output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_size_width  # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))
    images=[]
    # for im in images_s:
    #     slika=im[:,:,0]
    #     images.append(slika)
    orig_h = 1080
    orig_w = 1920

    resized_h = img_dims[0]  # 341
    resized_w = img_dims[1]  # 512

    # You must pass these two to the function
    # crop_left_px = <left side crop amount>
    # crop_width_px = <width of the cropped region>

    for im_ind, image in enumerate(images_s):
        scale_x = crop_width_list[im_ind] / resized_w
        scale_y = orig_h / resized_h
        img_orig=deepcopy(image)
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # image=image[:,:,0]
        #image=image.reshape(image.shape+(1,))
        # image_draw=image
        # image_draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        coords_list = []
        coords_list_full_hd=[]
        #cv2.imshow('sl',image)
        image_za_nms = deepcopy(images_color[im_ind])
        valid_bboxes = []  # list of bounding boxes with positive height and width

        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions
            #ako e klasa nevozilo ne crtaj i ne zacucuvaj koordinati
            if d[pred_ind]==2:
                continue
           #print(d[pred_ind])
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]*1.39128367347
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            # center_row_reg = int(center_row + delta_r)
            # center_col_reg = int(center_col + delta_c)

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]
            print(h)
            print(w)
            # bbox top left point
            min_row = int(center_row - np.round(h / 2))
            min_col = int(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = int(min_row + delta_r)
            min_col_adj = int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, int(min_col_adj))
            min_row_adj = max(0, int(min_row_adj))
            max_col_adj = min(int(max_col_adj), img_dims[1])
            max_row_adj = min(int(max_row_adj), img_dims[0])

            # scale back
            full_min_r = int(min_row_adj * scale_y)
            full_max_r = int(max_row_adj * scale_y)

            full_min_c = int(min_col_adj * scale_x + crop_left_list[im_ind])
            full_max_c = int(max_col_adj * scale_x + crop_left_list[im_ind])

            center_row_reg = int(min_row_adj + np.round(h / 2))
            center_col_reg = int(min_col_adj + np.round(w / 2))
            #print(center_row_reg)
            #print(center_col_reg)

            if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:
                valid_bboxes.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])
                #cv2.circle(image, (center_col, center_row), 1, color=(255, 0, 0), thickness=2)     # plot object centers
                # cv2.circle(image, (center_col_reg, center_row_reg), 1, color=(255, 0, 0), thickness=1)     # plot object centers
                # image=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
                cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=colors_list[d[pred_ind]], thickness=1)
                # coords_list.append([min_row_adj, min_col_adj, max_row_adj - min_row_adj, max_col_adj - min_col_adj, 6])
                coords_list.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])
                coords_list_full_hd.append([full_min_r, full_min_c,
                                            full_max_r, full_max_c,
                                            d[pred_ind]])

            # if d[pred_ind]==1:
        #cv2.imwrite(os.path.join(results_path, images_names[im_ind]), image) #NOTE: ova koga sakam teski vozila da mi izvadi samo

           #NOTE: ova koga sakam teski vozila da mi izvadi samo
        #cv2.imshow('sl2',image)0
        #cv2.waitKey(0)
        if len(r)!=0:
            #cv2.imshow('sl',images_color[im_ind])
            #success =cv2.imwrite(os.path.join(results_path, str(images_names[im_ind][:-4]+'.png')),full_hd_list[im_ind])
            cv2.imwrite(os.path.join(results_path, str(images_names[im_ind][:-4] + '_alldets.png')), image)
            print("saved")
            #print(os.path.join(results_path, str(full_hd_list[im_ind][:-4]+'.png')))
        # cv2.imwrite(os.path.join(results_path, str(images_names[im_ind])[:-4]+'_ORIG.bmp'),img_orig)
        #cv2.imwrite(os.path.join(results_path, images_names[im_ind]), image)
            if flag_save_coords:
                coords_list = np.array(coords_list)
                ime = str(images_names[im_ind][:-4])
                ime = ime + '.txt'
                print(ime)
                np.savetxt(os.path.join(results_path, ime), coords_list, delimiter=',', fmt='%i')
            if flag_save_coords_full_hd:
                coords_list = np.array(coords_list_full_hd)
                ime = str(images_names[im_ind][:-4])
                ime = ime + '.txt'
                print(ime)
                np.savetxt(os.path.join(results_path, ime), coords_list, delimiter=',', fmt='%i')

            result=helper_postprocessing.nms_new(image_za_nms, os.path.join(results_path_nms, str(images_names[im_ind])), valid_bboxes, thr_clustering, colors_list, num_classes, left_crop, nacrtaj,flag_save_coords, flag_choose)
        # if isinstance(result, tuple):
        #     img_postproc, finalwindows=result
        #     if flag_choose:
        #         cv2.imshow('slika', img_postproc)
        #         k = cv2.waitKey(0)
        #         if k==ord('Y'):
        #             print('Good')
        #             cv2.imwrite(os.path.join(results_path_nms, str(images_names[im_ind])), img_orig)
        #             np.savetxt(os.path.join(results_path_nms, str(images_names[im_ind][:-4]) + '.txt'), finalwindows, delimiter=',', fmt='%i')


def save_results_anchorless_limits_probability_maps(results_path, results_path_nms, results_path_kamioni, images_s, images_names, output_cls, output_reg, anchor_stride, prob_thr,
                                   reg_norm_coef_position_rows,
                                   reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width,
                                   thr_clustering, colors_list,  flag_save_coords):
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
    # print(len(images_names))
    # print(np.shape(images))
    img_dims = (images_s[0].shape[0], images_s[0].shape[1])  # height, width
    num_classes = output_cls.shape[3] - 1

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows  # regressor output
    output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_cols  # regressor output
    output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_size_height  # regressor output
    output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_size_width  # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))
    images=[]
    # for im in images_s:
    #     slika=im[:,:,0]
    #     images.append(slika)

    for im_ind, image in tqdm(enumerate(images_s)):
        # image=image[:,:,0]
        #image=image.reshape(image.shape+(1,))
        # image_draw=image
        # image_draw = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        coords_list = []
        #cv2.imshow('sl',image)
        image_za_nms = deepcopy(image)
        valid_bboxes = []  # list of bounding boxes with positive height and width

        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions

           #print(d[pred_ind])
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            # center_row_reg = int(center_row + delta_r)
            # center_col_reg = int(center_col + delta_c)

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]
            print(h)
            print(w)
            # bbox top left point
            min_row = int(center_row - np.round(h / 2))
            min_col = int(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = int(min_row + delta_r)
            min_col_adj = int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, int(min_col_adj))
            min_row_adj = max(0, int(min_row_adj))
            max_col_adj = min(int(max_col_adj), img_dims[1])
            max_row_adj = min(int(max_row_adj), img_dims[0])

            center_row_reg = int(min_row_adj + np.round(h / 2))
            center_col_reg = int(min_col_adj + np.round(w / 2))
            #print(center_row_reg)
            #print(center_col_reg)

            if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:
                valid_bboxes.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])

                # cv2.circle(image, (center_col, center_row), 1, color=(255, 255, 255), thickness=1)     # plot object centers
                # cv2.circle(image, (center_col_reg, center_row_reg), 1, color=(255, 0, 0), thickness=1)     # plot object centers
                cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=colors_list[d[pred_ind]], thickness=1)
                cv2.putText(image,str(output_cls[im_ind,r[pred_ind],c[pred_ind],d[pred_ind]]),(min_col_adj,min_row_adj),color=colors_list[d[pred_ind]],thickness=2)

                coords_list.append([min_row_adj, min_col_adj, max_row_adj - min_row_adj, max_col_adj - min_col_adj, 6])
            # if d[pred_ind]==1:
            #     cv2.imwrite(os.path.join(results_path_kamioni, str(im_ind).zfill(5) + '.bmp'), image1) NOTE: ova koga sakam teski vozila da mi izvadi samo
        #cv2.imshow('sl2',image)
        #cv2.waitKey(0)
        if len(r)!=0:
            cv2.imwrite(os.path.join(results_path, str(images_names[im_ind])),image)

            if flag_save_coords:
                coords_list = np.array(coords_list)
                ime = str(images_names[im_ind][:-4])
                ime = ime + '.txt'
                print(ime)
                np.savetxt(os.path.join(results_path, ime), coords_list, delimiter=',', fmt='%i')

        # helper_postprocessing.nms_new(image_za_nms, os.path.join(results_path_nms, str(images_names[im_ind])), valid_bboxes, thr_clustering, colors_list, num_classes,
        #                                 flag_save_coords)
def save_results_anchorless_limits_save_empty_frames(results_path, images_save, images, images_names, output_cls,  anchor_stride, prob_thr,broi,coords_list):
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
    # print(len(images_names))
    # print(np.shape(images))
    img_dims = (images[0].shape[0], images[0].shape[1])  # height, width
    # num_classes = output_cls.shape[3] - 1
    # binarize classifier output probabilities
    # output_cls[output_cls >= prob_thr] = 1
    # output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    # output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows  # regressor output
    # output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_cols  # regressor output
    # output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_size_height  # regressor output
    # output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_size_width  # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))

    for im_ind, image in tqdm(enumerate(images_save)):
        image1 = image.copy()
        # coords_list = []

        image_za_nms = deepcopy(image)
        valid_bboxes = []  # list of bounding boxes with positive height and width

        # res = output_cls[im_ind, :, :, :-2]  # classifier output, probability maps for positive objects only

        # [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        # if len(r)!=0:
        #     return broi
        # else:

        cv2.imwrite(os.path.join(results_path, str(images_names[im_ind])), image)
        # coords_list=[40,330,90,370,6]
        np.savetxt(os.path.join(results_path,str(images_names[im_ind])[:-4]+'.txt'), coords_list, delimiter=',', fmt='%i')
        broi+=1

    return broi

            # coords_list.append([0, 0, 540, 960, 6])

        #
        #
        # for pred_ind in range(len(r)):  # iterate over positive predictions
        #
        #     print(d[pred_ind])
        #     center_row = r[pred_ind] * anchor_stride + start_r
        #     center_col = c[pred_ind] * anchor_stride + start_c
        #
        #     # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
        #     delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
        #     delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
        #     h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
        #     w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]
        #
        #     # center_row_reg = int(center_row + delta_r)
        #     # center_col_reg = int(center_col + delta_c)
        #
        #     h = h_percent * img_dims[0]
        #     w = w_percent * img_dims[1]
        #
        #     # bbox top left point
        #     min_row = int(center_row - np.round(h / 2))
        #     min_col = int(center_col - np.round(w / 2))
        #
        #     # adjust position and size with regressor predictions
        #     min_row_adj = int(min_row + delta_r)
        #     min_col_adj = int(min_col + delta_c)
        #
        #     max_row_adj = min_row_adj + h
        #     max_col_adj = min_col_adj + w
        #
        #     # plot bounding box onto image
        #     # clip bounding boxes falling out of the image borders
        #     min_col_adj = max(0, int(min_col_adj))
        #     min_row_adj = max(0, int(min_row_adj))
        #     max_col_adj = min(int(max_col_adj), img_dims[1])
        #     max_row_adj = min(int(max_row_adj), img_dims[0])
        #
        #     center_row_reg = int(min_row_adj + np.round(h / 2))
        #     center_col_reg = int(min_col_adj + np.round(w / 2))
        #
        #     if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:
        #         valid_bboxes.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])
        #
        #         # cv2.circle(image, (center_col, center_row), 1, color=(255, 0, 0), thickness=1)     # plot object centers
        #         # cv2.circle(image, (center_col_reg, center_row_reg), 1, color=(0, 0, 255), thickness=1)     # plot object centers
        #         cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=colors_list[d[pred_ind]], thickness=1)
        #
        #         coords_list.append([min_row_adj, min_col_adj, max_row_adj - min_row_adj, max_col_adj - min_col_adj, d[pred_ind]])
        #     # if d[pred_ind]==1:
        #     #     cv2.imwrite(os.path.join(results_path_kamioni, str(im_ind).zfill(5) + '.bmp'), image1) NOTE: ova koga sakam teski vozila da mi izvadi samo
        #
        # cv2.imwrite(os.path.join(results_path, str(images_names[im_ind])), image)
        #
        # # if flag_save_coords:
        # #     coords_list = np.array(coords_list)
        # #     ime = str(images_names[im_ind][:-4])
        # #     ime = ime + '.txt'
        # #     print(ime)
        # #     np.savetxt(os.path.join(results_path, ime), coords_list, delimiter=',', fmt='%i')
        #
        # helper_postprocessing.nms_new(image_za_nms, os.path.join(results_path_nms, str(images_names[im_ind])), valid_bboxes, thr_clustering, colors_list, num_classes,
        #                                 flag_save_coords)

def save_results_anchorless_limits_distance(results_path, results_path_nms,images, images_names, output_cls, output_reg, anchor_stride, prob_thr,reg_norm_coef_position_rows
                                            ,reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width,
                                            thr_clustering, colors_list, flag_save_coords,flag_after_regressor, normAll):
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

    # print(len(images_names))
    # print(np.shape(images))
    img_dims = (images[0].shape[0], images[0].shape[1])  # height, width
    num_classes = output_cls.shape[3] - 1
    color_plot_small=(0,255,0)
    color_plot_big=(255,0,0)
    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    if normAll:
        output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows  # regressor output
        output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_cols  # regressor output
        output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_size_height # regressor output
        output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_size_width  # regressor output
    else:
        output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows  # regressor output
        output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_rows  # regressor output
        output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_position_rows  # regressor output
        output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_position_rows # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))

    for im_ind, image in enumerate(images):

        coords_list = []  # za zapishuvanje

        image_za_nms = deepcopy(image)
        valid_bboxes = []  # list of bounding boxes with positive height and width

        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # pomestuvanja na granicite na bboxot
            min_r = max(0,int(center_row - output_reg[im_ind, r[pred_ind], c[pred_ind], 0]))
            min_c = max(0,int(center_col - output_reg[im_ind, r[pred_ind], c[pred_ind], 1]))
            max_r = max(0,int(center_row + output_reg[im_ind, r[pred_ind], c[pred_ind], 2]))
            max_c = max(0,int(center_col + output_reg[im_ind, r[pred_ind], c[pred_ind], 3]))

            min_r_adj=min_r
            min_c_adj=min_c
            max_r_adj=max_r
            max_c_adj=max_c
            # if np.any(output_reg[im_ind, r[pred_ind],c[pred_ind], :]<0):
            #     continue
            if max_r<min_r or max_c<min_c:
                continue
            if d[pred_ind] == 2:
                continue
            if not flag_after_regressor:
                coords_list.append([min_r, min_c, max_r, max_c, d[pred_ind]])
                if d[pred_ind] == 0:

                    # cv2.circle(image, (center_col, center_row), 1, color=color_plot_small, thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_c, min_r), (max_c, max_r), color=color_plot_small, thickness=1)
                elif d[pred_ind] == 1:
                    # cv2.circle(image, (center_col, center_row), 1, color=color_plot_big, thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_c, min_r), (max_c, max_r), color=color_plot_big, thickness=1)
                else:
                    cv2.rectangle(image, (min_c, min_r), (max_c, max_r), color=(255, 255, 0), thickness=1)

                # save test image with bounding boxes of detected objects
                # if flag_save_coords:
                #     coords_list1 = np.array(coords_list)
                #     ime = str(images_names[im_ind][:-4])
                #     ime = ime + '_class.txt'
                #     # print(ime)
                #     np.savetxt(os.path.join(results_path, ime), coords_list1, delimiter=',', fmt='%i')

            else:
                coords_list.append([min_r_adj, min_c_adj, max_r_adj, max_c_adj, d[pred_ind]])
                if d[pred_ind] == 0:
                    # cv2.circle(image, (center_col, center_row), 1, color=(255, 255, 0), thickness=1)  # plot object centers for classifier
                    #
                    # cv2.circle(image, (center_col_reg, center_row_reg), 1, color=(255, 0, 0), thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_c_adj, min_r_adj), (max_c_adj, max_r_adj), color=color_plot_small, thickness=1)
                elif d[pred_ind] == 1:
                    # cv2.circle(image, (center_col, center_row), 1, color=(255, 255, 0), thickness=1)  # plot object centers
                    #
                    # cv2.circle(image, (center_col_reg, center_row_reg), 1, color=color_plot_big, thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_c_adj, min_r_adj), (max_c_adj, max_r_adj), color=color_plot_big, thickness=1)
                else:
                    cv2.rectangle(image, (min_c, min_r), (max_c, max_r), color=(255, 255, 0), thickness=1)

        if flag_save_coords:
            coords_list1 = np.array(coords_list)
            ime = str(images_names[im_ind][:-4])
            if flag_after_regressor:
                ime = ime + '_reg.txt'
            else:
                ime = ime + '_class.txt'
            # print(ime)
            np.savetxt(os.path.join(results_path, ime), coords_list1, delimiter=',', fmt='%i')

        #cnt += 1
        if not flag_after_regressor:
            cv2.imwrite(os.path.join(results_path, str(images_names[im_ind][:-4]) + '_class.bmp'), image)
        else:
            cv2.imwrite(os.path.join(results_path, str(images_names[im_ind][:-4]) + '_reg.bmp'), image)



        #
        # if (max_r - min_r) > 0 and (max_c - min_c) > 0:
        #     valid_bboxes.append([min_r, min_c, max_r, max_c, d[pred_ind]])
        #
        #         # cv2.circle(image, (center_col, center_row), 1, color=(255, 0, 0), thickness=1)  # plot object centers
        #         cv2.rectangle(image, (min_c, min_r), (max_c, max_r), color=colors_list[d[pred_ind]], thickness=1)
        #
        #         coords_list.append([min_r, min_c, max_r - min_r, max_c - min_c, d[pred_ind]])  # za zapishuvanje
        #
        # cv2.imwrite(os.path.join(results_path, str(images_names[im_ind])), image)
        #
        # if flag_save_coords:
        #     coords_list = np.array(coords_list)
        #     ime = str(images_names[im_ind][:-4])
        #     ime = ime + '.txt'
        #     print(ime)
        #     np.savetxt(os.path.join(results_path, ime), coords_list, delimiter=',', fmt='%i')
        #


        # helper_postprocessing.nms_tanja(image_za_nms, os.path.join(results_path_nms, str(images_names[im_ind])), coords_list, thr_clustering, colors_list, num_classes,
        #                                 flag_save_coords)


def save_results_anchorless_limits_distance_centerness(results_path, results_path_nms,images, images_names, output_cls, out_centerness, output_reg, anchor_stride, prob_thr,reg_norm_coef_position_rows
                                            ,reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width,
                                            thr_clustering, colors_list, flag_save_coords,flag_after_regressor, normAll):
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
    # print(len(images_names))
    # print(np.shape(images))
    img_dims = (images[0].shape[0], images[0].shape[1])  # height, width
    num_classes = output_cls.shape[3] - 1
    color_plot_small=(0,255,0)
    color_plot_big=(255,0,0)
    # binarize classifier output probabilities
    output_cls=np.sqrt(np.multiply(output_cls,out_centerness))
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    if normAll:
        output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows  # regressor output
        output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_cols  # regressor output
        output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_size_height # regressor output
        output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_size_width  # regressor output
    else:
        output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows  # regressor output
        output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_rows  # regressor output
        output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_position_rows  # regressor output
        output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_position_rows # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))

    for im_ind, image in tqdm(enumerate(images)):

        coords_list = []  # za zapishuvanje

        image_za_nms = deepcopy(image)
        valid_bboxes = []  # list of bounding boxes with positive height and width

        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # pomestuvanja na granicite na bboxot
            min_r = max(0,int(center_row - output_reg[im_ind, r[pred_ind], c[pred_ind], 0]))
            min_c = max(0,int(center_col - output_reg[im_ind, r[pred_ind], c[pred_ind], 1]))
            max_r = max(0,int(center_row + output_reg[im_ind, r[pred_ind], c[pred_ind], 2]))
            max_c = max(0,int(center_col + output_reg[im_ind, r[pred_ind], c[pred_ind], 3]))

            min_r_adj=min_r
            min_c_adj=min_c
            max_r_adj=max_r
            max_c_adj=max_c

            if not flag_after_regressor:
                coords_list.append([min_r, min_c, max_r, max_c, d[pred_ind]])
                if d[pred_ind] == 0:

                    # cv2.circle(image, (center_col, center_row), 1, color=color_plot_small, thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_c, min_r), (max_c, max_r), color=color_plot_small, thickness=1)
                elif d[pred_ind] == 1:
                    # cv2.circle(image, (center_col, center_row), 1, color=color_plot_big, thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_c, min_r), (max_c, max_r), color=color_plot_big, thickness=1)
                else:
                    print('Invalid class.')
                    exit(1)

                # save test image with bounding boxes of detected objects
                # if flag_save_coords:
                #     coords_list1 = np.array(coords_list)
                #     ime = str(images_names[im_ind][:-4])
                #     ime = ime + '_class.txt'
                #     # print(ime)
                #     np.savetxt(os.path.join(results_path, ime), coords_list1, delimiter=',', fmt='%i')

            else:
                coords_list.append([min_r_adj, min_c_adj, max_r_adj, max_c_adj, d[pred_ind]])
                if d[pred_ind] == 0:
                    # cv2.circle(image, (center_col, center_row), 1, color=(255, 255, 0), thickness=1)  # plot object centers for classifier
                    #
                    # cv2.circle(image, (center_col_reg, center_row_reg), 1, color=(255, 0, 0), thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_c_adj, min_r_adj), (max_c_adj, max_r_adj), color=(255, 0, 0), thickness=1)
                elif d[pred_ind] == 1:
                    # cv2.circle(image, (center_col, center_row), 1, color=(255, 255, 0), thickness=1)  # plot object centers
                    #
                    # cv2.circle(image, (center_col_reg, center_row_reg), 1, color=color_plot_big, thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_c_adj, min_r_adj), (max_c_adj, max_r_adj), color=color_plot_big, thickness=1)
                else:
                    print('Invalid class.')
                    exit(1)

        if flag_save_coords:
            coords_list1 = np.array(coords_list)
            ime = str(images_names[im_ind][:-4])
            if flag_after_regressor:
                ime = ime + '_reg.txt'
            else:
                ime = ime + '_class.txt'
            # print(ime)
            np.savetxt(os.path.join(results_path, ime), coords_list1, delimiter=',', fmt='%i')

        #cnt += 1
        if not flag_after_regressor:
            cv2.imwrite(os.path.join(results_path, str(images_names[im_ind][:-4]) + '_class.bmp'), image)
        else:
            cv2.imwrite(os.path.join(results_path, str(images_names[im_ind][:-4]) + '_reg.bmp'), image)

        helper_postprocessing.nms_tanja(image_za_nms, os.path.join(results_path_nms, str(images_names[im_ind])), coords_list, thr_clustering, colors_list, num_classes,
                                        flag_save_coords)

def save_results_anchorless_limits_position_size(results_path, results_path_nms, results_path_kamioni, images, images_names, output_cls, output_reg, anchor_stride, prob_thr,
                                                 reg_norm_coef_position,
                                                 reg_norm_coef_size,
                                                 thr_clustering, colors_list, flag_save_coords):
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

    img_dims = (images[0].shape[0], images[0].shape[1])  # height, width
    num_classes = output_cls.shape[3] - 1

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    output_reg[:, :, :, 0:2] = output_reg[:, :, :, 0:2] * reg_norm_coef_position  # regressor output
    output_reg[:, :, :, 2:] = output_reg[:, :, :, 2:] * reg_norm_coef_size  # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))

    for im_ind, image in tqdm(enumerate(images)):
        image1 = image.copy()
        coords_list = []

        image_za_nms = deepcopy(image)
        valid_bboxes = []  # list of bounding boxes with positive height and width

        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]

            # bbox top left point
            min_row = int(center_row - np.round(h / 2))
            min_col = int(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = int(min_row + delta_r)
            min_col_adj = int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, int(min_col_adj))
            min_row_adj = max(0, int(min_row_adj))
            max_col_adj = min(int(max_col_adj), img_dims[1])
            max_row_adj = min(int(max_row_adj), img_dims[0])

            if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:
                valid_bboxes.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])

                # cv2.circle(image, (center_col, center_row), 3, color=(255, 0, 0), thickness=3)     # plot object centers
                cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=colors_list[d[pred_ind]], thickness=1)

                coords_list.append([min_row_adj, min_col_adj, max_row_adj - min_row_adj, max_col_adj - min_col_adj, d[pred_ind]])
            # if d[pred_ind]==1:
            #     cv2.imwrite(os.path.join(results_path_kamioni, str(im_ind).zfill(5) + '.bmp'), image1) NOTE: ova koga sakam teski vozila da mi izvadi samo

        cv2.imwrite(os.path.join(results_path, str(images_names[im_ind])), image)

        if flag_save_coords:
            coords_list = np.array(coords_list)
            ime = str(images_names[im_ind][:-4])
            ime = ime + '.txt'
            np.savetxt(os.path.join(results_path, ime), coords_list, delimiter=',', fmt='%i')

        helper_postprocessing.nms_tanja(image_za_nms, os.path.join(results_path_nms, str(images_names[im_ind])), valid_bboxes, thr_clustering, colors_list, num_classes,
                                        flag_save_coords)


def save_results_anchorless_limits_normalizedByRegion(results_path, results_path_nms, results_path_kamioni, images, output_cls, output_reg, anchor_stride, prob_thr, regions,
                                                      norm_coeff, dest_matrix,
                                                      thr_clustering, colors_list, flag_save_coords):
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

    img_dims = (images[0].shape[0], images[0].shape[1])  # height, width
    num_classes = output_cls.shape[3] - 1

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    for ind, region in enumerate(regions):
        print(region)
        print(norm_coeff[ind])
        output_reg = helper_anchorless.reverse_normalization(region, norm_coeff[ind], dest_matrix[ind], output_reg, anchor_stride)

    # output_reg[:, :, :, 0:2] = output_reg[:, :, :, 0:2] * norm_coef_position     # regressor output
    # output_reg[:, :, :, 2:] = output_reg[:, :, :, 2:] * norm_coef_size     # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))

    for im_ind, image in tqdm(enumerate(images)):
        image1 = image.copy()
        coords_list = []

        image_za_nms = deepcopy(image)
        valid_bboxes = []  # list of bounding boxes with positive height and width

        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]

            # bbox top left point
            min_row = int(center_row - np.round(h / 2))
            min_col = int(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = int(min_row + delta_r)
            min_col_adj = int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, int(min_col_adj))
            min_row_adj = max(0, int(min_row_adj))
            max_col_adj = min(int(max_col_adj), img_dims[1])
            max_row_adj = min(int(max_row_adj), img_dims[0])

            if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:
                valid_bboxes.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])

                # cv2.circle(image, (center_col, center_row), 3, color=(255, 0, 0), thickness=3)     # plot object centers
                cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=colors_list[d[pred_ind]], thickness=1)

                coords_list.append([min_row_adj, min_col_adj, max_row_adj - min_row_adj, max_col_adj - min_col_adj, d[pred_ind]])
            # if d[pred_ind]==1:
            #     cv2.imwrite(os.path.join(results_path_kamioni, str(im_ind).zfill(5) + '.bmp'), image1) NOTE: ova koga sakam teski vozila da mi izvadi samo

        cv2.imwrite(os.path.join(results_path, str(im_ind).zfill(5) + '.bmp'), image)

        if flag_save_coords:
            coords_list = np.array(coords_list)
            np.savetxt(os.path.join(results_path, str(im_ind).zfill(5) + '.txt'), coords_list, delimiter=',', fmt='%i')

        helper_postprocessing.nms_tanja(image_za_nms, os.path.join(results_path_nms, str(im_ind).zfill(5) + '.bmp'), valid_bboxes, thr_clustering, colors_list, num_classes,
                                        flag_save_coords)


def save_results_anchorless_limits_with_metrics(results_path, results_path_nms, results_path_kamioni, images, output_cls, output_reg, anchor_stride, prob_thr, norm_coef_position,
                                                norm_coef_size,
                                                thr_clustering, colors_list, flag_save_coords):
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

    img_dims = (images[0].shape[0], images[0].shape[1])  # height, width
    num_classes = output_cls.shape[3] - 1

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # round regressor output and cast to integer pixel values
    output_reg[:, :, :, 0:2] = output_reg[:, :, :, 0:2] * norm_coef_position  # regressor output
    output_reg[:, :, :, 2:] = output_reg[:, :, :, 2:] * norm_coef_size  # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))
    all_bboxes = []
    for im_ind, image in tqdm(enumerate(images)):
        image1 = image.copy()
        coords_list = []

        image_za_nms = deepcopy(image)
        valid_bboxes = []  # list of bounding boxes with positive height and width

        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]

            # bbox top left point
            min_row = int(center_row - np.round(h / 2))
            min_col = int(center_col - np.round(w / 2))

            # adjust position and size with regressor predictions
            min_row_adj = int(min_row + delta_r)
            min_col_adj = int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, int(min_col_adj))
            min_row_adj = max(0, int(min_row_adj))
            max_col_adj = min(int(max_col_adj), img_dims[1])
            max_row_adj = min(int(max_row_adj), img_dims[0])

            if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:
                valid_bboxes.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])

                # cv2.circle(image, (center_col, center_row), 3, color=(255, 0, 0), thickness=3)     # plot object centers
                cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=colors_list[d[pred_ind]], thickness=1)

                coords_list.append([min_row_adj, min_col_adj, max_row_adj - min_row_adj, max_col_adj - min_col_adj, d[pred_ind]])
            # if d[pred_ind]==1:
            #     cv2.imwrite(os.path.join(results_path_kamioni, str(im_ind).zfill(5) + '.bmp'), image1) NOTE: ova koga sakam teski vozila da mi izvadi samo

        cv2.imwrite(os.path.join(results_path, str(im_ind).zfill(5) + '.bmp'), image)

        if flag_save_coords:
            coords_list = np.array(coords_list)
            np.savetxt(os.path.join(results_path, str(im_ind).zfill(5) + '.txt'), coords_list, delimiter=',', fmt='%i')

        bboxes_afternms = helper_postprocessing.nms_tanja_w_metrics(image_za_nms, os.path.join(results_path_nms, str(im_ind).zfill(5) + '.bmp'), valid_bboxes, thr_clustering,
                                                                    colors_list, num_classes, flag_save_coords)
        # all_bboxes.append(bboxes_afternms)
        # print(all_bboxes)
        #
        # print(1)

def save_results_anchorless_limits_cls_proba(results_path, images, images_names, output_cls, output_reg, anchor_stride, prob_thr, reg_norm_coef_position_rows,
                                       reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width, flag_normalizeAll, flag_save_coords, flag_after_regressor):
    """
    plot bounding boxes of detected objects onto test images and save as images
    NOTE: small vehicles - green, big vehicles - red
    :param results_path: path of destination folder [str]
    :param images: test images [ndarray]
    :param plot_color: BGR values of the color of the annotations (tuple)
    :param output_cls: output of the classifier [ndarray]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef: normalization coefficient for regression data [float]
    :return: None
    """
    cnt = 0
    img_dims = (images[0].shape[0], images[0].shape[1])  # height, width
    print('tukaaa')
    color_plot_small = (0, 255, 0)
    color_plot_big = (0, 0, 255)

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows  # regressor output
    output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_cols  # regressor output
    output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_size_height  # regressor output
    output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_size_width  # regressor output
    # round regressor output and cast to integer pixel values
    # output_reg = output_reg * norm_coef     # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))

    for im_ind, image in enumerate(images):
        coords_list = []
        # print(images.shape)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions

            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            # ovie dve linii ne trebaat samo za klasifikator output ako sakame
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]

            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]

            # bbox top left point
            min_row = int(center_row - np.round(h / 2))
            min_col = int(center_col - np.round(w / 2))

            # NOTE tocki na bbox posle pomestuvanje ododadeno od regresor
            # adjust position and size with regressor predictions
            min_row_adj = int(min_row + delta_r)
            min_col_adj = int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, int(min_col_adj))
            min_row_adj = max(0, int(min_row_adj))
            max_col_adj = min(int(max_col_adj), img_dims[1])
            max_row_adj = min(int(max_row_adj), img_dims[0])

            center_row_reg = int(min_row_adj + np.round(h / 2))
            center_col_reg = int(min_col_adj + np.round(w / 2))

            max_col = int(min(min_col + w, img_dims[1]))
            max_row = int(min(min_row + h, img_dims[0]))
            min_col = int(max(0, min_col))
            min_row = int(max(0, min_row))

            #ako e nevozilo ne go zapisuvaj vo txt fajlot
            if d[pred_ind]==2:
                continue
            # cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=plot_color, thickness=1)
            if not flag_after_regressor:
                coords_list.append([min_row, min_col, max_row, max_col, d[pred_ind]])
                if d[pred_ind] == 0:

                    # cv2.circle(image, (center_col, center_row), 1, color=color_plot_small, thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_col, min_row), (max_col, max_row), color=color_plot_small, thickness=1)
                    # cv2.imshow('sl',image)
                    # cv2.waitKey(0)
                elif d[pred_ind] == 1:
                    # cv2.circle(image, (center_col, center_row), 1, color=color_plot_big, thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_col, min_row), (max_col, max_row), color=color_plot_big, thickness=1)

                else:

                    cv2.rectangle(image, (min_col, min_row), (max_col, max_row), color=(255,255,0), thickness=1)

                    # print('Invalid class.')
                    # exit(1)

                # save test image with bounding boxes of detected objects
                # if flag_save_coords:
                #     coords_list1 = np.array(coords_list)
                #     ime = str(images_names[im_ind][:-4])
                #     ime = ime + '_class.txt'
                #     # print(ime)
                #     np.savetxt(os.path.join(results_path, ime), coords_list1, delimiter=',', fmt='%i')

            else:
                coords_list.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])
                if d[pred_ind] == 0:
                    # cv2.circle(image, (center_col, center_row), 1, color=(255, 255, 0), thickness=1)  # plot object centers for classifier
                    #
                    # cv2.circle(image, (center_col_reg, center_row_reg), 1, color=(255, 0, 0), thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=(255, 0, 0), thickness=1)
                elif d[pred_ind] == 1:
                    # cv2.circle(image, (center_col, center_row), 1, color=(255, 255, 0), thickness=1)  # plot object centers
                    #
                    # cv2.circle(image, (center_col_reg, center_row_reg), 1, color=color_plot_big, thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=color_plot_big, thickness=1)
                else:
                    cv2.rectangle(image, (min_col, min_row), (max_col, max_row), color=(255,255,0), thickness=1)

                    # print('Invalid class.')
                    # exit(1)

        if flag_save_coords:
            coords_list1 = np.array(coords_list)
            ime = str(images_names[im_ind][:-4])
            if flag_after_regressor:
                ime = ime + '_reg.txt'
            else:
                ime=ime+'_class.txt'
            # print(ime)
            np.savetxt(os.path.join(results_path, ime), coords_list1, delimiter=',', fmt='%i')

        cnt += 1
        if not flag_after_regressor:
            cv2.imwrite(os.path.join(results_path, str(images_names[im_ind][:-4]) + '_class.bmp'), image)
        else:
            cv2.imwrite(os.path.join(results_path, str(images_names[im_ind][:-4]) + '_reg.bmp'), image)
    print(cnt)
    print(1)


def save_results_anchorless_limits_cls(results_path, images, images_names, output_cls, output_reg, anchor_stride, prob_thr, reg_norm_coef_position_rows,
                                       reg_norm_coef_position_cols, reg_norm_coef_size_height, reg_norm_coef_size_width, num_classes,colors_list,thr_clustering, flag_save_coords, flag_after_regressor,flag_nms):
    """
    plot bounding boxes of detected objects onto test images and save as images
    NOTE: small vehicles - green, big vehicles - red
    :param results_path: path of destination folder [str]
    :param images: test images [ndarray]
    :param plot_color: BGR values of the color of the annotations (tuple)
    :param output_cls: output of the classifier [ndarray]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef: normalization coefficient for regression data [float]
    :return: None
    """
    cnt = 0
    img_dims = (images[0].shape[0], images[0].shape[1])
    num_classes_nms=2 # height, width
    print('tukaaa')
    color_plot_small = (0, 255, 0)
    color_plot_big = (0, 0, 255)
    # image_nms=deepcopy(images[0])
    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    output_reg[:, :, :, 0] = output_reg[:, :, :, 0] * reg_norm_coef_position_rows  # regressor output
    output_reg[:, :, :, 1] = output_reg[:, :, :, 1] * reg_norm_coef_position_cols  # regressor output
    output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_size_height  # regressor output
    output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_size_width  # regressor output
    # round regressor output and cast to integer pixel values
    # output_reg = output_reg * norm_coef     # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))

    for im_ind, image in enumerate(images):
        coords_list = []
        # print(images.shape)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_nms=deepcopy(image)
        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions
            # ako e nevozilo ne go zapisuvaj vo txt fajlot
            if d[pred_ind] == 2:
                continue
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            # ovie dve linii ne trebaat samo za klasifikator output ako sakame
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]

            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]
            if h<0 or w<0:
                print(f"losoooo:{h}")
                print(f"losoooo:{w}")
                print(images_names[im_ind])
            # bbox top left point
            min_row = int(center_row - np.round(h / 2))
            min_col = int(center_col - np.round(w / 2))

            # NOTE tocki na bbox posle pomestuvanje ododadeno od regresor
            # adjust position and size with regressor predictions
            min_row_adj = int(min_row + delta_r)
            min_col_adj = int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, int(min_col_adj))
            min_row_adj = max(0, int(min_row_adj))
            max_col_adj = min(int(max_col_adj), img_dims[1])
            max_row_adj = min(int(max_row_adj), img_dims[0])

            center_row_reg = int(min_row_adj + np.round(h / 2))
            center_col_reg = int(min_col_adj + np.round(w / 2))

            max_col = int(min(min_col + w, img_dims[1]))
            max_row = int(min(min_row + h, img_dims[0]))
            min_col = int(max(0, min_col))
            min_row = int(max(0, min_row))



            # cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=plot_color, thickness=1)
            if not flag_after_regressor:
                if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:
                    coords_list.append([min_row, min_col, max_row, max_col, d[pred_ind]])
                if not flag_nms:
                    if d[pred_ind] == 0:

                        # cv2.circle(image, (center_col, center_row), 1, color=color_plot_small, thickness=1)  # plot object centers

                        cv2.rectangle(image, (min_col, min_row), (max_col, max_row), color=color_plot_small, thickness=1)
                        # cv2.imshow('sl',image)
                        # cv2.waitKey(0)
                    elif d[pred_ind] == 1:
                        # cv2.circle(image, (center_col, center_row), 1, color=color_plot_big, thickness=1)  # plot object centers

                        cv2.rectangle(image, (min_col, min_row), (max_col, max_row), color=color_plot_big, thickness=1)

                    else:

                        cv2.rectangle(image, (min_col, min_row), (max_col, max_row), color=(255,255,0), thickness=1)

                    # print('Invalid class.')
                    # exit(1)

                # save test image with bounding boxes of detected objects
                # if flag_save_coords:
                #     coords_list1 = np.array(coords_list)
                #     ime = str(images_names[im_ind][:-4])
                #     ime = ime + '_class.txt'
                #     # print(ime)
                #     np.savetxt(os.path.join(results_path, ime), coords_list1, delimiter=',', fmt='%i')

            else:
                if (max_row_adj - min_row_adj) > 0 and (max_col_adj - min_col_adj) > 0:

                    coords_list.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])
                else:
                    print(f"KOORDINATI LOSI: {min_row_adj, min_col_adj, max_row_adj, max_col_adj}")
                if not flag_nms:
                    if d[pred_ind] == 0:
                        # cv2.circle(image, (center_col, center_row), 1, color=(255, 255, 0), thickness=1)  # plot object centers for classifier
                        #
                        # cv2.circle(image, (center_col_reg, center_row_reg), 1, color=(255, 0, 0), thickness=1)  # plot object centers

                        cv2.rectangle(image, (max_col_adj, max_row_adj), (min_col_adj, min_row_adj), color=(255, 0, 0), thickness=1)
                    elif d[pred_ind] == 1:
                        # cv2.circle(image, (center_col, center_row), 1, color=(255, 255, 0), thickness=1)  # plot object centers
                        #
                        # cv2.circle(image, (center_col_reg, center_row_reg), 1, color=color_plot_big, thickness=1)  # plot object centers

                        cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=color_plot_big, thickness=1)
                    # else:
                    #     cv2.rectangle(image, (min_col, min_row), (max_col, max_row), color=(255,255,0), thickness=1)

                    # print('Invalid class.')
                    # exit(1)
        if not flag_nms:
            if flag_save_coords:
                coords_list1 = np.array(coords_list)
                ime = str(images_names[im_ind][:-4])
                if flag_after_regressor:
                    ime = ime + '_reg.txt'
                else:
                    ime=ime+'_class.txt'
                # print(ime)
                np.savetxt(os.path.join(results_path, ime), coords_list1, delimiter=',', fmt='%i')

            cnt += 1
        if not flag_nms:
            if not flag_after_regressor:
                cv2.imwrite(os.path.join(results_path, str(images_names[im_ind][:-4]) + '_class.bmp'), image)
            else:
                cv2.imwrite(os.path.join(results_path, str(images_names[im_ind][:-4]) + '_reg.bmp'), image)
    # print(cnt)
    # print(1)
        #nms
        if flag_nms:
            if flag_after_regressor:
                im_name=str(images_names[im_ind][:-4]) + '_reg.bmp'
            else:
                im_name=str(images_names[im_ind][:-4]) + '_class.bmp'
            helper_postprocessing.nms_new(image_nms,os.path.join(results_path,im_name ),coords_list,thr_clustering,colors_list,num_classes_nms,0,True,True,False)

#
#
# save_results_anchorless_limits_cls(res, np.array(image_list), (0, 255, 0), out_class_list, out_reg_list, 8, 0.5, 20.0)
#
#
# file_path_reg_coef=r'D:\KlasifikacijaVozila\Miladinovci'
# filename='reg_norm_coef.txt'
# fid1 = open(os.path.join(file_path_reg_coef, filename), 'rb')
# reg_norm_coef=pickle.load(fid1)
# # reg_norm_coef = np.max(np.abs(out_reg_list))
#
#
# # anchor_stride=8
# # prob_thr=0.5
# # norm_coef=reg_norm_coef
# # #
# # # # print(out_class_list)
# # # save_results_anchorless_limits_cls(res, image_list, (0,0,0), out_class_list, out_reg_list, anchor_stride, prob_thr, norm_coef)


def save_results_anchorless_limits_cls1(results_path, images, images_names, output_cls, output_reg, anchor_stride, prob_thr, reg_norm_coef_position,
                                        reg_norm_coef_size, flag_normalizeAll, flag_save_coords, flag_after_regressor):
    """
    plot bounding boxes of detected objects onto test images and save as images
    NOTE: small vehicles - green, big vehicles - red
    :param results_path: path of destination folder [str]
    :param images: test images [ndarray]
    :param plot_color: BGR values of the color of the annotations (tuple)
    :param output_cls: output of the classifier [ndarray]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param norm_coef: normalization coefficient for regression data [float]
    :return: None
    """
    cnt = 0
    img_dims = (images[0].shape[0], images[0].shape[1])  # height, width

    color_plot_small = (0, 255, 0)
    color_plot_big = (0, 0, 255)

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    output_reg[:, :, :, :2] = output_reg[:, :, :, :2] * reg_norm_coef_position  # regressor output
    output_reg[:, :, :, 2:] = output_reg[:, :, :, 2:] * reg_norm_coef_size  # regressor output
    # output_reg[:, :, :, 2] = output_reg[:, :, :, 2] * reg_norm_coef_size_height  # regressor output
    # output_reg[:, :, :, 3] = output_reg[:, :, :, 3] * reg_norm_coef_size_width  # regressor output
    # round regressor output and cast to integer pixel values
    # output_reg = output_reg * norm_coef     # regressor output

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))

    for im_ind, image in enumerate(images):
        print(images.shape)
        coords_list = []
        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions
            if d[pred_ind]==2: #klasa_nevozilo
                continue
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            # ovie dve linii ne trebaat samo za klasifikator output ako sakame
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]

            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]

            # bbox top left point
            min_row = int(center_row - np.round(h / 2))
            min_col = int(center_col - np.round(w / 2))

            # NOTE tocki na bbox posle pomestuvanje ododadeno od regresor
            # adjust position and size with regressor predictions
            min_row_adj = int(min_row + delta_r)
            min_col_adj = int(min_col + delta_c)

            max_row_adj = min_row_adj + h
            max_col_adj = min_col_adj + w

            # plot bounding box onto image
            # clip bounding boxes falling out of the image borders
            min_col_adj = max(0, int(min_col_adj))
            min_row_adj = max(0, int(min_row_adj))
            max_col_adj = min(int(max_col_adj), img_dims[1])
            max_row_adj = min(int(max_row_adj), img_dims[0])

            center_row_reg = int(min_row_adj + np.round(h / 2))
            center_col_reg = int(min_col_adj + np.round(w / 2))

            max_col = int(min(min_col + w, img_dims[1]))
            max_row = int(min(min_row + h, img_dims[0]))
            min_col = int(max(0, min_col))
            min_row = int(max(0, min_row))


            # cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=plot_color, thickness=1)
            if not flag_after_regressor:
                coords_list.append([min_row, min_col, max_row, max_col, d[pred_ind]])
                if d[pred_ind] == 0:
                    cv2.circle(image, (center_col, center_row), 1, color=color_plot_small, thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_col, min_row), (max_col, max_row), color=color_plot_small, thickness=1)
                elif d[pred_ind] == 1:
                    cv2.circle(image, (center_col, center_row), 1, color=color_plot_big, thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_col, min_row), (max_col, max_row), color=color_plot_big, thickness=1)

                else:
                    print('Invalid class.')
                    exit(1)

                # save test image with bounding boxes of detected objects
                if flag_save_coords:
                    coords_list1 = np.array(coords_list)
                    ime = str(images_names[im_ind][:-4])
                    ime = ime + '_class.txt'
                    print(ime)
                    np.savetxt(os.path.join(results_path, ime), coords_list1, delimiter=',', fmt='%i')

            else:
                coords_list.append([min_row_adj, min_col_adj, max_row_adj, max_col_adj, d[pred_ind]])
                if d[pred_ind] == 0:
                    cv2.circle(image, (center_col_reg, center_row_reg), 1, color=color_plot_small, thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=color_plot_small, thickness=1)
                elif d[pred_ind] == 1:
                    cv2.circle(image, (center_col_reg, center_row_reg), 1, color=color_plot_big, thickness=1)  # plot object centers

                    cv2.rectangle(image, (min_col_adj, min_row_adj), (max_col_adj, max_row_adj), color=color_plot_big, thickness=1)
                else:
                    print('Invalid class.')
                    exit(1)

                if flag_save_coords:
                    coords_list1 = np.array(coords_list)
                    ime = str(images_names[im_ind][:-4])
                    ime = ime + '_reg.txt'
                    print(ime)
                    np.savetxt(os.path.join(results_path, ime), coords_list1, delimiter=',', fmt='%i')
        if not flag_after_regressor:
            cv2.imwrite(os.path.join(results_path, str(images_names[im_ind][:-4]) + '_class.bmp'), image)
        else:
            cv2.imwrite(os.path.join(results_path, str(images_names[im_ind][:-4]) + '_reg.bmp'), image)

        cnt+=1
    print(cnt)
    print(1)
