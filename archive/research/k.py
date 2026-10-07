def get_anchorless_ground_truth_ssd_plates(images, bboxes, anchor_dims, img_dims, anchor_stride, iou_low, iou_high, iou_width_low,
                               iou_width, thr_for_negatives, rfnac, debugflag, ratioofchosennegatives):
    """
      generate ground truth output for classifier and regressor
      :param bboxes: annotated bounding boxes [min_row, min_col, max_row, max_col] [ndarray]
      :param anchor_dims: tuple of anchor dimensions - (height, width) [tuple]
      :param img_dims: (rows, cols, depth) [tuple]
      :param anchor_stride: stride along rows and columns [int]
      :param iou_low: rectangles with lower iou are marked negative [int]
      :param iou_high: rectangles with higher iou are marked positive [int]
      :param rfnac: ratio_for_negatives_around_car
      :return: output_class_list - ground truth output of the classification branch [ndarray]
               output_reg_list - ground truth output of the regression branch [ndarray]
               valid_inds - indices of output samples containing at least one object [list]
      """
    # chosen_negatives = []
    # anchor_list_positives = []
    # for _ in anchor_dims:
    #     anchor_list_positives.append([])
    # #print(anchor_list_positives)
    output_class_list = []  # classification branch output
    output_reg_list = []  # regression branch output
    valid_inds = []  # indices of output samples containing at least one object

    num_anchors = len(anchor_dims)

    for img_ind, img_bboxes in tqdm(enumerate(bboxes)):  # iterate over ground truth files for each image

        # #print("img_ind ",img_ind)
        # xxx = images[img_ind].copy()
        # cv2.destroyAllWindows()
        # cv2.imshow(str(img_ind),images[img_ind])
        # cv2.waitKey(0)

        ind_to_keeep = []
        # calculate output dimensions
        output_dims_class = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), 2)
        output_dims_reg = (np.int(img_dims[0] / anchor_stride), np.int(img_dims[1] / anchor_stride), 4)
        # initialize output
        output_class = np.zeros(output_dims_class).astype(np.int)
        output_reg = np.zeros(output_dims_reg).astype(np.float64)
        # #print(output_class.shape,output_reg.shape)
        # first position of an anchor center (top left) - half of stride size
        start_r = np.int(np.round(anchor_stride / 2))
        start_c = np.int(np.round(anchor_stride / 2))
        # #print(start_r,start_c)
        # input("stop")
        xxx = images[img_ind].copy()
        imasliki = False
        for output_row, center_row in enumerate(
                range(start_r, img_dims[0], anchor_stride)):  # iterate over rows of centers
            for output_col, center_col in enumerate(
                    range(start_c, img_dims[1], anchor_stride)):  # iterate over columns of centers

                for bbox in img_bboxes:  # iterate over annotated bounding boxes row1 col1 row2 col2
                    # xxx=images[img_ind]
                    # cv2.rectangle(xxx, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0, 255, 0), thickness=2)

                    # --- assign classes and adjustments:
                    #     calculate IOU, place 1 or 0 at the required position
                    #     claculate position and size adjustments for positive and neutral samples (IOU > iou_low) ---

                    # calculate parameters of current anchor
                    # prevent anchors going over the image borders

                    bbox_h = bbox[2] - bbox[0]
                    bbox_w = bbox[3] - bbox[1]

                    half_bbox_dim_h = np.int(np.round(bbox_h / 2))
                    half_bbox_dim_w = np.int(np.round(bbox_w / 2))

                    if (bbox[2] - bbox[0] == 0) or (bbox[3] - bbox[1] == 0):    # bounding box with area 0 px
                        # print(bbox)
                        # cv2.imshow("empty",images[img_ind])
                        # cv2.waitKey(0)
                        continue
                    # if (bbox[2] + bbox[3]) == 0:
                    #     continue
                    # if (center_row - half_bbox_dim_h) < 0:
                    #     continue
                    # if (center_col - half_bbox_dim_w) < 0:
                    #     continue
                    # if (center_row + half_bbox_dim_h) >= img_dims[0]:
                    #     continue
                    # if (center_col + half_bbox_dim_w) >= img_dims[1]:
                    #     continue

                    anchor = [max(0, center_row - half_bbox_dim_h),
                              max(0, center_col - half_bbox_dim_w),
                              min(center_row + half_bbox_dim_h, img_dims[0]),
                              min(center_col + half_bbox_dim_w, img_dims[1])]  # min_row, min_col, max_row, max_col

                    # xxx = images[img_ind]
                    # xxx_copy=xxx.copy()
                    # #print("img_bboxes", img_bboxes)
                    # #print(anchor, anchor_ind, img_ind, bbox)
                    # cv2.rectangle(xxx_copy, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0, 0, 0), thickness=2)
                    #
                    # cv2.imshow("kopija", xxx_copy)
                    # cv2.waitKey(0)

                    # print(f'bbox: {bbox}')
                    # print(f'anchor: {anchor}')

                    iou = helper_postprocessing.calc_iou(bbox, anchor)  # calculate IOU

                    # if debugflag:
                    #     xxx = images[img_ind]
                    #     xxx_copy = xxx.copy()
                    #     # print("img_bboxes", img_bboxes)
                    #     # print(anchor, anchor_ind, img_ind, bbox, iou)
                    #     # print("position", output_row, output_col, anchor_ind)
                    #     # print(center_row, center_col)
                    #     if (iou > 0.69):
                    #         # print("iou", iou)
                    #         cv2.circle(xxx_copy, (center_col, center_row), radius=1, color=(255, 255, 255),
                    #                    thickness=4)
                    #         cv2.rectangle(xxx_copy, (anchor[1], anchor[0]), (anchor[3], anchor[2]),
                    #                       color=(255, 255, 255),
                    #                       thickness=2)
                    #         cv2.rectangle(xxx_copy, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0, 0, 0),
                    #                       thickness=1)
                    #
                    #         cv2.imshow("iouneg", xxx_copy)
                    #         cv2.waitKey(0)
                    # if((anchor[0]-anchor_stride)<bbox[0] and (anchor[0]+anchor_stride)>bbox[0] and (anchor[1]-anchor_stride)<bbox[1] and (anchor[1]+anchor_stride)>bbox[1]):
                    #     #print(img_bboxes)
                    #     #print(anchor,anchor_ind,img_ind,bbox,iou)
                    #     input("stop")

                    # if iou > thr_for_negatives and iou < iou_low:
                    #     ind_to_keeep.append((output_row, output_col, 0))

                    if iou >= iou_high:
                        # imasliki = True
                        # iou_w = helper_postprocessing.calc_iou_width(bbox, anchor)
                        # #print(anchor_dims[anchor_ind],anchor,bbox)
                        # #print(output_row, output_col, anchor_ind)
                        # positive sample: set class, calculate deltas
                        # print(iou_w)
                        # if iou_w > iou_width:
                            # print(iou_w)
                            output_class[output_row, output_col, 0] = 1
                            # anchor_list_positives[0].append(bbox)
                            # --- set deltas ---
                            # current location minus correct location
                            delta_r = bbox[0] - anchor[0]
                            delta_c = bbox[1] - anchor[1]
                            h_percent = bbox_h / img_dims[0]
                            w_percent = bbox_w / img_dims[1]

                            output_reg[output_row, output_col, 0] = delta_r
                            output_reg[output_row, output_col, 1] = delta_c
                            output_reg[output_row, output_col, 2] = h_percent
                            output_reg[output_row, output_col, 3] = w_percent

                            # if debugflag:
                            #     # xxx = images[img_ind].copy()
                            #     # print("img_bboxes",img_bboxes)
                            #     # print(anchor,anchor_ind,img_ind,bbox,iou)
                            #     # print("position",output_row,output_col,anchor_ind)
                            #     # print(center_row,center_col)
                            #     cv2.rectangle(xxx, (anchor[1], anchor[0]), (anchor[3], anchor[2]),
                            #                   color=(255, 255, 255), thickness=1)
                            #     cv2.rectangle(xxx, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0, 0, 0),
                            #                   thickness=1)

                                # cv2.imshow("ioupos", xxx)
                                # cv2.waitKey(0)
                                # # impath=r'D:\Tatjana\NewBeginning\MachineVision\Results\GT/image'+str(img_ind).zfill(4)+'_'+str(anchor[1])+'_'+str(anchor[2])+'_'+str(anchor[3])+'_'+str(anchor[4])+'.bmp'
                                # impath=r'D:\Tatjana\NewBeginning\MachineVision\Results\GT/image'+str(img_ind).zfill(4)+'_'+str(anchor[0])+'_'+str(anchor[1])+'_'+str(anchor[2])+'_'+str(anchor[3])+'.bmp'
                                # # print(impath)
                                # # cv2.imwrite(impath,xxx)



                        # elif iou_w < iou_width_low:
                        #     # print(iou_w)
                        #
                        #     output_class[output_row, output_col, 1] = 1
                        #     chosen_negatives.append([output_row, output_col])
                        #     if debugflag:
                        #         # xxx = images[img_ind].copy()
                        #         # print("img_bboxes",img_bboxes)
                        #         # print(anchor,anchor_ind,img_ind,bbox,iou)
                        #         # print("position",output_row,output_col,anchor_ind)
                        #         # print(center_row,center_col)
                        #         cv2.rectangle(xxx, (anchor[1], anchor[0]), (anchor[3], anchor[2]),
                        #                       color=(255, 255, 255), thickness=1)
                        #         cv2.rectangle(xxx, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0, 0, 0),
                        #                       thickness=1)
                        #
                        #         cv2.imshow("iouposamanegW", xxx)
                        #         cv2.waitKey(0)
                        #         # # impath=r'D:\Tatjana\NewBeginning\MachineVision\Results\GT/image'+str(img_ind).zfill(4)+'_'+str(anchor[1])+'_'+str(anchor[2])+'_'+str(anchor[3])+'_'+str(anchor[4])+'.bmp'
                        #         # impath=r'D:\Tatjana\NewBeginning\MachineVision\Results\GT/image'+str(img_ind).zfill(4)+'_'+str(anchor[0])+'_'+str(anchor[1])+'_'+str(anchor[2])+'_'+str(anchor[3])+'.bmp'
                        #         # # print(impath)
                        #         # # cv2.imwrite(impath,xxx)

                    if (iou < iou_high) and (iou > iou_low):
                        # IOU between iou_min and iou_max: class - temporarily marked with 2, claculate deltas
                        output_class[output_row, output_col, 0] = 2
                        delta_r = bbox[0] - anchor[0]
                        delta_c = bbox[1] - anchor[1]
                        h_percent = bbox_h / img_dims[0]
                        w_percent = bbox_w / img_dims[1]

                        output_reg[output_row, output_col, 0] = delta_r
                        output_reg[output_row, output_col, 1] = delta_c
                        output_reg[output_row, output_col, 2] = h_percent
                        output_reg[output_row, output_col, 3] = w_percent

                        # xxx = images[img_ind].copy()
                        # #print("img_bboxes", img_bboxes)
                        # # #print(anchor, anchor_ind, img_ind, bbox, iou)
                        # #print("position", output_row, output_col, anchor_ind)
                        # #print(center_row, center_col)
                        # cv2.rectangle(xxx, (anchor[1], anchor[0]), (anchor[3], anchor[2]), color=(255, 255, 255), thickness=1)
                        # cv2.rectangle(xxx, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0, 0, 0), thickness=1)
                        #
                        # cv2.imshow("slika0.3", xxx)
                        # cv2.waitKey(0)

        # cv2.imwrite(r"C:\Users\tatja\Desktop\DesktopFolder\Dataset\GTDean/Whole/" + str(cnt) + "_" + str(
        #     center_row) + "_" + str(center_col) + ".jpg", xxx)
        # cnt = cnt + 1
        # cv2.imshow("ioupos", xxx)
        # cv2.waitKey(0)
        # impath=r'D:\Tatjana\NewBeginning\MachineVision\Results\GT/image'+str(img_ind).zfill(4)+'_'+str(anchor[1])+'_'+str(anchor[2])+'_'+str(anchor[3])+'_'+str(anchor[4])+'.bmp'
        impath = r'D:\Tatjana\NewBeginning\MachineVision\Results\GT3/image' + str(img_ind).zfill(4) + '.bmp'
        # print(impath)
        # if imasliki:
        # cv2.imwrite(impath,xxx)
        # np.savetxt(impath[:-4]+'.txt',img_bboxes[0],delimiter=',', fmt='%d')

        imasliki = False

        # assign background samples (centers where none of the anchor sizes are marked positive, or ignored (marked with 2))

        for out_row in range(output_class.shape[0]):  # iterate over rows of output
            for out_col in range(output_class.shape[1]):  # iterate over columns of output

                if sum(output_class[out_row, out_col, :]) == 0:  #
                    output_class[out_row, out_col, 1] = 1  # depth 3 - negative class

        # replace 2s with 0s - mark samples with iou between thresholds as ignored
        output_class = np.where(output_class == 2, 0, output_class)

        # mark border pixels (anchors not fully within the borders of the image) as ignored
        # border_padding = np.int((anchor_dim[0] / anchor_stride) / 2) + 1
        #
        # output_class[0:border_padding, :, anchor_ind] = 0
        # output_class[output_class.shape[0] - border_padding:, :, anchor_ind] = 0
        # output_class[:, 0:border_padding, anchor_ind] = 0
        # output_class[:, output_class.shape[1] - border_padding:, anchor_ind] = 0
        #
        # output_reg[0:border_padding, :, anchor_ind] = 0
        # output_reg[output_class.shape[0] - border_padding:, :, anchor_ind] = 0
        # output_reg[:, 0:border_padding, anchor_ind] = 0
        # output_reg[:, output_class.shape[1] - border_padding:, anchor_ind] = 0

        # --- select negative samples ---
        num_positives = np.sum(output_class[:, :, 0])  # count positives and negatives
        # print("positives",img_ind,num_positives)
        # find negatives
        negs = output_class[:, :, 1]
        [r, c] = np.where(negs == 1)
        # print(negs)
        # select negatives to remove

        neg_ind_to_keep = []
        np.random.shuffle(chosen_negatives)
        # print("img",img_ind)
        if len(chosen_negatives) > 0:
            if len(chosen_negatives) > ratioofchosennegatives * num_positives:
                for ind in range(ratioofchosennegatives * num_positives):
                    neg_ind_to_keep.append(chosen_negatives[ind][0])

            else:
                for negative in chosen_negatives:
                    neg_ind_to_keep.append(negative[0])
        # print(neg_ind_to_keep)
        # print(len(neg_ind_to_keep))

        r_new = []

        for row in r:
            if row not in neg_ind_to_keep:
                r_new.append(row)
        # print(len(r_new),r_new)
        r = r_new

        ind_to_remove = np.arange(len(r))
        # print(ind_to_remove)
        # print(len(ind_to_remove))
        np.random.shuffle(ind_to_remove)

        num_neg = min(len(r), num_positives * (ratioofchosennegatives + 1) - len(neg_ind_to_keep))  # number of positive to negative samples ratio: 1 to 10
        # print("negatives:",num_neg)

        num_to_remove = len(r) - num_neg
        temp_flag = False
        # if(num_positives+num_neg==0):
        #     temp_flag=True
        #     num_to_remove=2
        # print("remove: ",num_to_remove)
        ind_to_remove = ind_to_remove[:num_to_remove]

        # #print(ind_to_remove)
        # remove negatives
        for ind in ind_to_remove:
            output_class[r[ind], c[ind], :] = 0

        # print(type(ind_to_keeep))

        # np.random.shuffle(ind_to_keeep)
        # print(type(ind_to_keeep))
        # print(ind_to_keeep.shape)
        # cv2.imshow("image", images[img_ind])
        # cv2.waitKey(0)

        total_num_negatives = 20 * num_positives * rfnac
        cnt = 0
        for coord in ind_to_keeep:
            if (cnt == total_num_negatives):
                break
            output_class[coord[0], coord[1], 1] = 1
            cnt = cnt + 1

        # #print(num_positives)
        if num_positives > 0 or temp_flag:
            output_class_list.append(output_class)
            output_reg_list.append(output_reg)
            valid_inds.append(img_ind)
        temp_flag = False
        # input("stop")

    output_class_list = np.array(output_class_list)
    output_reg_list = np.array(output_reg_list)

    reg_norm_coef = np.max(np.abs(output_reg_list))

    output_reg_list = output_reg_list / reg_norm_coef  # normalize regression output to range same as the classifier output

    return output_class_list, output_reg_list, valid_inds, reg_norm_coef
