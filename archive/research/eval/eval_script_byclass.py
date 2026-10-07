
import os
import cv2
import numpy as np
# from tensorflow.python.ops.metrics_impl import false_positives
import helper_model
# from FCN_SSD_test_video import cropped_image
# from eval.eval_script import best_recall

debug=True

def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.
    box format: [ymin, xmin, ymax, xmax]
    """
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2

    inter_ymin = max(ymin1, ymin2)
    inter_xmin = max(xmin1, xmin2)
    inter_ymax = min(ymax1, ymax2)
    inter_xmax = min(xmax1, xmax2)

    inter_area = max(0, inter_ymax - inter_ymin) * max(0, inter_xmax - inter_xmin)
    box1_area = (ymax1 - ymin1) * (xmax1 - xmin1)
    box2_area = (ymax2 - ymin2) * (xmax2 - xmin2)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0
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

    print("union < 0")
    # return
    exit(1)
def adjust_coordinates(detections, original_size, new_size, crop_left):
    original_width, original_height = original_size
    new_width, new_height = new_size

    # Scaling factors
    width_scale = 1620 / 512
    height_scale = 1080 / 341

    adjusted_detections = []

    for bbox in detections:
        # Apply left crop shift
        xmin=bbox[1]
        ymin=bbox[0]
        xmax=bbox[3]
        ymax=bbox[2]

        # Rescale coordinates to the new image size
        new_xmin = int(xmin * width_scale)
        new_xmax = int(xmax * width_scale)
        new_ymin = int(ymin * height_scale)
        new_ymax = int(ymax * height_scale)

        new_xmin += crop_left
        new_xmax += crop_left
        adjusted_detections.append([new_ymin, new_xmin, new_ymax, new_xmax,bbox[4]])

    return adjusted_detections



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
def nms_new(lista, thr, num_classes):
    finalwindows = []

    if len(lista) == 0:
        return []

    all_clusters = getclustersforanchors(lista, thr)

    for cluster in all_clusters:

        maxwindows = []
        maxwindows_new = []

        if len(cluster) == 0:
            continue

        elements = [item[2] - item[0] for item in cluster]

        if len(elements) <= 2:
            first2max = sorted(elements)
        else:

            first2max = sorted(elements)[:]

        first2max_noduplicates = []
        for i in first2max:
            if i not in first2max_noduplicates:
                first2max_noduplicates.append(i)

        for maxnum in first2max_noduplicates:
            indices = [i for i, x in enumerate(elements) if x == maxnum]

            for idx in indices:
                maxwindows.append(cluster[idx])

        bboxes_per_class = [[] for i in range(num_classes)]  # bounding boxes grouped by class

        for maxwindow in maxwindows:
            bboxes_per_class[maxwindow[-1]].append(maxwindow)

        max_class = max((len(l), i) for i, l in enumerate(bboxes_per_class))[1]

        for maxwindow in maxwindows:
            if maxwindow[-1] == max_class:
                maxwindows_new.append(maxwindow)

        finalwindow = average_window(maxwindows_new)
        finalwindows.append(finalwindow)

    return finalwindows




def preprocess_data(image,dir_name):

    rows, cols = image.shape[:2]

    if 'videoframes' in dir_name:
        cropped_image = image[0:rows, 0:1141]
        left_crop = 0

    elif 'miladinovci' in dir_name:
        cropped_image = image[0:rows, 0:1621]
        left_crop = 0

    elif 'kamera2' in dir_name or 'kam23' in dir_name:
        cropped_image = image[0:rows, 160:1782]
        left_crop = 160


    elif ('M-30' in dir_name) or ('drugo' in dir_name):
        # break
        # print('m30')
        pass

    elif 'mil_res' in dir_name:
        pass

    elif 'kam25' in dir_name:
        cropped_image = image[0:rows, 0:1621]
        left_crop=0

    elif 'kam28' in dir_name:
        cropped_image = image[0:rows, 0:1621]
        left_crop = 0
    elif 'kam30' in dir_name:
        cropped_image = image[0:rows, 0:1621]
        left_crop = 0
    elif 'kam32' in dir_name:
        cropped_image = image[0:rows, 299:cols]
        left_crop=299
    elif 'kam33' in dir_name:
        cropped_image = image[0:rows, 99:1720]
        left_crop = 99
    elif 'kam36' in dir_name:
        cropped_image = image[0:rows, 189:1810]
        left_crop = 189
    elif 'kam40' in dir_name:
        cropped_image = image[0:rows, 255:255 + 1620]
        left_crop = 255
    elif 'kam42' in dir_name:
        cropped_image = image[0:rows, 299:cols]
        left_crop = 299
    elif 'kam44' in dir_name:
        cropped_image = image[0:rows, 299:cols]
        left_crop =299
    elif 'kam46' in dir_name:
        cropped_image = image[0:rows, 100:1721]
        left_crop = 100
    elif 'kam48' in dir_name:
        cropped_image = image[0:rows, 299:cols]
        left_crop = 299
    else:
        cropped_image=None
        left_crop=0

    resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)
    return resized,left_crop
def save_results_anchorless_limits(number_of_images, output_cls, output_reg, anchor_stride, prob_thr,
                                   thr_clustering):
    """
    plot bounding boxes of detected objects onto test images and save as images
    :param output_cls: output of the classifier [ndarray]
    :param output_reg: output of the regressor [ndarray]
    :param anchor_stride: stride along rows and columns [int]
    :param prob_thr: probability threshold for object classification (range: 0 to 1) [float]
    :param thr_clustering: iou threshold for combining detections into one cluster [float]
    :return: None
    """
    img_dims = (341, 512)  # height, width
    # print(images_s[0].shape[0])
    num_classes = output_cls.shape[3] - 1

    # binarize classifier output probabilities
    output_cls[output_cls >= prob_thr] = 1
    output_cls[output_cls < prob_thr] = 0

    # calculate location of first (top left) anchor center - start at half of stride size
    start_r = int(np.round(anchor_stride / 2))
    start_c = int(np.round(anchor_stride / 2))
    finalwindows_batch = []

    for im_ind in range(number_of_images):

        # image_za_nms = deepcopy(image)
        valid_bboxes = []  # list of bounding boxes with positive height and width

        res = output_cls[im_ind, :, :, :-1]  # classifier output, probability maps for positive objects only

        [r, c, d] = np.where(res > 0.5)  # get coordinate of positive anchors (data is already binarized)

        for pred_ind in range(len(r)):  # iterate over positive predictions
            # ako e klasa nevozilo ne crtaj i ne zacucuvaj koordinati
            if d[pred_ind] == 2:
                continue
            center_row = r[pred_ind] * anchor_stride + start_r
            center_col = c[pred_ind] * anchor_stride + start_c

            # 4 - 4 dimensions are fine-tuned: r, c (top left corner), h, w
            delta_r = output_reg[im_ind, r[pred_ind], c[pred_ind], 0]
            delta_c = output_reg[im_ind, r[pred_ind], c[pred_ind], 1]
            h_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 2]
            w_percent = output_reg[im_ind, r[pred_ind], c[pred_ind], 3]

            h = h_percent * img_dims[0]
            w = w_percent * img_dims[1]
            # print(h)
            # print(w)
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

        if len(r) != 0:

            finalwindows = nms_new(valid_bboxes, thr_clustering, num_classes)

            # finalwindows_batch.append(finalwindows)

        else:

            finalwindows=[]

    return finalwindows
def reformat_ground_truths(ground_truths):
    """
    Reformat ground truths from (ymin, xmin, h, w, class) to (ymin, xmin, ymax, xmax, class).

    Args:
        ground_truths (np.ndarray): Array of ground truths in the format (ymin, xmin, h, w, class).

    Returns:
        np.ndarray: Reformatted ground truths in the format (ymin, xmin, ymax, xmax, class).
    """
    reformatted = ground_truths.copy()
    reformatted[:, 2] = reformatted[:, 0] + reformatted[:, 2]  # ymax = ymin + h
    reformatted[:, 3] = reformatted[:, 1] + reformatted[:, 3]  # xmax = xmin + w
    return reformatted
def calculate_metrics_best_iou(ground_truths, detections, im_orig, im_path_dest, iou_threshold=0.3):
    matched_detections_class_0 = set()
    matched_ground_truths_class_0 = set()
    matched_detections_class_1 = set()
    matched_ground_truths_class_1 = set()
    errors_class_0 = []
    errors_class_1 = []

    # Separate ground truths and detections by class
    ground_truths_class_0 = [gt for gt in ground_truths if gt[4] == 1 or gt[4]==3]  # Class 0 (small vehicle)
    ground_truths_class_1 = [gt for gt in ground_truths if gt[4] == 2 or gt[4]==4 or gt[4]==5]  # Class 1 (big vehicle)

    detections_class_0 = [det for det in detections if det[4] == 0]  # Class 0 (small vehicle)
    detections_class_1 = [det for det in detections if det[4] == 1]  # Class 1 (big vehicle)

    # Compute IoU matrices for each class
    iou_matrix_class_0 = np.zeros((len(ground_truths_class_0), len(detections_class_0)))
    iou_matrix_class_1 = np.zeros((len(ground_truths_class_1), len(detections_class_1)))

    # Compute IoU for small vehicles (class 0)
    for i, gt in enumerate(ground_truths_class_0):
        for j, det in enumerate(detections_class_0):
            iou_matrix_class_0[i, j] = compute_iou(gt[:4], det[:4])

    # Compute IoU for big vehicles (class 1)
    for i, gt in enumerate(ground_truths_class_1):
        for j, det in enumerate(detections_class_1):
            iou_matrix_class_1[i, j] = compute_iou(gt[:4], det[:4])

    # Match detections with ground truths for small vehicles
    while True:
        if iou_matrix_class_0.size > 0:
            max_iou_class_0 = np.max(iou_matrix_class_0)
        else:
            max_iou_class_0 = 0

        if max_iou_class_0 < iou_threshold:
            break

        gt_idx_class_0, det_idx_class_0 = np.unravel_index(np.argmax(iou_matrix_class_0), iou_matrix_class_0.shape)
        matched_detections_class_0.add(det_idx_class_0)
        matched_ground_truths_class_0.add(gt_idx_class_0)

        gt_center = [(ground_truths_class_0[gt_idx_class_0][1] + ground_truths_class_0[gt_idx_class_0][3]) / 2,
                     ground_truths_class_0[gt_idx_class_0][2]]
        det_center = [(detections_class_0[det_idx_class_0][1] + detections_class_0[det_idx_class_0][3]) / 2,
                      detections_class_0[det_idx_class_0][2]]
        error = np.sqrt((gt_center[0] - det_center[0]) ** 2 + (gt_center[1] - det_center[1]) ** 2)
        errors_class_0.append(error)

        iou_matrix_class_0[gt_idx_class_0, :] = -1
        iou_matrix_class_0[:, det_idx_class_0] = -1

    # Match detections with ground truths for big vehicles
    while True:
        if iou_matrix_class_1.size > 0:
            max_iou_class_1 = np.max(iou_matrix_class_1)
        else:
            max_iou_class_1 = 0

        if max_iou_class_1 < iou_threshold:
            break

        gt_idx_class_1, det_idx_class_1 = np.unravel_index(np.argmax(iou_matrix_class_1), iou_matrix_class_1.shape)
        matched_detections_class_1.add(det_idx_class_1)
        matched_ground_truths_class_1.add(gt_idx_class_1)

        gt_center = [(ground_truths_class_1[gt_idx_class_1][1] + ground_truths_class_1[gt_idx_class_1][3]) / 2,
                     ground_truths_class_1[gt_idx_class_1][2]]
        det_center = [(detections_class_1[det_idx_class_1][1] + detections_class_1[det_idx_class_1][3]) / 2,
                      detections_class_1[det_idx_class_1][2]]
        error = np.sqrt((gt_center[0] - det_center[0]) ** 2 + (gt_center[1] - det_center[1]) ** 2)
        errors_class_1.append(error)

        iou_matrix_class_1[gt_idx_class_1, :] = -1
        iou_matrix_class_1[:, det_idx_class_1] = -1

    # Calculate TP, FP, FN for both classes
    tp_class_0 = len(matched_detections_class_0)
    fp_class_0 = len(detections_class_0) - tp_class_0
    fn_class_0 = len(ground_truths_class_0) - len(matched_ground_truths_class_0)

    tp_class_1 = len(matched_detections_class_1)
    fp_class_1 = len(detections_class_1) - tp_class_1
    fn_class_1 = len(ground_truths_class_1) - len(matched_ground_truths_class_1)

    # Visualize the results for false positives and false negatives (optional)
    if debug:
        false_positive_class_0 = False
        false_negative_class_0 = False
        # Visualize small vehicle false positives and false negatives
        for gt in ground_truths_class_0:
            ymin, xmin, ymax, xmax, _ = gt
            cv2.rectangle(im_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        for det in [detections_class_0[j] for j in range(len(detections_class_0)) if
                    j not in matched_detections_class_0]:
            ymin, xmin, ymax, xmax, _ = det
            cv2.rectangle(im_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            cv2.putText(im_orig, "FP", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) #RED FP SMALL
            false_positive_class_0 = True

        for gt in [ground_truths_class_0[i] for i in range(len(ground_truths_class_0)) if i not in matched_ground_truths_class_0]:
            ymin, xmin, ymax, xmax, _ = gt
            cv2.rectangle(im_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 255), 2) #YELLOW FN SMALL
            cv2.putText(im_orig, "FN", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            false_negative_class_0 = True

        # Visualize big vehicle false positives and false negatives
        false_positive_class_1 = False
        false_negative_class_1 = False
        for gt in ground_truths_class_1:
            ymin, xmin, ymax, xmax, _ = gt
            cv2.rectangle(im_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)

        for det in [detections_class_1[j] for j in range(len(detections_class_1)) if
                    j not in matched_detections_class_1]:
            ymin, xmin, ymax, xmax, _ = det
            cv2.rectangle(im_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 255), 2) #MAGENTA FP BIG
            cv2.putText(im_orig, "FP", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            false_positive_class_1 = True

        for gt in [ground_truths_class_1[i] for i in range(len(ground_truths_class_1)) if i not in matched_ground_truths_class_1]:
            ymin, xmin, ymax, xmax, _ = gt
            cv2.rectangle(im_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2) #cyan  FN BIG
            cv2.putText(im_orig, "FN", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            false_negative_class_1 = True

        if false_positive_class_0 or false_negative_class_0 or false_positive_class_1 or false_negative_class_1:
            # if len(detections_class_0) > 0 or len(detections_class_1) > 0:
                cv2.imwrite(im_path_dest, im_orig)

    return {
        'tp_class_0': tp_class_0, 'fp_class_0': fp_class_0, 'fn_class_0': fn_class_0, 'errors_class_0': errors_class_0,
        'tp_class_1': tp_class_1, 'fp_class_1': fp_class_1, 'fn_class_1': fn_class_1, 'errors_class_1': errors_class_1
    }


# Paths
# yolo_folder = r"D:\Monika\yolov5\runs\detect\exp11\exp11_renamed\reformatted_labels"
annot_root_path = r"D:\Monika\VideosTest\All_Cameras_annnotated\Annotations\Renamed"
image_folder = r"D:\Monika\VideosTest\All_Cameras_annnotated\Images\Renamed"
output_folder = "M:\Monika\magisterska\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva_1inception_fixedbsize2\cls2"
os.makedirs(output_folder, exist_ok=True)
srcModelPath=r'D:\Monika\Models\Trening_With_Translations-Up-Down-All_Cams_Nevozilo_bez_regresor1_All_Negatives_Renamed_diou_focal_notnormreg_anotiraniguzva_1inception_fixedbsize2'
# max_prob_thr=1
# Example usage
original_size = (512, 341)  # original image size (after crop and before resize)
new_size = (1920, 1080)  # target size





#Load model
model = helper_model.load_model(model_path=os.path.join(srcModelPath,'model.json'),
                                    weights_path=os.path.join(srcModelPath, 'model.h5'))  # build model architecture

# Iterate through ground truth files
frame_batch=1
anchor_stride=8
thr_clustering=0.3
best_f1 = 0
best_thr=0
# for prob_thr in np.arange(0.4, 1.05, 0.05):
results = []


# print(prob_thr)
num_vehicles=0
best_total_tp_class_0 = 0
best_total_tp_class_1 = 0
best_total_fp_class_0 = 0
best_total_fp_class_1 = 0
best_total_fn_class_0 = 0
best_total_fn_class_1 = 0
best_precision_class_0 = 0
best_precision_class_1 = 0
best_recall_class_0 = 0
best_recall_class_1 = 0
best_f1_class_0 = 0
best_f1_class_1 = 0
best_precision = 0
for prob_thr in np.arange(0.7, 0.75, 0.05):
    total_tp_class_0 = 0
    total_fp_class_0 = 0
    total_fn_class_0 = 0
    total_tp_class_1 = 0
    total_fp_class_1 = 0
    total_fn_class_1 = 0
    total_error_class_0 = 0
    total_error_class_1 = 0
    total_error_count_class_0 = 0
    total_error_count_class_1 = 0


    for ground_truth_folder in os.listdir(annot_root_path):
        # if not 'kam33_1' in ground_truth_folder:
        #     continue
        for gt_file in os.listdir(os.path.join(annot_root_path,ground_truth_folder)):
            if gt_file.endswith(".txt"):
                image_name = gt_file.replace(".txt", ".bmp")
                image_path = os.path.join(image_folder,ground_truth_folder, image_name)
                im_orig=cv2.imread(image_path,1)
                image=cv2.imread(image_path,0)
                image_res,left_crop=preprocess_data(image,ground_truth_folder)
                im_path_dest = os.path.join(output_folder, image_name)

                # Load ground truths
                gt_path = os.path.join(annot_root_path,ground_truth_folder, gt_file)
                ground_truths = np.loadtxt(gt_path, delimiter=',').reshape(-1, 5) if os.path.exists(gt_path) else []
                num_vehicles+=len(ground_truths)
                image_reshaped=np.reshape(image_res,image_res.shape+(1,))
                image_reshaped = np.expand_dims(image_reshaped, axis=0)
                output_combined = model.predict(image_reshaped, verbose=1)
                output_cls=output_combined[:,:,:,:4]
                output_reg=output_combined[:,:,:,4:]
                detections_model = save_results_anchorless_limits(frame_batch, output_cls,
                                                                                       output_reg,
                                                                                       anchor_stride, prob_thr,
                                                                                       thr_clustering
                                                                                       )

                detections=adjust_coordinates(detections_model,original_size,new_size,left_crop)
                # Load YOLO detections
                # for bbox in detections:
                #     cv2.rectangle(image,(bbox[1],bbox[0]),(bbox[3],bbox[2]),color=(255,0,0),thickness=1)
                # cv2.imshow('s',image)
                # cv2.waitKey(0)
                # yolo_path = os.path.join(yolo_folder, gt_file[:-4]+'_'+str(ground_truth_folder)+'_reg.txt')
                # detections = np.loadtxt(yolo_path, delimiter=',').reshape(-1, 5) if os.path.exists(yolo_path) else []

                # Calculate metrics

                # Reformat ground truths
                if ground_truths.size > 0:
                    ground_truths_ref = reformat_ground_truths(ground_truths)
                    # num_vehicles+=1
                    # print(ground_truths)
                result = calculate_metrics_best_iou(ground_truths_ref, detections, im_orig, im_path_dest,
                                                    iou_threshold=0.3)

                # Accumulate metrics for each class
                total_tp_class_0 += result['tp_class_0']
                total_fp_class_0 += result['fp_class_0']
                total_fn_class_0 += result['fn_class_0']
                total_error_class_0 += sum(result['errors_class_0'])
                total_error_count_class_0 += len(result['errors_class_0'])

                total_tp_class_1 += result['tp_class_1']
                total_fp_class_1 += result['fp_class_1']
                total_fn_class_1 += result['fn_class_1']
                total_error_class_1 += sum(result['errors_class_1'])
                total_error_count_class_1 += len(result['errors_class_1'])

    precision_class_0 = total_tp_class_0 / (total_tp_class_0 + total_fp_class_0) if (total_tp_class_0 + total_fp_class_0) > 0 else 0
    recall_class_0 = total_tp_class_0 / (total_tp_class_0 + total_fn_class_0) if (total_tp_class_0 + total_fn_class_0) > 0 else 0
    f1_score_class_0 = 2 * (precision_class_0 * recall_class_0) / (precision_class_0 + recall_class_0) if (precision_class_0 + recall_class_0) > 0 else 0

    precision_class_1 = total_tp_class_1 / (total_tp_class_1 + total_fp_class_1) if (total_tp_class_1 + total_fp_class_1) > 0 else 0
    recall_class_1 = total_tp_class_1 / (total_tp_class_1 + total_fn_class_1) if (total_tp_class_1 + total_fn_class_1) > 0 else 0
    f1_score_class_1 = 2 * (precision_class_1 * recall_class_1) / (precision_class_1 + recall_class_1) if (precision_class_1 + recall_class_1) > 0 else 0

    precision_avg=(precision_class_0+precision_class_1)/2
    recall_avg=(recall_class_0+recall_class_1)/2
    f1_avg=(f1_score_class_0+f1_score_class_1)/2

    if f1_avg >= best_f1:
        best_f1 = f1_avg
        best_f1_class_0=f1_score_class_0
        best_f1_class_1=f1_score_class_1

        best_thr = prob_thr

        best_precision=precision_avg
        best_precision_class_0=precision_class_0
        best_precision_class_1=precision_class_1

        best_recall=recall_avg
        best_recall_class_0=recall_class_0
        best_recall_class_1=recall_class_1

        best_total_tp_class_0=total_tp_class_0
        best_total_tp_class_1=total_tp_class_1

        best_total_fp_class_0 = total_fp_class_0
        best_total_fp_class_1 = total_fp_class_1

        best_total_fn_class_0 = total_fn_class_0
        best_total_fn_class_1 = total_fn_class_1

    # Print out the final results for each class
print("BEST Metrics for Class 0 (Small Vehicle):")
print(f"Precision: {best_precision_class_0:.4f}")
print(f"Recall: {best_recall_class_0:.4f}")
print(f"F1 Score: {best_f1_class_0:.4f}")
print(f"TP Small: {best_total_tp_class_0:.4f}")
print(f"FP Small: {best_total_fp_class_0:.4f}")
print(f"FN Small: {best_total_fn_class_0:.4f}")
# print(f"F1 Score: {f1_score_class_0:.4f}")


print("\nMetrics for Class 1 (Big Vehicle):")
print(f"Precision: {best_precision_class_1:.4f}")
print(f"Recall: {best_recall_class_1:.4f}")
print(f"F1 Score: {best_f1_class_1:.4f}")
print(f"TP Small: {best_total_tp_class_1:.4f}")
print(f"FP Small: {best_total_fp_class_1:.4f}")
print(f"FN Small: {best_total_fn_class_1:.4f}")

print("\nMetrics Averaged")
print(f"Precision: {best_precision:.4f}")
print(f"Recall: {best_recall:.4f}")
print(f"F1 Score: {best_f1:.4f}")
print(f"Best Threshold: {best_thr:.4f}")