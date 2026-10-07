"""
postproc
"""

# python imports
import numpy as np
import os
import cv2

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




def line_segments_intersect(x1, x2, y1, y2):
    """
    calculate intersection over union for two 1-d segments
    # Assumes x1 <= x2 and y1 <= y2; if this assumption is not safe, the code
    # can be changed to have x1 being min(x1, x2) and x2 being max(x1, x2) and
    # similarly for the ys.
    :param x1: min_point of first segment
    :param x2: max_point of first segment
    :param y1: min_point of second segment
    :param y2: max_point of second segment
    :return: IoU value
    """

    if x2 >= y1 and y2 >= x1:
        # the segments overlap
        intersection = min(x2, y2) - max(y1, x1)
        union = max(x2, y2) - min(x1, y1)
        iou = float(intersection) / float(union)

        return iou

    return 0


def calc_iou_partwise(box1, box2):
    """
    calculate intersection over union for height and width separately
    :param box1: list of coordinates: row1, col1, row2, col2 [list]
    :param box2: list of coordinates: row1, col1, row2, col2 [list]
    :return: iou_height: iou value by height [float]
             iou_width: iou value by width [float]
    """

    iou_height = line_segments_intersect(box1[0], box1[2], box2[0], box2[2])
    iou_width = line_segments_intersect(box1[1], box1[3], box2[1], box2[3])

    return iou_height, iou_width


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


def nms_tanja(image, img_dest, lista, thr, colors_list, num_classes, flag_save_coords):
    # print(img_dest)
    coords_list = []

    finalwindows = []

    if len(lista) == 0:
        cv2.imwrite(os.path.join(img_dest), image)
        return

    all_clusters = getclustersforanchors(lista, thr)

    for cluster in all_clusters:
        # print(cluster)
        maxwindows = []

        if len(cluster) == 0:
            continue

        elements = [item[2] - item[0] for item in cluster]

        if len(elements) <= 2:
            first2max = sorted(elements)
        else:
            # first2max = sorted(elements)[len(elements) - 2:len(elements)]
            first2max = sorted(elements)[:]

        first2max_noduplicates = []
        for i in first2max:
            if i not in first2max_noduplicates:
                first2max_noduplicates.append(i)

        for maxnum in first2max_noduplicates:
            indices = [i for i, x in enumerate(elements) if x == maxnum]

            for idx in indices:
                maxwindows.append(cluster[idx])

        bboxes_per_class = [[] for i in range(num_classes)]     # bounding boxes grouped by class

        for maxwindow in maxwindows:
            bboxes_per_class[maxwindow[-1]].append(maxwindow)

        max_class = max((len(l), i) for i, l in enumerate(bboxes_per_class))[1]

        color = colors_list[max_class]

        finalwindow = average_window(maxwindows)
        finalwindows.append(finalwindow)
        cv2.rectangle(image, (int(finalwindow[1]), int(finalwindow[0])), (int(finalwindow[3]), int(finalwindow[2])), color, thickness=1)

        coords_list.append([int(finalwindow[0]), int(finalwindow[1]), int(finalwindow[2]) - int(finalwindow[0]), int(finalwindow[3]) - int(finalwindow[1]), max_class])

    cv2.imwrite(os.path.join(img_dest), image)

    if flag_save_coords:
        coords_list = np.array(coords_list)
        # print(img_dest[:-4]+'.txt')
        np.savetxt(img_dest[:-4] + '.txt', coords_list, delimiter=',', fmt='%i')

    return finalwindows


from copy import deepcopy
def nms_new_save_original(image, img_dest, lista, thr, colors_list, num_classes,left_crop, nacrtaj, flag_save_coords,flag_choose, HNew=1080,HOld=341,WNew=1620,WOld=512):
    # print(img_dest)

    coords_list = []

    finalwindows = []
    image_orig=deepcopy(image)
    if len(lista) == 0:
        cv2.imwrite(os.path.join(img_dest), image)
        return

    all_clusters = getclustersforanchors(lista, thr)

    for cluster in all_clusters:
        # print(cluster)
        maxwindows = []
        maxwindows_new=[]

        if len(cluster) == 0:
            continue

        elements = [item[2] - item[0] for item in cluster]

        if len(elements) <= 2:
            first2max = sorted(elements)
        else:
            # first2max = sorted(elements)[len(elements) - 2:len(elements)]
            first2max = sorted(elements)[:]

        first2max_noduplicates = []
        for i in first2max:
            if i not in first2max_noduplicates:
                first2max_noduplicates.append(i)

        for maxnum in first2max_noduplicates:
            indices = [i for i, x in enumerate(elements) if x == maxnum]

            for idx in indices:
                maxwindows.append(cluster[idx])


        bboxes_per_class = [[] for i in range(num_classes)]     # bounding boxes grouped by class

        for maxwindow in maxwindows:
            bboxes_per_class[maxwindow[-1]].append(maxwindow)

        max_class = max((len(l), i) for i, l in enumerate(bboxes_per_class))[1]

        for maxwindow in maxwindows:
            if maxwindow[-1]==max_class:
                maxwindows_new.append(maxwindow)
        color = colors_list[max_class]

        finalwindow = average_window(maxwindows_new)
        finalwindow_hd=[0]*4
        finalwindow_hd[0]=finalwindow[0]*(HNew/HOld)
        finalwindow_hd[1]=finalwindow[1]*(WNew/WOld)+left_crop
        finalwindow_hd[2]=finalwindow[2]*(HNew/HOld)
        finalwindow_hd[3]=finalwindow[3]*(WNew/WOld)+left_crop

        finalwindows.append(finalwindow_hd)
        # if nacrtaj:
        #     cv2.rectangle(image, (int(finalwindow[1]*(1620/512))+left_crop, int(finalwindow[0]*(1080/341))), (int(finalwindow[3]*(1620/512))+left_crop, int(finalwindow[2]*(1080/341))), color, thickness=1)
        # cv2.rectangle(image, (int(finalwindow_hd[1]), int(finalwindow_hd[0])), (int(finalwindow_hd[3]), int(finalwindow_hd[2])), color, thickness=1)

        coords_list.append([int(finalwindow_hd[0]), int(finalwindow_hd[1]), int(finalwindow_hd[2]), int(finalwindow_hd[3]), max_class])
        # coords_list.append([int(finalwindow[0]*(1080/341)), int(finalwindow[1]*(1620/512))+left_crop, int(finalwindow[2]*(1080/341)), int(finalwindow[3]*(1620/512))+left_crop, max_class])
    if not flag_choose:
        cv2.imwrite(os.path.join(img_dest), image)
        # cv2.imwrite(os.path.join(img_dest), image_orig[:-4]+'_ORIG.bmp')

        if flag_save_coords:
            np.savetxt(img_dest[:-4] + '.txt', coords_list, delimiter=',', fmt='%i')
    #     # coords_list = np.array(coords_list)
    #     # coords_list[:,0]=coords_list[:,0]*int(1080/341)
    #     # coords_list[:,1]=coords_list[:,1]*int(1620/512)+left_crop
    #     # coords_list[:,2]=coords_list[:,2]*int(1080/341)
    #     # coords_list[:,3]=coords_list[:,3]*int(1620/512)+left_crop
    #
    #     # print(img_dest[:-4]+'.txt')


    return image, coords_list
def nms_new(image, img_dest, lista, thr, colors_list, num_classes, left_crop, nacrtaj, flag_save_coords,flag_choose):
    # print(img_dest)
    coords_list = []

    finalwindows = []
    image_orig=deepcopy(image)
    if len(lista) == 0:
        cv2.imwrite(os.path.join(img_dest), image)
        return

    all_clusters = getclustersforanchors(lista, thr)

    for cluster in all_clusters:
        # print(cluster)
        maxwindows = []
        maxwindows_new=[]

        if len(cluster) == 0:
            continue

        elements = [item[2] - item[0] for item in cluster]

        if len(elements) <= 2:
            first2max = sorted(elements)
        else:
            # first2max = sorted(elements)[len(elements) - 2:len(elements)]
            first2max = sorted(elements)[:]

        first2max_noduplicates = []
        for i in first2max:
            if i not in first2max_noduplicates:
                first2max_noduplicates.append(i)

        for maxnum in first2max_noduplicates:
            indices = [i for i, x in enumerate(elements) if x == maxnum]

            for idx in indices:
                maxwindows.append(cluster[idx])


        bboxes_per_class = [[] for i in range(num_classes)]     # bounding boxes grouped by class

        for maxwindow in maxwindows:
            bboxes_per_class[maxwindow[-1]].append(maxwindow)

        max_class = max((len(l), i) for i, l in enumerate(bboxes_per_class))[1]

        for maxwindow in maxwindows:
            if maxwindow[-1]==max_class:
                maxwindows_new.append(maxwindow)
        color = colors_list[max_class]

        finalwindow = average_window(maxwindows_new)
        finalwindows.append(finalwindow)
        # if nacrtaj:
        #     cv2.rectangle(image, (int(finalwindow[1]*(1620/512))+left_crop, int(finalwindow[0]*(1080/341))), (int(finalwindow[3]*(1620/512))+left_crop, int(finalwindow[2]*(1080/341))), color, thickness=1)
        cv2.rectangle(image, (int(finalwindow[1]), int(finalwindow[0])), (int(finalwindow[3]), int(finalwindow[2])), color, thickness=1)

        coords_list.append([int(finalwindow[0]), int(finalwindow[1]), int(finalwindow[2]), int(finalwindow[3]), max_class])
        # coords_list.append([int(finalwindow[0]*(1080/341)), int(finalwindow[1]*(1620/512))+left_crop, int(finalwindow[2]*(1080/341)), int(finalwindow[3]*(1620/512))+left_crop, max_class])
    # if not flag_choose:
    # cv2.imshow('sl',image)
    # cv2.waitKey(0)
    cv2.imwrite(img_dest, image)
    # cv2.imwrite(os.path.join(img_dest), image_orig[:-4]+'_ORIG.bmp')

    if flag_save_coords:
        np.savetxt(img_dest[:-4] + '.txt', coords_list, delimiter=',', fmt='%i')
    #     # coords_list = np.array(coords_list)
    #     # coords_list[:,0]=coords_list[:,0]*int(1080/341)
    #     # coords_list[:,1]=coords_list[:,1]*int(1620/512)+left_crop
    #     # coords_list[:,2]=coords_list[:,2]*int(1080/341)
    #     # coords_list[:,3]=coords_list[:,3]*int(1620/512)+left_crop
    #
    #     # print(img_dest[:-4]+'.txt')


    # return


def nms_tanja_w_metrics(image, img_dest, lista, thr, colors_list, num_classes, flag_save_coords):

    coords_list = []

    finalwindows = []

    if len(lista) == 0:
        cv2.imwrite(os.path.join(img_dest), image)
        return

    all_clusters = getclustersforanchors(lista, thr)

    for cluster in all_clusters:
        # print(cluster)
        maxwindows = []

        if len(cluster) == 0:
            continue

        elements = [item[2] - item[0] for item in cluster]

        if len(elements) <= 2:
            first2max = sorted(elements)
        else:
            # first2max = sorted(elements)[len(elements) - 2:len(elements)]
            first2max = sorted(elements)[:]

        first2max_noduplicates = []
        for i in first2max:
            if i not in first2max_noduplicates:
                first2max_noduplicates.append(i)

        for maxnum in first2max_noduplicates:
            indices = [i for i, x in enumerate(elements) if x == maxnum]

            for idx in indices:
                maxwindows.append(cluster[idx])

        bboxes_per_class = [[] for i in range(num_classes)]     # bounding boxes grouped by class

        for maxwindow in maxwindows:
            bboxes_per_class[maxwindow[-1]].append(maxwindow)

        max_class = max((len(l), i) for i, l in enumerate(bboxes_per_class))[1]

        color = colors_list[max_class]

        finalwindow = average_window(maxwindows)
        finalwindows.append(finalwindow)
        cv2.rectangle(image, (int(finalwindow[1]), int(finalwindow[0])), (int(finalwindow[3]), int(finalwindow[2])), color, thickness=1)

        coords_list.append([int(finalwindow[0]), int(finalwindow[1]), int(finalwindow[2]) - int(finalwindow[0]), int(finalwindow[3]) - int(finalwindow[1]), max_class])

    cv2.imwrite(os.path.join(img_dest), image)

    if flag_save_coords:
        coords_list = np.array(coords_list)
        np.savetxt(img_dest.split('.')[0] + '.txt', coords_list, delimiter=',', fmt='%i')
   #$NOTE: crtanje na detektirani prozorci posle nms na site objekti na edna slika
#   for window in finalwindows:
    #     cv2.rectangle(image, (int(window[1]), int(window[0])), (int(window[3]), int(window[2])), color, thickness=1)
    # cv2.imshow('slika',image)
    # cv2.waitKey(0)
    return finalwindows
