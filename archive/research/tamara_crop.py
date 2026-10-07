import os
import cv2
import numpy as np
trainOriginal = r'E:\Science\Monika1\tamara\filtered_photos\test'
# dst = r'E:\Science\Monika1\Novi\train-cam-cropped'
# srcAnnotationsPathTrainCopy1 = r'E:\Science\Monika\GRAM-RTMv4\Annotations\baseline_blurred'
# annot_path= r'E:\Science\Monika\GRAM-RTMv4\Annotations\M-30-anotacii'
listOrig=os.listdir(trainOriginal)
anotacii=r'E:\Science\Monika1\tamara\filtered_detections'
listOrig=os.listdir(trainOriginal)

for im_ind, im_name in enumerate(listOrig):

        src_image = cv2.imread(os.path.join(trainOriginal, im_name))
        rows, col = src_image.shape[:2]
        # desno = 50
        # levo = 248
        im_res=cv2.resize(src_image, (int(col/2), int(rows/2)), interpolation=cv2.INTER_AREA)
        rows_r, col_r=im_res.shape[:2]
        cropped_image = im_res[0:rows_r, 47:col_r - 34]
        rows_cr, col_cr = cropped_image.shape[:2]
        print(rows_cr,col_cr)
        resized = cv2.resize(cropped_image, (512, 341), interpolation=cv2.INTER_AREA)
        cv2.imwrite()
        objects = []
        HScale = 341 / rows_cr
        WScale = 512 / col_cr
        for annot_name in (os.listdir(anotacii)):
            if (im_name[:-4] + '_') in annot_name or (im_name[:-4] + '.txt') in annot_name:
                object = np.loadtxt(os.path.join(anotacii, annot_name), delimiter=',')

                object = [int(el) for el in object]

                pom = object[0]
                object[0] = object[1]
                object[1] = pom

                pom = object[2]
                object[2] = object[3]
                object[3] = pom
                object[1]=object[1]-47
                object[3]=object[3]-47
                object = [int(np.round(object[0] * HScale)), int(np.round(object[1] * WScale)), int(np.round(object[2] * HScale)), int(np.round(object[3] * WScale))]
                objects.append(object)

                if len(objects) < 0:
                    continue
                print(object)
        for obj in objects:
            cv2.rectangle(resized, (obj[1], obj[0]), (obj[3], obj[2]), color=(0, 255, 0), thickness=1)
                # print("crtam")
        cv2.imshow("slika", resized)
        cv2.imshow("slika1", im_res)
        cv2.waitKey(0)