import os
import numpy as np
import cv2

images_path=r'E:\Science\Monika1\Novi\train-cam'
annot_path=r'E:\Science\Monika1\Novi\D2_train\anotacii'
HScale=540/1080
WScale=960/1920
for im_name in (os.listdir(images_path)):
    image=cv2.imread(os.path.join(images_path,im_name))
    resized = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)

    # cv2.imshow("slika", resized)
    # cv2.waitKey(0)
    for annot_name in (os.listdir(annot_path)):
        if (im_name[:-4] + '_') in annot_name or (im_name[:-4] + '.txt') in annot_name:
            object = np.loadtxt(os.path.join(annot_path, annot_name), delimiter=',')

            object = [int(el) for el in object]
            object[2] = object[0] + object[2]
            object[3] = object[1] + object[3]
            with open(os.path.join(annot_path,annot_name), 'rb+') as fh:
                fh.seek(-1, os.SEEK_END)

                fh.truncate()
                fh.close()
            if(len(object)==4):
                cv2.rectangle(resized, (int(object[1]*WScale), int(object[0]*HScale)), (int(object[3]*WScale), int(object[2]*HScale)), color=(0, 0, 255), thickness=1)
                cv2.imshow(im_name,resized)
                cv2.waitKey(0)
                class_num=input("Klasa na vozilo, 1-kola, 2-kamion, 3-malo kombe, 4-golemo kombe")
                if int(class_num)==1 or int(class_num)==2 or int(class_num)==3 or int(class_num)==4:
                    f=open(os.path.join(annot_path,annot_name), 'a')
                    cl=','+class_num
                    f.write(cl)
                    f.close()
            else:
                continue


