import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import random
from PIL import Image, ImageDraw

# Set the paths to the dataset directory
dataset_dir = r'D:\Monika\Pascal_Voc_Data'
image_dir = os.path.join(dataset_dir, 'JPEGImages')
annotation_dir = os.path.join(dataset_dir, 'Annotations')
segmentation_dir = os.path.join(dataset_dir, 'SegmentationClass')
obj_images_path=os.path.join(dataset_dir, 'SegmentationObject')

src_images=r'D:\Monika\VideosTest\Frames_Predmet_na_patot'
dst_images=r'D:\Monika\VideosTest\Zivotni_ZaTestiranje'
dst_annot=r'D:\Monika\Cyclists_Animals_Data\Zivotni\Annotations\pascal'


dict_reg1={"cat": (10,20), "dog":(20,30), "cow":(40,55),"horse":(30,50), "sheep":(40,50), "bear":(70,60)}
dict_reg2={"cat": (20,30), "dog":(30,40), "cow":(55,70),"horse":(50,70), "sheep":(50,60), "bear":(80,70)}

klasi={'cat':8, 'cow':10, 'dog':12, 'horse':13, 'sheep':17 }

mask_s=Image.open(r'D:\Monika\Pascal_Voc_Data\SegmentationClass\2007_000063.png')
palette = mask_s.getpalette()   # rgb
palette = np.reshape(palette, (-1, 3))
print(palette)

images_names=os.listdir(src_images)
num_imgs=len(images_names)
i=0
m=0

# Define a list of animal class names
animal_classes = ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
brojac=0
# Loop through each image and its corresponding annotation file
# for filename in os.listdir(annotation_dir):
#     if not filename.endswith('.xml'):
#         continue
#
#     # Parse the XML annotation file
#     xml_path = os.path.join(annotation_dir, filename)
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#
#     # Check if the object in the annotation file is an animal
#     for obj in root.findall('object'):
#         name = obj.find('name').text
#         if name in animal_classes:
#             bb_xml=obj.find('bndbox')
#             bb = [np.int(bb_xml.find('xmin').text),  # min_col
#                   np.int(bb_xml.find('xmax').text),  # max_col
#                   np.int(bb_xml.find('ymin').text),  # min_row
#                   np.int(bb_xml.find('ymax').text),  # max_row
#                   ]
#             # Retrieve the corresponding image and segmentation mask
#             image_path = os.path.join(image_dir, root.find('filename').text)
#             segmentation_path = os.path.join(segmentation_dir, root.find('filename').text.split('.')[0] + '.png')
#
#             # Open the image and segmentation mask
#             image = Image.open(image_path)
#             segmentation_mask = Image.open(segmentation_path)
#
#             color=image.crop((bb[0], bb[2], bb[1], bb[3]))
#             color.show()
#             mask_s=segmentation_mask.crop((bb[0], bb[2], bb[1], bb[3]))
#             mask_s.show()
#             print(mask_s.getcolors())
#
#             mask_s.show()
#             print(name)
#             print(klasi[name])
#             print(palette[klasi[name]])
#
#             mask_s_pom = np.array(mask_s)
#             print(mask_s_pom)
#
#             kade=np.where(mask_s_pom==klasi[name])
#             print(kade)
#
#             min_r = np.min(kade[0])
#             min_c = np.min(kade[1])
#             max_r = np.max(kade[0])
#             max_c = np.max(kade[1])
#
#             mask_ss = mask_s[min_r:max_r, min_c:max_c]
#             # print(np.min(mask_s), np.max(mask_s))
#
#             mask_sss = Image.fromarray(mask_ss * 255)
#
#             mask_sss.show()
#             konj = color.copy()
#             konj.putalpha(mask_ss)
#             # cv2.imshow('slika', konj)
#             konj.show()
#             # color.putalpha(mask)
#
#             mil = Image.open(os.path.join(src_images, images_names[i]))
#
#             mil = mil.crop((117, 0, 1920, 1080))
#             # mil = image[0:image.shape[0], 117:image.shape[1]]
#             mil = mil.resize((469, 281))
#             reg_n = random.randint(1, 2)
#             x = random.randint(65, 350)
#             width, height = konj.size
#             if height == 0 or width == 0:
#                 continue
#             asp_ratio = height / width
#             category = name
#
#             if reg_n == 1:
#                 reg = random.randint(109, 190)
#                 dim = dict_reg1[category]
#                 print(dim)
#
#             else:
#                 reg = random.randint(190, 281 - 50)
#                 dim = dict_reg2[category]
#                 print(dim)
#
#             if height > width:
#                 new_height = dim[0]
#                 if new_height > height:
#                     continue
#                 new_w = int(dim[0] / asp_ratio)
#             else:
#                 new_w = dim[1]
#                 if new_w > width:
#                     continue
#                 new_height = int(dim[1] * asp_ratio)
#             print(konj.mode)
#             print(category)
#             print(konj.size)
#             konj_r = konj.resize((new_w, new_height))
#
#             # konj_r = cv2.resize(np.array(konj), (new_w, new_height), interpolation=cv2.INTER_AREA)
#
#             # konj_r=cv2.cvtColor(konj_r, cv2.COLOR_RGB2BGR)
#             # cv2.imshow('slika',konj_r)
#             # konj_r.show()
#
#             mil.paste(konj_r, (x - int(new_w / 2), reg - int(new_height / 2)), konj_r)
#             # mil.show()
#             new_name = str(brojac).zfill(6) + '.bmp'
#             mil.save(os.path.join(dst_images, new_name))
#             brojac += 1
#             objekt = [reg - int(new_height / 2), x - int(new_w / 2), reg + int(new_height / 2), x + int(new_w / 2), 2]
#
#             np.savetxt(os.path.join(dst_annot, new_name[:-4] + '.txt'), [objekt], fmt='%i', delimiter=',')
#             img1 = ImageDraw.Draw(mil)
#             # img1.rectangle([(x,reg), (x+new_w, reg+new_height)], outline='red')
#             img1.rectangle([(x - int(new_w / 2), reg - int(new_height / 2)), (x + int(new_w / 2), reg + int(new_height / 2))], outline='red')
#             # mil.show()
#             # cv2.waitKey(0)
#             print(1)
#
#             # Do something with the image and segmentation mask
#             # For example, you can display them using matplotlib
#             # import matplotlib.pyplot as plt
#             # fig, axs = plt.subplots(1, 2)
#             # axs[0].imshow(image)
#             # axs[1].imshow(segmentation_mask)
#             plt.show()
#             print(1)




obj_imgs=os.listdir(obj_images_path)
import cv2
for obj_img_name in obj_imgs:
    color=Image.open(os.path.join(image_dir,obj_img_name[:-4]+'.jpg'))
    obj_img=Image.open(os.path.join(obj_images_path,obj_img_name))
    seg_img=Image.open(os.path.join(segmentation_dir,obj_img_name))
    # obj_img.show()
    # seg_img.show()
    # color.show()

    obj_img_ind = np.array(obj_img)
    seg_img_ind = np.array(seg_img)
    indeksi = np.unique(obj_img_ind)

    for ind in indeksi:
        kade = np.where(obj_img_ind == ind)
        print(obj_img_ind.shape)
        print(len(kade[0]))
        if seg_img_ind[kade[0][0], kade[1][0]] in klasi.values():
            m+=1
            if m>=num_imgs:
                m=0
            print(obj_img_ind.shape)
            # print(np.unique(obj_img_ind))
            mask = np.zeros_like(obj_img_ind)
            mask[obj_img_ind == ind] = 255
            print(mask)
            #cv2.imshow('slika', mask)
            #cv2.waitKey(0)
            kade = np.where(mask)
            min_r =np.min(kade[0])
            min_c = np.min(kade[1])
            max_r = np.max(kade[0])
            max_c = np.max(kade[1])

            color = color.crop((min_c, min_r, max_c, max_r))  # left, top, right, bottom
            # color.show()

            mask_s = mask[min_r:max_r, min_c:max_c]

            mask_ss=Image.fromarray(mask_s)
            konj = color.copy()
            #color.show()

            konj.putalpha(mask_ss)
            # cv2.imshow('slika', konj)
            #konj.show()
            # color.putalpha(mask)

            mil = Image.open(os.path.join(src_images, images_names[i]))

            #mil = mil.crop((117, 0, 1920, 1080))
            # mil = image[0:image.shape[0], 117:image.shape[1]]
            mil = mil.resize((960, 540))
            reg_n = random.randint(1, 2)
            x = random.randint(100, 800)
            width, height = konj.size
            if height == 0 or width == 0:
                continue
            asp_ratio = height / width
            klasa_so_broj=seg_img_ind[kade[0][0], kade[1][0]]

            value = [i for i in klasi if klasi[i]==klasa_so_broj]
            print(value[0])

            category = value[0]

            if reg_n == 1:
                reg = random.randint(100, 250)
                dim = dict_reg1[category]
                print(dim)

            else:
                reg = random.randint(250, 500)
                dim = dict_reg2[category]
                print(dim)

            if height > width:
                new_height = dim[0]
                if new_height > height:
                    continue
                new_w = int(dim[0] / asp_ratio)
            else:
                new_w = dim[1]
                if new_w > width:
                    continue
                new_height = int(dim[1] * asp_ratio)
            print(konj.mode)
            print(category)
            print(konj.size)
            konj_r = konj.resize((new_w, new_height))

            # konj_r = cv2.resize(np.array(konj), (new_w, new_height), interpolation=cv2.INTER_AREA)

            # konj_r=cv2.cvtColor(konj_r, cv2.COLOR_RGB2BGR)
            # cv2.imshow('slika',konj_r)
            # konj_r.show()

            mil.paste(konj_r, (x - int(new_w / 2), reg - int(new_height / 2)), konj_r)
            #mil.show()
            new_name = str(brojac).zfill(6) + '.bmp'
            mil.save(os.path.join(dst_images, new_name))
            brojac += 1
            objekt = [reg - int(new_height / 2), x - int(new_w / 2), reg + int(new_height / 2), x + int(new_w / 2), 2]

            #np.savetxt(os.path.join(dst_annot, new_name[:-4] + '.txt'), [objekt], fmt='%i', delimiter=',')
            img1 = ImageDraw.Draw(mil)
            # img1.rectangle([(x,reg), (x+new_w, reg+new_height)], outline='red')
            img1.rectangle([(x - int(new_w / 2), reg - int(new_height / 2)), (x + int(new_w / 2), reg + int(new_height / 2))], outline='red')
            #mil.show()
            # cv2.waitKey(0)
            #print(1)




...

