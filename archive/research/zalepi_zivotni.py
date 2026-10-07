
import json
from pycocotools.coco import COCO
from pycocotools import mask
from PIL import Image, ImageDraw
import random
# Load COCO annotations for validation set
coco = COCO(r'D:\Monika\Animals_Data\annotations\instances_train.json')

# Load COCO annotations for stuff categories

src_images=r'D:\Monika\VideosTest\Frames_Predmet_na_patot'
dst_images=r'D:\Monika\VideosTest\Zivotni_ZaTestiranje'
dst_annot=r'D:\Monika\Cyclists_Animals_Data\Zivotni\Annotations'
# Define animal category IDs
animal_ids = coco.getCatIds(catNms=['cat', 'dog', 'horse', 'sheep', 'cow', 'bear'])
import os
import numpy as np
import cv2

dict_reg1={"cat": (10,20), "dog":(20,30), "cow":(40,55),"horse":(40,60), "sheep":(40,50), "bear":(70,60)}
dict_reg2={"cat": (20,30), "dog":(30,40), "cow":(55,70),"horse":(60,80), "sheep":(50,60), "bear":(80,70)}

images_names=os.listdir(src_images)
num_imgs=len(images_names)
i=0
m=0
brojac=0
broi_zivotni=0
# Loop over images in validation set
for img_id in coco.getImgIds():
    # Load image and its annotations
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(r'D:\Monika\Animals_Data\images\train', img_info['file_name'])

    img = Image.open(img_path).convert('RGB')
    # img.show()
    #
    # color = img.crop((20, 30, 100, 110))
    # color.show()

    # exit(1)

    #img = cv2.imread(img_path)
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=animal_ids, iscrowd=0)
    anns = coco.loadAnns(ann_ids)


    # Loop over annotations and extract masks
    masks = []
    categories=[]
    for i,ann in enumerate(anns):
        mask_data = coco.annToMask(ann)

        masks.append(mask_data)
        entity_id = anns[i]["category_id"]
        entity = coco.loadCats(entity_id)[0]["name"]
        categories.append(entity)

    # Check if any animal masks were extracted
    if masks:
        # Combine animal masks into a single mask
        # print('masks')

        if m>=num_imgs:
            m=0

        for ind, mask in enumerate(masks):
            # print('mask')
            m+=1
            kade=(np.where(mask>0))
            min_r=np.min(kade[0])
            min_c=np.min(kade[1])
            max_r=np.max(kade[0])
            max_c=np.max(kade[1])
            # print(min_c, min_r, max_c, max_r)
            # print(img.size)
            color = img.crop((min_c, min_r, max_c, max_r))  # left, top, right, bottom
            #color.show()

            mask_s=mask[min_r:max_r, min_c:max_c]
            # print(np.min(mask_s), np.max(mask_s))

            mask_ss=Image.fromarray(mask_s * 255)
            #mask_ss.show()

            konj = color.copy()
            konj.putalpha(mask_ss)
            # cv2.imshow('slika', konj)
            #konj.show()
            # color.putalpha(mask)

            mil=Image.open(os.path.join(src_images,images_names[i]))

            mil=mil.crop((117,0,1920,1080))
            #mil = image[0:image.shape[0], 117:image.shape[1]]
            mil=mil.resize((469,281))
            reg_n=random.randint(1,2)
            x = random.randint(65, 350)
            width, height = konj.size
            if height==0 or width==0:
                continue
            asp_ratio = height / width
            category = categories[ind]

            if reg_n==1:
                reg = random.randint(109, 190)
                dim=dict_reg1[category]
                # print(dim)

            else:
                reg = random.randint(190, 281 - 50)
                dim=dict_reg2[category]
                # print(dim)


            if height > width:
                new_height = dim[0]
                if new_height> height:
                    continue
                new_w = int(dim[0] / asp_ratio)
            else:
                new_w = dim[1]
                if new_w>width:
                    continue
                new_height = int(dim[1] * asp_ratio)
            # broi_zivotni+=1
            # if broi_zivotni<27000:
            #     continue
            # print(konj.mode)
            # print(category)
            # print(konj.size)
            konj_r=konj.resize((new_w,new_height))

            #konj_r = cv2.resize(np.array(konj), (new_w, new_height), interpolation=cv2.INTER_AREA)

            #konj_r=cv2.cvtColor(konj_r, cv2.COLOR_RGB2BGR)
            #cv2.imshow('slika',konj_r)
            #konj_r.show()

            mil.paste(konj_r, (x-int(new_w/2),reg-int(new_height/2)), konj_r)
            #mil.show()
            new_name=str(brojac).zfill(6)+'.bmp'
            mil.save(os.path.join(dst_images,new_name))
            brojac+=1
            objekt = [reg-int(new_height/2), x-int(new_w/2), reg+int(new_height/2), x+int(new_w/2), 2]


            #np.savetxt(os.path.join(dst_annot, new_name[:-4] + '.txt'), [objekt], fmt='%i', delimiter=',')
            img1=ImageDraw.Draw(mil)
            #img1.rectangle([(x,reg), (x+new_w, reg+new_height)], outline='red')
            img1.rectangle([(x-int(new_w/2),reg-int(new_height/2)), (x+int(new_w/2), reg+int(new_height/2))], outline='red')
            mil.show()
            cv2.waitKey(0)
            print(1)
            #half_h=int(konj_r.shape[0]/2)
            #half_w=int(konj_r.shape[1]/2)
            #mil[reg-half_h:reg-half_h+(konj_r.shape[0]),x-half_w:x-half_w+(konj_r.shape[1])]=konj_r


        # combined_mask = masks[0]
        #
        #
        #
        # for mask_data in masks[1:]:
        #     combined_mask = np.logical_or(combined_mask, mask_data)
        #
        #     # Apply combined mask to image and save result
        # masked_img = Image.fromarray(np.array(img) * np.expand_dims(combined_mask, axis=2))
        # dst_path = os.path.join(r'D:\Monika\Animals_Data\visual_res', img_info['file_name'])
        # masked_img.save(dst_path)
        #     combined_mask = mask.merge([combined_mask, mask_data], intersect=False)
        #
        # # Apply combined mask to image and save result
        # masked_img = Image.fromarray(mask.apply_mask(img, combined_mask))
        # dst_path=os.path.join(r'D:\Monika\Animals_Data\visual_res', img_info['file_name'])
        # masked_img.save(dst_path)
