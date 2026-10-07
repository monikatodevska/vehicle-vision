
import json
from pycocotools.coco import COCO
from pycocotools import mask
from PIL import Image, ImageDraw
import random
# Load COCO annotations for validation set
coco = COCO(r'D:\Monika\Animals_Data\annotations\instances_train.json')

# Load COCO annotations for stuff categories
coco_stuff = COCO(r'D:\Monika\Animals_Data\annotations\stuff.json')


src_images=r'D:\Monika\Results\RamkiZaTovar'
#src_images=r'D:\Monika\VideosTest\Frames_Predmet_na_patot'
dst_images=r'D:\Monika\Cyclists_Animals_Data\Tovar\Images'
dst_annot=r'D:\Monika\Cyclists_Animals_Data\Tovar\Annotations'
# Define animal category IDs
animal_ids = coco.getCatIds(catNms=['bicycle', 'motorcycle', 'fire hydrant', 'backpack', 'umbrella', 'handbag', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'basebal bat', 'skateboard',
         'surfboard','tennis racket', 'bowl', 'chair','couch','potted plant', 'tv', 'laptop', 'keyboard', 'microwave',
         'oven', 'toaster', 'refrigerator', 'book', 'vase', 'teddy bear', 'hair drier'] )
import os
import numpy as np
import cv2

# dict_reg1={"cat": (20,30), "dog":(30,40), "cow":(50,60),"horse":(50,80), "sheep":(50,60), "bear":(70,60)}
# dict_reg2={"cat": (30,40), "dog":(40,50), "cow":(70,80),"horse":(60,90), "sheep":(70,80), "bear":(80,70)}

images_names=os.listdir(src_images)
print(images_names)
num_imgs=len(images_names)
i=0
m=0
brojac=0
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
        print('masks')



        for ind, mask in enumerate(masks):
            print('mask')
            if m >= num_imgs:
                m = 0

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

            mil=Image.open(os.path.join(src_images,images_names[m]))

            mil=mil.crop((117,0,1920,1080))
            #mil = image[0:image.shape[0], 117:image.shape[1]]
            mil=mil.resize((469,281))
            reg_n=random.randint(1,2)
            x = random.randint(78, 410)
            width, height = konj.size
            if height<=5 or width<=5:
                continue
            asp_ratio = height / width
            category = categories[ind]

            if reg_n==1:
                reg = random.randint(86, 141)
                dim=(20,30)
                print(dim)

            else:
                reg = random.randint(141, 268)
                dim=(30,40)
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


            np.savetxt(os.path.join(dst_annot, new_name[:-4] + '.txt'), [objekt], fmt='%i', delimiter=',')
            m += 1

            img1=ImageDraw.Draw(mil)
            #img1.rectangle([(x,reg), (x+new_w, reg+new_height)], outline='red')
            img1.rectangle([(x-int(new_w/2),reg-int(new_height/2)), (x+int(new_w/2), reg+int(new_height/2))], outline='red')
            #mil.show()
            #cv2.waitKey(0)
            print(1)

