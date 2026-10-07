import os
import shutil
srcImages=r'E:\Science\Monika3\site_sliki_GT'
dst=r'E:\Science\Monika2\site_sliki_GT1'

srcGT=r'D:\KlasifikacijaVozila\M-30\GroundTruthM-30'
# dst=r'E:\Science\Monika1\Video2_frames_augmented_val1'
dstannot=r'D:\KlasifikacijaVozila\M-30\GT_val'
# # srcImagesTam=r'E:\Science\Monika1\tamara\filtered_photos\augmented'
anchorless_val1=
# empty=r'E:\Science\Monika1\prazni'
list=os.listdir(srcImages)
lista=list[0:]
m=1
for gt in list:
    # if 'image' in im_name:
    #     # os.remove(os.path.join(srcImages,im_name))
    #     continue

    annot_name=gt[:-4] +'.txt'
    if(os.path.exists(os.path.join(srcGT,annot_name))):
        im_name_new='image'+str(m).zfill(6)+'.jpg'
        annot_name_new=im_name_new[:-4] + '.txt'
        # annot_name_new=str(annot_name_new)+ '.txt'

        os.rename(os.path.join(srcImages, im_name) , os.path.join(dst, im_name_new))
        os.rename(os.path.join(srcGT, annot_name) , os.path.join(dstannot, annot_name_new))
        m+=1
    # else:
    #     shutil.move(os.path.join(srcImages,im_name), os.path.join(empty, im_name))


