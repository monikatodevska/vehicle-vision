import os
import shutil
import numpy as np
# annotations=r'\\Dip-t30\dip-nas2\Science\video3_anotacii'
# image_path=r'\\HP-1060\ForSharingD\VideoFrames\video3_frames'
# dst_image=r'E:\Science\Monika4\video3_val'
# annotations_empty=r'E:\Science\Monika4\empty_annot'
# annotations1=r'E:\Science\Monika4\video3_anotacii'

annotations=r'\\HP-1060\ForSharingD\AnotaciiKoliDenes\Annotations\Video1'
image_path=r'\\HP-1060\ForSharingD\VideoFrames\video1_frames'
dst_image=r'D:\KlasifikacijaVozila\Miladinovci\Sliki\videoframes_1'
annotations_empty=r'D:\KlasifikacijaVozila\Miladinovci\empty_annot\video1'
annotations1=r'D:\KlasifikacijaVozila\Miladinovci\Anotacii\videoframes_1'


for annot_name in os.listdir(annotations):
    im_name=annot_name[:-4] +'.jpg'
    # if os.stat(os.path.join(annotations, annot_name)).st_size==0:
    # shutil.copy(os.path.join(annotations,annot_name),os.path.join(annotations_empty,annot_name))
    # continue
    # else:
    if (os.path.exists(os.path.join(image_path,im_name))):
        shutil.copy(os.path.join(image_path,im_name),os.path.join(dst_image,im_name))
        # shutil.copy(os.path.join(annotations,annot_name),os.path.join(annotations1,annot_name))


