import os.path
import shutil

src_path=r'D:\Monika\yolov5\runs\detect\exp19\labels'
annot_path=r'D:\Monika\yolov5\runs\detect\exp19\labels'

dst_path=r'D:\Monika\yolov5\runs\detect\exp11\exp11_renamed'

filenames=[x for x in os.listdir(annot_path) if x.endswith('.txt')]
images_names=[x for x in os.listdir(annot_path) if x.endswith('.jpg')]
for f_n in filenames:
    new_fn=f_n[:-4]+'_miladinovci'+'_reg'+'.txt'
    new_imname=f_n[:-4]+'_miladinovci'+'_reg'+'.bmp'

    im_name=f_n[:-4]+'.jpg'
    shutil.copy(os.path.join(src_path,f_n),os.path.join(dst_path,new_fn))
    shutil.copy(os.path.join(src_path,im_name),os.path.join(dst_path,new_imname))



