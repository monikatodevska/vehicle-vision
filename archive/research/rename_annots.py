
import os
import shutil
src_path=r'D:\Monika\VideosTest\koli_zatestmnozestvo\ch28_guzva\anotacii'
dst_path=r'D:\Monika\VideosTest\All_Cameras_annnotated\Annotations\Renamed\kam48_1'
dst_path_im=r'D:\Monika\VideosTest\All_Cameras_annnotated\Images\Renamed\kam48_1'
m=583
fnames=[x for x in os.listdir(src_path) if x[-4:]=='.txt']

for fname in fnames:
    new_name='image'+str(m).zfill(6)+'.txt'
    new_name_im='image'+str(m).zfill(6)+'.bmp'
    fname_im=fname[:-4]+'.bmp'
    m+=1
    shutil.copy(os.path.join(src_path,fname),os.path.join(dst_path,new_name))
    shutil.copy(os.path.join(src_path,fname_im),os.path.join(dst_path_im,new_name_im))