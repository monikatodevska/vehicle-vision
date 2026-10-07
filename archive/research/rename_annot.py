import os
import shutil
# srcAnnots=r'E:\Science\Monika\GRAM-RTMv4\Annotations\flipped'
# dstAnnots=r'E:\Science\Monika\GRAM-RTMv4\Annotations\flipped-renamed'
# list=os.listdir(srcAnnots)
# lista=list[0:]

# m=33820
#
# for k in lista:
#     os.rename(os.path.join(srcAnnots, k) , os.path.join(dstAnnots, str(m) + '.xml'))
#     m=m+1
# lista_br=[]
#
# for k in lista:
#     br=int(k[:-4])
#     br_n=br+m
#     os.rename(os.path.join(srcAnnots, k) , os.path.join(dstAnnots, str(br_n) + '.xml'))
    # lista_br.append(br_n)

# print(lista_br)
# lista_sorted=lista_br.sort()
# print(1)
# for i in lista_br:
ImPath=r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams\Validacija\Images'
GTPath=r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams\Validacija\GT'
DstImPath=r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_Renamed\Images'
DstGTPath=r'D:\Monika\Trening_With_Translations-Up-Down-All_Cams_Renamed\GT'
lista=os.listdir(ImPath)
# m=2670
# for dir in lista:
#     filenames=os.listdir(os.path.join(ImPath,dir))
#
#     for filename in filenames:
#         txt_filename=filename[:-4]+'.txt'
#         new_filename=str(m).zfill(6)+'.txt'
#         shutil.move(os.path.join(GTPath,dir,txt_filename), os.path.join(DstGTPath, new_filename))
#         shutil.move(os.path.join(ImPath,dir,filename), os.path.join(DstImPath, new_filename[:-4]+'.jpg'))
#         m=m+1


filenames=os.listdir(DstImPath)

for filename in filenames:

    os.rename(os.path.join(DstImPath,filename), os.path.join(DstImPath,'image'+filename))
    os.rename(os.path.join(DstGTPath,filename[:-4]+'.txt'), os.path.join(DstGTPath,'image'+filename[:-4]+'.txt'))