import os
import shutil

root=r'\\HP-1060\ForSharingD\AnotaciiKoliDenes\DETRAC-Train-Annotations-Reformatted_2'
root1=r'\\HP-1060\SoobrakjajRPNData\DETRAC_Dataset\Insight-MVT_Annotation_Train'
kineski_dataset=r'D:\KlasifikacijaVozila\Kineski\kineski_dataset'
kineski_dataset1=r'D:\KlasifikacijaVozila\Kineski\kineski_dataset1'

sliki=r'D:\KlasifikacijaVozila\Kineski\sliki'
sliki1=r'D:\KlasifikacijaVozila\Kineski\sliki1'

dirnames=os.listdir(root)
print(dirnames)
m=1
for dir in dirnames:
    filenames=os.listdir(os.path.join(root,dir))
    for filename in filenames:
        name=os.path.join(root,dir)
        im_name=filename[:-4]+'.jpg'
        new_name_anot='im'+str(m).zfill(6)+'.txt'
        new_name='im'+str(m).zfill(6)+'.jpg'
        shutil.copy(os.path.join(name, filename), os.path.join(kineski_dataset,filename) )
        name1=os.path.join(root1,dir)
        shutil.copy(os.path.join(name1, im_name), os.path.join(sliki, new_name) )

        os.rename( os.path.join(kineski_dataset,filename), os.path.join(kineski_dataset1,new_name_anot))
        os.rename( os.path.join(sliki, new_name), os.path.join(sliki1,new_name))
        m+=1