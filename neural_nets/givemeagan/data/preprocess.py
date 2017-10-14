import glob
import os
import shutil
import pandas
import cv2 as cv

raw_data_root = '/opt/ProjectsPy/0_DATASETS/Cyrillic/'
small_root = '/opt/ProjectsPy/0_DATASETS/Cyrillic-small/'
target_path = '/opt/ProjectsPy/machine-learning/neural_nets/givemeagan/data/dataset/'


def rename():
    for directory in glob.iglob(raw_data_root + '*'):
        dir_name = basename(directory)
        sample = 0
        for filename in os.listdir(directory):
            if filename.endswith('.png'):
                title, ext = os.path.splitext(filename)
                os.rename(os.path.join(directory, filename),
                          os.path.join(directory, '{}.{}'.format(dir_name, sample)) + ext)
                sample += 1


def copy_to_project():
    for cdir, subdirs, files in os.walk(raw_data_root):
        for filename in files:
            if filename.endswith('.png'):
                shutil.copy(os.path.join(cdir, filename), target_path)


def prepare_dataframe_csv():
    ll = []
    for cdir, subdirs, files in os.walk(raw_data_root):
        print(cdir)
        for filename in files:
            if filename.endswith('.png'):
                ll.append((os.path.join(cdir, filename), filename.split('.')[0]))
        frame = pandas.DataFrame(ll, columns=['img_path', 'label'])
        frame.to_csv(target_path + 'dataset.csv', encoding='utf-8')
 
    
def make_small_dirs():
    for cdir, subdirs, files in os.walk(raw_data_root):
        if cdir != raw_data_root:
            cdir_basename = basename(cdir)
            target_dir = os.path.join(small_root, cdir_basename)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                print(target_dir)
   
    
def transfer_images_to_small():
    for cdir, subdirs, files in os.walk(raw_data_root):
        cdir_basename = basename(cdir)
        print(cdir_basename)
        for filename in files:
            if filename.endswith('.png'):
                processed = image_transform(os.path.join(cdir, filename), (40, 40))
                cv.imwrite(os.path.join(small_root, cdir_basename, filename), img=processed)
  
            
def image_transform(path_to_img, dim):
    image = cv.imread(path_to_img, cv.IMREAD_UNCHANGED)
    active_channel = cv.split(image)[3]
    active_channel = cv.resize(active_channel, dim)
    return active_channel

    
def basename(directory):
    return os.path.basename(os.path.normpath(directory))

    
if __name__ == '__main__':
    transfer_images_to_small()
    pass
