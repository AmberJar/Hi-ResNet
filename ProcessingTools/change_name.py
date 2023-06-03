import os
import shutil

def file_rename(file_path, save_path):
    print(file_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for filename in os.listdir(file_path):
        print(filename)
        names = filename[4:]
        print(names)
        # new_name = names[0] + names[1].split('_')[1]
        new_name = names
        # print(new_name)
        # img = cv2.imread(os.path.join(filename, file_path))
        # # print(img.shape)
        src = os.path.join(file_path, filename)
        dst = os.path.join(save_path, new_name)
        os.rename(src, dst)

print(file_rename(r'/data/fpc/data/streets_semantic/transferred_imgs_', r'/data/fpc/data/streets_semantic/transferred_imgs'))