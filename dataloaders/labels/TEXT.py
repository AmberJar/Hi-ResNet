import numpy as np
from base import BaseDataSet, BaseDataLoader
import os
from utils import palette
import cv2
from glob import glob
from PIL import Image
class HuaweiDataSet(BaseDataSet):
    def __init__(self,**kwargs):
        self.num_classes = 2
        self.palette = palette.AdaSpaceMaps_palette
        super(HuaWeiDataset, self).__init__(**kwargs)

    def _set_files_(self):
        self.images_dir = os.path.join(self.root,self.split,'images')
        self.labels_dir = os.path.join(self.root.self.split,'lables')
        self.files = [os.path.basename(path).split('.')[0] for path in
                      glob(self.images_dir + '/*jpg')]

    def _data_loder(self,index):
        image_id = self.files[index]
        labels_id = image_id
        images_path = os.path.join(self.images_dir,image_id,'jpg')
        labels_path = os.path.join(self.labels_dir,labels_id,'png')
        images = cv2.imread(images_path)
        images = cv2.cvtColor(images,cv2.COLOR_BGR2RGB)
        labels = np.asarray(Image.open(labels_path,dtype=np.int32))

        labels[labels == 5] = 1
        if labels.shape == 3:
            labels = labels[:, :, 0]

        return images, labels, image_id

class HuaweiDataLoader(BaseDataLoader):
    def __init__(self,data_dir, batch_size, split, crop_size=None,
                 base_size=None, scale=True, num_workers=1, val=False,
                 shuffle=False, flip=False, rotate=False, blur=False,
                 augment=False, val_split=None, return_id=False):

        self.MEAN = [0.271931, 0.277937, 0.266352]
        self.STD = [0.149572, 0.143942, 0.139900]
        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }
        self.dataset = HuaWeiDataset(**kwargs)
        super(HuaweiDataLoader, self).__init__(self.dataset, batch_size, num_workers=num_workers, val_split=val_split, shuffle=shuffle)

