from os import listdir
from os.path import join

import torch.utils.data as data
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, cb, cr = img.split()
    return y, cb, cr


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None, allColors=False):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.allColors = allColors

    def __getitem__(self, index):
        input_image, cb, cr = load_img(self.image_filenames[index])
        target = input_image.copy()
        
        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target = self.target_transform(target)

        if(self.allColors) :
            target_cb = cb.copy()
            target_cr = cr.copy()

            if self.input_transform: 
                cb = self.input_transform(cb)
                cr = self.input_transform(cr)
            if self.target_transform: 
                target_cb = self.target_transform(target_cb)
                target_cr = self.target_transform(target_cr)
            input_images = [input_image, cb, cr]
            targets = [target, target_cb, target_cr]
        else : 
            input_images = [input_image]
            targets = [target]

        return input_images, targets

    def __len__(self):
        return len(self.image_filenames)
