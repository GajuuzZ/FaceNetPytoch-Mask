import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


class LFWDataset:
    """LFW faces dataset loader. Require a pairs list of faces file that pair same or difference person.
    Note: The file name and folder structure should be in correct format.
        {main_folder}/{class_name}/{class_name}_{%04d}_{w/o|suffix}.ext
    """
    def __init__(self, lfw_folder, pairs_file, image_size=160, transform=None,
                 mask=False, mask_suffix='_mask-'):
        self.data_folder = lfw_folder
        self.pairs_file = pairs_file
        self.image_size = (image_size, image_size)
        self.mask = mask
        self.suffix = mask_suffix

        self.transform = transform
        self.path_list = self.get_list()

    def __len__(self):
        return len(self.path_list)

    def add_extension(self, path, mask=False):
        for i in range(1, 3):
            path_ = path
            if mask:
                path_ = path_ + self.suffix + str(i)

            if os.path.exists(path_ + '.jpg'):
                return path_ + '.jpg'
            elif os.path.exists(path_ + '.png'):
                return path_ + '.png'

        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    def get_list(self):
        pairs = []
        with open(self.pairs_file, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        #pairs = np.array(pairs)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            if len(pair) == 3:
                path0 = self.add_extension(os.path.join(
                    self.data_folder, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = self.add_extension(os.path.join(
                    self.data_folder, pair[0], pair[0] + '_' + '%04d' % int(pair[2])), self.mask)
                issame = True
            elif len(pair) == 4:
                path0 = self.add_extension(os.path.join(
                    self.data_folder, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = self.add_extension(os.path.join(
                    self.data_folder, pair[2], pair[2] + '_' + '%04d' % int(pair[3])), self.mask)
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    def get_image(self, path, pre_process=True):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        if pre_process and self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        assert idx < len(self), 'Out of index range!'

        path1, path2, issame = self.path_list[idx]
        image1, image2 = self.get_image(path1), self.get_image(path2)
        return image1, image2, issame

    def show_pair(self, idx):
        image1, image2, issame = self.__getitem__(idx)
        fig = plt.figure(figsize=(10, 5))
        for i, img in enumerate((image1, image2)):
            ax = fig.add_subplot(1, 2, i + 1)
            if self.transform:
                img = img.permute(1, 2, 0)
            ax.imshow(img)

        fig.suptitle('Same' if issame else 'Not Same')
        plt.show()


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils import data

    LFW_FOLDER = '../Data/LFW/lfw-masked'
    PAIRS_FILE = '../Data/LFW/LFW_pairs.txt'

    transforms_fn = transforms.Compose([transforms.ToTensor()])

    dataset = LFWDataset(LFW_FOLDER, PAIRS_FILE, mask=False, transform=transforms_fn)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=2,
        shuffle=True
    )


