import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from Utils import resize_padding


class FaceTriplets:
    """Faces Dataset Loader. Random load face image into set of three images
    (anchor, positive, negative) for triplets training.

    Args:
        data_folder: (str) a folder contains each class folders of face images. If the face is
            masking it file name should have '_mask' suffix.
        image_size: (int) square image size.
    """
    def __init__(self, data_folders, image_size=160, augment=None, transform=None):
        self.image_size = (image_size, image_size)
        self.augment = augment if augment is not None else lambda x, y: x
        self.transform = transform if transform is not None else lambda x: x

        self.data_folders = [data_folders] if not isinstance(data_folders, list) else data_folders
        self.data_list_file = 'data_list.csv'

        if not os.path.exists(self.data_list_file):
            self.data = self.get_data_list()
        else:
            self.data = pd.read_csv(self.data_list_file)

    def get_data_list(self):
        """Create csv list info file of image folders."""
        print('Create data list files...')
        list_rows = []
        for dataset in self.data_folders:
            set_name = os.path.split(dataset)[-1]
            list_fol = sorted(os.listdir(dataset))
            for fol in list_fol:
                folfil = os.path.join(dataset, fol)
                list_fil = sorted(os.listdir(folfil))
                for fil in list_fil:
                    if not fil.endswith(('.jpg', '.png', '.jpeg')):
                        continue
                    row = {'path': os.path.join(folfil, fil),
                           'masked': True if '_mask' in fil else False,
                           'name': set_name + '-' + fol}
                    list_rows.append(row)

        df = pd.DataFrame(list_rows)
        df['class'] = pd.factorize(df['name'])[0]
        df.to_csv(self.data_list_file, index=None)
        return df

    def get_image(self, index):
        row = self.data.iloc[index]
        image = cv2.imread(row['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image, is_mask = self.augment(image, not row['masked'])

        if image.shape[:2] != self.image_size:
            image, _, _ = resize_padding(image, self.image_size[0], self.image_size[1], cv2.BORDER_REPLICATE)
            #image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = self.transform(image)
        return image, is_mask

    def __getitem__(self, idx):
        anc_cls = self.data['class'][idx]

        pos_idx = np.random.choice(np.where(self.data['class'] == anc_cls)[0])
        pos_cls = self.data['class'][pos_idx]
        neg_idx = np.random.choice(np.where(self.data['class'] != anc_cls)[0])
        neg_cls = self.data['class'][neg_idx]

        anc_img = self.get_image(idx)
        pos_img = self.get_image(pos_idx)
        neg_img = self.get_image(neg_idx)

        sample = {
            'anc_img': anc_img[0],
            'pos_img': pos_img[0],
            'neg_img': neg_img[0],
            'pos_cls': torch.tensor([pos_cls], dtype=torch.long),
            'neg_cls': torch.tensor([neg_cls], dtype=torch.long),
            'is_mask': (anc_img[1] | pos_img[1] | neg_img[1])
        }
        return sample

    def __len__(self):
        return len(self.data)

    def show_triplets(self, idx):
        sample = self.__getitem__(idx)
        fig = plt.figure(figsize=(10, 6))
        for i, img in enumerate((sample['anc_img'], sample['pos_img'], sample['neg_img'])):
            ax = fig.add_subplot(1, 3, i + 1)
            if self.transform:
                img = img.permute(1, 2, 0)
            ax.imshow(img)

            cls = sample['pos_cls'] if i < 2 else sample['neg_cls']
            ax.set_xlabel(str(cls[0]))
        plt.show()


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils import data
    from FaceMasking import AugmentMasking
    from augmentation import face_augment_pipe

    folders = ['../Data/CASIA-WebFace', '../Data/RMFD/AFDB_face_dataset']

    transforms_fn = transforms.Compose([transforms.ToTensor(),
                                        #transforms.Normalize(
                                        #    mean=[0.5, 0.5, 0.5],
                                        #    std=[0.5, 0.5, 0.5])
                                        ])
    #transforms_fn = transforms.Compose([fixed_image_standardization])
    #dtl = FaceTriplets(folder, transform=None)

    dataset = FaceTriplets(folders, augment=AugmentMasking(post_augment=face_augment_pipe()),
                           transform=transforms_fn)

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=2,
        shuffle=True
    )

