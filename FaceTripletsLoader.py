import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


class FaceTriplets:
    """Faces Dataset Loader. Random load face image into set of three images
    (anchor, positive, negative) for triplets training.

    Args:
        data_folder: (str) a folder contains each class folders of face images. If the face is
            masking it file name should have '_mask' suffix.
        image_size: (int) square image size.
    """
    def __init__(self, data_folder, image_size=160, transform=None):
        self.image_size = (image_size, image_size)
        self.transform = transform

        self.data_folder = os.path.normpath(data_folder)
        self.datainfo_file = self.data_folder + '-info.csv'

        if not os.path.exists(self.datainfo_file):
            self.data = self.get_datainfo()
        else:
            self.data = pd.read_csv(self.datainfo_file)

    def __len__(self):
        return len(self.data)

    def get_image(self, index, pre_process=True):
        assert index < len(self.data), 'Out of index range!'
        image = cv2.imread(self.data['path'][index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        if pre_process and self.transform:
            image = self.transform(image)
        return image

    def get_datainfo(self):
        """Create csv list info file of image folder."""
        print('Create data folder infomation file...')
        list_rows = []
        list_fol = sorted(os.listdir(self.data_folder))
        for fol in list_fol:
            folfil = os.path.join(self.data_folder, fol)
            list_fil = sorted(os.listdir(folfil))
            for fil in list_fil:
                if not fil.endswith(('.jpg', '.png', '.jpeg')):
                    continue
                row = {'path': os.path.join(folfil, fil),
                       'masked': True if '_mask' in fil else False,
                       'name': fol}
                list_rows.append(row)

        df = pd.DataFrame(list_rows)
        df['class'] = pd.factorize(df['name'])[0]
        df.to_csv(self.datainfo_file, index=None)
        return df

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
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,
            'pos_cls': torch.tensor([pos_cls], dtype=torch.long),
            'neg_cls': torch.tensor([neg_cls], dtype=torch.long),
        }
        return sample

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


class FaceTripletsSet(FaceTriplets):
    """Generate pre hard-triplets matching list on every few epochs during training.
    *** Note: This still incomplete code because it uses a lot of time to embedding all
            samples and finds hard-triplets matches.
    """
    def __init__(self, data_folder, image_size=160, transform=None):
        super(FaceTripletsSet, self).__init__(data_folder, image_size, transform)
        self.embedded_file = self.data_folder + '-embedded.npy'
        self.embedded_file_tmp = self.data_folder + '-embedded_tmp.npy'
        self.triplets_file = self.data_folder + '-tripletslist.csv'

        if not os.path.exists(self.triplets_file):
            self.triplets_list = self.generate_triplets()
        else:
            self.triplets_list = pd.read_csv(self.triplets_file)

    def embed_all_data(self, embedder_model, batch_size=64):
        st = time.time()
        embedded = []
        bs_len = int(np.ceil(len(self.data) / batch_size))
        with tqdm(range(bs_len), desc='Embedding faces') as pbar:
            for i in range(bs_len):
                bs_ = len(self.data) - (i * batch_size) if i == bs_len - 1 else batch_size
                batch_img = torch.zeros((bs_, 3, self.image_size, self.image_size),
                                        dtype=torch.float32)
                for k, j in enumerate(range(batch_size * i, (batch_size * i) + bs_)):
                    batch_img[k] = self.get_image(j)

                emb = embedder_model(batch_img.cuda()).detach().cpu().numpy()
                embedded.extend(emb)
                pbar.update()

        embedded = np.array(embedded)
        np.save(self.embedded_file_tmp, embedded)

        # Dimension reduction.
        embedded = PCA(n_components=64).fit_transform(embedded)
        np.save(self.embedded_file, embedded)
        print('Save embedded faces to file: ' + self.embedded_file)
        elps = time.time() - st
        print('time used: %.0f m : %.0f s' % (elps // 60, elps % 60))
        return embedded

    def generate_triplets(self, batch_size=256, embedding_batch_size=64):
        if not os.path.exists(self.embedded_file):
            embedded = self.embed_all_data(embedding_batch_size)
        else:
            embedded = np.load(self.embedded_file)
        assert embedded.shape[0] == len(self.data), 'Embeded file and data folder not relative!'

        st = time.time()
        emb_class_mean = pd.DataFrame(
            np.concatenate((self.data['class'].values[:, None], embedded),
                           axis=1)).groupby(0).mean().values

        triplets = np.zeros((len(self.data), 4), dtype=np.int)
        bs_len = int(np.ceil(len(self.data) / batch_size))
        with tqdm(range(bs_len), desc='Generate triplets faces') as pbar:
            for i in range(bs_len):
                bs_ = len(self.data) - (i * batch_size) if i == bs_len - 1 else batch_size
                anc_class = self.data['class'][i * batch_size: (i * batch_size) + bs_].values
                anc_embed = embedded[i * batch_size: (i * batch_size) + bs_]

                anc_class_cm = cosine_similarity(anc_embed, emb_class_mean)

                """pos_idx = np.ma.array(
                    anc_cm, mask=np.repeat(self.data['class'][None, :], bs_, 0) != anc_class[:, None]).argmin(1)
                neg_idx = np.ma.array(
                    anc_cm, mask=np.repeat(self.data['class'][None, :], bs_, 0) == anc_class[:, None]).argmin(1)

                triplets[i * bs: (i * bs) + bs_] = np.array([np.arange(i * bs, (i * bs) + bs_),
                                                             pos_idx, neg_idx]).transpose()"""

                """for j in range(bs_):
                    if np.random.rand() > 0.5:
                        pos_idx = np.ma.array(anc_cm[j], mask=self.data['class'] != anc_class[j]).argmin()
                        neg_idx = np.ma.array(anc_cm[j], mask=self.data['class'] == anc_class[j]).argmax()
                    else:
                        pos_idx = np.random.choice(np.where(self.data['class'] == anc_class[j])[0])
                        neg_idx = np.random.choice(np.where(self.data['class'] != anc_class[j])[0])

                    triplets[(i * batch_size) + j] = (i * batch_size) + j, pos_idx, neg_idx"""

                neg_cls = np.ma.array(anc_class_cm, mask=np.array(
                    [np.arange(0, anc_class_cm.shape[1])] * bs_) == anc_class[:, None]).argmax(1)
                for j in range(bs_):
                    if np.random.rand() > 0.5:
                        p_i = np.where(self.data['class'] == anc_class[j])[0]
                        pos_idx = p_i[cosine_similarity(anc_embed[j][None, :], embedded[p_i])[0].argmin()]
                        n_i = np.where(self.data['class'] == neg_cls[j])[0]
                        neg_idx = n_i[cosine_similarity(anc_embed[j][None, :], embedded[n_i])[0].argmax()]
                        hard_pair = 1
                    else:
                        pos_idx = np.random.choice(np.where(self.data['class'] == anc_class[j])[0])
                        neg_idx = np.random.choice(np.where(self.data['class'] != anc_class[j])[0])
                        hard_pair = 0

                    triplets[(i * batch_size) + j] = (i * batch_size) + j, pos_idx, neg_idx, hard_pair

                pbar.update()

        triplets = pd.DataFrame(triplets, columns=['anc', 'pos', 'neg', 'hard'])
        triplets.to_csv(self.triplets_file, index=None)
        print('Save triplets pairs list to file: ' + self.triplets_file)
        elps = time.time() - st
        print('time used: %.0f m : %.0f s' % (elps // 60, elps % 60))
        return triplets

    def __getitem__(self, idx):
        a_i, p_i, n_i, ishard = self.triplets_list.loc[idx]
        anc_cls, pos_cls, neg_cls = self.data['class'][[a_i, p_i, n_i]].values
        assert anc_cls == pos_cls and anc_cls != neg_cls, 'Invalid triplets pair!'

        anc_img = self.get_image(a_i)
        pos_img = self.get_image(p_i)
        neg_img = self.get_image(n_i)

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,
            'pos_cls': torch.tensor([pos_cls], dtype=torch.long),
            'neg_cls': torch.tensor([neg_cls], dtype=torch.long),
            'ishard': torch.tensor([ishard], dtype=torch.long)
        }
        return sample

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

        fig.suptitle('Hard triplet' if sample['ishard'] else 'Random triplet')
        plt.show()


if __name__ == '__main__':
    import torchvision.transforms as transforms
    folder = '../Data/CASIA-masked'

    transforms_fn = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
                                        ])
    #transforms_fn = transforms.Compose([fixed_image_standardization])
    dtl = FaceTripletsSet(folder, transform=transforms_fn)
