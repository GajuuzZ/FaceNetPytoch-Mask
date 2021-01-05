import os
import cv2
import time
import dlib
import pandas as pd
import numpy as np
from PIL import Image

from FaceMasking import FaceMasker
from FaceDetector import HOGFaceDetector, MTCNNFaceDetector


source_folder = '../Data/LFW/lfw-deepfunneled'
output_folder = '../Data/LFW/lfw-masked'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_size = 160

mtcnn = MTCNNFaceDetector()
masker = [FaceMasker('default'), FaceMasker('n95')]

list_fol = os.listdir(source_folder)
num_fol = len(list_fol)
less_filfol = []
morethanone = []
nondetected = []

st = time.time()
for i, fol in enumerate(list_fol):
    print('folder: {}/{}'.format(i+1, num_fol))
    folfil = os.listdir(os.path.join(source_folder, fol))
    num_fil = len(folfil)
    if len(folfil) <= 0:
        less_filfol.append(fol)
        continue

    save_fol = os.path.join(output_folder, fol)
    if not os.path.exists(save_fol):
        os.makedirs(save_fol)

    for j, fil in enumerate(folfil):
        print('  file: {}/{}'.format(j+1, num_fil))
        fil = os.path.join(source_folder, fol, fil)

        image = cv2.imread(fil)[:, :, ::-1]
        face, shaped, _ = mtcnn.get_faces2(image, return_shape=True)

        if shaped is None or len(shaped) == 0:
            nondetected.append(fil)
            continue
        if len(shaped) > 1:
            morethanone.append(fil)
            continue

        # Save normal-extracted face.
        face = face[0][:, :, ::-1]
        cv2.imwrite(os.path.join(save_fol, os.path.basename(fil)), face)

        # Save masked-extracted face.
        #for m, mkr in enumerate(masker):
        m = np.random.randint(2)
        mkr = masker[m]

        image_mask = mkr.wear_mask_to_face(image, shaped[0].parts())

        face_mask = dlib.get_face_chip(image_mask, shaped[0], size=image_size)[:, :, ::-1]
        cv2.imwrite(os.path.join(save_fol, os.path.basename(fil).split('.')[0] + '_mask-{}.jpg'.format(m+1)),
                    face_mask)

elps = time.time() - st
print('time used: %.0f m : %.0f s' % (elps // 60, elps % 60))

set_name = os.path.dirname(output_folder).split('/')[-1]
nondetected = pd.DataFrame(nondetected)
nondetected.to_csv(os.path.join('../Data', set_name + '-nondetected.csv'), header=None, index=None)
morethanone = pd.DataFrame(morethanone)
morethanone.to_csv(os.path.join('../Data', set_name + '-morethanone.csv'), header=None, index=None)