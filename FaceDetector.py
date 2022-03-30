import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from Models.mtcnn import MTCNN, extract_face


class HOGFaceDetector:
    def __init__(self, out_size=160):
        self.out_size = out_size
        self.detector = dlib.get_frontal_face_detector()
        self.shaper = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

    def get_faces(self, image, scale=1, get_largest=True, return_shape=False):
        res = None
        shp = None
        faces = self.detector(image, scale)
        if len(faces) > 0:
            if get_largest:
                areas = [rec.width() * rec.height() for rec in faces]
                idx = np.argmax(areas)
                faces = [faces[idx]]

            res = np.zeros((len(faces), self.out_size, self.out_size, 3), dtype=np.uint8)
            shp = []
            for i, face in enumerate(faces):
                shaped = self.shaper(image, face)
                shp.append(shaped)
                aligned = dlib.get_face_chip(image, shaped, size=self.out_size)
                res[i] = aligned

        if return_shape:
            return res, shp
        return res

    def show_detected(self, image, scale=1):
        face, shp, rect = self.get_faces(image, scale, return_shape=True)
        res = image.copy()
        for i, rec in enumerate(rect):
            res = cv2.rectangle(res, (rec.left(), rec.top()), (rec.right(), rec.bottom()),
                                (0, 255, 0), 2)
            for pt in shp[i]:
                res = cv2.circle(res, (pt.x, pt.y), 2, (0, 255, 0), -1)

        plt.figure()
        plt.imshow(res)
        plt.show()


class MTCNNFaceDetector:
    def __init__(self, out_size=160):
        self.out_size = out_size
        self.detector = MTCNN(image_size=out_size, keep_all=True).eval()
        self.shaper = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

    def get_faces(self, image, get_largest=True, return_shape=False):
        res = None
        shp = None
        faces, _, pts = self.detector.detect(Image.fromarray(image), landmarks=True)
        if faces is not None and len(faces) > 0:
            if get_largest:
                faces = [faces[0]]

            res = np.zeros((len(faces), self.out_size, self.out_size, 3), dtype=np.uint8)
            shp = []
            for i, face in enumerate(faces):
                # Adjust to square.
                w = face[2] - face[0]
                h = face[3] - face[1]
                ex = np.abs(h - w) / 2.
                if h > w:
                    face[[0, 2]] = face[0] - ex, face[2] + ex
                else:
                    face[[1, 3]] = face[1] - ex, face[3] + ex

                # Move rec to center of the face.
                cent_face = np.array((sum(pts[i, :, 0]) / 5, sum(pts[i, :, 1]) / 5))
                cent_rect = np.array(((face[0] + face[2]) / 2, (face[1] + face[3]) / 2))
                d = cent_face - cent_rect
                face[:] = face[0] + d[0], face[1] + d[1], face[2] + d[0], face[3] + d[1]

                rec = dlib.rectangle(*face)
                shaped = self.shaper(image, rec)
                shp.append(shaped)
                aligned = dlib.get_face_chip(image, shaped, size=self.out_size)
                res[i] = aligned

        if return_shape:
            return res, shp, pts
        return res

    def get_faces2(self, image, get_largest=True, return_shape=False):
        res = None
        shp = None
        _, _, pts = self.detector.detect(Image.fromarray(image), landmarks=True)
        if pts is not None and len(pts) > 0:
            if get_largest:
                pts = [pts[0]]

            res = np.zeros((len(pts), self.out_size, self.out_size, 3), dtype=np.uint8)
            shp = []
            for i, pt in enumerate(pts):
                face = np.array((min(pt[:, 0]), min(pt[:, 1]), max(pt[:, 0]), max(pt[:, 1])))
                w, h = face[2] - face[0], face[3] - face[1]
                ex = np.abs(h - w) / 2.
                if h > w:
                    face[[0, 2]] = face[0] - ex, face[2] + ex
                else:
                    face[[1, 3]] = face[1] - ex, face[3] + ex

                ex = max(h, w) * 0.5
                face[:] = face[0] - ex, face[1] - ex, face[2] + ex, face[3] + ex

                rec = dlib.rectangle(*face)
                shaped = self.shaper(image, rec)
                shp.append(shaped)
                aligned = dlib.get_face_chip(image, shaped, size=self.out_size)
                res[i] = aligned

        if return_shape:
            return res, shp, pts
        return res

    def show_detected(self, image):
        face, shp, pts = self.get_faces2(image, return_shape=True)
        res = image.copy()
        for i, sh in enumerate(shp):
            rec = sh.rect
            res = cv2.rectangle(res, (rec.left(), rec.top()), (rec.right(), rec.bottom()),
                                (0, 0, 255), 2)
            for pt in sh.parts():
                res = cv2.circle(res, (pt.x, pt.y), 2, (0, 0, 255), -1)

            for pt in pts[i]:
                res = cv2.circle(res, (pt[0], pt[1]), 3, (255, 0, 0), 2)

        plt.figure()
        plt.imshow(res)
        plt.show()


if __name__ == '__main__':
    data_folder = '../Data/CASIA-WebFace/'
    image = cv2.imread(data_folder + '0203221/043.jpg')[:, :, ::-1]

    hogdt = HOGFaceDetector()
    mtcnn = MTCNNFaceDetector()

    hogdt.show_detected(image)
    mtcnn.show_detected(image)
