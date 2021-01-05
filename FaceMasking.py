import cv2
import numpy as np
from PIL import Image


class FaceMasker:
    def __init__(self, mask_name='default'):
        self.num_pts = 9
        self.tri_mask_idx = [[0, 1, 3], [3, 1, 4], [3, 4, 6], [6, 4, 7], [4, 7, 8],
                             [4, 5, 8], [1, 5, 4], [1, 2, 5]]
        self.load_mask(mask_name)

    def load_mask(self, name):
        if name == 'default':
            self.image_mask = cv2.imread('./mask_images/mask.png', cv2.IMREAD_UNCHANGED)
            self.pt_mask = np.array([(30, 12), (125, 5), (220, 12), (20, 80), (125, 80),
                                     (230, 80), (65, 140), (125, 160), (185, 140)])
            self.tri_face_idx = [[1, 28, 3], [3, 28, 30], [3, 30, 5], [5, 30, 8],
                                 [30, 8, 11], [30, 13, 11], [28, 13, 30], [28, 15, 13]]
        elif name == 'n95':
            self.image_mask = cv2.imread('./mask_images/blue-mask.png', cv2.IMREAD_UNCHANGED)
            self.pt_mask = np.array([(5, 15), (110, 15), (210, 15), (5, 100), (110, 90),
                                     (210, 100), (45, 160), (110, 200), (170, 160)])
            self.tri_face_idx = [[1, 28, 3], [3, 28, 51], [3, 51, 5], [5, 51, 8], [51, 8, 11],
                                 [51, 13, 11], [28, 13, 51], [28, 15, 13]]

        self.image_mask = cv2.cvtColor(self.image_mask, cv2.COLOR_BGRA2RGBA)
        self.get_tri_mask_points()

    def get_tri_mask_points(self):
        self.tri_mask = np.zeros((len(self.tri_mask_idx), 6), dtype=np.float32)
        for i in range(len(self.tri_mask_idx)):
            self.tri_mask[i] = self.pt_mask[self.tri_mask_idx[i]].ravel()

    def get_tri_face_points(self, shape_pts):
        tri_face = np.zeros((len(self.tri_face_idx), 6), dtype=np.float32)
        for i in range(len(self.tri_face_idx)):
            for j in range(3):
                pt = shape_pts[self.tri_face_idx[i][j]]
                tri_face[i, [j+j, j+j+1]] = pt.x, pt.y
        return tri_face

    def wear_mask_to_face(self, image, face_shape):
        tri_face = self.get_tri_face_points(face_shape)

        image_face = Image.fromarray(image)
        for pts1, pts2 in zip(self.tri_mask, tri_face):
            pts1 = pts1.copy().reshape(3, 2)
            pts2 = pts2.copy().reshape(3, 2)

            rect1 = cv2.boundingRect(pts1)
            pts1[:, 0] = pts1[:, 0] - rect1[0]
            pts1[:, 1] = pts1[:, 1] - rect1[1]

            croped_tri_mask = self.image_mask[rect1[1]:rect1[1]+rect1[3], rect1[0]:rect1[0]+rect1[2]]

            rect2 = cv2.boundingRect(pts2)
            pts2[:, 0] = pts2[:, 0] - rect2[0]
            pts2[:, 1] = pts2[:, 1] - rect2[1]

            mask_croped = np.zeros((rect2[3], rect2[2]), np.uint8)
            cv2.fillConvexPoly(mask_croped, pts2.astype(np.int32), 255)

            M = cv2.getAffineTransform(pts1, pts2)
            warped = cv2.warpAffine(croped_tri_mask, M, (rect2[2], rect2[3]))
            warped = cv2.bitwise_and(warped, warped, mask=mask_croped)

            warped = Image.fromarray(warped)
            image_face.paste(warped, (rect2[0], rect2[1]), warped)

        return np.array(image_face)


if __name__ == '__main__':
    import dlib
    import matplotlib.pyplot as plt

    masker = FaceMasker()

    detector = dlib.get_frontal_face_detector()
    shaper = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if ret:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)

            faces = detector(frame, 1)
            if len(faces) > 0:
                shaped = shaper(frame, faces[0])
                frame = masker.wear_mask_to_face(frame, shaped.parts())

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cam.release()
    cv2.destroyAllWindows()
