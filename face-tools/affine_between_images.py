import cv2
import imageio
import dlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp
from skimage.transform import AffineTransform

res = 224

def shape_to_np(shape, dtype):
    coord = np.zeros(shape=(68, 2), dtype=dtype)
    for i in range(0, 68):
        coord[i] = (shape.part(i). x, shape.part(i).y)
    return coord

def extract_landmarks(image):
    p = "common-data/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    print('training model loaded...')
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(image_cv,0)
    rect = rects[0]
    shape = predictor(image_cv,rect)
    shape= shape_to_np(shape,"int")
    lmk_pos = []
    for p in shape:
        lmk_pos.append([p[0], p[1]])
    return np.array(lmk_pos)

def get_affine(image_render, image_real):
    lmk_pos_real = extract_landmarks(image_real)
    lmk_pos_render = extract_landmarks(image_render)
    lmk_pos_render_homo = np.hstack((lmk_pos_render, np.ones([lmk_pos_render.shape[0], 1])))
    P = np.linalg.lstsq(lmk_pos_render_homo, lmk_pos_real, rcond=1)[0].T
    P_homo = np.vstack((P, np.array([0,0,1])))
    affine = AffineTransform(matrix=P_homo)
    return affine

if __name__ == '__main__':

    image_real = imageio.imread("00025/00025_00058.png")
    image_render = imageio.imread("00025/00025_00058_render.png")

    affine = get_affine(image_render, image_real)
    warped = warp(image_render, affine.inverse)
    warped = (255*warped).astype(np.uint8)

    plt.imshow((warped[:,:,:3] * 0.5 + image_real[:,:,:3] * 0.5).astype(np.uint8))
    plt.show()