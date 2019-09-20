import sys
import soft_renderer as sr
import imageio
from skimage.transform import warp
from skimage.transform import AffineTransform
import numpy as np
import cv2
import dlib

import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

res = 224

def recover(rt):
    rots = []
    trans = []
    for tt in range(rt.shape[0]):
        ret = rt[tt,:3]
        r = R.from_rotvec(ret)
        ret_R = r.as_dcm()
        ret_t = rt[tt, 3:]
        ret_t = ret_t.reshape(3,1)
        rots.append(ret_R)
        trans.append(ret_t)
    return (np.array(rots), np.array(trans))

def load_obj(obj_file):
    vertices = []
    triangles = []
    colors = []

    with open(obj_file) as infile:
        for line in infile.read().splitlines():
            if len(line) > 2 and line[:2] == "v ":
                ts = line.split()
                x = float(ts[1])
                y = float(ts[2])
                z = float(ts[3])
                r = float(ts[4])
                g = float(ts[5])
                b = float(ts[6])
                vertices.append([x,y,z])
                colors.append([r,g,b])
            elif len(line) > 2 and line[:2] == "f ":
                ts = line.split()
                fx = int(ts[1]) - 1
                fy = int(ts[2]) - 1
                fz = int(ts[3]) - 1
                triangles.append([fx,fy,fz])
    
    return (np.array(vertices), np.array(triangles).astype(np.int), np.array(colors))

def frontalize(vertices):
    canonical_vertices = np.load('common-data/canonical_vertices.npy')
    vertices_homo = np.hstack((vertices, np.ones([vertices.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(vertices_homo, canonical_vertices, rcond=1)[0].T # Affine matrix. 3 x 4
    front_vertices = vertices_homo.dot(P.T)
    return front_vertices, P

def display_image(image):
    image = image.detach().cpu().numpy()
    image = image.transpose((1,2,0))
    plt.imshow(image)
    plt.show()

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

def setup_renderer(vertices):
    center = vertices.mean(axis=0)
    eye_pos = center + np.array([0.0, 0.0, 600.0])
    cam_up = [0.0,1.0,0.0]
    cam_dir = [0.0,0.0,-1.0]
    renderer = sr.SoftRenderer(camera_mode="look", far=10000, camera_direction=cam_dir, camera_up=cam_up, eye=eye_pos.astype(np.float32), image_size=res, viewing_angle=15, light_intensity_ambient=1)
    return renderer

def get_np_uint8_image(mesh, renderer):
    images = renderer.render_mesh(mesh)
    image = images[0]
    image = torch.flip(image, [2])
    image = image.detach().cpu().numpy().transpose((1,2,0))
    image = (255*image).astype(np.uint8)
    return image

if __name__ == "__main__":
    key_id = 58 #
    model_id = "00025"
    video_len = 8000 # milliseconds

    vertices_org, triangles, colors = load_obj("{}/{}_original.obj".format(model_id, model_id)) # get unfrontalized vertices position
    
    # set up the renderer
    renderer = setup_renderer(vertices_org)

    # generate rendered key image 
    face_mesh = sr.Mesh(vertices_org, triangles, colors, texture_type="vertex")
    key_image_render = get_np_uint8_image(face_mesh, renderer)

    # calculate the affine matrix between rendered key image and real key image
    key_image_real = imageio.imread("{}/{}_{:05}.png".format(model_id, model_id, key_id))
    affine = get_affine(key_image_render, key_image_real)

    # load RTs
    rots, trans = recover(np.load("{}/{}_sRT.npy".format(model_id, model_id)))

    # calculate frontalized vertices and the associated affine matrix
    # decompose the affine matrix to rotation and translation for latering usage
    vertices, p_affine = frontalize(vertices_org)
    pr = p_affine[:,:3] # 3x3
    pt = p_affine[:,3:] # 3x1
    pr_inv = np.linalg.inv(pr)

    # generate animation
    fig = plt.figure()
    ims = []
    # writer = imageio.get_writer('rotation.gif', mode='I')
    for i in range(rots.shape[0]):

        # dark magic transformation
        # update the unfrontalized vertices based on RTs in canonical coordinates
        new_vertices = (pr_inv @ (rots[key_id] @ np.linalg.inv(rots[i]) @ (vertices.T - trans[i]) + trans[key_id] - pt)).T 

        # do the rendering
        face_mesh = sr.Mesh(new_vertices, triangles, colors, texture_type="vertex")
        image = get_np_uint8_image(face_mesh, renderer)

        # apply the image affine transformation
        warped_image = warp(image, affine.inverse)

        # push into video frames
        im = plt.imshow(warped_image, animated=True)
        ims.append([im])
        # writer.append_data((255*warped_image).astype(np.uint8))

        print("[{}/{}]".format(i+1, rots.shape[0]))    
    # writer.close()

    itvl = video_len / (rots.shape[0]-1)
    ani = animation.ArtistAnimation(fig, ims, interval=itvl, blit=True, repeat_delay=1000)
    ani.save('{}/{}_render_ani.mp4'.format(model_id, model_id))
    plt.show()