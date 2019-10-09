import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import cv2
from api import PRN

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture
import pickle

import soft_renderer as sr
import soft_renderer.cuda.create_texture_image as create_texture_image_cuda
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
import os
root  = '/mnt/Data/lchen63/voxceleb/'
def get_3d(bbb):
   
    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # GPU number, -1 for CPU
    prn = PRN(is_dlib = True)

    # ------------- load data
    # frame_id = "test_video/id00419/3U0abyjM2Po/00024"
    # mesh_file = os.path.join(root, frame_id + ".obj") 
    # rt_file = os.path.join(root, frame_id + "_sRT.npy")
    # image_path 

    # _file = open(os.path.join(root, 'txt',  "front_rt.pkl"), "rb")
    # data = pickle._Unpickler(_file)
    # data.encoding = 'latin1'
    # data = data.load()
    _file = open(os.path.join(root, 'txt',  "train_clean.pkl"), "rb")
    data = pickle.load(_file)
    _file.close()
    gg = len(data)
    print (len(data))
    data = data[int(gg * 0.1 *bbb ); int(gg * 0.1 * (bbb + 1) ) ]
    for kk ,item in enumerate(data) :
        print (kk)
        
        target_id = item[-1]
        video_path = os.path.join(root, 'unzip', item[0] + '.mp4')        
        print (video_path)
        if not os.path.exists(video_path):
            print (video_path) 
            continue
        if os.path.exists(video_path[:-4] + '.obj'):
            continue
        cap = cv2.VideoCapture(video_path)
        for i in range(target_id):
            ret, frame = cap.read()
        ret, target_frame = cap.read()
        cv2.imwrite(video_path[:-4] + '_%05d.png'%target_id,target_frame)
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)



        image = target_frame
        # read image
        [h, w, c] = image.shape
        
        pos = prn.process(image) # use dlib to detect face
        
        image = image/255.
        if pos is None:
            continue
        

        # landmark
        kpt = prn.get_landmarks(pos)
        kpt[:,1] = 224 - kpt[:,1]

        np.save(video_path[:-4] + '_prnet.npy', kpt)
        print (video_path[:-4])
        # 3D vertices
        vertices = prn.get_vertices(pos)
        # save_vertices, p = frontalize(vertices)
        # np.save(video_path[:-4] + '_p.npy', p) 
        # if os.path.exists(video_path[:-4] + '.obj'):
        #     continue
        save_vertices = vertices.copy()
        save_vertices[:,1] = h - 1 - save_vertices[:,1]

        
        # corresponding colors
        colors = prn.get_colors(image, vertices)
        
        # print (colors.shape)
        # print ('=========')
        # cv2.imwrite('./mask.png', colors * 255)
        write_obj_with_colors(video_path[:-4] + '_original.obj', save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

        
        # print (video_path)
        # break
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b",
                        type=int,
                        default=0)
    return parser.parse_args()
config = parse_args()

get_3d(config.b)    


