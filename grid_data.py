#!/usr/bin/env python
# coding: utf-8

# In[29]:


import argparse
import fnmatch
import os
import shutil
import subprocess
import h5py
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import pickle
import cv2
import face_alignment
import librosa
from util import utils
from tqdm import tqdm
import dlib
import matplotlib.animation as manimation

from torch import multiprocessing

import time  

def parse_args():
    parser = argparse.ArgumentParser()

   
    parser.add_argument('-i','--in_file', type=str, default='56')
      
    return parser.parse_args()
config = parse_args()



fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)#,  device='cpu')


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./basics/shape_predictor_68_face_landmarks.dat')


def unzip_video(path):
    
    tar_files = os.listdir(path)
    valid = []
    for f in tar_files:
        if 'mpg' in f:
            valid.append(f)
    for f in valid:
        f_path = os.path.join(path, f)
        print (f_path)
        command = ' tar -C ' +'/data/lchen63/grid/zip/video/ ' + ' -xvf ' + f_path
        print (command)
        os.system(command)


def unzip_audio(path):
    tar_files = os.listdir(path)
    valid = []
    for f in tar_files:
        if '50kHz' in f:
            valid.append(f)
    for f in valid:
        f_path = os.path.join(path, f)
        print (f_path)
        command = ' tar -C ' +'/data/lchen63/grid/zip/audio/ ' + ' -xvf ' + f_path
        print (command)
        os.system(command)
        

    
def get3DLmarks(frame_list, v_path):
    frame_num = len(frame_list)
    lmarks = np.zeros((frame_num, 68,3))
    for i in range(frame_num):
        lmark = fa.get_landmarks(frame_list[i])        
        if lmark is not None:
            landmark =  lmark[0]
        else:
            landmark = -np.ones((68, 3))
        lmarks[i] = landmark
    np.save(v_path[:-4] + '.npy', lmarks)




def get_v_txt(folder):
    root = '/data/lchen63/grid/zip'

    file_list = []
    txt_f = open( os.path.join(root,'txt', 'v_dev.txt'), 'wb') 
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith(('.mpg', '.mov', '.mp4')):
                filepath = os.path.join(root, filename)
#                 print (filepath, filename)
                file_list.append(filepath)
    for line in file_list:
        txt_f.writelines(line)
        txt_f.write('\n')
    txt_f.close()


def get_a_txt(folder):
    root = '/data/lchen63/grid/zip'

    file_list = []
    txt_f = open( os.path.join(root,'txt', 'a_dev.txt'), 'wb') 
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith(('.wav')):
                filepath = os.path.join(root, filename)
#                 print (filepath, filename)
                file_list.append(filepath)
    for line in file_list:
        txt_f.writelines(line)
        txt_f.write('\n')
    txt_f.close()
    

def crop_image(image, id = 0, kxy = []): # if id ==0, image is a cv2 format,else: image is a image path
    if id == 1:
        image = cv2.imread(image)
    if kxy != []:
        [k, x, y] = kxy
        roi = image[y - int(0.2 * k):y + int(1.6 * k), x- int(0.4 * k):x + int(1.4 * k)]
        roi = cv2.resize(roi, (224,224), interpolation = cv2.INTER_AREA)
        return roi, kxy 
    else:        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = utils.shape_to_np(shape)
            (x, y, w, h) = utils.rect_to_bb(rect)
            center_x = x + int(0.5 * w)
            center_y = y + int(0.5 * h)
            k = min(w, h)
            roi = image[y - int(0.2 * k):y + int(1.6 * k), x- int(0.4 * k):x + int(1.4 * k)]
            roi = cv2.resize(roi, (224,224), interpolation = cv2.INTER_AREA)
            return roi ,[k,x,y]
    
    
def _video2img2lmark(v_path):
    root = '/data/lchen63/grid/zip/'

    count = 0
    kxy = []
    tmp = v_path.split('/')
    frame_list = []
    if not os.path.exists(os.path.join(root , 'img') ):
        os.mkdir(os.path.join(root , 'img'))
    if not os.path.exists(os.path.join(root , 'img', tmp[-4]) ):
        os.mkdir(os.path.join(root , 'img', tmp[-4]))
    if not os.path.exists(os.path.join(root , 'img', tmp[-4], tmp[-1][:-4]) ):
        os.mkdir(os.path.join(root , 'img', tmp[-4], tmp[-1][:-4]))
   
    cap  =  cv2.VideoCapture(v_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            try:
                count += 1
                img,kxy = crop_image(frame, 0,kxy)
                frame_name = os.path.join(root , 'img', tmp[-4], tmp[-1][:-4] ,'%05d.png'%count)
                cv2.imwrite(frame_name,img)
                frame_list.append(img)
            except:
                break
        else:
            break
    if len(frame_list) == 75:
        get3DLmarks(frame_list,v_path)

# _video2img2lmark('/data/lchen63/grid/zip/video/s20/video/mpg_6000/pbaa1n.mpg')    
    
def video2img2lmark(list):
    length = len(list)
    cmt = 0
    for p in list:
        current = time.time()
        print ('{}/{}'.format(cmt, length))
        if os.path.exists(p[:-5] + '.npy'):
            if np.load(p[:-5] + '.npy').size != 0:
                cmt += 1
                continue
        
        _video2img2lmark(p[:-1])
        print (time.time() - current)
        cmt += 1
        
def video_transfer(txt):
    txt_f = open(txt, 'rb')
    list = txt_f.readlines()
    print (len(list))
    batch_size = int(len(list))
    i = int(config.in_file)
    video2img2lmark(list[i*batch_size:i*batch_size + batch_size])
       
        
        
        
# get_txt('/data2/lchen63/voxceleb/unzip/dev_video')



######### visualization code
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import cv2
import os

def vis(root):
    v_path = '/data/lchen63/grid/zip/video/s20/video/mpg_6000/pbaa1n.mp4'
    lmark_path = '/data/lchen63/grid/zip/video/s20/video/mpg_6000/pbaa1n.npy'
    cap  =  cv2.VideoCapture(v_path)
    lmark = np.load(lmark_path)
    count = 0
    tmp = v_path.split('/')
    for count in range(1,100):
        frame_name = os.path.join(root , 'img', tmp[-4], tmp[-1][:-4] ,'%05d.png'%count)
        input = io.imread(frame_name)
        preds = lmark[count]
        #TODO: Make this nice
        fig = plt.figure(figsize=plt.figaspect(.5))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(input)
        ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
        ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1) 
        ax.axis('off')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
        ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
        ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
        ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
        ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
        ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
        ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
        ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
        ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )

        ax.view_init(elev=90., azim=90.)
        ax.set_xlim(ax.get_xlim()[::-1])
        plt.show()
# vis('/data/lchen63/grid/zip')        
        
################ get training data#################################
import os
import numpy as np
# print ('++++')
def get_new_txt(txt):
    root = '/data2/lchen63/voxceleb/'

    file_list = []
    txt_w = open( os.path.join(root,'txt', 'fv_dev.txt'), 'wb') 
    
    
    txt_f = open(txt, 'rb')
    list = txt_f.readlines()
    length = len(list)
    cmt = 0
    finished = []
    for p in list:
        print (cmt)
        if os.path.exists(p[:-5] + '.npy'):
            if np.load(p[:-5] + '.npy').size != 0:
                cmt += 1
#                 print (np.load(p[:-5] + '.npy').shape )
                finished.append(p)
#     print (finished)    
    print (len(finished))
    for line in finished:
        txt_w.writelines(line)
    txt_f.close()
    
# get_new_txt('/data2/lchen63/voxceleb/txt/v_dev.txt')

import os
import numpy as np
import pickle

print ('++++')

def get_train_pair(txt):
    root = '/data/lchen63/grid/zip/'

    file_list = []
    
    txt_f = open(txt, 'rb')
    list = txt_f.readlines()
    length = len(list)
    cmt = 0
    finished = []
    test = []
    for p in list:
        try:
            cmt += 1
            print ("{}/{}".format(cmt, length))
            v_path = p[:-5]            
            lmark_length = np.load( v_path + '.npy').shape[0] 
            if lmark_length != 75:
                continue
                
            id = v_path.split('/')[-4]
            
            if id == 's29':
                test.append(v_path)
            else:
                finished.append(v_path)
        except:
            continue
    print (len(finished))
    print (len(test))
    print (finished[:3])
    with open(os.path.join('/data/lchen63/grid/zip/txt','dev.pkl'), 'wb') as handle:
        pickle.dump(finished, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join('/data/lchen63/grid/zip/txt','test.pkl'), 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

###################audi2mfcc
import librosa
import python_speech_features
print ('===========')
def audio2mfcc(txt):
    count = 0
    txt_f = open(txt, 'rb')
    list = txt_f.readlines()
    length = len(list)            
    for line in list:
        mfcc1_name = line[:-5]  + '_16k.npy'
        if not os.path.exists(mfcc1_name):
            try:
                print ('{}/{}'.format(count, length))
                audio_path = line[:-1]
                audio,fs = librosa.core.load(audio_path,sr = 16000 )
                mfcc = python_speech_features.mfcc(audio,16000,winstep=0.01)

                np.save(mfcc1_name, mfcc)
                count += 1
                
                audio,fs = librosa.core.load(audio_path,sr = 100 )
                mfcc = python_speech_features.mfcc(audio,100,winstep=0.01)
                mfcc2_name = line[:-5]  + '.npy'
                np.save(mfcc2_name, mfcc)
            except:
                print ('====')
                continue
        else:
            print ('====')

        
from random import shuffle
import torch
from numpy import *
from scipy.spatial.transform import Rotation as R
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

def compute_RT():
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    print (len(consider_key))
    k = 20
    _file = open( "/data/lchen63/grid/zip/txt/test.pkl", "rb")
    data = pickle.load(_file)
    landmarks = []
    RT_list = []
    source = np.zeros((len(consider_key),3))
    ff = np.load('/data2/lchen63/voxceleb/unzip/dev_video/id02343/08TabUdunsU/00001.npy')[30]
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]
        
    source = mat(source)
    for index in range(len(data)):
        print ('{}/{}'.format(index,len(data) ))
        print ( data[index])
#         lmark = np.load( data[index] + '.npy' )
        srt_path = data[index] +  '_sRT.npy'
        front_path = data[index] +  '_front.npy'
        if os.path.exists(srt_path) and os.path.exists(front_path):                
            continue
                
#             lmark_path = '/data2/lchen63/voxceleb/unzip/dev_video/id00967/4glDEPWbvKk/00030.npy'

        t_lmark = np.load( data[index] + '.npy' )
        if t_lmark.shape[0] < 64:
            continue
        lmark_part = np.zeros((t_lmark.shape[0],len(consider_key),3))
        RTs =  np.zeros((t_lmark.shape[0],6))
            
        nomalized =  np.zeros((t_lmark.shape[0],68,3))
        t = time.time()
        for j in range(lmark_part.shape[0]  ):

            for m in range(len(consider_key)):
                lmark_part[:,m] = t_lmark[:,consider_key[m]] 

            target = mat(lmark_part[j])
            ret_R, ret_t = rigid_transform_3D( target, source)

            source_lmark  = mat(t_lmark[j])

            A2 = ret_R*source_lmark.T
            A2+= tile(ret_t, (1, 68))
            A2 = A2.T
            nomalized[j] = A2
            r = R.from_dcm(ret_R)
            vec = r.as_rotvec()             
            RTs[j,:3] = vec
            RTs[j,3:] =  np.squeeze(np.asarray(ret_t))            
        np.save(srt_path, RTs)
        np.save(front_path, nomalized)
        print (time.time() - t )


def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)
    
    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA) * BB

    U, S, Vt = linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2,:] *= -1
        R = Vt.T * U.T
    
#     print(centroid_A)
#     print(centroid_B)
    t = -R*centroid_A.T + centroid_B.T

#     print( t)

    return R, t
def vis(img,lmark1,lmark2):
#     img  = io.imread(img_path)
    preds = lmark1
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1) 
    ax.axis('off')
    
    preds = lmark2
    
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img)
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1) 
    ax.axis('off')
    plt.show()



def smooth_lmark():
    from scipy.signal import savgol_filter
    window_size1 =21
    window_size2 =7
    degree1 = 3
    degree2 = 1
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    _file = open( "/data/lchen63/grid/zip/txt/dev.pkl", "rb")
    train_data = pickle.load(_file)
     
    for index in range(len(train_data)):
        print (index , len(train_data))
        data = np.load( train_data[index] + '_front.npy' )
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                if i < 48:
                    data[:,i,j] = savgol_filter(data[:,i,j], window_size1, degree2)
                else:
                    data[:,i,j] = savgol_filter(data[:,i,j], window_size2, degree1)
        
        real_open = np.zeros((3,75))
        for k in range(3):
            real_open[k] = (data[:,open_pair[k][0], 1] - data[:,open_pair[k][1] , 1])
        real_open = np.abs(real_open)
        real_open = np.sum(real_open, axis = 0)
        for  i in range(data.shape[0]):
            if real_open[i] < 8 :
                for k in range(3):
                    data[i, open_pair[k][0], 1 ] =  data[i, open_pair[k][1], 1 ] = ( data[i, open_pair[k][0], 1]  + data[i, open_pair[k][1], 1 ]) / 2.0
                    
        np.save(train_data[index] + '_front_norm.npy',data)
                
            
    
    

def compute_PCA():
    import random
    _file = open( "/data/lchen63/grid/zip/txt/dev.pkl", "rb")
    train_data = pickle.load(_file)
    lmark_list = []
    random.shuffle(train_data)
    
    for index in range(len(train_data)):
        if index == 4000:
            break
        lmark = np.load( train_data[index] + '_front_norm.npy' )
        lmark_list.append(torch.FloatTensor(lmark))
#     print (lmark_list) 
    lmark_list = torch.stack(lmark_list,dim= 0)
    print (lmark_list.shape)
    
    lmark_roni_list = lmark_list.view(lmark_list.size(0) *lmark_list.size(1) ,  -1)
    mean = torch.mean(lmark_roni_list,0)
    std = torch.std(lmark_roni_list,0)
    np.save('./basics/mean_grid_norm.npy', mean.numpy())
    np.save('./basics/std_grid_norm.npy', std.numpy())
    derivatives = lmark_roni_list - mean.expand_as(lmark_roni_list)
    U,S,V  = torch.svd(torch.t(lmark_roni_list))
    np.save('./basics/U_grid_norm.npy', U.numpy())
    
    lmark_roni_list = lmark_list[:,:, :48,:]
    lmark_roni_list = lmark_roni_list.view(lmark_roni_list.size(0) *lmark_roni_list.size(1) ,  48* 3)
    mean = torch.mean(lmark_roni_list,0)
    std = torch.std(lmark_roni_list,0)
    np.save('./basics/mean_grid_roni_norm.npy', mean.numpy())
    np.save('./basics/std_grid_roni_norm.npy', std.numpy())
    derivatives = lmark_roni_list - mean.expand_as(lmark_roni_list)
    U,S,V  = torch.svd(torch.t(lmark_roni_list))
    np.save('./basics/U_grid_roni_norm.npy', U.numpy())
    
    
    lmark_lip_list = lmark_list[:,:, 48:,:]
    lmark_lip_list = lmark_lip_list.view(lmark_lip_list.size(0) *lmark_lip_list.size(1) ,  20* 3)
    mean = torch.mean(lmark_lip_list,0)
    std = torch.std(lmark_lip_list,0)
    np.save('./basics/mean_grid_lip_norm.npy', mean.numpy())
    np.save('./basics/std_grid_lip_norm.npy', std.numpy())
    derivatives = lmark_lip_list - mean.expand_as(lmark_lip_list)
    U,S,V  = torch.svd(torch.t(lmark_lip_list))
    np.save('./basics/U_grid_lip_norm.npy', U.numpy())


def compose_dataset():
    root  = '/data/lchen63/grid/zip/'
    lstm = False
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    _file = open(os.path.join(root, 'txt' ,  "dev.pkl"), "rb")
    data = pickle.load(_file)
    new_data = []
    for index in range(len(data)):
#         print (index, len(data))
        tmp = data[index].split('/')
        lmark = np.load( data[index] + '_front_norm.npy' )
        v_id = os.path.join(tmp[-4],tmp[-1])
        open_rate = []
        for k in range(3):
            open_rate.append(lmark[:,open_pair[k][0],1] - lmark[:,open_pair[k][1], 1])
        open_rate = np.asarray(open_rate)
        mean_open = np.mean(open_rate, axis = 0)
        mean_open = np.absolute(mean_open)            
        min_index=  np.argmin(mean_open)            
        lmark_length = lmark.shape[0]
        if lstm:
            gg = lmark_length- 34
        else:
            gg = lmark_length-1
        for sample_id in range(0, gg ):
#                 vialisze the results
#                 video_path = os.path.join('/data2/lchen63/voxceleb/unzip/', data[index][0], data[index][2][0] +  '.mp4')
#                 cap = cv2.VideoCapture(video_path)
#                 if sample_id != min_index:
#                     continue
#                 for t in range(lmark_length):
#                     ret, frame = cap.read()
#                     if ret and t == sample_id:
#                         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#                         vis(frame,lmark[sample_id], lmark[sample_id])
#                         break
#                 print (open_rate[:,sample_id])
            tmp = []
            tmp.append(data[index] + '.mpg')
            tmp.append(sample_id)
            tmp.append(min_index)
            tmp.append(v_id)
            new_data.append(tmp)
    print (len(new_data))
    print (new_data[0])
    if lstm:
        nname =  os.path.join('/data/lchen63/grid/zip/txt','train_clean_lstm.pkl')
    else:
        nname = os.path.join('/data/lchen63/grid/zip/txt','train_clean_new.pkl') 
    with open(nname, 'wb') as handle:
        pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)  
# smooth_lmark()
# compose_dataset()
# unzip('/data/lchen63/grid/zip/zip')
# unzip_audio('/data/lchen63/grid/backup_zip')
# get_a_txt('/data/lchen63/grid/zip/audio')
# get_train_pair('/data/lchen63/grid/zip/txt/v_dev.txt')
compute_PCA()  
compose_dataset()
# clean_by_RT()
# video2img2lmark()
# compute_RT()
# audio2mfcc('/data/lchen63/grid/zip/txt/a_dev.txt')
# video_transfer('/data/lchen63/grid/zip/txt/v_dev.txt')
# compose_front()
# get_train_pair('/data2/lchen63/voxceleb/txt/v_test.txt')
