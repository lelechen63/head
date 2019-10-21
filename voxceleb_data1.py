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
# import face_alignment
import librosa
from util import utils
from tqdm import tqdm
import dlib
import matplotlib.animation as manimation
from zipfile import ZipFile
from torch import multiprocessing
from mpl_toolkits.mplot3d import Axes3D
import os
import mmcv
import time  

def parse_args():
    parser = argparse.ArgumentParser()

   
    parser.add_argument('-i','--in_file', type=str, default='0')

    parser.add_argument('-t','--txt_start', type=int, default=0)
    return parser.parse_args()
config = parse_args()
# root = '/mnt/Data/lchen63/voxceleb'
root = '/data2/lchen63/voxceleb/'
# root ='/home/cxu-serve/p1/lchen63/voxceleb/'
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)#,  device='cpu')


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




def get_txt(folder):
    file_list = []
    txt_f = open( os.path.join(root,'txt', 'v_dev.txt'), 'wb') 
    for r, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith(( '.mp4')) and  '_ani' not in filename :
                filepath = os.path.join(r, filename)
#                 print (filepath, filename)
                file_list.append(filepath)
    # print (file_list[:10])
    for line in file_list:
        txt_f.writelines(line)
        txt_f.write('\n')
    txt_f.close()
    
def _video2img2lmark(v_path):
    # root = '/data2/lchen63/voxceleb/'

    count = 0
    # tmp = v_path.split('/')
    # frame_list = []
    # if not os.path.exists(os.path.join(root , 'img') ):
    #     os.mkdir(os.path.join(root , 'img'))
    # if not os.path.exists(os.path.join(root , 'img', tmp[-3]) ):
    #     os.mkdir(os.path.join(root , 'img', tmp[-3]))
    # if not os.path.exists(os.path.join(root , 'img', tmp[-3], tmp[-2]) ):
    #     os.mkdir(os.path.join(root , 'img', tmp[-3], tmp[-2]))
    # if not os.path.exists(os.path.join(root , 'img', tmp[-3], tmp[-2], tmp[-1][:-4]) ):
    #     os.mkdir(os.path.join(root , 'img', tmp[-3], tmp[-2], tmp[-1][:-4]))
    frame_list  = mmcv.VideoReader(v_path)
#     cap  =  cv2.VideoCapture(v_path)
#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if ret == True:            
# #             frame_name = os.path.join(root , 'img', tmp[-3], tmp[-2], tmp[-1][:-4] ,'%05d.png'%count)
# #             cv2.imwrite(frame_name,frame)
#             frame_list.append(frame)
#             count += 1
#         else:
#             break
    get3DLmarks(frame_list,v_path)

    
    
def video2img2lmark(list):
    length = len(list)
    cmt = 0
    for p in list:
        current = time.time()
        print ('{}/{}'.format(cmt, length))
        p = os.path.join( root  , p[24:])
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
    batch_size = int(len(list)/10)
    i = int(config.in_file)
    video2img2lmark(list[i*batch_size:i*batch_size + batch_size])
       
        
        
        




def vis3d():
    v_path = os.path.join(root, 'unzip', 'test_video/id03862/jc6k4sbenMY/00366' + '.mp4' )
    lmark_path = os.path.join(root, 'unzip', 'test_video/id03862/jc6k4sbenMY/00366_prnet' + '.npy' )
    # lmark_path = os.path.join(root, 'unzip', 'test_video/id04276/k0zLls_oen0/00341_prnet' + '.npy' )
    cap  =  cv2.VideoCapture(v_path)
    lmark = np.load(lmark_path)
    print (lmark.shape)
    print (lmark[0])
    count = 0
    tmp = v_path.split('/')
    real_video  = mmcv.VideoReader(v_path)
    for count in range(0,100):
        input = real_video[count]
        input = mmcv.bgr2rgb(input)
        preds = lmark#[count]
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

################ get training data#################################
import os
import numpy as np
# print ('++++')
def get_new_txt(txt):

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



def get_train_pair(txt):

    file_list = []
    txt_w = open( os.path.join(root,'txt', 'pair_train2.txt'), 'wb') 
    
    
    txt_f = open(txt, 'rb')
    list = txt_f.readlines()
    length = len(list)
    cmt = 0
    finished = {}
    for p in list:
        try:
            cmt += 1
            print ("{}/{}".format(cmt, length))
            v_path = p[:-5]
    #         img_path = v_path.replace('unzip/dev_video', 'img')
    #         img_length = len(os.listdir(img_path))
            tmp =  v_path.split('/')
            key = os.path.join(tmp[-4],tmp[-3],tmp[-2])
            lmark_length = np.load( v_path + '.npy').shape[0] 
            if lmark_length < 64:
                continue
            if key not in finished.keys():
                finished[key] = [0,[]]        

    #         if lmark_length != img_length:
    #             print (lmark_length, img_length)
            finished[key][1].append(tmp[-1])
    #         print (lmark_length)
            finished[key][0] += lmark_length
    #     print (finished)
        except:
            continue
    print (len(finished))
    kk = []
    for line in finished:
#         print (line)
#         print finished[line]
        if finished[line][0] < 64:
            del finished[line]
        else:
            kk.append([line,finished[line][0],finished[line][1]])
            txt_w.writelines(line + ':' + str(finished[line]))
    txt_f.close()
    print (kk[:2])
    with open(os.path.join( root, 'txt','train.pkl'), 'wb') as handle:
        pickle.dump(kk, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

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
        mfcc_name = line[:-5].replace('_video', '_audio')  + '.npy'
        if not os.path.exists(mfcc_name):
            try:
                print ('{}/{}'.format(count, length))
                audio_path = line[:-5].replace('_video', '_audio') + '.m4a'
 

                audio,fs = librosa.core.load(audio_path,sr = 16000 )
                mfcc = python_speech_features.mfcc(audio,16000,winstep=0.01)

                np.save(mfcc_name, mfcc)
                count += 1
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

def compute_RT(pickle_file):
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    print (len(consider_key))
    k = 20
    _file = open( os.path.join( root, pickle_file) , "rb")
    train_data = pickle.load(_file)
    # train_data = train_data[50000:]
    gg = len(train_data)
    print (gg)
    # train_data = train_data[int(config.txt_start * 0.1 * gg) : int(( 1+ config.txt_start) * 0.1 * gg )]
    landmarks = []
    RT_list = []
    source = np.zeros((len(consider_key),3))
    ff = np.load('./basics/00001.npy')[30]
    # ff = np.load( os.path.join(root, 'unzip', 'dev_video/id02343/08TabUdunsU/00001.npy'))[30]
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]
        
    source = mat(source)
    for index in range(len(train_data)):
        print ('{}/{}'.format(index,len(train_data) ))
        if not os.path.exists(os.path.join( root, 'unzip', train_data[index][0])):
            print (os.path.join( root, 'unzip', train_data[index][0]))
            continue
        # break
        for i in range(len(train_data[index][2])):
            lmark_path = os.path.join( root, 'unzip', train_data[index][0], train_data[index][2][i] + '.npy' )
            srt_path = os.path.join(root, 'unzip', train_data[index][0], train_data[index][2][i] + '_sRT.npy')
            front_path = os.path.join(root, 'unzip', train_data[index][0], train_data[index][2][i] + '_front.npy')
            if os.path.exists(srt_path) and os.path.exists(front_path):                
                continue
                
#             lmark_path = '/data2/lchen63/voxceleb/unzip/dev_video/id00967/4glDEPWbvKk/00030.npy'

            t_lmark = np.load(lmark_path)
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
            np.save(os.path.join( root, 'unzip', train_data[index][0], train_data[index][2][i] + '_sRT.npy'), RTs)
            np.save(os.path.join( root, 'unzip', train_data[index][0], train_data[index][2][i] + '_front.npy'), nomalized)
            print (time.time() - t )


def vis(lists): # a list [frame, landmark, frame, landmark]
    windows = len(lists)/2
    print (windows)
    fig = plt.figure(figsize=plt.figaspect(.5))

    for i in range(windows):
#     img  = io.imread(img_path)
        preds = lists[i*2 + 1 ]
        ax = fig.add_subplot(1, windows, i + 1)
        ax.imshow(lists[i* 2])
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
# def vis(img,lmark1,lmark2):
# #     img  = io.imread(img_path)
#     preds = lmark1
#     fig = plt.figure(figsize=plt.figaspect(.5))
#     ax = fig.add_subplot(1, 2, 1)
#     ax.imshow(img)
#     ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1) 
#     ax.axis('off')
    
#     preds = lmark2
    
#     ax = fig.add_subplot(1, 2, 2)
#     ax.imshow(img)
#     ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
#     ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1) 
#     ax.axis('off')
#     plt.show()
def clean_by_RT(pickle_file):
    n = 68
    _file = open(os.path.join( root , "txt/", pickle_file ), "rb")
    data = pickle.load(_file)
    _file.close()
    k = len(data)
    data_copy = []
    for index in range(len(data)):
#         if index == 100:
#             break
        print ('{}/{}'.format(index,len(data)))
        data_copy.append([data[index][0],data[index][1],[]])
        for j in range(len(data[index][2])):
            rt_path = os.path.join(root, 'unzip', data[index][0], data[index][2][j] + '_sRT.npy' )
            flmark_path = os.path.join(root, 'unzip', data[index][0], data[index][2][j] + '_front.npy' )
            olmark_path = os.path.join(root, 'unzip', data[index][0], data[index][2][j] + '.npy' )
            obj_path = os.path.join(root, 'unzip', data[index][0], data[index][2][j] + '_original.obj' )
            if os.path.exists(obj_path):
                continue
            if os.path.exists( rt_path) and os.path.exists(flmark_path) and os.path.exists(olmark_path):
                RT = np.load( rt_path)
                flmark  = np.load( flmark_path)                               
                olmark  = np.load( olmark_path)
                flag = True
                for tt in range(flmark.shape[0]):                    
                    # recover the transformation
                    A3 = utils.reverse_rt(flmark[tt], RT[tt])
                    target = mat(olmark[tt])

                    # Find the error
                    err = A3 - target

                    err = multiply(err, err)
                    err = sum(err)
                    rmse = sqrt(err/n);
        #             print ("RMSE:", rmse)
                    if rmse> 20:
                        print ("RMSE:", rmse)
                        print ( os.path.join(data[index][0], data[index][2][j]))
                        flag = False
#                         #vialisze the results
#                         video_path = os.path.join('/data2/lchen63/voxceleb/unzip/', data[index][0], data[index][2][j] +  '.mp4')
#                         cap = cv2.VideoCapture(video_path)
#                         for t in range(olmark.shape[0]):
#                             ret, frame = cap.read()
#                             if ret and t == tt:
#                                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#                                 vis(frame,flmark[tt],olmark[tt])
#                                 vis(frame,A3,olmark[tt])      
                        break
                                        
                if flag == True:
                    data_copy[index][2].append(data[index][2][j])
            else:
                print (rt_path)

    final = []
    print (len(data_copy))
    for index in range(len(data_copy)):
        if len(data_copy[index][2])!= 0:
            final.append(data_copy[index])
    print (len(final))
    print(k)
    with open(os.path.join(root, 'txt',pickle_file.replace('.pkl', '_clean.pkl')), 'wb') as handle:
        pickle.dump(final, handle, protocol=pickle.HIGHEST_PROTOCOL)




def bbox2(lmark):
    x_min = np.amin(lmark[:,0])
    x_max = np.amax(lmark[:,0])
    y_min = np.amin(lmark[:,1])
    y_max = np.amax(lmark[:,1])
    return x_min, y_min, x_max, y_max 
def visualization_lmark():
    n = 79
    v_id = 'test_video/id04276/k0zLls_oen0/00341' 
    rt_path = os.path.join(root, 'unzip', v_id + '_sRT.npy' )
    flmark_path = os.path.join(root, 'unzip', v_id + '_front.npy' )
    olmark_path = os.path.join(root, 'unzip',  v_id + '.npy' )

    prnet_lmark_path = os.path.join(root, 'unzip',  v_id + '_prnet.npy' )

    RT = np.load( rt_path)
    flmark  = np.load( flmark_path)                               
    olmark  = np.load( olmark_path)
    prnet_lmark = np.load(prnet_lmark_path)
    prnet_lmark[:,1] = 224- prnet_lmark[:,1]
    flag = True
    video_path = os.path.join(root, 'unzip', v_id + '.mp4')

    ani_video_path = os.path.join(root, 'unzip', v_id  + '_ani.mp4')
    # ani_video_path ='fuck.mp4'
    cap = cv2.VideoCapture(video_path)
    cap_ani = cv2.VideoCapture(ani_video_path)
    
    print (flmark.shape)
    for tt in range(flmark.shape[0]):                    
        # recover the transformation
        A3 = utils.reverse_rt(flmark[tt], RT[tt])

        #vialisze the results            
        ret, frame = cap.read()

        ret, ani_frame = cap_ani.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ani_frame = cv2.cvtColor(ani_frame, cv2.COLOR_BGR2RGB)
        if tt == n :
            vis([frame,flmark[tt], frame, A3, frame, olmark[tt],ani_frame, olmark[tt], frame, prnet_lmark  ])
            print (tt)
                # break
                                
def label_id():
    root = '/data2/lchen63/voxceleb/'
    _file = open(os.path.join("/data2/lchen63/voxceleb/txt/", "train_clean.pkl"), "rb")
    data = pickle.load(_file)
    _file.close()
    print ('=========')
    print(len(data))
    for index in range(len(data)):
        print ('{}/{}'.format(index,len(data)))
        print (data[index][0])
        print (data[index])
        break
        
#         for j in range(len(data[index][2])):        
#             RT = np.load( os.path.join(root, 'unzip', data[index][0], data[index][2][j] + '_RT.npy' ))
#             lmark  = np.load( os.path.join(root, 'unzip', data[index][0], data[index][2][j] + '.npy' ))
#             flag = True
#             for tt in range(lmark.shape[0]-1):
#                 source =  mat(lmark[tt])
#                 target = mat(lmark[tt+ 1])

#                 # recover the transformation
#                 rec = RT[tt,:3]
#                 r = R.from_rotvec(rec)
#                 ret_R = r.as_dcm()
#                 ret_t = RT[tt,3:]
#                 ret_t = ret_t.reshape(3,1)

#                 A2 = (ret_R*source.T) + tile(ret_t, (1, n))
#                 A2 = A2.T

#                 # Find the error
#                 err = A2 - target

#                 err = multiply(err, err)
#                 err = sum(err)
#                 rmse = sqrt(err/n);
#     #             print ("RMSE:", rmse)
#                 if rmse> 10:
#                     print ("RMSE:", rmse)
#                     print ( os.path.join(data[index][0], data[index][2][j]))
# #                     data_copy[index][2].remove(data_copy[index][2][j])
#                     flag = False
#                     break
#             if flag == True:
#                 data_copy[index][2].append(data[index][2][j])
#         print(data[index])
#         print (data_copy[index])
#             if len(data_copy[index][2]) == 0:
#                 del data_copy[index]
#                 break
    #                 vid = os.path.join(data[index][0], data[index][2][0])
    #                 fid = tt 
    #                 vis(vid, fid)
    #             if index == 100:
    #                 break
#     final = []
#     for index in range(len(data_copy)):
#         if len(data_copy[index][2])!= 0:
#             final.append(data_copy[index])
#     print (len(final))
#     print(k)
#     with open(os.path.join('/data2/lchen63/voxceleb/txt','train_clean.pkl'), 'wb') as handle:
#         pickle.dump(final, handle, protocol=pickle.HIGHEST_PROTOCOL)        
# label_id()        
# clean_by_RT()    
def compute_PCA():
    _file = open( os.path.join(root, "txt/train2.pkl"),      "rb")
    train_data = pickle.load(_file)
    random.shuffle(train_data)

    landmarks = []
    lmark_list = []
    k = 20
    print (len(train_data))
    for index in range(len(train_data)):
        if index == 9000:
            break
        for i in range(len(train_data[index][2])):
            lmark_path = os.path.join(  root ,  'unzip', train_data[index][0], train_data[index][2][i] + '_front.npy' )
            t_lmark = np.load(lmark_path)
            if t_lmark.shape[0] < 64:
                continue
            t_lmark = t_lmark[:t_lmark.shape[0]:5,:,:]
            t_lmark = utils.smooth(t_lmark)
            t_lmark = torch.FloatTensor(t_lmark)
        lmark_list.append(t_lmark)
    lmark_list = torch.cat(lmark_list,dim= 0)
    lmark_list = lmark_list.view(lmark_list.size(0) ,  68* 3)
    mean = torch.mean(lmark_list,0)

    np.save('./basics/mean_front_smooth_vox.npy', mean.cpu().numpy())
    derivatives = lmark_list - mean.expand_as(lmark_list)

    U,S,V  = torch.svd(torch.t(lmark_list))
    
    print (U[:,:10])
    C = torch.mm(lmark_list, U[:,:k])
    np.save('./basics/U_front_smooth_vox.npy', U.cpu().numpy())

def compose_audio_lmark_dataset():
    lstm = False
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    _file = open(os.path.join(root, 'txt' ,  "train_clean.pkl"), "rb")
    data = pickle.load(_file)
    new_data = []
    
    for index in range(len(data)):
        print (index, len(data))
        tmp = data[index][0].split('/')
        if len(data[index][2]) ==1:
            
            lmark = np.load( os.path.join(root, 'unzip',data[index][0], data[index][2][0] + '_front.npy' ))
            v_id = os.path.join(data[index][0], data[index][2][0])
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
                tmp.append(v_id)
                tmp.append(sample_id)
                tmp.append(min_index)
                new_data.append(tmp)
                           
        else:  
            for r in range(len(data[index][2])):
                
                v_id = os.path.join(data[index][0], data[index][2][r])
                lmark = np.load( os.path.join(root, 'unzip', data[index][0], data[index][2][r] + '_front.npy' ))
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
                for sample_id in range(0, gg  ):
#                     if sample_id != min_index:
#                         continue
#                     video_path = os.path.join('/data2/lchen63/voxceleb/unzip/', data[index][0], data[index][2][0] +  '.mp4')
#                     cap = cv2.VideoCapture(video_path)
#                     for t in range(lmark_length):
#                         ret, frame = cap.read()
#                         if ret and t == sample_id:
#                             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#                             vis(frame,lmark[sample_id], lmark[sample_id])
#                             break
#                     print (open_rate[:,sample_id])
                    
                    tmp = []
                    tmp.append(v_id)
                    tmp.append(sample_id)
                    tmp.append(min_index)
                    new_data.append(tmp)

    print (len(new_data))
    print (new_data[0])
    if lstm:
        nname =  os.path.join('/data2/lchen63/voxceleb/txt','train_clean_lstm.pkl')
    else:
        nname = os.path.join('/data2/lchen63/voxceleb/txt','train_clean_new.pkl') 
    with open(nname, 'wb') as handle:
        pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def compose_lmark_face_dataset():
    root  = '/data2/lchen63/voxceleb'
    lstm = False    
    _file = open(os.path.join(root, 'txt' ,  "test_clean.pkl"), "rb")
    data = pickle.load(_file)
    new_data = []
    for index in range(len(data)):
        print (index, len(data))
        tmp = data[index][0].split('/')
        if len(data[index][2]) ==1:
            
            front_lmark = np.load( os.path.join(root, 'unzip',data[index][0], data[index][2][0] + '_front.npy' ))
            rt = np.load( os.path.join(root, 'unzip',data[index][0], data[index][2][0] + '_sRT.npy' ))
            lmark = np.load( os.path.join(root, 'unzip',data[index][0], data[index][2][0] + '.npy' ))
            v_id = os.path.join(data[index][0], data[index][2][0])
            
            
            lmark_length = lmark.shape[0]
            for start_id in range( lmark_length - 65):
                tmp = []
                for sample_id in [0,start_id] + [start_id+ 64, lmark_length - 1]:
                    sample_rt = rt[sample_id]
                    
                    r_diff = rt[start_id:start_id + 64:, :3] - sample_rt[:3]
                    
                    t_diff = rt[start_id:start_id + 64:, 3:] - sample_rt[3:]
                    
                    r_diff = np.absolute(r_diff)
                    r_diff = np.mean(r_diff, axis =1)
                    min_r_index=  np.argmin(r_diff)  + start_id
                    tmp.append(v_id)
                    tmp.append(start_id)
                    tmp.append(sample_id)
                    tmp.append(min_r_index)
                    new_data.append(tmp)

        else:
            for r in range(len(data[index][2])):
                front_lmark = np.load( os.path.join(root, 'unzip',data[index][0], data[index][2][r] + '_front.npy' ))
                rt = np.load( os.path.join(root, 'unzip',data[index][0], data[index][2][r] + '_sRT.npy' ))
                lmark = np.load( os.path.join(root, 'unzip',data[index][0], data[index][2][r] + '.npy' ))
                v_id = os.path.join(data[index][0], data[index][2][r])
                
                lmark_length = lmark.shape[0]
                for start_id in range( lmark_length - 65):
                    
                    for sample_id in [0,start_id] + [start_id+ 64, lmark_length - 1]:
                        tmp = []
                        sample_rt = rt[sample_id]

                        r_diff = rt[start_id:start_id + 64:, :3] - sample_rt[:3]

                        t_diff = rt[start_id:start_id + 64:, 3:] - sample_rt[3:]

                        r_diff = np.absolute(r_diff)
                        r_diff = np.mean(r_diff, axis =1)
                        min_r_index=  np.argmin(r_diff)  + start_id
                        tmp.append(v_id)
                        tmp.append(sample_id)
                        tmp.append(min_r_index)
                        new_data.append(tmp)


    print (len(new_data))
    print (new_data[0])
    if lstm:
        nname =  os.path.join('/data2/lchen63/voxceleb/txt','lmark_video_train_clean_lstm.pkl')
    else:
        nname = os.path.join('/data2/lchen63/voxceleb/txt','lmark_video_test_clean_new.pkl') 
    with open(nname, 'wb') as handle:
        pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
        
def compose_front(pickle_file):
    n = 68
    _file = open(os.path.join(root, 'txt' , pickle_file ), "rb")
    data = pickle.load(_file)
    new_data = []
    # data= [['test_video/id03127/Zss2vvY2aLo',1254, ['00231']]]
    for index in range(len(data)):
        print(index, len(data))
        # if index == 10:
        #     break
        tmp = data[index][0].split('/')
        if len(data[index][2]) ==1:
            f_lmark =  os.path.join(root, 'unzip',data[index][0], data[index][2][0] + '_front.npy' )
            rt_path =  os.path.join(root, 'unzip',data[index][0], data[index][2][0] + '_sRT.npy' )
            o_lmark =  os.path.join(root, 'unzip',data[index][0], data[index][2][0] + '.npy' )
            v_path =  os.path.join(root, 'unzip',data[index][0], data[index][2][0] + '.mp4' )
            rt = np.load( rt_path)
            v_id = os.path.join(data[index][0], data[index][2][0])
            lmark = np.load(o_lmark)
            new_data.append([v_id])
            # video_path = os.path.join(root, 'unzip', v_id +  '.mp4')
            # cap = cv2.VideoCapture(video_path)
            # frames = []
            lmark_length = lmark.shape[0]
            find_rt = []
            for t in range(0, lmark_length):
                find_rt.append(sum(np.absolute(rt[t,:3])))

            #     ret, frame = cap.read()
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #     frames.append(frame)
                # vis(frame, lmark[t],frame, lmark[t],frame, lmark[t])
            # new_data.append([v_path, f_lmark, o_lmark , rt])
            find_rt = np.asarray(find_rt)

            min_index = np.argmin(find_rt)

            new_data[-1].append(min_index)
            
        else:  
            for r in range(len(data[index][2])):
                f_lmark =  os.path.join(root, 'unzip',data[index][0], data[index][2][r] + '_front.npy' )
                rt_path =  os.path.join(root, 'unzip',data[index][0], data[index][2][r] + '_sRT.npy' )
                
                o_lmark =  os.path.join(root, 'unzip',data[index][0], data[index][2][r] + '.npy' )
                v_path =  os.path.join(root, 'unzip',data[index][0], data[index][2][r] + '.mp4' )

                v_id = os.path.join(data[index][0], data[index][2][r])
                new_data.append([v_id])
                rt = np.load(rt_path )
                lmark = np.load(o_lmark)
                # video_path = os.path.join(root, 'unzip', v_id +  '.mp4')
                # cap = cv2.VideoCapture(video_path)
                # frames = []
                lmark_length = lmark.shape[0]
                find_rt = []
                for t in range(0, lmark_length):
                    find_rt.append(sum(np.absolute(rt[t,:3])))

                #     ret, frame = cap.read()
                #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #     frames.append(frame)
                    # vis(frame, lmark[t],frame, lmark[t],frame, lmark[t])
                # new_data.append([v_path, f_lmark, o_lmark , rt])
                find_rt = np.asarray(find_rt)

                min_index = np.argmin(find_rt)

                new_data[-1].append(min_index)
        # vis(frames[min_index], lmark[min_index],frames[min_index], lmark[min_index],frames[min_index], lmark[min_index])

    print (new_data[0])
    print (len(new_data))
    with open(os.path.join(root, 'txt','front_rt.pkl'), 'wb') as handle:
        pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def for_3d_to_rgb(): # based on front_rt.pkl, remove the videos which not contain ani video
    _pickle_file = os.path.join(root, 'txt','front_rt.pkl')
    _file = open(_pickle_file, "rb")
    data = pickle.load(_file)
    new_data = []
  
    for line in data:
        print(line)
        
        ani_video_path = os.path.join(root, 'unzip', line[0] + '_ani.mp4')
        if os.path.exists(ani_video_path):
            # ani_video = mmcv.VideoReader(ani_video_path)
            # real_video = mmcv.VideoReader(os.path.join(root, 'unzip', line[0] + '.mp4'))
            # ani_length = len(ani_video)
            # real_length = len(real_video)
            # reference_id = line[1]
            # if ani_length != real_length:
            #     print (ani_video_path, ani_length, real_length)
            #     continue
            # if reference_id >= ani_length:
            #     print (ani_video_path, reference_id, real_length)
            #     continue
            new_data.append(line)
    print (len(new_data))
    with open(os.path.join(root, 'txt','train_front_rt2.pkl'), 'wb') as handle:
        pickle.dump(new_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def file2folder():
    # _file = open(os.path.join(root, 'txt',  "front_rt.pkl"), "rb")
    # data = pickle._Unpickler(_file)
    # data.encoding = 'latin1'
    # data = data.load()
    # _file.close()
    
    _pickle_file = os.path.join(root, 'txt','front_rt.pkl')
    print (_pickle_file)
    _file = open(_pickle_file, "rb")
    data = pickle.load(_file)


    dir_set = set()
    new_list = data[-11000: -10000]
    for k,line in enumerate( new_list):
        video_path = os.path.join(root, 'unzip', line[0] + '.mp4') 
        print (video_path)
        folder_name = os.path.dirname(video_path)
        folder_name = os.path.dirname(folder_name)
        dir_set.add(folder_name)
    print (dir_set)
    file_list = []
    for dir_t in dir_set:
        # command_line = 'rsync -a ' + dir_t + ' ' + '/data/lchen63/vox' 
        # print (command_line)
        # os.system(command_line)
        for r,directories, files in os.walk(dir_t):
            for filename in files:
                file_list.append(os.path.join(r, filename))
    with ZipFile( os.path.join('/data/lchen63', 'zip_%03d.zip'%2), 'w', allowZip64=True) as zip:
        for file in file_list:
            zip.write(file)


# def rotate_3d (rt, obj):


# visualization_lmark()  
# 
# vis3d()
# for_3d_to_rgb()
# compute_PCA()

# rotate_3d('/test_video/id03127/Zss2vvY2aLo/00231_sRT.npy', '/test_video/id03127/Zss2vvY2aLo/00231.npy')
# compose_front()

# compose_dataset()
# compose_lmark_face_dataset()
# clean_by_RT()
# video2img2lmark()

# audio2mfcc('/data2/lchen63/voxceleb/txt/v_test.txt')
# video_transfer(os.path.join(root,'txt/v_test.txt'))
# compose_front()

# get_txt(os.path.join(root, 'unzip/test_video'))
file2folder()
####################
# get_txt(os.path.join(root, 'unzip/dev_video'))
# get_new_txt(os.path.join(root, 'txt/v_dev.txt'))
# get_train_pair( os.path.join(root, 'txt/fv_dev.txt')  )



# compute_RT("txt/train.pkl")

# clean_by_RT("train.pkl")
# compose_front("train_clean.pkl")
# for_3d_to_rgb()


