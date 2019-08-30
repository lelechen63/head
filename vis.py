import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import cv2
import os
import dlib
import utils
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./basics/shape_predictor_68_face_landmarks.dat')
import librosa
def dlibdect(img):
#     cv2.imshow('image',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = utils.shape_to_np(shape)
        print (shape.shape)
        
    return shape
    
def visualize(input, lmark1, lmark2, audio):     #shape 68,3
    preds = lmark1 
    #TODO: Make this nice
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(input)
    
    ax = fig.add_subplot(1, 4, 2)
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
    preds = lmark2
    ax = fig.add_subplot(1, 4, 3)
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
    
    
    ax = fig.add_subplot(1, 4, 4)
    ax.imshow(audio)
#     ax.colorbar()
    
#     ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()
    
    
root = '/data2/lchen63/voxceleb'
# /data2/lchen63/voxceleb/unzip/dev_video/id05880/55qXtfTk-oI/00018_RT.npy
#/data2/lchen63/voxceleb/unzip/dev_video/id02495/67VqQpS9dmw/00043_RT.npy
# /data2/lchen63/voxceleb/unzip/dev_video/id06453/Kt5_8WlBI8E/00124_RT.npy
#/data2/lchen63/voxceleb/unzip/dev_video/id02821/L6H3RV08O98/00108_RT.npy
#/data2/lchen63/voxceleb/unzip/dev_video/id07358/OovnuNo0vic/00163_RT.npy
v_id = 'id07294/PtHmwQygx8Q/00010'
v_path = '/data2/lchen63/voxceleb/unzip/dev_video/'+ v_id + '.mp4'
lmark_path = '/data2/lchen63/voxceleb/unzip/dev_video/' + v_id  + '.npy'
mfcc_path = '/data2/lchen63/voxceleb/unzip/dev_audio/' + v_id + '.npy'
lmark = np.load(lmark_path)
audio = np.load(mfcc_path)

lmark_length = lmark.shape[0]
audio_length = audio.shape[0]
audio_pad = np.zeros((lmark.shape[0] * 4, 13))
if audio_length < lmark_length * 4 :
    audio_pad[:audio_length] = audio
    audio = audio_pad
count = 0
tmp = v_path.split('/')
cap  =  cv2.VideoCapture(v_path)

for sample_id in range(0,lmark.shape[0]):
    if sample_id < 3:
        sample_audio = np.zeros((28,12))
        sample_audio[4 * (3- sample_id): ] = audio[4 * (0) : 4 * ( 3 + sample_id + 1 ), 1: ]

    elif sample_id > lmark_length - 4:
        sample_audio = np.zeros((28,12))
        sample_audio[:4 * ( lmark_length + 3 - sample_id )] = audio[4 * (sample_id -3) : 4 * ( lmark_length ), 1: ]

    else:
        sample_audio = audio[4 * (sample_id -3) : 4 * ( sample_id + 4 ) , 1: ]
    ret, frame = cap.read()
#     frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    olmark = lmark[sample_id]
    nlmark = dlibdect(frame)
    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print (sample_audio.shape)
    visualize(frame, olmark, nlmark, sample_audio)
    
    