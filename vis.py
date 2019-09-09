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
    
def visualize(input, lmark, bkground, lmark2):     #shape 68,3
    preds = lmark 
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
    preds = lmark
    ax = fig.add_subplot(1, 4, 3)
    ax.imshow(bkground)                     
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='b',lw=1) 
    ax.axis('off')

    preds = lmark2
    ax = fig.add_subplot(1, 4, 4)
    ax.imshow(bkground)                     
    ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
    ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='b',lw=1) 
    ax.axis('off')
    
    
#     ax.colorbar()
    
#     ax.set_xlim(ax.get_xlim()[::-1])
    # plt.show()
    return fig
    

def smooth(kps, ALPHA1=0.2, ALPHA2=0.7):
    
    n = kps.shape[0]

    kps_new = np.zeros_like(kps)

    for i in range(n):
        if i==0:
            kps_new[i,:,:] = kps[i,:,:]
        else:
            kps_new[i,:48,:] = ALPHA1 * kps[i,:48,:] + (1-ALPHA1) * kps_new[i-1,:48,:]
            kps_new[i,48:,:] = ALPHA2 * kps[i,48:,:] + (1-ALPHA2) * kps_new[i-1,48:,:]

    # np.save(out_file, kps_new)
    return kps_new
root = '/home/cxu-serve/p1/lchen63/voxceleb/'
# /data2/lchen63/voxceleb/unzip/dev_video/id05880/55qXtfTk-oI/00018_RT.npy
#/data2/lchen63/voxceleb/unzip/dev_video/id02495/67VqQpS9dmw/00043_RT.npy
# /data2/lchen63/voxceleb/unzip/dev_video/id06453/Kt5_8WlBI8E/00124_RT.npy
#/data2/lchen63/voxceleb/unzip/dev_video/id02821/L6H3RV08O98/00108_RT.npy
#/data2/lchen63/voxceleb/unzip/dev_video/id07358/OovnuNo0vic/00163_RT.npy
v_id = 'id00017/01dfn2spqyE/00001'
v_path = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/test_video/'+ v_id + '.mp4'
lmark_path = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/test_video/' + v_id  + '_front.npy'
lmark = np.load(lmark_path)

lmark_smooth = smooth(lmark)

lmark_length = lmark.shape[0]
count = 0
tmp = v_path.split('/')
cap  =  cv2.VideoCapture(v_path)
audio_file = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/test_audio/'+ v_id + '.m4a'
if not os.path.exists('./tmp/olmark'):
    os.mkdir('./tmp/olmark')

if not os.path.exists('./tmp/slmark'):
    os.mkdir('./tmp/slmark')

background =  '/u/lchen63/github/head/basics/background.jpg'
background = cv2.imread(background)
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
for sample_id in range(0,lmark.shape[0]):
    ret, frame = cap.read()
#     frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    olmark = lmark[sample_id]
    flmark = lmark_smooth[sample_id]
    # nlmark = dlibdect(frame)
    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    fig = visualize(frame, olmark, background, flmark)
    fig.savefig('./tmp/olmark/%05d.png'%sample_id)
    # flmark = lmark_smooth[sample_id]
    # fig = visualize(frame, flmark, background)
    # fig.savefig('./tmp/slmark/%05d.png'%sample_id)
    

            
video_name = os.path.join('./tmp2' , 'orignal.mp4')
utils.image_to_video(os.path.join('./', 'tmp/olmark'), video_name )
utils.add_audio(video_name, audio_file)
print ('The generated video is: {}'.format(os.path.join('./tmp2' , 'orignal.mov')))
# video_name = os.path.join('./tmp2' , 'smooth.mp4')
# utils.image_to_video(os.path.join('./', 'tmp/slmark'), video_name )
# utils.add_audio(video_name, audio_file)
# print ('The generated video is: {}'.format(os.path.join('./tmp2' , 'smooth.mov')))