import os
import argparse

import librosa
from ATVG import AT_single as atnet
# from ATVG import AT_single2_no_pca as atnet
import cv2
import scipy.misc
import utils
from tqdm import tqdm
import torchvision.transforms as transforms
import shutil
from collections import OrderedDict
import python_speech_features
from skimage import transform as tf
from copy import deepcopy
from scipy.spatial import procrustes
import face_alignment
from scipy.spatial.transform import Rotation 

import glob
import time
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.modules.module import _addindent
import numpy as np
from collections import OrderedDict
import argparse
from logger import Logger
import cv2
from dataset import  Voxceleb_audio_lmark_single
from torch.nn import init
from util import utils
import dlib
from numpy import *
def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict



def initialize_weights( net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

        
        

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",
                     type=int,
                     default=1)
    parser.add_argument("--cuda",
                     default=True)
    parser.add_argument("--pca",
                     default=True)
#     parser.add_argument("--lstm",
#                      default=True)
#     parser.add_argument("--vg_model",
#                      type=str,
#                      default="../model/generator_23.pth")
#     parser.add_argument("--at_model",
#                      type=str,
#                      default="./model/at2_no_pca_openrate_loss2/anet2_single.pth")
    parser.add_argument("--at_model",
                     type=str,
                     default="./model/at/anet_single.pth")
    parser.add_argument( "--sample_dir",
                    type=str,
                    default="./results")
    parser.add_argument('-i','--in_file', type=str, default='./audio/test.wav')
    parser.add_argument('-data','--dataset_name', type=str, default='vox')
    parser.add_argument('-d','--data_path', type=str, default='./basics')
    parser.add_argument('-p','--person', type=str, default='./image/test2.jpg')
    parser.add_argument('-v','--video', type=str, default='./video/test.mp4')
    parser.add_argument('--device_ids', type=str, default='2')
    parser.add_argument('--num_thread', type=int, default=1)   
    return parser.parse_args()
config = parse_args()




detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./basics/shape_predictor_68_face_landmarks.dat')


def crop_image(image, id = 1, kxy = []):
#     kxy = []
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

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)#,  device='cpu')


def _crop_video(video):
    cap  =  cv2.VideoCapture(video)
    count = 0
    kxy =[]
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            count += 1
            print (count)
            img,kxy = crop_image(frame, 0,kxy)
            cv2.imwrite('./tmp/%04d.png'%count, img)
        else:
            break
    command = 'ffmpeg -framerate 25  -i ' +   './tmp/%04d.png  -vcodec libx264 -y -vf format=yuv420p ' + video[:-4] + '_crop.mp4' 
    os.system(command)

def get3DLmarks_single_image(image_path):
#     img = crop_image(image_path)
    img = cv2.imread(image_path)
    lmark = fa.get_landmarks(img)        
    np.save(image_path[:-4] + '.npy', lmark)



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

def _video2img2lmark(v_path):

    count = 0    
    frame_list = []
   
    cap  =  cv2.VideoCapture(v_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:            
            frame_list.append(frame)
            count += 1
        else:
            break
    get3DLmarks(frame_list, v_path)


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
    
    t = -R*centroid_A.T + centroid_B.T

    return R, t
def compute_RT(video_path):
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    print (len(consider_key))
    k = 2
    source = np.zeros((len(consider_key),3))
    
#     ff = np.load('/data/lchen63/grid/zip/video/s20/video/mpg_6000/bbad3n.npy')[0]
    ff = np.load('./vox_sample/08TabUdunsU/00001.npy')[30]
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]
        
    source = mat(source)

    lmark_path = video_path[:-4] + '.npy' 
    srt_path = video_path[:-4] +  '_sRT.npy'
    front_path =video_path[:-4] +  '_front.npy'
    
    
    t_lmark = np.load(lmark_path)
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
        r = Rotation.from_dcm(ret_R)
        vec = r.as_rotvec()             
        RTs[j,:3] = vec
        RTs[j,3:] =  np.squeeze(np.asarray(ret_t))            
    np.save(srt_path, RTs)
    np.save(front_path, nomalized)
#     return nomalized, RTs

def compute_RT_single(img_path):
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    print (len(consider_key))
    k = 2
    source = np.zeros((len(consider_key),3))
    
#     ff = np.load('/data/lchen63/grid/zip/video/s20/video/mpg_6000/bbad3n.npy')[0]
    
    ff = np.load('./vox_sample/08TabUdunsU/00001.npy')[30]
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]
        
    source = mat(source)

    lmark_path = img_path[:-4] + '.npy' 
    srt_path = img_path[:-4] +  '_sRT.npy'
    front_path =img_path[:-4] +  '_front.npy'
    
    
    t_lmark = np.load(lmark_path)
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
        r = Rotation.from_dcm(ret_R)
        vec = r.as_rotvec()             
        RTs[j,:3] = vec
        RTs[j,3:] =  np.squeeze(np.asarray(ret_t))            
    # after frontilize, we need to 
    
    np.save(srt_path, RTs)
    np.save(front_path, nomalized)




    
    return nomalized

def _extract_audio(video):
        
    command = 'ffmpeg -i ' + video + ' -ar 16000  -ac 1 -y ' + video.replace('video','audio').replace('.mp4','.wav') 
    try:
        # pass
        os.system(command)
    except BaseException:
        print (video)
    
    
def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
    
    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    os.mkdir('./tmp')  
    
    if config.dataset_name == 'vox':
        pca = torch.FloatTensor( np.load('./basics/U_front_smooth_vox.npy')[:,:6]).cuda()
        mean =torch.FloatTensor( np.load('./basics/mean_front_smooth_vox.npy')).cuda() 


       
    elif config.dataset_name == 'grid':
        face_pca = torch.FloatTensor( np.load('./basics/U_grid_roni.npy')[:,:6]).cuda()
        face_mean =torch.FloatTensor( np.load('./basics/mean_grid_roni.npy')).cuda()
        lip_pca = torch.FloatTensor( np.load('./basics/U_grid_lip.npy')[:,:6]).cuda()
        lip_mean =torch.FloatTensor( np.load('./basics/mean_grid_lip.npy')).cuda()

        lip_std = torch.FloatTensor( np.load('./basics/std_grid_lip.npy')).cuda()
        face_std = torch.FloatTensor( np.load('./basics/std_grid_roni.npy')).cuda()

    #change frame rate to 25FPS
    command = 'ffmpeg -i ' +  config.video +   ' -r 25 -y  ' + config.video
    
#     os.system(command)

    #extract audio from video
    _extract_audio(config.video)

#     # crop video into correct ratio
    _crop_video(config.video)
    config.video = config.video[:-4] + '_crop.mp4'
    
# #     genrate ground truth
    _video2img2lmark(config.video)
    
# #     #compute_RT and get front view ground truth
    compute_RT(config.video)
    
#     command = 'cp ./tmp/0141.png ./image/test.png'
#     os.system(command)
    
#     #extract exmaple landmark from single image
    
    
    RTs = np.load(config.video[:-4] +  '_sRT.npy' )
    gt_front = np.load(config.video[:-4] +  '_front.npy')
    
    
    get3DLmarks_single_image(config.person)
    example_lmark = compute_RT_single(config.person)
    
    example_lmark =torch.FloatTensor(example_lmark).view(1,-1).cuda()

    example_lmark_pca = (example_lmark - mean.expand_as(example_lmark))

    example_lmark_pca = torch.mm(example_lmark_pca, pca)

    
    example_lmark_pca = Variable(example_lmark_pca)
    
      
    if os.path.exists('./lmark'):
        shutil.rmtree('./lmark')
    os.mkdir('./lmark')
    if not os.path.exists('./lmark/real'):
        os.mkdir('./lmark/real')
    if not os.path.exists('./lmark/fake'):
        os.mkdir('./lmark/fake')
    if not os.path.exists('./lmark/fake_rt'):
        os.mkdir('./lmark/fake_rt')
    
    encoder = atnet()
    if config.cuda:
        encoder = encoder.cuda()

#     state_dict = multi2single(config.at_model, 0)
    encoder.load_state_dict( torch.load(config.at_model))
    encoder.eval()
    test_file = config.in_file
 
    # Load speech and extract features
    speech, sr = librosa.load(test_file, sr=16000)
    mfcc = python_speech_features.mfcc(speech ,16000,winstep=0.01)
#     speech = np.insert(speech, 0, np.zeros(1920))
#     speech = np.append(speech, np.zeros(1920))
    sound, _ = librosa.load(test_file, sr=44100)

    print ('=======================================')
    print ('Start to generate images')
    t =time.time()
    ind = 3
    with torch.no_grad(): 
        fake_lmark = []
        input_mfcc = []
        while ind <= int(mfcc.shape[0]/4) - 4:
            t_mfcc =mfcc[( ind - 3)*4: (ind + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc).cuda()
            input_mfcc.append(t_mfcc)
            ind += 1
        input_mfcc = torch.stack(input_mfcc,dim = 0)
      
        fake_lmark = encoder(input_mfcc, example_lmark_pca.repeat(input_mfcc.shape[0] ,1) )
        if config.pca:
            fake_lmark =  fake_lmark.view(fake_lmark.size(0)  , 6)
            fake_lmark = torch.mm( fake_lmark, pca.t() ) 
        else:
            fake_lmark =  fake_lmark.view(fake_lmark.size(0)  , 204)
        fake_lmark += mean.expand_as(fake_lmark)
        
        fake_lmark = fake_lmark.data.cpu().numpy()
        
        
        

        for gg in range(fake_lmark.shape[0]):
            backgroung = cv2.imread('./tmp/%04d.png'%(gg+ 1))
            backgroung= cv2.cvtColor(backgroung, cv2.COLOR_BGR2RGB) 
            lmark_name  = "./lmark/fake/%05d.png"%gg
            plt = utils.lmark2img(fake_lmark[gg].reshape((68,3)), backgroung, c = 'b')
            plt.savefig(lmark_name)
            A3 = utils.reverse_rt(fake_lmark[gg].reshape((68,3)), RTs[gg])
            lmark_name  = "./lmark/fake_rt/%05d.png"%gg
            plt = utils.lmark2img(A3.reshape((68,3)), backgroung, c = 'b')
            plt.savefig(lmark_name)
            
        video_name = os.path.join(config.sample_dir , 'fake.mp4')
        utils.image_to_video(os.path.join('./', 'lmark/fake'), video_name )
        utils.add_audio(video_name, config.in_file)
        print ('The generated video is: {}'.format(os.path.join(config.sample_dir , 'fake.mov')))
#         
        video_name = os.path.join(config.sample_dir , 'fake_rt.mp4')
        utils.image_to_video(os.path.join('./', 'lmark/fake_rt'), video_name )
        utils.add_audio(video_name, config.in_file)
        print ('The generated video is: {}'.format(os.path.join(config.sample_dir , 'fake_rt.mov')))
        

        for gg in range(gt_front.shape[0]):
            backgroung = cv2.imread('./tmp/%04d.png'%(gg+ 1))
            backgroung= cv2.cvtColor(backgroung, cv2.COLOR_BGR2RGB) 
            lmark_name  = "./lmark/real/%05d.png"%gg
            plt = utils.lmark2img(gt_front[gg].reshape((68,3)), backgroung, c = 'b')
            plt.savefig(lmark_name)
            
            reversed_lmark = utils.reverse_rt(gt_front[gg], RTs[gg])
            if not os.path.exists('./lmark/original'):
                os.mkdir ('./lmark/original')
            lmark_name  = "./lmark/original/%05d.png"%gg
            plt = utils.lmark2img(reversed_lmark.reshape((68,3)), backgroung, c = 'b')
            plt.savefig(lmark_name)
            
        video_name = os.path.join(config.sample_dir , 'real.mp4')
        utils.image_to_video(os.path.join('./', 'lmark/real'), video_name )
        utils.add_audio(video_name, config.in_file)
        print ('The generated video is: {}'.format(os.path.join(config.sample_dir , 'real.mov')))
        
        video_name = os.path.join(config.sample_dir , 'orig.mp4')
        utils.image_to_video(os.path.join('./', 'lmark/original'), video_name )
        utils.add_audio(video_name, config.in_file)
        print ('The generated video is: {}'.format(os.path.join(config.sample_dir , 'orig.mov')))
        

test()

# video_name = os.path.join(config.sample_dir , 'results.mp4')
# utils.image_to_video(os.path.join('./', 'lmark'), video_name )
# utils.add_audio(video_name, config.in_file)
# print ('The generated video is: {}'.format(os.path.join(config.sample_dir , 'results.mov')))
        
        
        
        