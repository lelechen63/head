import os
import random
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import librosa
import time
import copy
import python_speech_features
from numpy import *
from scipy.spatial.transform import Rotation as R
import time
from util import utils


def readVideo(self, videoFile):
    # Open the video file
    cap = cv2.VideoCapture(videoFile)
    # nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = torch.FloatTensor(self.channels, self.timeDepth, self.xSize, self.ySize)
    failedClip = False
    for f in range(self.timeDepth):

        ret, frame = cap.read()
        if ret:
            frame = torch.from_numpy(frame)
            # HWC2CHW
            frame = frame.permute(2, 0, 1)
            frames[:, f, :, :] = frame

        else:
            print("Skipped!")
            failedClip = True
            break

    for c in range(3):
        frames[c] -= self.mean[c]
    frames /= 255
    return frames, failedClip
class Voxceleb_head_movements_derivative(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train'):
        self.train = train
        self.num_frames = 64  
        self.root  = '/data2/lchen63/voxceleb/'
                 
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "train.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "test.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()

        self.consider_key = [1,5,8,11,15,27,28,29,30,31,32,33,34,35,39,42]
    def __getitem__(self, index):
        try:
            tmp = self.data[index][0].split('/')
            mean = np.load( os.path.join(self.root, 'unzip', self.data[index][0],'mean.npy' ))
            if len(self.data[index][2]) ==1:
#                 print ('----')
                landmark = np.load( os.path.join(self.root, 'unzip', self.data[index][0], self.data[index][2][0] + '.npy' ))
                middle = int(self.data[index][1] / 2)
                in_max = middle - 64
                if (in_max <= 0):
                    in_max = 10
                out_max = self.data[index][1] - 64
                if out_max < middle:
                    middle -= 15
                elif out_max == middle:
                    middle -= 5
                in_start  = random.choice([x for x in range(0,in_max)])
                out_start = random.choice([x for x in range(middle,out_max)])
                in_lmark = landmark[in_start:in_start+ 64] 
                out_lmark = landmark[out_start:out_start+ 64] 
#                 print (in_lmark.shape,out_lmark.shape )
#                 print ('~--')
            else:
#                 print ('++++')
                in_landmark = np.load( os.path.join(self.root, 'unzip', self.data[index][0], self.data[index][2][0] + '.npy' ))
                r  = random.choice([x for x in range(1,len(self.data[index][2]))])
                out_landmark = np.load( os.path.join(self.root, 'unzip', self.data[index][0], self.data[index][2][r] + '.npy' ))
                in_max = in_landmark.shape[0] - 64
                out_max = out_landmark.shape[0] - 64
                if (in_max <= 0):
                    in_max = 10
                if (out_max <= 0):
                    out_max = 10 
                in_start  = random.choice([x for x in range(0,in_max)])
                out_start = random.choice([x for x in range(0,out_max)])                  
                in_lmark = in_landmark[in_start:in_start + 64] 
                out_lmark = out_landmark[out_start:out_start + 64]
#                 print (in_lmark.shape,out_lmark.shape )
#                 print ('~++')
            assert in_lmark.shape[0]== 64 and out_lmark.shape[0] == 64
            in_lmark = torch.FloatTensor(in_lmark)
            
            out_lmark = torch.FloatTensor(out_lmark)
            mean = torch.FloatTensor(mean)
            in_lmark_part = torch.zeros((64,len(self.consider_key),3),dtype=torch.float32)
            
            out_lmark_part = torch.zeros((64,len(self.consider_key),3),dtype=torch.float32)
            mean_lmark_part = torch.zeros((len(self.consider_key),3),dtype=torch.float32)
            for i in range(len(self.consider_key)):
                in_lmark_part[:,i] = in_lmark[:,self.consider_key[i]]
                out_lmark_part[:,i] = out_lmark[:,self.consider_key[i]]
                mean_lmark_part[i] = mean[self.consider_key[i]]

#             in_lmark = in_lmark.view(in_lmark.shape[0],-1)
#             out_lmark = out_lmark.view(out_lmark.shape[0],-1)
#             mean = mean.view(1, -1)
            input_dict = {'in_lmark': in_lmark_part,  'out_lmark': out_lmark_part,'mean': mean_lmark_part}
#             print (in_lmark.shape, out_lmark.shape, mean.shape)
            return input_dict   
        except:
            self.__getitem__((index + 1)% len(self.data))
    def __len__(self):
        
            return len(self.data)


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



class Voxceleb_lmark_rgb_single(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train',
                 gan = True):
        self.train = train
        self.gan = gan
        self.output_shape   = tuple([256, 256])
        self.num_frames = 64  
        self.root  = '/home/cxu-serve/p1/lchen63/voxceleb/'      
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_video_test_clean_new.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_video_test_clean_new.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

#         self.consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    def __getitem__(self, index):
#         try:
            tmp = self.data[index][0].split('/')
           
            
            lmark = np.load( os.path.join(self.root, 'unzip', self.data[index][0] + '.npy' ))
            length = lmark.shape[0]
            lmark = utils.smooth(lmark)
            lmark = torch.FloatTensor(lmark)

            
            example_id  = self.data[index][2]
            sample_id = self.data[index][1]
            if self.gan:
                while True:
                    other_id = np.random.choice([0, length - 1])
                    if other_id != sample_id:
                        break
                
            lip_base = torch.FloatTensor(3,256, 256)
            
            lmark_base = torch.FloatTensor(68,2)
            video_path =  os.path.join('/data2/lchen63/voxceleb/unzip/',self.data[index][0]+  '.mp4')
            cap = cv2.VideoCapture(video_path)
            for t in range(length):
                ret, frame = cap.read()
                if ret :
                    if self.gan and other_id == t:
                        mismatch_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mismatch_img = cv2.resize(mismatch_img, self.output_shape)
                        mismatch_img = self.transform(mismatch_img)
                    if t == sample_id:
                        gt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        gt_img = cv2.resize(gt_img, self.output_shape)
                        gt_img = self.transform(gt_img)
                    if t == example_id:
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.output_shape)
                        lip = self.transform(img)
                        lip_base = lip
                        lmark_base = lmark[t,:, :-1]
                    
            input_lmark = lmark[sample_id,:, :-1]
            input_dict = {'in_lmark': input_lmark ,'lmark_base': lmark_base,  'lip_base': lip_base, 'gt_img': gt_img, 'mismatch_img': mismatch_img}
            return (input_dict)   
            













class Voxceleb_audio_lmark_single(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train',
                 gan = True):
        self.train = train  
        self.root  = '/home/cxu-serve/p1/lchen63/voxceleb/' 
        # self.lip_pca = torch.FloatTensor( np.load('./basics/U_front.npy')[:,:6])#.cuda()
        # self.lip_mean =torch.FloatTensor( np.load('./basics/mean_front.npy'))#.cuda()
                
        # self.other_pca = torch.FloatTensor( np.load('./basics/U_front_roni.npy')[:,:6])#.cuda()
        # self.other_mean =torch.FloatTensor( np.load('./basics/mean_front_roni.npy'))#.cuda()
        
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "test_clean_new.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "test_clean_new.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    def __getitem__(self, index):
#         try:
            tmp = self.data[index][0].split('/')
            lmark = np.load( os.path.join(self.root, 'unzip', self.data[index][0] + '_front.npy' ))
            if self.train == 'train':
                audio_path = os.path.join(self.root, 'unzip', 'test_audio' , tmp[1],tmp[2], tmp[3] +'.npy' )
            else:
                audio_path = os.path.join(self.root, 'unzip', 'test_audio' , tmp[1],tmp[2], tmp[3] +'.npy' )
                rts = np.load(os.path.join(self.root, 'unzip', self.data[index][0] + '_sRT.npy' ))
            audio = np.load(audio_path)
            sample_id = self.data[index][1]
            lmark_length = lmark.shape[0]
            audio_length = audio.shape[0]
            lmark = utils.smooth(lmark)
            audio_pad = np.zeros((lmark.shape[0] * 4, 13))
            if audio_length < lmark_length * 4 :
                audio_pad[:audio_length] = audio
                audio = audio_pad
            
           
            if sample_id < 3:
                sample_audio = np.zeros((28,12))
                sample_audio[4 * (3- sample_id): ] = audio[4 * (0) : 4 * ( 3 + sample_id + 1 ), 1: ]

            elif sample_id > lmark_length - 4:
                sample_audio = np.zeros((28,12))
                sample_audio[:4 * ( lmark_length + 3 - sample_id )] = audio[4 * (sample_id -3) : 4 * ( lmark_length ), 1: ]

            else:
                sample_audio = audio[4 * (sample_id -3) : 4 * ( sample_id + 4 ) , 1: ]
            
            sample_lmark = lmark[sample_id]
            sample_audio =torch.FloatTensor(sample_audio)
            sample_lmark =torch.FloatTensor(sample_lmark)            
            sample_lmark = sample_lmark.view(-1)

    
            ex_id = self.data[index][2]
            ex_lmark = lmark[ex_id]
            ex_lmark =torch.FloatTensor(ex_lmark)
            
            ex_lmark = ex_lmark.view(-1)
            
            # input_dict = {'audio': sample_audio , 'lip_region': lip_region, 'other_region': other_region, 'ex_other_region':ex_other_region,'ex_lip_region':ex_lip_region,   'img_path': self.data[index][0], 'sample_id' : sample_id, 'sample_rt': rts[sample_id] if self.train == 'test' else 1}
            input_dict = {'audio': sample_audio , 'sample_lmark': sample_lmark, 'ex_lmark': ex_lmark,   'img_path': self.data[index][0], 'sample_id' : sample_id, 'sample_rt': rts[sample_id] if self.train == 'test' else 1}

            return (input_dict)   
#         except:
#             return self.__getitem__((index + 1)% len(self.data))
    def __len__(self):
        
            return len(self.data)        



class Voxceleb_audio_lmark_single_short(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train',
                 gan = True):
        self.train = train  
        self.root  = '/data2/lchen63/voxceleb/' 
        self.lip_pca = torch.FloatTensor( np.load('./basics/U_front.npy')[:,:6])#.cuda()
        self.lip_mean =torch.FloatTensor( np.load('./basics/mean_front.npy'))#.cuda()
                
        self.other_pca = torch.FloatTensor( np.load('./basics/U_front_roni.npy')[:,:6])#.cuda()
        self.other_mean =torch.FloatTensor( np.load('./basics/mean_front_roni.npy'))#.cuda()
        
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "train2_clean_new.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "test_clean_new.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    def __getitem__(self, index):
#         try:
            tmp = self.data[index][0].split('/')
            lmark = np.load( os.path.join(self.root, 'unzip', self.data[index][0] + '_front.npy' ))
            if self.train == 'train':
                audio_path = os.path.join(self.root, 'unzip', 'dev_audio' , tmp[1],tmp[2], tmp[3] +'.npy' )
            else:
                audio_path = os.path.join(self.root, 'unzip', 'test_audio' , tmp[1],tmp[2], tmp[3] +'.npy' )
                rts = np.load(os.path.join(self.root, 'unzip', self.data[index][0] + '_sRT.npy' ))
            audio = np.load(audio_path)
            sample_id = self.data[index][1]
            lmark_length = lmark.shape[0]
            audio_length = audio.shape[0]
            audio_pad = np.zeros((lmark.shape[0] * 4, 13))
            if audio_length < lmark_length * 4 :
                audio_pad[:audio_length] = audio
                audio = audio_pad
            
           
            if sample_id < 1:
                sample_audio = np.zeros((12,12))
                sample_audio[4 * (1- sample_id): ] = audio[4 * (0) : 4 * ( 1 + sample_id + 1 ), 1: ]

            elif sample_id > lmark_length - 2:
                sample_audio = np.zeros((12,12))
                sample_audio[:4 * ( lmark_length + 1 - sample_id )] = audio[4 * (sample_id  - 1) : 4 * ( lmark_length ), 1: ]

            else:
                sample_audio = audio[4 * (sample_id -1) : 4 * ( sample_id + 2 ) , 1: ]

            sample_lmark = lmark[sample_id]
            sample_audio =torch.FloatTensor(sample_audio)
            sample_lmark =torch.FloatTensor(sample_lmark)
            lip_region =  sample_lmark[ 48: ]
            lip_region = lip_region.view(-1)

            other_region =  sample_lmark[ :48 ]
            other_region = other_region.view( -1)
            ex_id = self.data[index][2]
            ex_lmark = lmark[ex_id]
            ex_lmark =torch.FloatTensor(ex_lmark)
            
            ex_other_region =  ex_lmark[:48 ]
            ex_other_region = ex_other_region.view(-1)
            
            ex_lip_region =  ex_lmark[48: ]
            ex_lip_region = ex_lip_region.view(-1)            
            input_dict = {'audio': sample_audio , 'lip_region': lip_region, 'other_region': other_region, 'ex_other_region':ex_other_region,'ex_lip_region':ex_lip_region,   'img_path': self.data[index][0], 'sample_id' : sample_id, 'sample_rt': rts[sample_id] if self.train == 'test' else 1}
#             print (img_id)
#             print ('========')
            return (input_dict)   
#         except:
#             return self.__getitem__((index + 1)% len(self.data))
    def __len__(self):
        
            return len(self.data)           
        
class Voxceleb_audio_lmark_lstm(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train',
                 gan = True):
        self.train = train 
        self.time_length = 32
        self.root  = '/data2/lchen63/voxceleb/' 
        self.lip_pca = torch.FloatTensor( np.load('./basics/U_front.npy')[:,:6])#.cuda()
        self.lip_mean =torch.FloatTensor( np.load('./basics/mean_front.npy'))#.cuda()
                
        self.other_pca = torch.FloatTensor( np.load('./basics/U_front_roni.npy')[:,:6])#.cuda()
        self.other_mean =torch.FloatTensor( np.load('./basics/mean_front_roni.npy'))#.cuda()
        
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "train2_clean_lstm.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "test_clean_lstm.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    def __getitem__(self, index):
#         try:
            tmp = self.data[index][0].split('/')
            lmark = np.load( os.path.join(self.root, 'unzip', self.data[index][0] + '_front.npy' ))
            if self.train == 'train':
                audio_path = os.path.join(self.root, 'unzip', 'dev_audio' , tmp[1],tmp[2], tmp[3] +'.npy' )
            else:
                audio_path = os.path.join(self.root, 'unzip', 'test_audio' , tmp[1],tmp[2], tmp[3] +'.npy' )
                rts = np.load(os.path.join(self.root, 'unzip', self.data[index][0] + '_sRT.npy' ))
            audio = np.load(audio_path)
            start_id = self.data[index][1]
            lmark_length = lmark.shape[0]
            audio_length = audio.shape[0]
            audio_pad = np.zeros((lmark.shape[0] * 4, 13))
            if audio_length < lmark_length * 4 :
                audio_pad[:audio_length] = audio
                audio = audio_pad
            
            lmarks = torch.zeros((self.time_length,68,3),dtype=torch.float32)
            audios = torch.zeros((self.time_length,28,12),dtype=torch.float32)
            for t in range(self.time_length):
                sample_id = start_id + t
            
                if sample_id < 3:
                    sample_audio = np.zeros((28,12))
                    sample_audio[4 * (3- sample_id): ] = audio[4 * (0) : 4 * ( 3 + sample_id + 1 ), 1: ]

                elif sample_id > lmark_length - 4:
                    sample_audio = np.zeros((28,12))
                    sample_audio[:4 * ( lmark_length + 3 - sample_id )] = audio[4 * (sample_id -3) : 4 * ( lmark_length ), 1: ]

                else:
                    sample_audio = audio[4 * (sample_id -3) : 4 * ( sample_id + 4 ) , 1: ]
                
                sample_lmark = lmark[sample_id]
                sample_audio =torch.FloatTensor(sample_audio)
                sample_lmark =torch.FloatTensor(sample_lmark)
                
                lmarks[t] = sample_lmark
                audios[t] = sample_audio
                
                
            lip_region =  lmarks[:, 48: ]
            lip_region = lip_region.view(self.time_length, -1)

            other_region =  lmarks[:,  :48 ]
            other_region = other_region.view(self.time_length, -1)

            
            ex_id =self.data[index][2]
            ex_lmark = lmark[ex_id]
            ex_lmark =torch.FloatTensor(ex_lmark)
            
            ex_other_region =  ex_lmark[:48 ]
            ex_other_region = ex_other_region.view(-1)
            
            ex_lip_region =  ex_lmark[48: ]
            ex_lip_region = ex_lip_region.view(-1)

            
            input_dict = {'audio': audios , 'lip_region': lip_region, 'other_region': other_region, 'ex_other_region':ex_other_region,'ex_lip_region':ex_lip_region,   'img_path': self.data[index][0], 'sample_id' : start_id, 'sample_rt': rts[sample_id] if self.train == 'test' else 1 }
            return (input_dict)   
#         except:
#             return self.__getitem__((index + 1)% len(self.data))
    def __len__(self):
        
            return len(self.data)          
class Grid_audio_lmark_single(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train',
                 gan = True):
        self.train = train  
        self.root  = '/data/lchen63/grid/zip/' 
     
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "train_clean_new.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "test_clean_new.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    def __getitem__(self, index):
            lmark = np.load( self.data[index][0][:-4] + '_front.npy' )
            audio_path = os.path.join(self.root, 'audio', self.data[index][-1] + '_16k.npy' )
            audio = np.load(audio_path)
            sample_id = self.data[index][1]
            lmark_length = lmark.shape[0]
            lmark = utils.smooth(lmark)
            audio_length = audio.shape[0]
            audio_pad = np.zeros((lmark.shape[0] * 4, 13))
            if audio_length < lmark_length * 4 :
                audio_pad[:audio_length] = audio
                audio = audio_pad
                      
            if sample_id < 3:
                sample_audio = np.zeros((28,12))
                sample_audio[4 * (3- sample_id): ] = audio[4 * (0) : 4 * ( 3 + sample_id + 1 ), 1: ]

            elif sample_id > lmark_length - 4:
                sample_audio = np.zeros((28,12))
                sample_audio[:4 * ( lmark_length + 3 - sample_id )] = audio[4 * (sample_id -3) : 4 * ( lmark_length ), 1: ]

            else:
                sample_audio = audio[4 * (sample_id -3) : 4 * ( sample_id + 4 ) , 1: ]
            
            sample_lmark = lmark[sample_id]
            sample_audio =torch.FloatTensor(sample_audio)
            sample_lmark =torch.FloatTensor(sample_lmark)
            lip_region =  sample_lmark[ 48: ]
            lip_region = lip_region.view(-1)

            other_region =  sample_lmark[ :48 ]
            other_region = other_region.view( -1)
    
            ex_id = self.data[index][2]
            ex_lmark = lmark[ex_id]
            ex_lmark =torch.FloatTensor(ex_lmark)
            
            ex_other_region =  ex_lmark[:48 ]
            ex_other_region = ex_other_region.view(-1)
            
            ex_lip_region =  ex_lmark[48: ]
            ex_lip_region = ex_lip_region.view(-1)
            img_path =  os.path.join(self.root, 'img', self.data[index][-1] , '%05d.png'%(sample_id +1) )
            input_dict = {'audio': sample_audio , 'lip_region': lip_region, 'other_region': other_region, 'ex_other_region':ex_other_region,'ex_lip_region':ex_lip_region,   'img_path': img_path , 'sample_id' : sample_id}
#             print (img_id)
#             print ('========')
            return (input_dict)   
#         except:
#             return self.__getitem__((index + 1)% len(self.data))
    def __len__(self):
        
            return len(self.data)        


class Grid_audio_lmark_single_whole(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train',
                 gan = True):
        self.train = train  
        self.root  = '/data/lchen63/grid/zip/' 
     
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "train_clean_new.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "test_clean_new.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    def __getitem__(self, index):
            lmark = np.load( self.data[index][0][:-4] + '_front_norm.npy' )
            audio_path = os.path.join(self.root, 'audio', self.data[index][-1] + '_16k.npy' )
            audio = np.load(audio_path)
            sample_id = self.data[index][1]
            lmark_length = lmark.shape[0]
            audio_length = audio.shape[0]
            audio_pad = np.zeros((lmark.shape[0] * 4, 13))
            if audio_length < lmark_length * 4 :
                audio_pad[:audio_length] = audio
                audio = audio_pad
                      
            if sample_id < 3:
                sample_audio = np.zeros((28,12))
                sample_audio[4 * (3- sample_id): ] = audio[4 * (0) : 4 * ( 3 + sample_id + 1 ), 1: ]

            elif sample_id > lmark_length - 4:
                sample_audio = np.zeros((28,12))
                sample_audio[:4 * ( lmark_length + 3 - sample_id )] = audio[4 * (sample_id -3) : 4 * ( lmark_length ), 1: ]

            else:
                sample_audio = audio[4 * (sample_id -3) : 4 * ( sample_id + 4 ) , 1: ]
            
            sample_lmark = lmark[sample_id]
            sample_audio =torch.FloatTensor(sample_audio)
            sample_lmark =torch.FloatTensor(sample_lmark)
            
            target_lmark  = sample_lmark.view( -1)
    
            ex_id = self.data[index][2]
            ex_lmark = lmark[ex_id]
            ex_lmark =torch.FloatTensor(ex_lmark)
            ex_lmark = ex_lmark.view(-1)
            img_path =  os.path.join(self.root, 'img', self.data[index][-1] , '%05d.png'%(sample_id +1) )
            input_dict = {'audio': sample_audio , 'target_lmark': target_lmark,'ex_lmark':ex_lmark,   'img_path': img_path , 'sample_id' : sample_id}
#             print (img_id)
#             print ('========')
            return (input_dict)   
#         except:
#             return self.__getitem__((index + 1)% len(self.data))
    def __len__(self):
        
            return len(self.data)         

        
class Voxceleb_face_region_single(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train',
                 gan = True):
        self.train = train
        self.gan = gan
        self.output_shape   = tuple([256, 256])
        self.num_frames = 64  
        self.root  = '/data2/lchen63/voxceleb/'      
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "lmark_video_train2_clean_new.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "lmark_video_test_clean_new.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

#         self.consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    def __getitem__(self, index):
#         try:
            tmp = self.data[index][0].split('/')
           
            
            lmark = np.load( os.path.join(self.root, 'unzip', self.data[index][0] + '.npy' ))
            length = lmark.shape[0]
            
            lmark = torch.FloatTensor(lmark)
            
            example_id  = self.data[index][2]
            sample_id = self.data[index][1]
            if self.gan:
                while True:
                    other_id = np.random.choice([0, length - 1])
                    if other_id != sample_id:
                        break
                
            lip_base = torch.FloatTensor(3,256, 256)
            
            lmark_base = torch.FloatTensor(68,2)
            video_path =  os.path.join('/data2/lchen63/voxceleb/unzip/',self.data[index][0]+  '.mp4')
            cap = cv2.VideoCapture(video_path)
            for t in range(length):
                ret, frame = cap.read()
                if ret :
                    if self.gan and other_id == t:
                        mismatch_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mismatch_img = cv2.resize(mismatch_img, self.output_shape)
                        mismatch_img = self.transform(mismatch_img)
                    if t == sample_id:
                        gt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        gt_img = cv2.resize(gt_img, self.output_shape)
                        gt_img = self.transform(gt_img)
                    if t == example_id:
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.output_shape)
                        lip = self.transform(img)
                        lip_base = lip
                        lmark_base = lmark[t,:, :-1]
                    
            input_lmark = lmark[sample_id,:, :-1]
            input_dict = {'in_lmark': input_lmark ,'lmark_base': lmark_base,  'lip_base': lip_base, 'gt_img': gt_img, 'mismatch_img': mismatch_img}
            return (input_dict)   
            
#         except:
#             return self.__getitem__((index + 1)% len(self.data))
    def __len__(self):
        
            return len(self.data)  

class Voxceleb_head_movements_RT(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train'):
        self.train = train
        self.num_frames = 64  
        self.root  = '/data2/lchen63/voxceleb/'
                 
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "train_clean.pkl"), "rb")
            self.data = pickle.load(_file)[:10]
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "test.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()

        self.consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    def __getitem__(self, index):
        try:
            tmp = self.data[index][0].split('/')
            if len(self.data[index][2]) ==1:
                RT = np.load( os.path.join(self.root, 'unzip', self.data[index][0], self.data[index][2][0] + '_0RT.npy' ))
               
                middle = int(self.data[index][1])
                in_max = middle - 64
                in_start  =  np.random.choice(in_max)
                in_lmark = RT[in_start:in_start+ 64] 
               
            else:
                r  = np.random.choice(len(self.data[index][2]))
                RT = np.load( os.path.join(self.root, 'unzip', self.data[index][0], self.data[index][2][r] + '_0RT.npy' ))
                in_max = RT.shape[0] - 64
                in_start  = np.random.choice(in_max)
                in_lmark = RT[in_start:in_start + 64]
            assert in_lmark.shape[0]== 64
#             in_lmark = torch.FloatTensor(in_lmark)
#             in_lmark_part = torch.zeros((64,len(self.consider_key),3),dtype=torch.float32)
#             in_lmark_part = np.zeros((64,len(self.consider_key),3))
#             for i in range(len(self.consider_key)):
#                 in_lmark_part[:,i] = in_lmark[:,self.consider_key[i]]
#             RTs =  np.zeros((64,6))
#             for j in range(in_lmark_part.shape[0] -1 ):    
#                 source = mat(in_lmark_part[j])
#                 target = mat(in_lmark_part[j+1])
#                 ret_R, ret_t = rigid_transform_3D(source, target)
#                 r = R.from_dcm(ret_R)
#                 vec = r.as_rotvec()             
#                 RTs[j,:3] = vec
#                 RTs[j,3:] =  np.squeeze(np.asarray(ret_t))      
#             print (RTs)
            
#             input_dict = {'in_lmark': in_lmark_part,  'out_lmark': out_lmark_part,'mean': mean_lmark_part}
#             print (in_lmark.shape, out_lmark.shape, mean.shape)
#             print (in_lmark.shape)    
            return in_lmark*1.0/(np.array([1.1, 1.9, 1.0, 181.0, 102.0, 154.0]))   
        except:
            return self.__getitem__((index + 1)% len(self.data))
    def __len__(self):
        
            return len(self.data)


class Voxceleb_lip_region(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train',
                 gan = True):
        self.train = train
        self.gan = gan
        self.num_frames = 64  
        self.root  = '/data2/lchen63/voxceleb/'      
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "train_clean.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "test.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    def __getitem__(self, index):
#         try:
            tmp = self.data[index][0].split('/')
            if len(self.data[index][2]) ==1:
                lmark = np.load( os.path.join(self.root, 'unzip', self.data[index][0], self.data[index][2][0] + '.npy' ))
                tmp = os.path.join(self.data[index][0], self.data[index][2][0] ).split('/')                
                img_folder =  os.path.join(self.root, 'img', tmp[1],tmp[2], self.data[index][2][0] )
                
                middle = lmark.shape[0]
                in_max = middle - 64
                 
                in_start  =  np.random.choice(in_max)
 
                sample_id = np.random.choice([0,in_start] + [in_start+ 63, middle - 1])
                if self.gan:
                    while True:
                        other_id = np.random.choice([0,in_start] + [in_start+ 63, middle - 1])
                        if other_id != sample_id:
                            break
                    
            else:
                r  = np.random.choice(len(self.data[index][2]))
                lmark = np.load( os.path.join(self.root, 'unzip', self.data[index][0], self.data[index][2][r] + '.npy' ))
                
                tmp = os.path.join(self.data[index][0], self.data[index][2][r] ).split('/')
                img_folder =  os.path.join(self.root, 'img', tmp[1],tmp[2], self.data[index][2][r]) 
                in_max = lmark.shape[0] - 64
                in_start  = np.random.choice(in_max)
                sample_id = np.random.choice([0, in_start] + [in_start+ 63,lmark.shape[0] -1 ])
                if self.gan:
                    while True:
                        other_id = np.random.choice([0,in_start] + [in_start+ 63, lmark.shape[0] - 1])
                        if other_id != sample_id:
                            break
            lip_base = torch.FloatTensor(64*3,64,64)
            for i in range(in_start,in_start + 64):
                img_path =  os.path.join(img_folder, '%05d.png'% i)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                lip = utils.crop_mouth(img, lmark[i])
                lip = self.transform(lip)
                lip_base[(i- in_start) * 3 : (i- in_start + 1) * 3] = lip         
            input_lmark = lmark[sample_id, 48:68,:-1]
            gt_img = utils.crop_mouth(cv2.imread(os.path.join(img_folder, '%05d.png'% sample_id)), lmark[sample_id])
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            gt_img = self.transform(gt_img)
            if self.gan:
                mismatch_img = utils.crop_mouth(cv2.imread(os.path.join(img_folder, '%05d.png'% other_id)), lmark[other_id])
                mismatch_img = cv2.cvtColor(mismatch_img, cv2.COLOR_BGR2RGB)
                mismatch_img = self.transform(mismatch_img)
                input_dict = {'in_lmark': input_lmark , 'lip_base': lip_base, 'gt_img': gt_img, 'mismatch_img': mismatch_img}
            else:
                input_dict = {'in_lmark': input_lmark , 'lip_base': lip_base, 'gt_img': gt_img}
            return (input_dict)   
#         except:
#             return self.__getitem__((index + 1)% len(self.data))
    def __len__(self):
        
            return len(self.data)        


class Voxceleb_face_region(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train',
                 gan = True):
        self.train = train
        self.gan = gan
        self.output_shape   = tuple([256, 256])
        self.num_frames = 64  
        self.root  = '/data2/lchen63/voxceleb/'      
        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "train2_clean_lstm.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "test.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

#         self.consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    def __getitem__(self, index):
#         try:
            tmp = self.data[index][0].split('/')
           
            
            lmark = np.load( os.path.join(self.root, 'unzip', self.data[index][0] + '.npy' ))
            length = lmark.shape[0]
            
            lmark = torch.FloatTensor(lmark)
            in_max = length - 64
            in_start  =  np.random.choice(in_max)
            sample_id = np.random.choice([0,in_start] + [in_start+ 63, length - 1])
            if self.gan:
                while True:
                    other_id = np.random.choice([0,in_start] + [in_start+ 63, length - 1])
                    if other_id != sample_id:
                        break
                
            lip_base = torch.FloatTensor(64,3,256, 256)
            
            lmark_base = torch.FloatTensor(64,68,2)
            video_path =  os.path.join('/data2/lchen63/voxceleb/unzip/',self.data[index][0]+  '.mp4')
            cap = cv2.VideoCapture(video_path)
            for t in range(length):
                ret, frame = cap.read()
                if ret :
                    if self.gan and other_id == t:
                        mismatch_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mismatch_img = cv2.resize(mismatch_img, self.output_shape)
                        mismatch_img = self.transform(mismatch_img)
                    if t == sample_id:
                        gt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        gt_img = cv2.resize(gt_img, self.output_shape)
                        gt_img = self.transform(gt_img)
                    if t >= in_start and  t <  in_start + 64:
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.output_shape)
                        lip = self.transform(img)
                        lip_base[(t- in_start)  : (t- in_start + 1)] = lip
                        lmark_base[(t- in_start)  : (t- in_start + 1)] = lmark[t,:, :-1]
                    
            input_lmark = lmark[sample_id,:, :-1]
            input_dict = {'in_lmark': input_lmark ,'lmark_base': lmark_base,  'lip_base': lip_base, 'gt_img': gt_img, 'mismatch_img': mismatch_img}
            return (input_dict)   
            
#         except:
#             return self.__getitem__((index + 1)% len(self.data))
    def __len__(self):
        
            return len(self.data)    
        

        

# import os
# import glob
# import time
# import torch
# import torch.utils
# import torch.nn as nn
# import torchvision
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torch.nn.modules.module import _addindent
# import numpy as np
# from collections import OrderedDict
# import argparse        
# dataset = Grid_audio_lmark_single('/data/lchen63/grid/zip/txt/', 'test')
# data_loader = DataLoader(dataset,
#                           batch_size=1,
#                           num_workers=1,
#                           shuffle=False, drop_last=True)  
# t1 = time.time()
# for step, data in enumerate(data_loader):
#     print (step)
#     print ( time.time() - t1 )
#     t1 = time.time()
#     print(step)
#     print (data['img_path'])
#     if step == 1:
#         break
#     print (data['audio'].shape)
#     print (data['lip_region'].shape)
