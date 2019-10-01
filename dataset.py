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
import mmcv

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


def create_rectangle_mask(  xcorners, ycorners   ):
    '''
    Give image and x/y coners to create a rectangle mask    
    image: 2d array
    xcorners, list, points of x coners
    ycorners, list, points of y coners
    Return:
    the polygon mask: 2d array, the polygon pixels with values 1 and others with 0
    
    Example:
    
    
    '''
    from skimage.draw import line_aa, line, polygon, circle    
    bst_mask = np.ones( (224,224,1) , dtype = float32)   
    rr, cc = polygon( ycorners,xcorners)
    bst_mask[ rr,cc, 0] = 0    
    #full_mask= ~bst_mask    
    return bst_mask 


class Voxceleb_mfcc_rgb_single(data.Dataset):
    def __init__(self, dataset_dir, train='train'):
        self.train = train
        self.output_shape   = tuple([256, 256])
        self.num_frames = 64  
        self.root  = dataset_dir    
        if self.train =='train':
            _file = open(os.path.join(dataset_dir, 'txt', "front_rt2.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, 'txt', "front_rt2.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        print (len(self.data))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        # self.lip_region= [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    def __getitem__(self, index):
        
        v_id = self.data[index][0]
        reference_id = self.data[index][1]

        video_path = os.path.join(self.root, 'unzip', v_id + '.mp4')
        
        ani_video_path = os.path.join(self.root, 'unzip', v_id + '_ani.mp4')

        mfcc_path = os.path.join(self.root, 'unzip',v_id.replace('video','audio')+'.npy' )

        lmark_path = os.path.join(self.root, 'unzip', v_id + '.npy')

        

        

        real_video  = mmcv.VideoReader(video_path)
        ani_video = mmcv.VideoReader(ani_video_path)

        v_length = len(real_video)

        mfcc = np.load(mfcc_path)
        audio_length = mfcc.shape[0]

        # if mfcc length is not v_length * 4 , we pad 0s at the end of mfcc.
        audio_pad = np.zeros((v_length * 4, 13))
        if audio_length < v_length * 4 :
            audio_pad[:audio_length] = mfcc
            mfcc = audio_pad
        
        # we randomly choose a target frame
        while True:
            target_id =  np.random.choice([0, v_length - 1])
            if target_id != reference_id:
                break
        
        target_rgb = real_video[target_id]

        # if target frame is at the begining or end, we need to pad 0s to mfcc.
        if target_id < 3:
            sample_audio = np.zeros((28,12))
            sample_audio[4 * (3- target_id): ] = mfcc[4 * (0) : 4 * ( 3 + target_id + 1 ), 1: ]

        elif target_id > v_length - 4:
            sample_audio = np.zeros((28,12))
            sample_audio[:4 * ( v_length + 3 - target_id )] = mfcc[4 * (target_id -3) : 4 * ( v_length ), 1: ]

        else:
            sample_audio = mfcc[4 * (target_id -3) : 4 * ( target_id + 4 ) , 1: ]

        reference_rgb = real_video[reference_id]

        reference_ani = ani_video[reference_id]

        target_ani = ani_video[target_id]
        
       
        # target_lmark = np.load(lmark_path)[target_id][48:60,:-1]
        # mask = create_rectangle_mask(target_lmark[:,0],target_lmark[:,1])
#   
#         mask = mask.astype(float32)
# # 
        # mask = cv2.resize(mask , self.output_shape)  
        # mask = np.expand_dims(mask, axis=2)
        # print (np.unique(mask))

        target_rgb = mmcv.bgr2rgb(target_rgb)
        # target_rgb = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2RGB)
        target_rgb = cv2.resize(target_rgb, self.output_shape)
        target_rgb = self.transform(target_rgb)

        target_ani = mmcv.bgr2rgb(target_ani)
        target_ani = cv2.resize(target_ani, self.output_shape)
        # target_ani =  target_ani * mask
        target_ani = self.transform(target_ani)
        mask = target_ani[0].clone()
        
        mask = mask >  -0.9
        mask = mask.type(torch.FloatTensor)

    

        # target_ani = ((target_ani + 1 )/2.0
        # print (target_ani.max())
        # print (target_ani.min())

        # mask  = target_ani[0] > 0
        # mask = mask.float()
        # print (mask.shape)
        # print (mask.max())
        # print(mask.min())


        reference_rgb = mmcv.bgr2rgb(reference_rgb)
        # reference_rgb = cv2.cvtColor(reference_rgb, cv2.COLOR_BGR2RGB)
        reference_rgb = cv2.resize(reference_rgb, self.output_shape)
        reference_rgb = self.transform(reference_rgb)
        
        reference_ani = mmcv.bgr2rgb(reference_ani)
        # reference_ani = cv2.cvtColor(reference_ani, cv2.COLOR_BGR2RGB)
        reference_ani = cv2.resize(reference_ani, self.output_shape)
        reference_ani = self.transform(reference_ani)


        sample_audio =torch.FloatTensor(sample_audio)
        sample_audio = sample_audio.unsqueeze(0)
     
        
        final_img = reference_rgb * torch.abs(1 - mask) + (target_ani) * mask
       
        ### we will not write mismatch in this version.
        input_dict = { 'v_id' : v_id, 'reference_rgb': reference_rgb ,'reference_ani': reference_ani,
                        'sample_audio': sample_audio, 'target_rgb': target_rgb, 'target_ani': target_ani, 'final_img': final_img}
        # input_dict = { 'id' : v_id, 'reference_rgb': reference_rgb ,'reference_ani': reference_ani,
        #                 'sample_audio': sample_audio, 'target_rgb': target_rgb, 'target_ani': target_ani, 'img' : img}
        return (input_dict)   
    def __len__(self):        
            return len(self.data)

class Voxceleb_lmark_rgb_single(data.Dataset):
    def __init__(self, dataset_dir, train='train', cuda = False):
        self.train = train
        self.output_shape   = tuple([256, 256])
        self.num_frames = 64  
        self.cuda = cuda
        self.root  = dataset_dir     
        if self.train =='train':
            _file = open(os.path.join(dataset_dir, 'txt',  "front_rt2.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, 'txt', "front_rt2.pkl"), "rb")
            self.data = pickle.load(_file)
            _file.close()
        print (len(self.data))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        # self.lip_region= [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    def __getitem__(self, index):
        try:
            v_id = self.data[index][0]
            reference_id = self.data[index][1]

            video_path = os.path.join(self.root, 'unzip', v_id + '.mp4')
            
            ani_video_path = os.path.join(self.root, 'unzip', v_id + '_ani.mp4')


            lmark_path = os.path.join(self.root, 'unzip', v_id + '.npy')


            lmark = np.load(lmark_path)



            real_video  = mmcv.VideoReader(video_path)
            ani_video = mmcv.VideoReader(ani_video_path)

            v_length = len(ani_video)

            
            # we randomly choose a target frame
            while True:
                target_id =  np.random.choice([0, v_length - 1])
                if target_id != reference_id:
                    break
            
            target_rgb = real_video[target_id]

            reference_rgb = real_video[reference_id]

            reference_ani = ani_video[reference_id]

            target_ani = ani_video[target_id]
            
            target_rgb = mmcv.bgr2rgb(target_rgb)
            # target_rgb = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2RGB)
            target_rgb = cv2.resize(target_rgb, self.output_shape)
            target_rgb = self.transform(target_rgb)
            if self.cuda:
                target_rgb = target_rgb.cuda()

            target_ani = mmcv.bgr2rgb(target_ani)
            target_ani = cv2.resize(target_ani, self.output_shape)
            # target_ani =  target_ani * mask
            target_ani = self.transform(target_ani)
            if self.cuda:
                target_ani = target_ani.cuda()
            mask = target_ani[0].clone()
            
            mask = mask >  -0.9
            mask = mask.type(torch.FloatTensor)
            if self.cuda:
                mask = mask.cuda()
            


            reference_rgb = mmcv.bgr2rgb(reference_rgb)
            # reference_rgb = cv2.cvtColor(reference_rgb, cv2.COLOR_BGR2RGB)
            reference_rgb = cv2.resize(reference_rgb, self.output_shape)
            reference_rgb = self.transform(reference_rgb)
            if self.cuda:
                reference_rgb = reference_rgb.cuda()
            
            reference_ani = mmcv.bgr2rgb(reference_ani)
            # reference_ani = cv2.cvtColor(reference_ani, cv2.COLOR_BGR2RGB)
            reference_ani = cv2.resize(reference_ani, self.output_shape)
            reference_ani = self.transform(reference_ani)



            target_lmark  =torch.FloatTensor(lmark[target_id]).view(-1)
            if self.cuda:
                target_lmark = target_lmark.cuda()
            
        
            
            final_img = reference_rgb * torch.abs(1 - mask) + (target_ani) * mask
            if self.cuda:
                final_img = final_img.cuda()
        
            ### we will not write mismatch in this version.
            input_dict = { 'v_id' : v_id, 'reference_rgb': reference_rgb ,'reference_ani': reference_ani,
                            'target_lmark': target_lmark, 'target_rgb': target_rgb, 'target_ani': target_ani, 'final_img': final_img}
            # input_dict = { 'id' : v_id, 'reference_rgb': reference_rgb ,'reference_ani': reference_ani,
            #                 'sample_audio': sample_audio, 'target_rgb': target_rgb, 'target_ani': target_ani, 'img' : img}
            return (input_dict)   
        except:
            self.__getitem__((index+1)%(self.__len__))
    def __len__(self):        
            return len(self.data)        
# import torchvision
# data = torch.Tensor(100, 1, 24, 24).random_(0, 255)
# print(data)

# # Apply threshold
# data = data > 128
# data = data.float()
# print(data)

# # Create fake masks for every image
# mask = torch.Tensor(100, 1, 24, 24).random_(0, 2)
# data = data * mask
# import time
# dataset = Voxceleb_mfcc_rgb_single('/home/cxu-serve/p1/lchen63/voxceleb/txt/', 'train')
# data_loader = data.DataLoader(dataset,
#                           batch_size=1,
#                           num_workers=4,
#                           shuffle=False, drop_last=True)  
# t1 = time.time()
# print (len(data_loader))
# for (step, gg)  in enumerate(data_loader):
#     print (time.time() -  t1)
#     print  (gg['v_id'])
#     inputs = [gg['final_img'], gg['reference_rgb'], gg['target_ani'], gg['target_rgb']]
#     fake_im = torch.stack(inputs, dim = 1)
#     fake_store = fake_im.data.contiguous().view(4*1,3,256,256)
#     torchvision.utils.save_image(fake_store, 
#         "./tmp/vis_%05d.png"%step,normalize=True)
    
#     if step == 10:
#         break
   




class Voxceleb_audio_lmark_single(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 train='train',
                 gan = True):
        self.train = train  
        # self.root = '/data2/lchen63/voxceleb/'
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
                 train='train'):
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
        

        

