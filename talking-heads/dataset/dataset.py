"""
This package performs the pre-processing of the VoxCeleb dataset in order to have it ready for training, speeding the
process up.
"""
import logging
import os
from datetime import datetime
import pickle as pkl
import random
from multiprocessing import Pool

import PIL
import cv2
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt


import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import mmcv


# region DATASET PREPARATION


# region DATASET RETRIEVAL


class Lmark2rgbDataset(Dataset):
    """ Dataset object used to access the pre-processed VoxCelebDataset """

    def __init__(self, dataset_dir, train = 'train'):
        """
        Instantiates the Dataset.

        :param root: Path to the folder where the pre-processed dataset is stored.
        :param extension: File extension of the pre-processed video files.
        :param shuffle: If True, the video files will be shuffled.
        :param transform: Transformations to be done to all frames of the video files.
        :param shuffle_frames: If True, each time a video is accessed, its frames will be shuffled.
        """
        self.train = train
        self.output_shape   = tuple([256, 256])
        self.num_frames = 64  
        self.root  = dataset_dir     
        if self.train =='train':
            _file = open(os.path.join(dataset_dir, 'txt',  "front_rt2.pkl"), "rb")
            # self.data = pkl.load(_file)
            self.data = pkl._Unpickler(_file)
            self.data.encoding = 'latin1'
            self.data = self.data.load()
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, 'txt', "front_rt2.pkl"), "rb")
            self.data = pkl._Unpickler(_file)
            self.data.encoding = 'latin1'
            self.data = self.data.load()
            # self.data = pkl.load(_file)
            _file.close()
        print (len(self.data))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.data)   

    def __getitem__(self, index):
        print ('+++++++++++++++++++++++++++')
        v_id = self.data[index][0]
        reference_id = self.data[index][1]

        video_path = os.path.join(self.root, 'unzip', v_id + '.mp4')
        
        ani_video_path = os.path.join(self.root, 'unzip', v_id + '_ani.mp4')


        lmark_path = os.path.join(self.root, 'unzip', v_id + '.npy')


        lmark = np.load(lmark_path)[:,:,:-1]

        v_length = lmark.shape[0]

        real_video  = mmcv.VideoReader(video_path)
        ani_video = mmcv.VideoReader(ani_video_path)

        # sample frames for embedding network
        input_indexs  = set(random.sample(range(0,64), 8))

        # we randomly choose a target frame 
        target_id =  np.random.choice([0, v_length - 1])
            
        reference_frames = []
        for t in input_indexs:
            rgb_t =  mmcv.bgr2rgb(real_video[t]) 
            lmark_t = lmark[t]
            lmark_rgb = plot_landmarks( lmark_t)
            # lmark_rgb = np.array(lmark_rgb) 

            # resize 224 to 256
            rgb_t  = cv2.resize(rgb_t, self.output_shape)
            lmark_rgb  = cv2.resize(lmark_rgb, self.output_shape)
            
            # to tensor
            rgb_t = self.transform(rgb_t)
            lmark_rgb = self.transform(lmark_rgb)


            reference_frames.append(torch.stack([rgb_t, lmark_rgb]))  # (6, 256, 256)   

        reference_frames = torch.stack(reference_frames)
        ############################################################################
        target_rgb = real_video[target_id]
        reference_rgb = real_video[reference_id]
        reference_ani = ani_video[reference_id]
        target_ani = ani_video[target_id]
        target_lmark = lmark[target_id]

        target_rgb = mmcv.bgr2rgb(target_rgb)
        target_rgb = cv2.resize(target_rgb, self.output_shape)
        target_rgb = self.transform(target_rgb)

        target_ani = mmcv.bgr2rgb(target_ani)
        target_ani = cv2.resize(target_ani, self.output_shape)
        target_ani = self.transform(target_ani)

        # reference_rgb = mmcv.bgr2rgb(reference_rgb)
        # reference_rgb = cv2.resize(reference_rgb, self.output_shape)
        # reference_rgb = self.transform(reference_rgb)

        # reference_ani = mmcv.bgr2rgb(reference_ani)
        # reference_ani = cv2.resize(reference_ani, self.output_shape)
        # reference_ani = self.transform(reference_ani)

        target_lmark = plot_landmarks(target_lmark)
        # target_lmark = np.array(target_lmark) 
        target_lmark  = cv2.resize(target_lmark, self.output_shape)
        target_lmark  = cv2.resize(target_lmark, self.output_shape)
        target_lmark = self.transform(target_lmark)


        input_dic = {'v_id' : v_id, 'target_lmark': target_lmark, 'reference_frames': reference_frames,
        'target_rgb': target_rgb, 'target_ani': target_ani
        }
        print ('--')
        return input_dic


def plot_landmarks1( landmarks):
    """
    Creates an RGB image with the landmarks. The generated image will be of the same size as the frame where the face
    matching the landmarks.

    The image is created by plotting the coordinates of the landmarks using matplotlib, and then converting the
    plot to an image.

    Things to watch out for:
    * The figure where the landmarks will be plotted must have the same size as the image to create, but matplotlib
    only accepts the size in inches, so it must be converted to pixels using the DPI of the screen.
    * A white background is printed on the image (an array of ones) in order to keep the figure from being flipped.
    * The axis must be turned off and the subplot must be adjusted to remove the space where the axis would normally be.

    :param frame: Image with a face matching the landmarks.
    :param landmarks: Landmarks of the provided frame,
    :return: RGB image with the landmarks as a Pillow Image.
    """
    dpi = 100
    fig = plt.figure(figsize=(224/ dpi,224 / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(np.ones((224,224)))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Head
    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=2)
    # Eyebrows
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=2)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=2)
    # Nose
    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=2)
    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=2)
    # Eyes
    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=2)
    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=2)
    # Mouth
    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=2)

    fig.canvas.draw()
    data = PIL.Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)
    plt.close(fig)
    return data


def plot_landmarks( landmarks):
    # landmarks = np.int32(landmarks)
    """
    Creates an RGB image with the landmarks. The generated image will be of the same size as the frame where the face
    matching the landmarks.

    The image is created by plotting the coordinates of the landmarks using matplotlib, and then converting the
    plot to an image.

    Things to watch out for:
    * The figure where the landmarks will be plotted must have the same size as the image to create, but matplotlib
    only accepts the size in inches, so it must be converted to pixels using the DPI of the screen.
    * A white background is printed on the image (an array of ones) in order to keep the figure from being flipped.
    * The axis must be turned off and the subplot must be adjusted to remove the space where the axis would normally be.

    :param frame: Image with a face matching the landmarks.
    :param landmarks: Landmarks of the provided frame,
    :return: RGB image with the landmarks as a Pillow Image.
    """
    # print (landmarks[0:17].shape)
    # print(type(landmarks))

    # points = np.array([[1, 4], [5, 6], [7, 8], [4, 4]])
    # print (points.shape)


    blank_image = np.zeros((224,224,3), np.uint8)

    # cv2.polylines(blank_image, np.int32([points]), True, (0,255,255), 1)

    cv2.polylines(blank_image, np.int32([landmarks[0:17]]) , True, (0,255,255), 1)
 
    cv2.polylines(blank_image,  np.int32([landmarks[17:22]]), True, (255,0,255), 1)

    cv2.polylines(blank_image, np.int32([landmarks[22:27]]) , True, (255,0,255), 1)

    cv2.polylines(blank_image, np.int32([landmarks[27:31]]) , True, (255,255, 0), 1)

    cv2.polylines(blank_image, np.int32([landmarks[31:36]]) , True, (255,255, 0), 1)

    cv2.polylines(blank_image, np.int32([landmarks[36:42]]) , True, (255,0, 0), 1)
    cv2.polylines(blank_image, np.int32([landmarks[42:48]]) , True, (255,0, 0), 1)

    cv2.polylines(blank_image, np.int32([landmarks[48:60]]) , True, (0, 0, 255), 1)





    return blank_image




# import torchvision


# import time
# dataset = Lmark2rgbDataset('/home/cxu-serve/p1/lchen63/voxceleb/', 'train')
# data_loader = torch.utils.data.DataLoader(dataset,
#                           batch_size=1,
#                           num_workers=1,
#                           shuffle=False, drop_last=True)  
# t1 = time.time()
# print (len(data_loader))
# for (step, gg)  in enumerate(data_loader):
#     print (time.time() -  t1)
#     print  (gg['v_id'])
#     print (gg['reference_frames'].shape)
#     inputs = [gg['target_rgb'], gg['target_ani'], gg['target_lmark'], gg['target_rgb']]
#     fake_im = torch.stack(inputs, dim = 1)
#     fake_store = fake_im.data.contiguous().view(4*1,3,256,256)
#     torchvision.utils.save_image(fake_store, 
#         "./tmp/vis_%05d.png"%step,normalize=True)
    
#     if step == 10:
#         break
   