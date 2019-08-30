import argparse
import os
import glob
import time
import numpy as np
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import librosa
from ATVG import AT_net
import cv2
import scipy.misc
from util import utils
from tqdm import tqdm
import torchvision.transforms as transforms
import shutil
from collections import OrderedDict
import python_speech_features
from skimage import transform as tf
from copy import deepcopy
from scipy.spatial import procrustes
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import dlib

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",
                     type=int,
                     default=1)
    parser.add_argument("--cuda",
                     default=True)
    parser.add_argument("--lstm",
                     default=True)
    parser.add_argument("--vg_model",
                     type=str,
                     default="./model/generator_23.pth")
    parser.add_argument("--at_model",
                     type=str,
                     default="./model/atnet/atnet_lstm.pth")
    parser.add_argument( "--sample_dir",
                    type=str,
                    default="./results")
    parser.add_argument('-i','--in_file', type=str, default='./audio/1.wav')
    parser.add_argument('-d','--data_path', type=str, default='./basics')
    parser.add_argument('-p','--person', type=str, default='./image/test.png')
    parser.add_argument('-rt','--rts', type=str, default='./video/1_crop_sRT.npy')
    parser.add_argument('--device_ids', type=str, default='2')
    parser.add_argument('--num_thread', type=int, default=1)   
    return parser.parse_args()
config = parse_args()


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./basics/shape_predictor_68_face_landmarks.dat')
ms_img = np.load('./basics/mean_shape_img.npy')
ms_norm = np.load('./basics/mean_shape_norm.npy')
S = np.load('./basics/S.npy')

MSK = np.reshape(ms_norm, [1, 68*2])
SK = np.reshape(S, [1, S.shape[0], 68*2])





def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def normLmarks(lmarks):
    norm_list = []
    idx = -1
    max_openness = 0.2
    mouthParams = np.zeros((1, 100))
    mouthParams[:, 1] = -0.06
    tmp = deepcopy(MSK)
    tmp[:, 48*2:] += np.dot(mouthParams, SK)[0, :, 48*2:]
    open_mouth_params = np.reshape(np.dot(S, tmp[0, :] - MSK[0, :]), (1, 100))
    if len(lmarks.shape) == 2:
        lmarks = lmarks.reshape(1,68,2)
    for i in range(lmarks.shape[0]):
        mtx1, mtx2, disparity = procrustes(ms_img, lmarks[i, :, :])
        mtx1 = np.reshape(mtx1, [1, 136])
        mtx2 = np.reshape(mtx2, [1, 136])
        norm_list.append(mtx2[0, :])
    pred_seq = []
    init_params = np.reshape(np.dot(S, norm_list[idx] - mtx1[0, :]), (1, 100))
    for i in range(lmarks.shape[0]):
        params = np.reshape(np.dot(S, norm_list[i] - mtx1[0, :]), (1, 100)) - init_params - open_mouth_params
        predicted = np.dot(params, SK)[0, :, :] + MSK
        pred_seq.append(predicted[0, :])
    return np.array(pred_seq), np.array(norm_list), 1
   
def crop_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = utils.shape_to_np(shape)
        (x, y, w, h) = utils.rect_to_bb(rect)
        center_x = x + int(0.5 * w)
        center_y = y + int(0.5 * h)
        r = int(0.64 * h)
        new_x = center_x - r
        new_y = center_y - r
        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]

        roi = cv2.resize(roi, (163,163), interpolation = cv2.INTER_AREA)
        scale =  163. / (2 * r)

        shape = ((shape - np.array([new_x,new_y])) * scale)

        return roi, shape 
def generator_demo_example_lips(img_path):
    name = img_path.split('/')[-1]
    landmark_path = os.path.join('./image/', name.replace('png', 'npy')) 
    region_path = os.path.join('./image/', name.replace('.png', '_region.png')) 
    roi, landmark= crop_image(img_path)
    if  np.sum(landmark[37:39,1] - landmark[40:42,1]) < -9:

        # pts2 = np.float32(np.array([template[36],template[45],template[30]]))
        template = np.load( './basics/base_68.npy')
    else:
        template = np.load( './basics/base_68_close.npy')
    # pts2 = np.float32(np.vstack((template[27:36,:], template[39,:],template[42,:],template[45,:])))
    pts2 = np.float32(template[27:45,:])
    # pts2 = np.float32(template[17:35,:])
    # pts1 = np.vstack((landmark[27:36,:], landmark[39,:],landmark[42,:],landmark[45,:]))
    pts1 = np.float32(landmark[27:45,:])
    # pts1 = np.float32(landmark[17:35,:])
    tform = tf.SimilarityTransform()
    tform.estimate( pts2, pts1)
    dst = tf.warp(roi, tform, output_shape=(163, 163))

    dst = np.array(dst * 255, dtype=np.uint8)
    dst = dst[1:129,1:129,:]
    cv2.imwrite(region_path, dst)


    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = utils.shape_to_np(shape)
        shape, _ ,_ = normLmarks(shape)
        np.save(landmark_path, shape)
        lmark= shape.reshape(68,2)
        name = region_path.replace('region.jpg','lmark.png')

        utils.plot_flmarks(lmark, name, (-0.2, 0.2), (-0.2, 0.2), 'x', 'y', figsize=(10, 10))
    return dst, lmark


# def vis(lmark):
       
       
#     preds = lmark
#     #TODO: Make this nice
#     fig = plt.figure(figsize=plt.figaspect(.5))
#     ax = fig.add_subplot(1, 2, 1)

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
#     print ( preds[:17,2])
#     print ( preds[:17,1])
#     print ( preds[:17,0])
#     ax = fig.add_subplot(1, 2, 2, projection='3d')
#     surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
#     ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
#     ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
#     ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
#     ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
#     ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
#     ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
#     ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
#     ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )

#     ax.view_init(elev=90., azim=90.)
#     ax.set_xlim(ax.get_xlim()[::-1])
# #     plt.show()
#     return plt
#     print ('=======')
def vis(lmark):
    preds = lmark
    preds = np.asarray(preds)
   
  
    
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 1, 1)       
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.scatter(preds[:,0],preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
    ax.plot3D(preds[:17,0],preds[:17,1], preds[:17,2], color='gray' )
    ax.plot3D(preds[17:22,0],preds[17:22,1],preds[17:22,2], color='gray')
    ax.plot3D(preds[22:27,0],preds[22:27,1],preds[22:27,2], color='gray')
    ax.plot3D(preds[27:31,0],preds[27:31,1],preds[27:31,2], color='gray')
    ax.plot3D(preds[31:36,0],preds[31:36,1],preds[31:36,2], color='gray')
    ax.plot3D(preds[36:42,0],preds[36:42,1],preds[36:42,2], color='gray')
    ax.plot3D(preds[42:48,0],preds[42:48,1],preds[42:48,2], color='gray')
    ax.plot3D(preds[48:,0],preds[48:,1],preds[48:,2], color='gray' )

    ax.view_init(elev=45., azim= 60.)
#     ax.set_xlim(ax.get_xlim()[::-1])
#     plt.show();
    return plt

def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
    if os.path.exists('./temp'):
        shutil.rmtree('./temp')
        
    if os.path.exists('./lmark'):
        shutil.rmtree('./lmark')
        os.mkdir('./lmark')
    if not os.path.exists('./lmark/real'):
        os.mkdir('./lmark/real')
    if not os.path.exists('./lmark/fake'):
        os.mkdir('./lmark/fake')
    if not os.path.exists('./lmark/fake_rt'):
        os.mkdir('./lmark/fake_rt')
        os.mkdir('./lmark/fake_rt_3d')
    os.mkdir('./temp')
    os.mkdir('./temp/img')
    pca = torch.FloatTensor( np.load('./basics/U_lrw1.npy')[:,:6]).cuda()
    mean =torch.FloatTensor( np.load('./basics/mean_lrw1.npy')).cuda()
    
    grid_mean =torch.FloatTensor( np.load('./basics/mean_grid.npy')).cuda()

    grid_std = torch.FloatTensor( np.load('./basics/std_grid.npy')).cuda()
    grid_pca = torch.FloatTensor( np.load('./basics/U_grid.npy')).cuda()
    
    encoder = AT_net()
    if config.cuda:
        encoder = encoder.cuda()

    state_dict = multi2single(config.at_model, 1)
    encoder.load_state_dict(state_dict)

    encoder.eval()

    test_file = config.in_file

    example_image, example_landmark = generator_demo_example_lips( config.person)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
     ])        
    example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
    example_image = transform(example_image)

    example_landmark =  example_landmark.reshape((1,example_landmark.shape[0]* example_landmark.shape[1]))

    if config.cuda:
        example_image = Variable(example_image.view(1,3,128,128)).cuda()
        example_landmark = Variable(torch.FloatTensor(example_landmark.astype(float)) ).cuda()
    else:
        example_image = Variable(example_image.view(1,3,128,128))
        example_landmark = Variable(torch.FloatTensor(example_landmark.astype(float)))
    # Load speech and extract features
    example_landmark = example_landmark * 5.0
    example_landmark  = example_landmark - mean.expand_as(example_landmark)
    example_landmark = torch.mm(example_landmark,  pca)
    speech, sr = librosa.load(test_file, sr=16000)
    mfcc = python_speech_features.mfcc(speech ,16000,winstep=0.01)
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)

    sound, _ = librosa.load(test_file, sr=44100)

    print ('=======================================')
    print ('Start to generate images')
    t =time.time()
    ind = 3
    RTs = np.load(config.rts)
    gt_front = np.load(config.rts.replace('sRT', 'front'))
    gt = np.load(config.rts.replace('_sRT',''))
    with torch.no_grad(): 
        fake_lmark = []
        input_mfcc = []
        while ind <= int(mfcc.shape[0]/4) - 4:
            t_mfcc =mfcc[( ind - 3)*4: (ind + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc).cuda()
            input_mfcc.append(t_mfcc)
            ind += 1
        input_mfcc = torch.stack(input_mfcc,dim = 0)
        input_mfcc = input_mfcc.unsqueeze(0)
        fake_lmark = encoder(example_landmark, input_mfcc)
        fake_lmark = fake_lmark.view(fake_lmark.size(0) *fake_lmark.size(1) , 6)
        example_landmark  = torch.mm( example_landmark, pca.t() ) 
        example_landmark = example_landmark + mean.expand_as(example_landmark)
        fake_lmark[:, 1:6] *= 2*torch.FloatTensor(np.array([1.1, 1.2, 1.3, 1.4, 1.5])).cuda() 
        fake_lmark = torch.mm( fake_lmark, pca.t() )
        fake_lmark = fake_lmark + mean.expand_as(fake_lmark)
    


       

        print ( 'In total, generate {:d} images, cost time: {:03f} seconds'.format(fake_lmark.size(0), time.time() - t) )
        
        fake_lmark = fake_lmark.data.cpu().numpy()
        np.save( os.path.join( config.sample_dir,  'obama_fake.npy'), fake_lmark)
        fake_lmark3d = np.zeros((fake_lmark.shape[0], 68, 3))      
        fake_lmark = np.reshape(fake_lmark, (fake_lmark.shape[0], 68, 2))
        fake_lmark3d[:,:,:-1] =  fake_lmark * 200
        meanshape = np.load('./basics/mean_grid.npy').reshape(68,3)
        fake_lmark3d[:,:,-1] += meanshape[:,-1]
        fake_lmark = fake_lmark3d
        
        gt_front = torch.FloatTensor(gt_front.reshape(gt_front.shape[0],-1)).cuda()
        gt_front = gt_front - grid_mean.expand_as(gt_front)
        gt_front = gt_front / grid_std
        gt_front = torch.mm(gt_front,  grid_pca)
        
        gt_front  = torch.mm( gt_front, grid_pca.t() ) * grid_std
        
        gt_front = gt_front + grid_mean.expand_as(gt_front)
        
        
        gt_front = gt_front.cpu().numpy()
        
        for gg in range(fake_lmark.shape[0]):
            backgroung = cv2.imread('./image/background.png')
            backgroung= cv2.cvtColor(backgroung, cv2.COLOR_BGR2RGB) 
            lmark_name  = "./lmark/fake/%05d.png"%gg
            plt = utils.lmark2img(fake_lmark[gg].reshape((68,3)), backgroung, c = 'b')
            plt.savefig(lmark_name)
            A3 = utils.reverse_rt(fake_lmark[gg].reshape((68,3)), RTs[gg])
            lmark_name  = "./lmark/fake_rt/%05d.png"%gg
            plt = utils.lmark2img(A3.reshape((68,3)), backgroung, c = 'b')
            plt.savefig(lmark_name)
            
            lmark_name  = "./lmark/fake_rt_3d/%05d.png"%gg
            plt = vis(A3.reshape(68,3))
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
        
        video_name = os.path.join(config.sample_dir , 'fake_rt_3d.mp4')
        utils.image_to_video(os.path.join('./', 'lmark/fake_rt_3d'), video_name )
        utils.add_audio(video_name, config.in_file)
        print ('The generated video is: {}'.format(os.path.join(config.sample_dir , 'fake_rt_3d.mov')))
        
        
        
        for gg in range(gt_front.shape[0]):
            backgroung = cv2.imread('./image/background.png')
#             backgroung = cv2.imread('./tmp/%04d.png'%(gg+ 1))
            backgroung= cv2.cvtColor(backgroung, cv2.COLOR_BGR2RGB) 
            lmark_name  = "./lmark/real/%05d.png"%gg
            plt = utils.lmark2img(gt_front[gg].reshape((68,3)), backgroung, c = 'b')
            plt.savefig(lmark_name)
            
          
            if not os.path.exists('./lmark/original'):
                os.mkdir ('./lmark/original')
            lmark_name  = "./lmark/original/%05d.png"%gg
            plt = utils.lmark2img(gt[gg].reshape((68,3)), backgroung, c = 'b')
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