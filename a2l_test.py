import os
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
import cv2
from logger import Logger

from dataset import  Voxceleb_audio_lmark_single  as Voxceleb_audio_lmark
# from dataset import  Voxceleb_audio_lmark_single_short as Voxceleb_audio_lmark

from torch.nn import init
from util import utils
np.set_printoptions(precision=4)
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

def test( config):            
        generator = atnet()
        l1_loss_fn =  nn.L1Loss()
        mse_loss_fn = nn.MSELoss()
        config = config

        if config.cuda:
            generator     = generator.cuda()
            mse_loss_fn   = mse_loss_fn.cuda()

        initialize_weights(generator)
        if config.dataset_name == 'vox':
            dataset =  audio_lmark('/home/cxu-serve/p1/lchen63/voxceleb/txt/', config.is_train)
        else:
            dataset =  audio_lmark('/home/cxu-serve/p1/lchen63/voxceleb/txt/', config.is_train)
        data_loader = DataLoader(dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=False, drop_last=True)
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

        num_steps_per_epoch = len(data_loader)
        cc = 0
        t0 = time.time()
        
        generator.load_state_dict( torch.load(config.model_dir))
        generator.eval()

        logger = Logger(config.log_dir)
        total_mse  = []
        total_openrate_mse = []
        
        std_total_mse  = []
        std_total_openrate_mse = []
        for step, data in enumerate(data_loader):
            print (step)
            if step == 100 :
                break
            t1 = time.time()
            sample_audio =data['audio']
            sample_lmark =  data['sample_lmark'] 
            ex_lmark = data['ex_lmark']  
            
            if config.dataset_name == 'vox':
                sample_rt = data['sample_rt'].numpy()



            if config.cuda:
                sample_audio    = Variable(sample_audio.float()).cuda()
                sample_lmark = sample_lmark.float().cuda()
                ex_lmark = ex_lmark.float().cuda()
            

            sample_lmark_pca = (sample_lmark- mean.expand_as(sample_lmark))
            
            if config.pca:
                sample_lmark_pca = torch.mm(sample_lmark_pca,  pca)

            ex_lmark_pca = (ex_lmark- mean.expand_as(ex_lmark))
            if config.pca:
                ex_lmark_pca = torch.mm(ex_lmark_pca,  pca)

            sample_lmark_pca = Variable(sample_lmark_pca)
            ex_lmark_pca = Variable(ex_lmark_pca)


            fake_lmark = generator(sample_audio,ex_lmark_pca)              
            
            
            loss  = mse_loss_fn(fake_lmark , sample_lmark_pca)
           

            logger.scalar_summary('loss', loss,   step+1)
                
            if config.pca:        
                fake_lmark =  fake_lmark.view(fake_lmark.size(0)  , 6)
                fake_lmark = torch.mm( fake_lmark, pca.t() ) 
            else:
                fake_lmark =  fake_lmark.view(fake_lmark.size(0)  , 204)
            fake_lmark +=  mean.expand_as(fake_lmark)
           

            fake_lmark = fake_lmark.data.cpu().numpy()
            
            if config.pca:        
                real_lmark =  sample_lmark_pca.view(sample_lmark_pca.size(0)  , 6)
                real_lmark = torch.mm( real_lmark, pca.t() ) 
            else:
                real_lmark =  real_lmark.view(real_lmark.size(0)  , 204)
            real_lmark +=  mean.expand_as(real_lmark)
                       
            

            real_lmark = real_lmark.data.cpu().numpy()
            
            if not os.path.exists( os.path.join(config.sample_dir, str(step)) ):
                os.mkdir(os.path.join(config.sample_dir, str(step)))

            for gg in range(int(config.batch_size)):
                
                if config.dataset_name == 'vox':
                #convert lmark to rted landmark
                    fake_A3 = utils.reverse_rt(fake_lmark[gg].reshape((68,3)), sample_rt[gg])
                    #convert lmark to rted landmark
                    A3 = utils.reverse_rt(real_lmark[gg].reshape((68,3)), sample_rt[gg])
                if config.visualize:
                    if config.dataset_name == 'vox':
                        v_path = os.path.join('/home/cxu-serve/p1/lchen63/voxceleb/unzip',data['img_path'][gg]  + '.mp4') 
                        cap = cv2.VideoCapture(v_path)
                        for t in range(real_lmark.shape[0]):
                            ret, frame = cap.read()
                            if ret :
                                if t == int(data['sample_id'][gg]) :
                                    gt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    break
                    else:
                         gt_img = cv2.cvtColor(cv2.imread(data['img_path'][gg]), cv2.COLOR_BGR2RGB)   
                    lmark_name  = "{}/fake_{}.png".format(os.path.join(config.sample_dir, str(step)), gg)
                    plt = utils.lmark2img(fake_lmark[gg].reshape((68,3)), gt_img)
                    plt.savefig(lmark_name)
                    
                    if config.dataset_name == 'vox':
                        lmark_name  = "{}/fake_rt_{}.png".format(os.path.join(config.sample_dir, str(step)), gg)
                        plt = utils.lmark2img(fake_A3.reshape((68,3)), gt_img)
                        plt.savefig(lmark_name)
                    
                    lmark_name  = "{}/real_{}.png".format(os.path.join(config.sample_dir, str(step)), gg)
                    plt = utils.lmark2img(real_lmark[gg].reshape((68,3)), gt_img)
                    plt.savefig(lmark_name)
                
                    if config.dataset_name == 'vox':
                        lmark_name  = "{}/real_rt_{}.png".format(os.path.join(config.sample_dir, str(step)), gg)
                        plt = utils.lmark2img(A3.reshape((68,3)), gt_img)
                        plt.savefig(lmark_name)
                if config.dataset_name == 'vox':
                    mse = utils.mse_metrix(A3, fake_A3)
                    openrate = utils.openrate_metrix(A3, fake_A3)
                    if mse > 50 or openrate > 50:
                        continue
                    total_openrate_mse.append(openrate)
                    total_mse.append(mse)
                
                std_mse = utils.mse_metrix(real_lmark[gg].reshape((68,3)), fake_lmark[gg].reshape((68,3)))
                std_openrate = utils.openrate_metrix(real_lmark[gg].reshape((68,3)), fake_lmark[gg].reshape((68,3)))
                std_total_openrate_mse.append(std_openrate)
                std_total_mse.append(std_mse)
                
                
                #compute different evaluation matrix  input: (68,3) real A3, fake A3 
        if config.dataset_name == 'vox':
            total_mse = np.asarray(total_mse)
            total_openrate_mse = np.asarray(total_openrate_mse)
        
        std_total_mse = np.asarray(std_total_mse)
        std_total_openrate_mse = np.asarray(std_total_openrate_mse)
        
        
        if config.dataset_name == 'vox':
            print (total_mse.mean())
            print (total_openrate_mse.mean())
        print (std_total_mse.mean())
        print (std_total_openrate_mse.mean())
                

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.0002)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--lambda1",
                        type=int,
                        default=100)
    parser.add_argument("--batch_size",
                        type=int,
                        default=512)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=10000)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/home/cxu-serve/p1/lchen63/voxceleb/txt/'")
    parser.add_argument("--model_name",
                        type=str,
                        default="at")
    parser.add_argument("--model_dir",
                        type=str,
                        default="./model/at/anet_single.pth")
    parser.add_argument("--sample_dir",
                        type=str,
                        default="./test_result/")   
    parser.add_argument('--dataset_name', type=str, default='vox')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--pca', type=bool, default=True, help='True or False')
    parser.add_argument('--openrate', type=bool, default=True, help='True or False')
    parser.add_argument('--visualize', type=bool, default=True, help='True or False')
    return parser.parse_args()



if __name__ == "__main__":
    config = parse_args()
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
    config.is_train = 'test'
            
    from ATVG import AT_single as atnet
    from dataset import  Voxceleb_audio_lmark_single  as audio_lmark
    config.model_dir = "./model/at/anet_single.pth"

    
            
    config.sample_dir = os.path.join(config.sample_dir, config.model_name)
    config.log_dir = os.path.join('./logs', config.model_name)
    print (config.model_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    print (config.sample_dir)
    config.cuda1 = torch.device('cuda:{}'.format(config.device_ids))
    test(config)


