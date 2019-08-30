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
from logger import Logger
import shutil
from torch.nn import init
# torch.set_printoptions(precision=5,sci_mode=False)
import random
from dataset import  Voxceleb_face_region
from ATVG import FaceNet2, FaceNet_Discriminator
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

def initialize_weights( net, init_type='normal', gain=0.0002):
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

class Trainer():
    def __init__(self, config):            
        self.generator = FaceNet2( )
        
        self.l1_loss_fn =  nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()
        self.config = config
        if self.config.gan:
            self.discriminator = FaceNet_Discriminator()
            self.bce_loss_fn = nn.BCELoss()
            self.ones = Variable(torch.ones(config.batch_size), requires_grad=False)
            self.zeros = Variable(torch.zeros(config.batch_size), requires_grad=False)
        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(',')]
            self.generator = nn.DataParallel(self.generator, device_ids=device_ids).cuda()
            if config.gan:
                self.discriminator     = nn.DataParallel(self.discriminator, device_ids=device_ids).cuda()
                self.bce_loss_fn = self.bce_loss_fn.cuda(device=config.cuda1)
            self.mse_loss_fn   = self.mse_loss_fn.cuda(device=config.cuda1)
            self.l1_loss_fn = self.l1_loss_fn.cuda(device=config.cuda1)
            self.ones          = self.ones.cuda(device=config.cuda1)
            self.zeros          = self.zeros.cuda(device=config.cuda1)
# #########single GPU#######################

#         if config.cuda:
#             self.generator     = self.generator.cuda()
#             if config.gan:
#                 self.discriminator     = self.discriminator.cuda()
#                 self.bce_loss_fn = self.bce_loss_fn.cuda()
#             self.ones          = self.ones.cuda()
#             self.zeros          = self.zeros.cuda()
#             self.mse_loss_fn   = self.mse_loss_fn.cuda()
#             self.l1_loss_fn =  nn.L1Loss().cuda()
        initialize_weights(self.generator)
        self.start_epoch = 0
        if config.load_model:
            self.start_epoch = config.start_epoch
            self.load(config.pretrained_dir, config.pretrained_epoch)
        self.opt_g = torch.optim.Adam( self.generator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        if config.gan:
            self.opt_d = torch.optim.Adam( self.discriminator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        self.dataset = Voxceleb_face_region( config.dataset_dir, 'train')
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
        
    def fit(self):
        config = self.config

        num_steps_per_epoch = len(self.data_loader)
        print(num_steps_per_epoch)
        cc = 0
        t0 = time.time()
        logger = Logger(config.log_dir)
        for epoch in range(self.start_epoch, config.max_epochs):
            
            for step, (data) in enumerate(self.data_loader):
                t1 = time.time()
                in_lmark = data['in_lmark'].float()
                lip_base = data['lip_base'].float()
                lmark_base = data['lmark_base'].float()
                gt_img = data['gt_img'].float()
                
                in_lmark = Variable(in_lmark.view(in_lmark.shape[0],-1).cuda())
                lip_base = Variable(lip_base).cuda()
                gt_img = Variable(gt_img.cuda())
                lmark_base = Variable(lmark_base.view(in_lmark.shape[0],64,-1).cuda())
                if config.gan:
                    mis_img = data['mismatch_img'].float()
                    mis_img = Variable(mis_img.cuda())
                if config.gan:
                         #train the discriminator
                    for p in self.discriminator.parameters():
                        p.requires_grad =  True
#                     print (lip_base.shape, lmark_base.shape, in_lmark.shape)
                    
                    fake_img = self.generator(lip_base,lmark_base, in_lmark)
                    D_real = self.discriminator(gt_img, in_lmark)
                    D_wrong = self.discriminator(mis_img, in_lmark)
                    loss_real = self.bce_loss_fn(D_real, self.ones)

                    
                    # train with fake image
                    D_fake = self.discriminator(fake_img.detach(), in_lmark)
                    loss_fake = self.bce_loss_fn(D_fake, self.zeros)
                    loss_wrong = self.bce_loss_fn(D_wrong, self.zeros)
                    

                    loss_disc = loss_real   + 0.5 * (loss_fake + loss_wrong) 

                    loss_disc.backward()
                    self.opt_d.step()
                    self._reset_gradients()
                    logger.scalar_summary('loss_real', loss_real, epoch * num_steps_per_epoch +  step+1)
                    logger.scalar_summary('loss_fake', loss_fake,epoch * num_steps_per_epoch + step+1)
                    logger.scalar_summary('loss_wrong', loss_wrong,epoch * num_steps_per_epoch + step+1)
                    logger.scalar_summary('loss_disc', loss_disc,epoch * num_steps_per_epoch + step+1)

                    #train the generator
                    for p in self.discriminator.parameters():
                        p.requires_grad = False  # to avoid computation

                    fake_img = self.generator( lip_base,lmark_base, in_lmark)
                    D_fake = self.discriminator(fake_img.detach(), in_lmark)

                    loss_gen = self.bce_loss_fn(D_fake, self.ones)

                    loss_pix =  self.l1_loss_fn(fake_img, gt_img) 
                   
                    loss =10 * loss_pix + loss_gen                     
                    loss.backward() 
                    self.opt_g.step()
                    self._reset_gradients()
                    logger.scalar_summary('loss_pix', loss_pix, epoch * num_steps_per_epoch +  step+1)
                    logger.scalar_summary('loss_gen', loss_gen,epoch * num_steps_per_epoch + step+1)
                    if (step+1) % 10 == 0 or (step+1) == num_steps_per_epoch:
                        steps_remain = num_steps_per_epoch-step+1 + \
                            (config.max_epochs-epoch+1)*num_steps_per_epoch

                        print("[{}/{}][{}/{}]   loss_disc: {:.8f},loss_gan: {:.8f},loss_pix: {:.8f},data time: {:.4f},  model time: {} second"
                              .format(epoch+1, config.max_epochs,
                                      step+1, num_steps_per_epoch,loss_disc,loss_gen , loss_pix,  t1-t0,  time.time() - t1))
                    t0 = time.time()
                    if (step) % (int(num_steps_per_epoch  / 32 )) == 0:
                    
                        fake_store = fake_img.data.contiguous().view(config.batch_size,3,256,256)
                        torchvision.utils.save_image(fake_store,
                            "{}/img_fake_{}.png".format(config.sample_dir,cc),normalize=True)

                        real_store = gt_img.data.contiguous().view(config.batch_size ,3,256,256)
                        torchvision.utils.save_image(real_store,
                            "{}/img_real_{}.png".format(config.sample_dir,cc),normalize=True)
                        cc += 1
                        torch.save(self.generator.state_dict(),
                               "{}/vg_net.pth"
                               .format(config.model_dir))
                
                else: 
                    fake_img = self.generator( lip_base, in_lmark)
                    loss =  self.mse_loss_fn(fake_img, gt_img) 
                    self._reset_gradients()
                    loss.backward() 
                    self.opt_g.step()
                
                    if (step+1) % 1 == 0 or (step+1) == num_steps_per_epoch:
                        steps_remain = num_steps_per_epoch-step+1 + \
                            (config.max_epochs-epoch+1)*num_steps_per_epoch
                        print("[{}/{}][{}/{}]   loss1: {:.8f},data time: {:.4f},  model time: {} second"
                              .format(epoch+1, config.max_epochs,
                                      step+1, num_steps_per_epoch, loss,  t1-t0,  time.time() - t1))
                
                if (step) % (int(num_steps_per_epoch  / 32 )) == 0: #and step != 0:
                    
                    fake_store = fake_img.data.contiguous().view(config.batch_size,3,256,256)
                    torchvision.utils.save_image(fake_store,
                        "{}/img_fake_{}.png".format(config.sample_dir,cc),normalize=True)
                    
                    real_store = gt_img.data.contiguous().view(config.batch_size ,3,256,256)
                    torchvision.utils.save_image(real_store,
                        "{}/img_real_{}.png".format(config.sample_dir,cc),normalize=True)
#                     cc += 1
                    
                        
      
                    cc += 1
                 
                    t0 = time.time()

                    torch.save(self.generator.state_dict(),
                                   "{}/vg_net_{}.pth"
                                   .format(config.model_dir,cc))
                    
                    torch.save(self.discriminator.state_dict(),
                                   "{}/dis_{}.pth"
                                   .format(config.model_dir,cc))
 
    def _reset_gradients(self):
        self.generator.zero_grad()
        
        

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
                        default=3)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=10000)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/data2/lchen63/voxceleb/txt/")
    
    parser.add_argument("--model_name",
                        type=str,
                        default="face_encoder2")
    parser.add_argument("--model_dir",
                        type=str,
                        default="./model/")
    parser.add_argument("--sample_dir",
                        type=str,
                        default="./sample/")
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--num_thread', type=int, default= 4)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--gan', type=bool, default=True)

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    t.fit()

if __name__ == "__main__":

    config = parse_args()
    config.is_train = 'train'
    import face_encoder2 as trainer
    config.model_dir = os.path.join(config.model_dir, config.model_name)
    config.sample_dir = os.path.join(config.sample_dir, config.model_name)
    config.log_dir = os.path.join('./logs', config.model_name)
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    else:
        shutil.rmtree(config.log_dir)
        os.mkdir(config.log_dir)
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    config.cuda1 = torch.device('cuda:{}'.format('0'))
    main(config)
