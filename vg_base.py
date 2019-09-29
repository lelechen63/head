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
from dataset import  Voxceleb_mfcc_rgb_single 
from ATVG import MFCC_Face_single , VG_base_Discriminator
from logger import Logger

from torch.nn import init
import utils
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

class Trainer():
    def __init__(self, config):

        self.generator = MFCC_Face_single()
        self.discriminator = VG_base_Discriminator()
        self.bce_loss_fn = nn.BCELoss()
        self.l1_loss_fn =  nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()
        self.config = config
        self.ones = Variable(torch.ones(config.batch_size), requires_grad=False)
        self.zeros = Variable(torch.zeros(config.batch_size), requires_grad=False)

        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(',')]
            # self.encoder = nn.DataParallel(self.encoder.cuda(device=config.cuda1), device_ids=device_ids)
            self.generator     = nn.DataParallel(self.generator, device_ids=device_ids).cuda()
            self.discriminator     = nn.DataParallel(self.discriminator, device_ids=device_ids).cuda()

            self.bce_loss_fn   = self.bce_loss_fn.cuda(device=config.cuda1)
            self.mse_loss_fn   = self.mse_loss_fn.cuda(device=config.cuda1)
            self.l1_loss_fn = self.l1_loss_fn.cuda(device=config.cuda1)
            self.ones          = self.ones.cuda(device=config.cuda1)
            self.zeros          = self.zeros.cuda(device=config.cuda1)
# #########single GPU#######################

#         if config.cuda:
#             device_ids = [int(i) for i in config.device_ids.split(',')]
#             self.generator     = self.generator.cuda(device=config.cuda1)
#             self.encoder = self.encoder.cuda(device=config.cuda1)
#             self.mse_loss_fn   = self.mse_loss_fn.cuda(device=config.cuda1)
#             self.l1_loss_fn =  nn.L1Loss().cuda(device=config.cuda1)
        initialize_weights(self.generator)
        self.start_epoch = 0
        if config.load_model:
            self.start_epoch = config.start_epoch
            self.load(config.pretrained_dir, config.pretrained_epoch)
        print ('-----------')
       

        self.opt_g = torch.optim.Adam( self.generator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        self.opt_d = torch.optim.Adam( self.discriminator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        self.dataset = Voxceleb_mfcc_rgb_single(config.dataset_dir, train=config.is_train)
        
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
        data_iter = iter(self.data_loader)
        data_iter.next()

    def fit(self):

        config = self.config
        logger = Logger(config.log_dir)

        num_steps_per_epoch = len(self.data_loader)
        cc = 0
        t0 = time.time()
       

        for epoch in range(self.start_epoch, config.max_epochs):
            for step, (batch_data) in enumerate(self.data_loader):
                t1 = time.time()
                reference_img = batch_data['final_img']
                reference_rgb = batch_data['reference_rgb']
                target_rgb = batch_data['target_rgb']
                mfcc = batch_data['sample_audio']
                if config.cuda:
                    reference_img = Variable(reference_img.float()).cuda(device=config.cuda1)
                    reference_rgb = Variable(reference_rgb.float()).cuda(device=config.cuda1)
                    target_rgb    = Variable(target_rgb.float()).cuda(device=config.cuda1)
                    mfcc = Variable(mfcc.float()).cuda(device=config.cuda1)
                else:
                    raise Exception('no cuda gpu avaliable')

                #train the discriminator
                for p in self.discriminator.parameters():
                    p.requires_grad =  True
                # print (reference_img.shape)
                # print (mfcc.shape)
                fake_im = self.generator( reference_img, mfcc)
                D_real= self.discriminator(target_rgb, mfcc)
                loss_real = self.bce_loss_fn(D_real, self.ones)

                # train with fake image
                D_fake  = self.discriminator(fake_im.detach(), mfcc)
                loss_fake = self.bce_loss_fn(D_fake, self.zeros)

                loss_disc = loss_real  + loss_fake 

                loss_disc.backward()
                self.opt_d.step()
                self._reset_gradients()

                #train the generator
                for p in self.discriminator.parameters():
                    p.requires_grad = False  # to avoid computation

                fake_im  = self.generator( reference_img, mfcc)
                D_fake = self.discriminator(fake_im, mfcc)

                loss_gen = self.bce_loss_fn(D_fake, self.ones)

                
                loss_pix = self.l1_loss_fn(fake_im, target_rgb)
                loss =10 * loss_pix + loss_gen  
                loss.backward() 
                self.opt_g.step()
                self._reset_gradients()
                logger.scalar_summary('loss_disc', loss_disc,epoch * num_steps_per_epoch + step+1)
                logger.scalar_summary('loss_gen', loss_gen,epoch * num_steps_per_epoch + step+1)
                logger.scalar_summary('loss_pix', loss_pix,epoch * num_steps_per_epoch + step+1)
                t2 = time.time()
                
                if (step+1) % 10 == 0 or (step+1) == num_steps_per_epoch:
                    steps_remain = num_steps_per_epoch-step+1 + \
                        (config.max_epochs-epoch+1)*num_steps_per_epoch

                    print("[{}/{}][{}/{}]  ,  loss_disc: {:.8f},   loss_gen: {:.8f}  ,  loss_pix: {:.8f} , data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss_disc.data[0],  loss_gen.data[0],loss_pix.data[0],  t1-t0,  t2 - t1))

                if (step) % (int(num_steps_per_epoch  / 20 )) == 0 :
                    
                    fake_store = fake_im.data.contiguous().view(config.batch_size,3,256,256)
                    
                    torchvision.utils.save_image(fake_store,
                        "{}/img_fake_{}.png".format(config.sample_dir,cc),normalize=True)

                    ref_store = reference_img.data.contiguous().view(config.batch_size,3,256,256)
                    
                    torchvision.utils.save_image(ref_store,
                        "{}/img_ref_{}.png".format(config.sample_dir,cc),normalize=True)

                    real_store = target_rgb.data.contiguous().view(config.batch_size ,3,256,256)
                    torchvision.utils.save_image(real_store,
                        "{}/img_real_{}.png".format(config.sample_dir,cc),normalize=True)
                    cc += 1
                    torch.save(self.generator.state_dict(),
                               "{}/vg_net_{}.pth"
                               .format(config.model_dir,cc))
                 
                t0 = time.time()
    def load(self, directory, epoch):
        gen_path = os.path.join(directory, 'generator_{}.pth'.format(epoch))

        self.generator.load_state_dict(torch.load(gen_path))

        dis_path = os.path.join(directory, 'discriminator_{}.pth'.format(epoch))

        self.discriminator.load_state_dict(torch.load(dis_path))


    def _reset_gradients(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()

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
                        default=8)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=5)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/home/cxu-serve/p1/lchen63/voxceleb/txt")
                        # default="/mnt/ssd0/dat/lchen63/grid/pickle/")
                        # default = '/media/lele/DATA/lrw/data2/pickle')
    parser.add_argument("--model_dir",
                        type=str,
                        default="./model/")
                        # default="/mnt/disk1/dat/lchen63/grid/model/model_gan_r")
                        # default='/media/lele/DATA/lrw/data2/model')
    parser.add_argument("--sample_dir",
                        type=str,
                        default="./sample/")
                        # default="/mnt/disk1/dat/lchen63/grid/sample/model_gan_r/")
                        # default='/media/lele/DATA/lrw/data2/sample/lstm_gan')
    parser.add_argument("--model_name",
                        type=str,
                        default="mfcc2rgb_single_base")
                        # default="/mnt/ssd0/dat/lchen63/grid/pickle/")
                        # default = '/media/lele/DATA/lrw/data2/pickle')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--perceptual', type=bool, default=False)

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    t.fit()

if __name__ == "__main__":
    config = parse_args()
    config.is_train = 'train'
    import vg_base as trainer
    config.model_dir = os.path.join(config.model_dir, config.model_name)
    config.sample_dir = os.path.join(config.sample_dir, config.model_name)
    config.log_dir = os.path.join('./logs', config.model_name)

    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)

    config.cuda1 = torch.device('cuda:{}'.format(config.device_ids))
    main(config)
