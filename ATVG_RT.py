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
from torch.nn import init
torch.set_printoptions(precision=5,sci_mode=False)
import random
from dataset import  Voxceleb_head_movements_RT as Voxceleb_head_movements_derivative
from ATVG import AT_net_RT_FC as AT_net
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

class Trainer():
    def __init__(self, config):            
        self.generator = AT_net()
        self.l1_loss_fn =  nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()
        self.config = config

        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(',')]
            self.generator     = nn.DataParallel(self.generator, device_ids=device_ids).cuda()
            # self.generator     = self.generator.cuda()
            self.mse_loss_fn   = self.mse_loss_fn.cuda()
            self.l1_loss_fn = self.l1_loss_fn.cuda()
# #########single GPU#######################

#         if config.cuda:
#             device_ids = [int(i) for i in config.device_ids.split(',')]
#             self.generator     = self.generator.cuda()
#             self.encoder = self.encoder.cuda()
#             self.mse_loss_fn   = self.mse_loss_fn.cuda()
#             self.l1_loss_fn =  nn.L1Loss().cuda()
        initialize_weights(self.generator)
        self.start_epoch = 0
        if config.load_model:
            self.start_epoch = config.start_epoch
            self.load(config.pretrained_dir, config.pretrained_epoch)
        self.opt_g = torch.optim.Adam( self.generator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        
#         self.opt_g = torch.optim.SGD(self.generator.parameters(), lr=config.lr)
        self.dataset = Voxceleb_head_movements_derivative( config.dataset_dir, 'train')


        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
        
    def fit(self):
        config = self.config
        log_name = os.path.join(config.sample_dir,'loss')
        num_steps_per_epoch = len(self.data_loader)
        print(num_steps_per_epoch)
        cc = 0
        t0 = time.time()
        xLim=(-224.0, 224.0)
        yLim=(-224.0, 224.0)
        xLab = 'x'
        yLab = 'y'
        for epoch in range(self.start_epoch, config.max_epochs):
            random.seed(time.time())
            for step, (data) in enumerate(self.data_loader):
                t1 = time.time()
                in_derivative = data.type(torch.FloatTensor).cuda()
                
                in_derivative = Variable(in_derivative)
                
                
        
                fake_derivative= self.generator( in_derivative)

                loss =  self.mse_loss_fn(fake_derivative , in_derivative) 
                self._reset_gradients()
                loss.backward() 
                self.opt_g.step()
                


                if (step+1) % 1 == 0 or (step+1) == num_steps_per_epoch:
                    steps_remain = num_steps_per_epoch-step+1 + \
                        (config.max_epochs-epoch+1)*num_steps_per_epoch

                    print("[{}/{}][{}/{}]   loss1: {:.8f},data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss,  t1-t0,  time.time() - t1))
                    with open(log_name, "a") as log_file:
                        
                        log_file.write('fake%s\n' % "[{}/{}][{}/{}]   loss1: {:.8f},data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss,  t1-t0,  time.time() - t1))
                
                    print (fake_derivative[0,3:10])
                    print (in_derivative[0,3:10])
                if (step) % (int(num_steps_per_epoch  / 1 )) == 0: #and step != 0:
                    with open(log_name, "w") as log_file:
                        
                        log_file.write('fake%s\n' % str(fake_derivative[0,3:5]))
                        log_file.write('gt%s\n' % str(in_derivative[0,3:5]))
#                     lmark = data['out_lmark']
#                     fake_lmark = torch.zeros(lmark.size()) 
#                     fake_lmark[:,0] = data['out_lmark'][:,0]

#                     tmp_fake = fake_derivative.view(config.batch_size, 64,16,3)


#                     fake_lmark[:, 1:] = data['out_lmark'][:, :-1] + fake_derivative[:,:-1]
#                     lmark = lmark.view(config.batch_size, 64, 136)
#                     lmark = lmark.numpy()
#                     fake_lmark = fake_lmark.view(config.batch_size, 64, 136)
#                     fake_lmark = fake_lmark.numpy()
#                     if not os.path.exists(os.path.join(config.sample_dir, cc)):
#                         os.mkdir(os.path.join(config.sample_dir, cc))



#                     for indx in range(3):
#                         for jj in range(64):

#                             name = "{}/{}real_{}_{}.png".format(config.sample_dir,cc, indx,jj)
#                             utils.plot_flmarks(lmark[indx,jj], name, xLim, yLim, xLab, yLab, figsize=(10, 10))
#                             name = "{}/{}fake_{}_{}.png".format(config.sample_dir,cc, indx,jj)
#                             utils.plot_flmarks(fake_lmark[indx,jj], name, xLim, yLim, xLab, yLab, figsize=(10, 10))
#                     torch.save(self.generator.state_dict(),
#                                "{}/at_rt_{}.pth"
#                                .format(config.model_dir,cc))
                  
                    cc += 1
                 
                t0 = time.time()
 
    def _reset_gradients(self):
        self.generator.zero_grad()
        
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.002)
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
                        default=1)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=10000)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/data2/lchen63/voxceleb/txt/")
                        # default="/mnt/ssd0/dat/lchen63/grid/pickle/")
                        # default = '/media/lele/DATA/lrw/data2/pickle')
    parser.add_argument("--model_dir",
                        type=str,
                        default="./checkpoints/at_rt/")
                        # default="/mnt/disk1/dat/lchen63/grid/model/model_gan_r")
                        # default='/media/lele/DATA/lrw/data2/model')
    parser.add_argument("--sample_dir",
                        type=str,
                        default="./sample/at_rt/")
                        # default="/mnt/disk1/dat/lchen63/grid/sample/model_gan_r/")
                        # default='/media/lele/DATA/lrw/data2/sample/lstm_gan')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--rnn', type=bool, default=True)

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    t.fit()

if __name__ == "__main__":

    config = parse_args()
    config.is_train = 'train'
    import ATVG_RT as trainer
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    config.cuda1 = torch.device('cuda:{}'.format(config.device_ids))
    main(config)
