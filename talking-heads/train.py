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
from dataset.dataset import  Lmark2rgbDataset 
from network.network import Embedder  , Lmark2img_Discriminator
from network.network import Lmark2img_Generator2 as Lmark2img_Generator
from torch.nn import init
from network.loss import LossCnt
from logger import Logger
def weights_init(init_type='kaiming'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


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

class Trainer():
    def __init__(self, config):

        self.generator = Lmark2img_Generator()
        self.discriminator = Lmark2img_Discriminator()
        self.embedder = Embedder()


        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.l1_loss_fn =  nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()


        self.config = config
        self.ones = Variable(torch.FloatTensor(config.batch_size).fill_(1.0), requires_grad=False)

        self.zeros = Variable(torch.FloatTensor(config.batch_size).fill_(0.0), requires_grad=False)
        # self.ones = Variable(torch.ones(config.batch_size, 1), requires_grad=False)
        # self.zeros = Variable(torch.zeros(config.batch_size,1 ), requires_grad=False)

        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(',')]
            config.device_ids = device_ids
            # self.encoder = nn.DataParallel(self.encoder.cuda(device=config.cuda1), device_ids=device_ids)
            self.generator     = nn.DataParallel(self.generator, device_ids=device_ids).cuda()
            self.discriminator     = nn.DataParallel(self.discriminator, device_ids=device_ids).cuda()
            self.embedder     = nn.DataParallel(self.embedder, device_ids=device_ids).cuda()

            self.loss_cnt     = nn.DataParallel(LossCnt( config.root), device_ids=device_ids).cuda()

            self.bce_loss_fn   = self.bce_loss_fn.cuda(device=config.cuda1)
            self.mse_loss_fn   = self.mse_loss_fn.cuda(device=config.cuda1)
            self.l1_loss_fn = self.l1_loss_fn.cuda(device=config.cuda1)
            # Adversarial ground truths
            # self.valid = 1
            # self.fake = 0
            self.ones          = self.ones.cuda(device=config.cuda1)
            self.zeros          = self.zeros.cuda(device=config.cuda1)
# #########single GPU#######################

#         if config.cuda:
#             device_ids = [int(i) for i in config.device_ids.split(',')]
#             self.generator     = self.generator.cuda(device=config.cuda1)
#             self.encoder = self.encoder.cuda(device=config.cuda1)
#             self.mse_loss_fn   = self.mse_loss_fn.cuda(device=config.cuda1)
#             self.l1_loss_fn =  nn.L1Loss().cuda(device=config.cuda1)
        self.start_epoch = 0
        if config.load_model:
            self.start_epoch = config.start_epoch
            self.load(config.pretrained_dir, config.pretrained_epoch)
       

        self.opt_g = torch.optim.Adam( list(self.generator.parameters()) + list( self.embedder.parameters()) ,
            lr=config.LEARNING_RATE_E_G)
        self.opt_d = torch.optim.Adam( self.discriminator.parameters(),
            lr=config.LEARNING_RATE_D)
        self.dataset = Lmark2rgbDataset(config.root, resolution = 256, train=config.is_train)
        
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
        # data_iter = iter(self.data_loader)
        # data_iter.next()
        self.embedder.apply(weights_init)
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
    def fit(self):

        config = self.config
        logger = Logger(config.log_dir)
        num_steps_per_epoch = len(self.data_loader)
        cc = 0
        t0 = time.time()
        self.embedder.train()
        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.start_epoch, config.max_epochs):
            
            for step, (batch_data) in enumerate(self.data_loader):
                t1 = time.time()
                references = batch_data['reference_frames']
                target_rgb = batch_data['target_rgb']
                target_lmark = batch_data['target_lmark']
                target_ani = batch_data['target_ani']
                if config.cuda:
                    references = Variable(references.float()).cuda(device=config.cuda1)
                    target_ani = Variable(target_ani.float()).cuda(device=config.cuda1)
                    target_rgb    = Variable(target_rgb.float()).cuda(device=config.cuda1)
                    target_lmark = Variable(target_lmark.float()).cuda(device=config.cuda1)
                else:
                    raise Exception('no cuda gpu avaliable')
                

                # embed the reference frames 
                dims = references.shape
                # print (dims)
                references = references.reshape( dims[0] * dims[1], dims[2], dims[3], dims[4]  )
                # reference_frames = references[:,0,:,:,:]
                # reference_lmark = references[:,1, :,:,:]

                self.opt_g.zero_grad()
                e_vectors = self.embedder(references).reshape(dims[0] , dims[1], -1)

                e_hat = e_vectors.mean(dim = 1)

                # train G, E
                
                for p in self.discriminator.parameters():
                    p.requires_grad = False  # to avoid computation
                fake_img  = self.generator( target_ani,target_lmark, e_hat)
                
                D_fake = self.discriminator(fake_img, target_lmark)

                loss_adv = self.mse_loss_fn(D_fake, self.ones)
                loss = []
                loss.append(loss_adv)
                if config.perceptual:
                    loss_cnt = self.loss_cnt(target_rgb, fake_img).mean()
                    loss.append(loss_cnt)
                if config.pixel:
                    loss_pix = self.l1_loss_fn(fake_img, target_rgb)
                    loss.append(loss_pix )
                loss_gen = sum(loss)
                loss_gen.backward()
                self.opt_g.step()

                #train the discriminator
                for p in self.discriminator.parameters():
                    p.requires_grad =  True

                self.opt_d.zero_grad()

                fake_img = self.generator( target_ani,target_lmark, e_hat.detach())
                

                # train with real image
                D_real= self.discriminator(target_rgb, target_lmark)
                loss_real = self.mse_loss_fn(D_real, self.ones)

                # train with fake image
                D_fake  = self.discriminator(fake_img.detach(), target_lmark)
                loss_fake = self.mse_loss_fn(D_fake, self.zeros)


                # train with ani image
                # D_ani  = self.discriminator(target_ani,  target_lmark)
                # loss_ani = self.mse_loss_fn(D_ani, self.zeros)

                loss_disc = loss_real  +  loss_fake #+ loss_ani

                loss_disc.backward()
                self.opt_d.step()

                logger.scalar_summary('loss_disc', loss_disc.item(),epoch * num_steps_per_epoch + step+1)
                logger.scalar_summary('loss_gen', loss_gen.item(),epoch * num_steps_per_epoch + step+1)
                if config.pixel:
                    logger.scalar_summary('loss_pix', loss_pix.item(),epoch * num_steps_per_epoch + step+1)
                if config.perceptual:
                    logger.scalar_summary('loss_cnt_G', loss_cnt.item(),epoch * num_steps_per_epoch + step+1)
                # logger.scalar_summary('loss_ani', loss_ani.item(),epoch * num_steps_per_epoch + step+1)
                t2 = time.time()
                
                # if (step) % 10 == 0 :
#                 print("[{}/{}][{}/{}]  ,  loss_disc: {:.8f},   loss_gen: {:.8f}   , loss_cnt: {:.8f}, data time: {:.4f},  model time: {} second".format(epoch+1, config.max_epochs, step+1, num_steps_per_epoch, loss_disc.item(),  loss_gen.item(), loss_cnt.item(),  t1-t0,  t2 - t1))
                if not config.perceptual:
                    loss_cnt = loss_gen 
                if not config.pixel:
                    loss_pix = loss_gen
                print("[{}/{}][{}/{}]  ,  loss_disc: {:.8f},   loss_gen: {:.8f}  ,  loss_pix: {:.8f} , loss_cnt: {:.8f}, data time: {:.4f},  model time: {} second".format(epoch+1, config.max_epochs, step+1, num_steps_per_epoch, loss_disc.item(),  loss_gen.item(),loss_pix.item(), loss_cnt.item(),  t1-t0,  t2 - t1))
                # print("[{}/{}][{}/{}]  , loss_pix: {:.8f} , data time: {:.4f},  model time: {} second".format(epoch+1, config.max_epochs, step+1, num_steps_per_epoch, loss_pix.item(),  t1-t0,  t2 - t1))

#                 if (step) % (int(num_steps_per_epoch  / 2 )) == 0 :
                t0 = time.time()
                if epoch == cc:
                    fake_store = fake_img.data.contiguous().view(config.batch_size,3,256,256)

                    torchvision.utils.save_image(fake_store,
                        "{}/img_fake_{}.png".format(config.sample_dir,cc),normalize=True)

                    references = references.reshape( config.batch_size, 4, dims[2], dims[3], dims[4]  )
                    reference_frames = references[:,0,:3,:,:].data.contiguous().view(config.batch_size,3,256,256)

                    reference_lmark = references[:,0,3:,:,:].data.contiguous().view(config.batch_size,3,256,256)
                        # ref_store = reference_img.data.contiguous().view(config.batch_size,3,256,256)
                    torchvision.utils.save_image(reference_frames,
                        "{}/img_ref_{}.png".format(config.sample_dir,cc),normalize=True)
                    
                    torchvision.utils.save_image(reference_lmark,
                        "{}/lmark_ref_{}.png".format(config.sample_dir,cc),normalize=True)
                    

                    target_lmark = target_lmark.data.contiguous().view(config.batch_size ,3,256,256)
                    torchvision.utils.save_image(target_lmark,
                        "{}/lmark_real_{}.png".format(config.sample_dir,cc),normalize=True)

                    target_ani = target_ani.data.contiguous().view(config.batch_size ,3,256,256)
                    torchvision.utils.save_image(target_ani,
                        "{}/ani_real_{}.png".format(config.sample_dir,cc),normalize=True)

                    real_store = target_rgb.data.contiguous().view(config.batch_size ,3,256,256)
                    torchvision.utils.save_image(real_store,
                        "{}/img_real_{}.png".format(config.sample_dir,cc),normalize=True)
                    cc += 1
                if epoch% 5 == 0:
                    torch.save(self.generator.state_dict(),
                                "{}/vg_net_{}.pth"
                                .format(config.model_dir, epoch))

            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--LEARNING_RATE_E_G",
                        type=float,
                        default=5e-4)
    parser.add_argument("--LEARNING_RATE_D",
                        type=float,
                        default=2e-4)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--lambda1",
                        type=int,
                        default=100)
    parser.add_argument("--pixel",
                        type=bool,
                        default=True)
    parser.add_argument("--gan",
                        type=bool,
                        default=True)
    parser.add_argument("--perceptual",
                        type=bool,
                        default=True)
    parser.add_argument("--use_ani",
                        type=bool,
                        default=True)
    parser.add_argument("--batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=100)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--root",
                        type=str,
                        default="/home/cxu-serve/p1/lchen63/voxceleb")
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
                        default="base")
                        # default="/mnt/ssd0/dat/lchen63/grid/pickle/")
                        # default = '/media/lele/DATA/lrw/data2/pickle')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    t.fit()

if __name__ == "__main__":
    config = parse_args()
    config.is_train = 'train'
    import train as trainer
    config.model_dir = os.path.join(config.model_dir, config.model_name)
    config.sample_dir = os.path.join(config.sample_dir, config.model_name)
    config.log_dir = os.path.join('./logs', config.model_name)

    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)

    config.cuda1 = torch.device('cuda:{}'.format('0'))
    main(config)
