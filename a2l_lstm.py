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
import cv2
from dataset import  Voxceleb_audio_lmark_lstm
from ATVG import AT_lstm as atnet
from torch.nn import init
from util import utils
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
        self.generator = atnet()
        self.l1_loss_fn =  nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()
        self.config = config

        if config.cuda:
            self.generator     = self.generator.cuda()
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
        self.dataset = Voxceleb_audio_lmark_lstm('/data2/lchen63/voxceleb/txt/', 'train')


        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
        

    def fit(self):
        time_length = 32
        config = self.config
        face_pca = torch.FloatTensor( np.load('./basics/U_front_roni.npy')[:,:6]).cuda()
        face_mean =torch.FloatTensor( np.load('./basics/mean_front_roni.npy')).cuda()
        lip_pca = torch.FloatTensor( np.load('./basics/U_front.npy')[:,:6]).cuda()
        lip_mean =torch.FloatTensor( np.load('./basics/mean_front.npy')).cuda()
        
        
        lip_std = torch.FloatTensor( np.load('./basics/std_front.npy')).cuda()
        face_std = torch.FloatTensor( np.load('./basics/std_front_roni.npy')).cuda()

        num_steps_per_epoch = len(self.data_loader)
        cc = 0
        t0 = time.time()
        logger = Logger(config.log_dir)

        

        for epoch in range(self.start_epoch, config.max_epochs):
            for step, data in enumerate(self.data_loader):
                t1 = time.time()
                sample_audio =data['audio']
                lip_region =  data['lip_region'] 
                other_region = data['other_region']  
                ex_other_region = data['ex_other_region'] 
                ex_lip_region = data['ex_lip_region'] 


                if config.cuda:
                    sample_audio    = Variable(sample_audio.float()).cuda()
                    lip_region = lip_region.float().cuda()
                    other_region = other_region.float().cuda()
                    ex_other_region = ex_other_region.float().cuda()
                    ex_lip_region = ex_lip_region.float().cuda()
                else:
                    sample_audio    = Variable(sample_audio.float())
                    lip_region = Variable(lip_region.float())
                    other_region = Variable(other_region.float())
                    ex_other_region = Variable(ex_other_region.float())
                    ex_lip_region = ex_lip_region.float()
                
                
                lip_region_pca = (lip_region- lip_mean.expand_as(lip_region))/lip_std
                lip_region_pca = lip_region_pca.view(lip_region_pca.shape[0] *lip_region_pca.shape[1], -1)
                lip_region_pca = torch.mm(lip_region_pca,  lip_pca)
                
                lip_region_pca = lip_region_pca.view(config.batch_size ,time_length,6)
                
                
#                 lip_region_pca = lip_region_pca.view(lip_region_pca.shape[0] *lip_region_pca.shape[1], -1)
                ex_lip_region_pca = (ex_lip_region- lip_mean.expand_as(ex_lip_region))/lip_std
                
#                 lip_region_pca = lip_region_pca.view(lip_region_pca.shape[0] *lip_region_pca.shape[1], -1)
                
                ex_lip_region_pca = torch.mm(ex_lip_region_pca,  lip_pca)
                
                ex_other_region_pca = (ex_other_region - face_mean.expand_as(ex_other_region))/face_std
                ex_other_region_pca = torch.mm(ex_other_region_pca,  face_pca)    
                
                other_region_pca = (other_region - face_mean.expand_as(other_region))/face_std
                other_region_pca = other_region_pca.view(other_region_pca.shape[0] *other_region_pca.shape[1], -1)
                other_region_pca = torch.mm(other_region_pca,  face_pca)
                
                other_region_pca = other_region_pca.view(config.batch_size ,time_length,6)
                
                ex_other_region_pca = Variable(ex_other_region_pca)
                lip_region_pca = Variable(lip_region_pca)
                other_region_pca = Variable(other_region_pca)
                
                
                
           
                fake_lip, fake_face = self.generator(sample_audio,ex_other_region_pca, ex_lip_region_pca)
#                 print (fake_lip.shape, fake_face.shape, lip_region_pca.shape, other_region_pca.shape)
                                                     
                loss_lip =  self.mse_loss_fn(fake_lip, lip_region_pca)
                loss_face =  self.mse_loss_fn(fake_face , other_region_pca)
                loss= loss_lip + 0.001* loss_face
                loss.backward() 
                self.opt_g.step()
                self._reset_gradients()
                
                logger.scalar_summary('loss_lip', loss_lip, epoch * num_steps_per_epoch +  step+1)
                logger.scalar_summary('loss_face', loss_face,epoch * num_steps_per_epoch + step+1)

                    
                
                if step % 100 == 0:
                    print("[{}/{}:{}/{}]]   loss_face: {:.8f},loss_lip: {:.8f},data time: {:.4f},  model time: {} second"
                      .format(epoch+1, config.max_epochs, step, num_steps_per_epoch,
                              loss_face, loss_lip,  t1-t0,  time.time() - t1))
                t0 = time.time()    
            fake_lip =  fake_lip.view(config.batch_size * time_length, 6)
            fake_lip = torch.mm( fake_lip, lip_pca.t() ) 
            fake_lip =  fake_lip.view(config.batch_size, time_length, -1)
            fake_lip = fake_lip  * lip_std+ lip_mean.expand_as(fake_lip)

            fake_face =  fake_face.view(config.batch_size * time_length, 6) 
            fake_face = torch.mm( fake_face, face_pca.t() ) 
            fake_face =  fake_face.view(config.batch_size, time_length, -1)
            fake_face = fake_face * face_std+ face_mean.expand_as(fake_face)
            fake_lmark = torch.cat([fake_face, fake_lip], 2)
            fake_lmark = fake_lmark.data.cpu().numpy()


            real_lip =  lip_region_pca.view(config.batch_size * time_length  , 6) 
            real_lip = torch.mm( real_lip, lip_pca.t() ) 
            real_lip =  real_lip.view(config.batch_size, time_length, -1)
            real_lip = real_lip  * lip_std + lip_mean.expand_as(real_lip)

            real_face =  other_region_pca.view(config.batch_size * time_length  , 6) 
            real_face = torch.mm( real_face, face_pca.t() ) 
            real_face =  real_face.view(config.batch_size, time_length, -1)
            real_face = real_face  * face_std+ face_mean.expand_as(real_face)

            real_lmark = torch.cat([real_face, real_lip], 2)

            real_lmark = real_lmark.data.cpu().numpy()

            for gg in range(int(config.batch_size/40)):
                v_path = os.path.join('/data2/lchen63/voxceleb/unzip',data['img_path'][gg]  + '.mp4') 
                cap = cv2.VideoCapture(v_path)
                ret = True
                count = 0
                while ret:
                    ret, frame = cap.read()
                    if ret and count == int(data['sample_id'][gg]) :
                        break
                    if not ret:
                        break
                    count += 1
                for g in range(time_length):
                    lmark_name  = "{}/fake_{}_{}_{}.png".format(config.sample_dir,epoch, gg, g)

                    gt_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    plt = utils.lmark2img(fake_lmark[gg,g].reshape((68,3)), gt_img)
                    plt.savefig(lmark_name)


                    lmark_name  = "{}/real_{}_{}_{}.png".format(config.sample_dir,epoch, gg, g)
                    plt = utils.lmark2img(real_lmark[gg,g].reshape((68,3)), gt_img)
                    plt.savefig(lmark_name)
                    ret, frame = cap.read()
                cap.release()
                cv2.destroyAllWindows()
                    
            torch.save(self.generator.state_dict(),
                       "{}/anet_lstm.pth"
                       .format(config.model_dir))

                
 
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
                        default=512)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=10000)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="'/data2/lchen63/voxceleb/txt/'")
    parser.add_argument("--model_name",
                        type=str,
                        default="at_lstm")
    parser.add_argument("--model_dir",
                        type=str,
                        default="./model/")
                        # default="/mnt/disk1/dat/lchen63/grid/model/model_gan_r")
                        # default='/media/lele/DATA/lrw/data2/model')
    parser.add_argument("--sample_dir",
                        type=str,
                        default="./sample/")    
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--num_thread', type=int, default = 8)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    t.fit()

if __name__ == "__main__":

    config = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
    config.is_train = 'train'
    import a2l_lstm as trainer
    config.model_dir = os.path.join(config.model_dir, config.model_name)
    config.sample_dir = os.path.join(config.sample_dir, config.model_name)
    config.log_dir = os.path.join('./logs', config.model_name)
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    config.cuda1 = torch.device('cuda:{}'.format(config.device_ids))
    main(config)



