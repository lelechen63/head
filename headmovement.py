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
import argparse
from dataset import  Voxceleb_head_movements_derivative

from torch.nn import init
from util import utils
from util.visualizer import Visualizer
import time
import os
import numpy as np
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0
from torch.utils.data import DataLoader
from models.models import create_model
import argparse
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
                        default=1280)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=500)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument('--name', type=str, default='hm_derivative',
                         help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--model', type=str, default='MoveNet', help='which model to use')
    parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32], 
                        help="Supported data type i.e. 8, 16, 32 bit")
    
    #visualization
    parser.add_argument('--tf_log', action='store_true',
                         help='if specified, use tensorboard logging. Requires tensorflow installed')
    parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
    parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=2, 
                                 help='frequency of saving checkpoints at the end of epochs')        
    parser.add_argument('--no_html', action='store_true', 
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/data2/lchen63/voxceleb/txt/")
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    
    # for continue training
    parser.add_argument('--continue_train', action='store_true', 
                             help='continue training: load the latest model')
    parser.add_argument('--load_pretrain', type=str, default='', 
                             help='load the pretrained model from the specified location')
    parser.add_argument('--which_epoch', type=str, default='latest', 
                             help='which epoch to load? set to latest to use latest cached model') 
    parser.add_argument('--perceptual_loss', type=bool, default=False)
    parser.add_argument('--niter_decay', type=int, default=500, 
                        help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--no_lsgan', action='store_true', 
                        help='do *not* use least square GAN, if false, use vanilla GAN')
    parser.add_argument('--train', type=str, default='train')
    return parser.parse_args()
opt = parse_args()

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batch_size)    

dataset = Voxceleb_head_movements_derivative( '/data2/lchen63/voxceleb/txt', 'train')
data_loader = DataLoader(dataset,
                         batch_size=2,
                         num_workers=1,                          
                         shuffle=True, drop_last=True)
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model = create_model(opt)
visualizer = Visualizer(opt)
optimizer_G, optimizer_D = model.optimizer_G, model.optimizer_D


total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
for epoch in range(start_epoch, opt.max_epochs + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % opt.batch_size
    for i, data in enumerate(data_loader, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta
        ############## Forward Pass ######################
        losses, generated = model(data['in_lmark'], data['out_lmark'],data['mean'], infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.loss_names, losses))
        # calculate final loss scalar
#         loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
#         loss_G = loss_dict['G_GAN'] + loss_dict.get('P',0)  + loss_dict['G_L1']
        loss_G = loss_dict['G_L1']
        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        loss_G.backward()          
        optimizer_G.step()

        # update discriminator weights
#         optimizer_D.zero_grad()        
#         loss_D.backward()        
#         optimizer_D.step() 
        
        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 
        
        # display output images
        if save_fake:
            print('======================')
            print (generated.data[0][5:10])
            print ('~++++++++++++++++++++++')
            gt = (data['out_lmark'][0][1:] - data['out_lmark'][0][:-1])* 2.0/torch.FloatTensor([16,16,32])
            gt = gt.view(gt.shape[0],-1)
            print (gt[5:10])
#             in_lmark = data['in_lmark'][0]
#             gt_lmark = data['out_lmark'][0]
#             fake_lmark = torch.zeros(in_lmark.size()) 
#             fake_lmark[0] = data['out_lmark'][0][0]
#             fake_lmark[1:]=  data['out_lmark'][0][:-1] +  generated.data[0][:-1]
#             print (gt_lmark)
#             print ()
#             visuals = OrderedDict([('in_lmark', utils.util.lmark2im(data['in_lmark'][0])),
#                                     ('out_lmark', utils.util.lmark2im(data['out_lmark'][0])),
#                                     ('mean', utils.util.lmark2im(data['mean'][0])),
#                                     ('fake_lmark', utils.util.tensor2im(generated.data[0]))])
#             # print (data['gt_path'][0])
#             visualizer.display_current_results(visuals, epoch, total_steps)

#         break
#     break