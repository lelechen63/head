from network.vgg import vgg_face, VGG_Activations
from torchvision.models import vgg19

import torch
from torch import nn
from torch.nn import functional as F

import config

class Loss_cnt(nn.Module):
    def __init__(self):
        super(Loss_cnt, self).__init__()

        self.VGG_FACE_AC = VGG_Activations(vgg_face(pretrained=True), [1, 6, 11, 18, 25])
        self.VGG19_AC = VGG_Activations(vgg19(pretrained=True), [1, 6, 11, 20, 29])
        self.l1_loss_fn =  nn.L1Loss()
        self.IMG_NET_MEAN = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]), requires_grad= False)
        self.IMG_NET_STD = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]), requires_grad= False)
        

    def loss_cnt(self, x, x_hat):

        x = (x - self.IMG_NET_MEAN) / self.IMG_NET_STD
        x_hat = (x_hat - self.IMG_NET_MEAN) / self.IMG_NET_STD

        # VGG19 Loss
        vgg19_x_hat = self.VGG19_AC(x_hat)
        vgg19_x = self.VGG19_AC(x)

        vgg19_loss = 0
        for i in range(0, len(vgg19_x)):
            vgg19_loss += self.l1_loss_fn(vgg19_x_hat[i], vgg19_x[i])

        # VGG Face Loss
        vgg_face_x_hat = self.VGG_FACE_AC(x_hat)
        vgg_face_x = self.VGG_FACE_AC(x)

        vgg_face_loss = 0
        for i in range(0, len(vgg_face_x)):
            vgg_face_loss += self.l1_loss_fn(vgg_face_x_hat[i], vgg_face_x[i])

        return vgg19_loss * config.LOSS_VGG19_WEIGHT + vgg_face_loss * config.LOSS_VGG_FACE_WEIGHT


    def forward(self, x, x_hat):
        

        cnt = self.loss_cnt(x, x_hat)

        return cnt.reshape(1)


class LossD(nn.Module):
    def __init__(self, gpu=None):
        super(LossD, self).__init__()
        self.gpu = gpu
        if gpu is not None:
            self.cuda(gpu)

    def forward(self, r_x, r_x_hat):
        if self.gpu is not None:
            r_x = r_x.cuda(self.gpu)
            r_x_hat = r_x_hat.cuda(self.gpu)
        return (F.relu(1 + r_x_hat) + F.relu(1 - r_x)).mean().reshape(1)
