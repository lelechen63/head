""" Implementation of the three networks that make up the Talking Heads generative model. """
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

from network.components import ResidualBlock, AdaptiveResidualBlock, ResidualBlockDown, AdaptiveResidualBlockUp, SelfAttention
from network.blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock



def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params


class Embedder(nn.Module):
    """
    The Embedder network attempts to generate a vector that encodes the personal characteristics of an individual given
    a head-shot and the matching landmarks.
    """
    def __init__(self):
        super(Embedder, self).__init__()

        self.conv1 = ResidualBlockDown(6, 64)
        self.conv2 = ResidualBlockDown(64, 128)
        self.conv3 = ResidualBlockDown(128, 256)
        self.att = SelfAttention(256)
        self.conv4 = ResidualBlockDown(256, 512)
        self.conv5 = ResidualBlockDown(512, 512)
        self.conv6 = ResidualBlockDown(512, 512)

        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        # self.apply(weights_init)
       

    def forward(self, x):   #(x: img, y: lmark)
        # assert x.dim() == 4 and x.shape[1] == 3, "Both x and y must be tensors with shape [BxK, 3, W, H]."
        # assert x.shape == y.shape, "Both x and y must be tensors with shape [BxK, 3, W, H]."
        # # Concatenate x & y
        # out = torch.cat((x, y), dim=1)  # [BxK, 6, 256, 256]

        # Encode
        out = (self.conv1(out))  # [BxK, 64, 128, 128]
        out = (self.conv2(out))  # [BxK, 128, 64, 64]
        out = (self.conv3(out))  # [BxK, 256, 32, 32]
        out = self.att(out)
        out = (self.conv4(out))  # [BxK, 512, 16, 16]
        out = (self.conv5(out))  # [BxK, 512, 8, 8]
        out = (self.conv6(out))  # [BxK, 512, 4, 4]
        # Vectorize
        out = F.relu(self.pooling(out).view(-1, 512))
        return out


# from torch.autograd import Variable

# import numpy as np

# model = Embedder().cuda()
# torch.set_printoptions(precision=6)
# print (model)
# # a = torch.zeros(2, 136).cuda()

# # a = a + 0.0001
# # a = Variable(a)

# # print (a.data)
# # a.shape
# b = torch.FloatTensor(2, 3, 256, 256).cuda()
# b = Variable(b)

# g= model(b, b)
# # print ('================')

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dim, n_blk, norm, activ):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))




class  Lmark2img_Generator(nn.Module):
    ADAIN_LAYERS = OrderedDict([
        ('res1', (512, 512)),
        ('res2', (512, 512)),
        ('res3', (512, 512)),
        ('res4', (512, 512)),
        ('res5', (512, 512)),
        ('deconv6', (512, 512)),
        ('deconv5', (512, 512)),
        ('deconv4', (512, 256)),
        ('deconv3', (256, 128)),
        ('deconv2', (128, 64)),
        ('deconv1', (64, 32))
    ])

    def __init__(self):
        super( Lmark2img_Generator, self).__init__()

        self.conv1 = ResidualBlockDown(3, 64)            #(64,128,128)
        self.in1_e = nn.InstanceNorm2d(64, affine=True)

        self.conv2 = ResidualBlockDown(64, 128)      #(128,64,64)
        self.in2_e = nn.InstanceNorm2d(128, affine=True)

        self.conv3 = ResidualBlockDown(128, 256)      #(256,32,32)
        self.in3_e = nn.InstanceNorm2d(256, affine=True)

        self.att1 = SelfAttention(256)

        self.conv4 = ResidualBlockDown(256, 512)        #(512,16,16)
        self.in4_e = nn.InstanceNorm2d(512, affine=True)

        self.conv5 = ResidualBlockDown(512, 512)        #(512,8,8)
        self.in5_e = nn.InstanceNorm2d(512, affine=True)

        self.conv6 = ResidualBlockDown(512, 512)        #(512,4,4)
        self.in6_e = nn.InstanceNorm2d(512, affine=True)
        self.model = []

        activ='relu'
        pad_type='reflect'
        self.model += [ResBlocks(2, 512, norm  = 'adain', activation=activ, pad_type='reflect')]
        self.model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]    # 512, 8 , 8 
        self.model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 512, 16 , 16 
        self.model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 32, 32 
        self.model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(256, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 64, 64 
        self.model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(256, 128, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 128, 128, 128 
        self.model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(128, 64, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 64, 256, 256 
        self.model += [Conv2dBlock(64, 3, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.decoder = nn.Sequential(*self.model)


        # self.apply(weights_init)


        self.mlp = MLP(512,
                       get_num_adain_params(self.decoder),
                       256,
                       3,
                       norm='none',
                       activ='relu')

        # self.apply(weights_init)
        
    def forward(self, y, e):

        out = y  # [B, 3, 256, 256]

      
        # Encode
        out = self.in1_e(self.conv1(out))  # [B, 64, 128, 128]
        out = self.in2_e(self.conv2(out))  # [B, 128, 64, 64]
        out = self.in3_e(self.conv3(out))  # [B, 256, 32, 32]
        out = self.att1(out)
        out = self.in4_e(self.conv4(out))  # [B, 512, 16, 16]
        out = self.in5_e(self.conv5(out))  # [B, 512, 8, 8]
        out = self.in6_e(self.conv6(out))  # [B, 512, 4, 4]
    
        # Decode
        adain_params = self.mlp(e)
        assign_adain_params(adain_params, self.decoder)

        image = self.decoder(out)
        return image





class  Lmark2img_Generator2(nn.Module):
    ADAIN_LAYERS = OrderedDict([
        ('res1', (512, 512)),
        ('res2', (512, 512)),
        ('res3', (512, 512)),
        ('res4', (512, 512)),
        ('res5', (512, 512)),
        ('deconv6', (512, 512)),
        ('deconv5', (512, 512)),
        ('deconv4', (512, 256)),
        ('deconv3', (256, 128)),
        ('deconv2', (128, 64)),
        ('deconv1', (64, 32))
    ])

    def __init__(self):
        super( Lmark2img_Generator2, self).__init__()

        # encoding layers
       
        self.conv1 = ResidualBlockDown(3, 64)            #(64,128,128)
        self.in1_e = nn.InstanceNorm2d(64, affine=True)

        self.conv2 = ResidualBlockDown(64, 128)      #(128,64,64)
        self.in2_e = nn.InstanceNorm2d(128, affine=True)

        self.conv3 = ResidualBlockDown(128, 256)      #(256,32,32)
        self.in3_e = nn.InstanceNorm2d(256, affine=True)

        self.att1 = SelfAttention(256)

        self.conv4 = ResidualBlockDown(256, 512)        #(512,16,16)
        self.in4_e = nn.InstanceNorm2d(512, affine=True)

        self.conv5 = ResidualBlockDown(512, 512)        #(512,8,8)
        self.in5_e = nn.InstanceNorm2d(512, affine=True)

        self.conv6 = ResidualBlockDown(512, 512)        #(512,4,4)
        self.in6_e = nn.InstanceNorm2d(512, affine=True)

        activ='relu'
        pad_type='reflect'
        self.model = []
        self.model += [Conv2dBlock(3, 64, 7, 1, 3,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type)]
        self.model += [Conv2dBlock(64, 128, 4, 2, 1,           # 128, 128, 128 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        self.model += [Conv2dBlock(128, 128, 4, 2, 1,           # 128, 64 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        self.model += [Conv2dBlock(128, 256, 4, 2, 1,           # 256 32 
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        self.model += [Conv2dBlock(256, 256, 4, 2, 1,           # 256 16
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        self.model += [Conv2dBlock(256, 512, 4, 2, 1,           # 512 8
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        self.model += [Conv2dBlock(512, 512, 4, 2, 1,           # 512 4
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]
        

        self.lmark_encoder = nn.Sequential(*self.model)


        self.model = []

        
        self.model += [ResBlocks(2, 512, norm  = 'adain', activation=activ, pad_type='reflect')]

        self.bottle =  nn.Sequential(*[Conv2dBlock(1024, 512, 3, 1, 1,           # 512 4
                                       norm= 'in',
                                       activation=activ,
                                       pad_type=pad_type)]

        )

        self.adainlayers = nn.Sequential(*self.model)
        self.model =[]
        self.model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]    # 512, 8 , 8 
        self.model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 512, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 512, 16 , 16 
        self.model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(512, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 32, 32 
        self.model += [nn.Upsample(scale_factor=2),
                        Conv2dBlock(256, 256, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)] # 256, 64, 64 
        self.model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(256, 128, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 128, 128, 128 
        self.model += [nn.Upsample(scale_factor=2), 
                        Conv2dBlock(128, 64, 5, 1, 2,
                                    norm='in',
                                    activation=activ,
                                    pad_type=pad_type)]  # 64, 256, 256 
        self.model += [Conv2dBlock(64, 3, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.decoder = nn.Sequential(*self.model)


        # self.apply(weights_init)


        self.mlp = MLP(512,
                       get_num_adain_params(self.decoder),
                       256,
                       3,
                       norm='none',
                       activ='relu')

        # self.apply(weights_init)
        
    def forward(self, ani, lmark, e):

        out = ani  # [B, 3, 256, 256]

      
        # Encode
        out = self.in1_e(self.conv1(out))  # [B, 64, 128, 128]
        out = self.in2_e(self.conv2(out))  # [B, 128, 64, 64]
        out = self.in3_e(self.conv3(out))  # [B, 256, 32, 32]
        out = self.att1(out)
        out = self.in4_e(self.conv4(out))  # [B, 512, 16, 16]
        out = self.in5_e(self.conv5(out))  # [B, 512, 8, 8]
        out = self.in6_e(self.conv6(out))  # [B, 512, 4, 4]
    
        # Decode
        adain_params = self.mlp(e)
        assign_adain_params(adain_params, self.adainlayers)

        pose_feature = self.adainlayers(out)

        lmark_feature = self.lmark_encoder(lmark)

        feature = torch.cat([pose_feature, lmark_feature], 1)

        feature = self.bottle(feature)


        image = self.decoder(feature)
        return image
# from torch.autograd import Variable

# import numpy as np
# model = Lmark2img_Generator2().cuda()
# torch.set_printoptions(precision=6)
# print (model)
# # a = torch.zeros(2, 136).cuda()

# # a = a + 0.0001
# # a = Variable(a)

# # print (a.data)
# # a.shape
# b = torch.FloatTensor(2, 6, 256, 256).cuda()
# b = Variable(b)

# a = torch.FloatTensor(2, 512)
# a = Variable(a).cuda()

# g= model(b, a)
# # print ('================')





class Lmark2img_Discriminator(nn.Module):
    def __init__(self, use_ani = False):
        super(Lmark2img_Discriminator, self).__init__()
        if use_ani == False:
            self.conv1 = ResidualBlockDown(6, 64)
        
        else:
            self.conv1 = ResidualBlockDown(9, 64)
        self.conv2 = ResidualBlockDown(64, 128)
        self.conv3 = ResidualBlockDown(128, 256)
        self.att = SelfAttention(256)
        self.conv4 = ResidualBlockDown(256, 512)
        self.conv5 = ResidualBlockDown(512, 512)
        self.conv6 = ResidualBlockDown(512, 512)
        self.res_block = ResidualBlock(512)
        self.last_fc = nn.Conv2d(512, 128, kernel_size=4, stride=1)
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.linear = nn.Linear(512, 1)

        # self.apply(weights_init)
    

    def forward(self, x, y): #x:  img, y: landmark 
        # assert x.dim() == 4 and x.shape[1] == 3, "Both x and y must be tensors with shape [BxK, 3, W, H]."
        # assert x.shape == y.shape, "Both x and y must be tensors with shape [BxK, 3, W, H]."


        # Concatenate x & y
        out = torch.cat((x, y), dim=1)  # [B, 6, 256, 256]

        # Encode
        out_0 = (self.conv1(out))  # [B, 64, 128, 128]
        out_1 = (self.conv2(out_0))  # [B, 128, 64, 64]
        out_2 = (self.conv3(out_1))  # [B, 256, 32, 32]
        out_3 = self.att(out_2)
        out_4 = (self.conv4(out_3))  # [B, 512, 16, 16]
        out_5 = (self.conv5(out_4))  # [B, 512, 8, 8]
        out_6 = (self.conv6(out_5))  # [B, 512, 4, 4]
        out_7 = (self.res_block(out_6))
        
        out = F.relu(self.pooling(out_7)).view(-1, 512)  # [B, 512]

        out = self.linear(out)

        return out  #, [out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7]

