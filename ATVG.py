import torch
import torch.nn as nn
import torchvision.models as models
import functools
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from collections import namedtuple
from torchvision import models


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
    
class AT_net(nn.Module):
    def __init__(self):
        super(AT_net, self).__init__()
        self.landmark_encoder = nn.Sequential(
            nn.Linear(48,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),

            )
        self.encode_lstm = nn.LSTM(512,256,3,batch_first = True)
        
        self.aggregator = nn.Sequential(
            nn.Linear(256*64,512),
#             nn.Dropout(p=dropout),
            nn.ReLU(True)
            )
        
        self.decode_lstm = nn.LSTM(512+128,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,48),
            nn.Tanh()
            )

    def forward(self, derivative):
        hidden1 = ( torch.autograd.Variable(torch.zeros(3, derivative.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, derivative.size(0), 256).cuda()))
        hidden2 = ( torch.autograd.Variable(torch.zeros(3, derivative.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, derivative.size(0), 256).cuda()))
        lstm_input = []        
        for step_t in range(derivative.size(1)):
            current_landmark = derivative[:, step_t ]            
            current_fea = self.landmark_encoder(current_landmark)
            lstm_input.append(current_fea)
        lstm_input = torch.stack(lstm_input, dim = 1)
        encoder_out, _ = self.encode_lstm(lstm_input, hidden1)
        encoder_out.data = encoder_out.data.contiguous()
        encoder_out = encoder_out.view(encoder_out.size(0), -1)
        
        movement_vector0 = self.aggregator(encoder_out)
        movement_vector = movement_vector0.unsqueeze(1).repeat(1,64,1)
        
        fixed_noise = torch.randn(encoder_out.size(0),derivative.size(1), 128).cuda()
        decoder_input = torch.cat([fixed_noise, movement_vector], -1 )
        
        decoder_output, decoder_hidden = self.decode_lstm(decoder_input,hidden2)
        
        
        
        fc_out   = []
        for step_t in range(derivative.size(1)):
            fc_in = decoder_output[:,step_t,:]
            fc_out.append(self.lstm_fc(fc_in))
        return torch.stack(fc_out, dim = 1)




def _apply(layer, activation, normalizer, channel_out=None):
    if normalizer:
        layer.append(normalizer(channel_out))
    if activation:
        if activation == nn.Sigmoid:
            layer.append(activation())
        else:    
            layer.append(activation)
    return layer

def conv2d(channel_in, channel_out,
           ksize=3, stride=1, padding=1,
           activation=nn.LeakyReLU(0.2, True),
           normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Conv2d(channel_in, channel_out,
                     ksize, stride, padding,
                     bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    # init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)
class AT_net_RT(nn.Module):
    def __init__(self):
        super(AT_net_RT, self).__init__()
        self.landmark_encoder = nn.Sequential(
            conv2d(1,64,3,(2,1),1),
            conv2d(64,64,3,(2,1),1),
            conv2d(64,128,3,(2,1),1),
            conv2d(128,128,3,2,1),
            conv2d(128,256,3,2,1),
#             conv2d(256,256,3,2,1)
            )
        
        
        self.encode_lstm = nn.LSTM(256,256,3,batch_first = True)
        
        self.aggregator = nn.Sequential(
            nn.Linear(256*4,128),
#             nn.Dropout(p=dropout),
            nn.ReLU(True)
            )
        
        self.decode_lstm = nn.LSTM(128,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,6)            
        )

    def forward(self, derivative):
        
        hidden= ( torch.autograd.Variable(torch.zeros(3, derivative.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, derivative.size(0), 256).cuda()))
        derivative = derivative.unsqueeze(1)      
        encoder_out = self.landmark_encoder(derivative)
        encoder_out = encoder_out.view(encoder_out.size(0), -1)
        
        movement_vector0 = self.aggregator(encoder_out)
        movement_vector = movement_vector0.unsqueeze(1).repeat(1,64,1)
        
        
        decoder_output, decoder_hidden = self.decode_lstm(movement_vector,hidden)
        
        fc_out   = []
        for step_t in range(derivative.size(2)):
            fc_in = decoder_output[:,step_t,:]
            fc_out.append(self.lstm_fc(fc_in))
        return torch.stack(fc_out, dim = 1)

    
class AT_net_RT_FC(nn.Module):
    def __init__(self):
        super(AT_net_RT_FC, self).__init__()
        self.landmark_encoder = nn.Sequential(
            nn.Linear(64*6,512),
#             nn.Dropout(p=dropout),
            nn.ReLU(True),
            nn.Linear(512,256),
#             nn.Dropout(p=dropout),
            nn.ReLU(True),
            nn.Linear(256,128),
#             nn.Dropout(p=dropout),
            nn.ReLU(True),
            nn.Linear(128,64),
#             nn.Dropout(p=dropout),
            nn.ReLU(True),

            )
        
        
       
        self.decoder = nn.Sequential(
            
            nn.Linear(64,128),
#             nn.Dropout(p=dropout),
            nn.ReLU(True),
            nn.Linear(128,256),
#             nn.Dropout(p=dropout),
            nn.ReLU(True),
            nn.Linear(256,512),
#             nn.Dropout(p=dropout),
            nn.ReLU(True),
            nn.Linear(512, 64*6),
            nn.Tanh()
#             nn.Dropout(p=dropout),
#             nn.ReLU(True),
        )

    def forward(self, derivative):
        
        derivative = derivative.view(derivative.size(0), -1)
        encoder_out = self.landmark_encoder(derivative)
        
        fc_out = self.decoder(encoder_out)
        
        fc_out = fc_out.view(derivative.size(0), 64 , 6)
        
       
        return fc_out 
    
class AT_net_RT_CNN(nn.Module):
    def __init__(self):
        super(AT_net_RT_CNN, self).__init__()
 
        self.landmark_encoder1 = nn.Sequential(
            conv2d(1,64,3,(2,1),1),
            conv2d(64,64,3,(2,1),1),
            conv2d(64,128,3,(2,1),1)
        )
        self.resblocks = []
        for i in range(6):
            self.resblocks += [ResnetBlock(128, padding_type='zero', norm_layer=nn.BatchNorm2d, use_dropout=True, use_bias=False)]
        self.landmark_encoder2 = nn.Sequential(
            conv2d(128,128,3,2,1),
            conv2d(128,256,3,2,1),
#             conv2d(256,256,3,2,1)
        )
        
        self.fc = nn.Sequential(nn.Linear(256* 4,256 * 4 * 3),
            nn.ReLU(True))
        
        self.resblocks = nn.Sequential(*self.resblocks)
        model =[]
        
        model += [nn.ConvTranspose2d(256, 128,
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=False),
                      nn.BatchNorm2d(128),
                      nn.ReLU(True)]
        for i in range(6):
            model += [ResnetBlock(128, padding_type='zero', norm_layer=nn.BatchNorm2d, use_dropout=True, use_bias=False)]
        self.decoder_conv = nn.Sequential( * model )
        
        self.decoder_fc = nn.Sequential(
            
            nn.Linear(128*8*6,1024),
            nn.ReLU(True),
            nn.Linear(1024,512),
            nn.ReLU(True),
            nn.Linear(512, 64*6),
        )

    def forward(self, derivative):
        
      
        derivative = derivative.unsqueeze(1)      
        encoder_out1 = self.landmark_encoder1(derivative)
        encoder_out2 = self.resblocks(encoder_out1)
        encoder_out3 = self.landmark_encoder2(encoder_out2)        
        intermedian = encoder_out3.view(encoder_out3.size(0), -1)    
        fc = self.fc(intermedian)
        
        conv_f = fc.view(fc.size(0), 256,4,3)                  
        conv_output = self.decoder_conv(conv_f)
        conv_output = conv_output.view(fc.size(0),-1)
        fc_out = self.decoder_fc(conv_output)
        fc_out = fc_out.view(fc.size(0), 64 , 6)
        return fc_out
    

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class LipNet(nn.Module):
    def __init__(self ):
        super(LipNet,self).__init__()
        dtype            = torch.FloatTensor
        self.lip_encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            conv2d(3 * 64, 64, 7,1, 0),
#             conv2d(64,64,3,2,1),
            conv2d(64,128,3,2,1),
            conv2d(128,128, 3,2,1),
            conv2d(128,256, 3,2,1),
            conv2d(256,512,3,2,1)
            )
#         self.fc = nn.Linear(512 , 256)

        use_bias = 'False'
        norm_layer = nn.BatchNorm2d
        self.landmark_encoder = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            )
        
        self.bottle = conv2d(512+ 256,512, 3,1,1)
        model = []
        model += [nn.ConvTranspose2d(512, int(256),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(256),
                      nn.ReLU(True)]
        
        model += [nn.ConvTranspose2d(256, int(256),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(256),
                      nn.ReLU(True)]
        model += [nn.ConvTranspose2d(256, 128,
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(128),
                      nn.ReLU(True)]
        
        
        
        model += [nn.ConvTranspose2d(128, 64,
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(64),
                      nn.ReLU(True)]
        model += [nn.Conv2d(64, 3, kernel_size=7, padding=3)]
        model += [nn.Tanh()]
        self.img_decoder = nn.Sequential(*model)
        
        
    
    def forward(self,lip_base, lmark ):
        lip_feature = self.lip_encoder(lip_base)
      
        lmark_feature = self.landmark_encoder(lmark)

        lmark_feature = lmark_feature.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4,4)
        merge_feature = torch.cat([lip_feature, lmark_feature], dim = 1)
        merge_feature = self.bottle(merge_feature)
        img = self.img_decoder(merge_feature)
                
        return img
        
        
class LipNet_Discriminator(nn.Module):
    def __init__(self):
        super(LipNet_Discriminator, self).__init__()

        self.lip_encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            conv2d(3 , 64, 7,1, 0),
#             conv2d(64,64,3,2,1),
            conv2d(64,128,3,2,1),
            conv2d(128,128, 3,2,1),
            conv2d(128,256, 3,2,1),
            conv2d(256,256, 3,2,1),
            conv2d(256,512,3,2,1)
            )
        self.landmark_encoder = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            )
        
        self.img_fc = nn.Sequential(
            nn.Linear(512*2*2, 512),
            nn.ReLU(True),
            )
        self.decision = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(True),
            nn.Linear(256,1),
            nn.Sigmoid()
            )
        
    def forward(self, lip , lmark):
        
        lmark_feature= self.landmark_encoder(lmark)
        img_feature = self.lip_encoder(lip)
        img_feature= img_feature.view(img_feature.shape[0], -1)
        img_feature = self.img_fc(img_feature)
        merge_feature = torch.cat([img_feature, lmark_feature], dim = 1)
        decision = self.decision(merge_feature)
        return decision.view(decision.size(0))

    
    
    
class FaceNet(nn.Module):
    def __init__(self ):
        super(FaceNet,self).__init__()
        dtype            = torch.FloatTensor
        self.lip_encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            conv2d(3 * 64, 64, 7,1, 0),
#             conv2d(64,64,3,2,1),
            conv2d(64,128,3,2,1),
            conv2d(128,128, 3,2,1),
            conv2d(128,256, 3,2,1),
            conv2d(256,256, 3,2,1),
            conv2d(256,512,3,2,1),
            conv2d(512,512,3,2,1)
            )
#         self.fc = nn.Linear(512 , 256)

        use_bias = 'False'
        norm_layer = nn.BatchNorm2d
        self.landmark_encoder = nn.Sequential(
            nn.Linear(136, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            )
        
        self.bottle = conv2d(512+ 256,512, 3,1,1)
        model = []
        model += [nn.ConvTranspose2d(512, int(512),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(512),
                      nn.ReLU(True)]
        
        model += [nn.ConvTranspose2d(512, int(256),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(256),
                      nn.ReLU(True)]
        
        model += [nn.ConvTranspose2d(256, int(256),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(256),
                      nn.ReLU(True)]
        model += [nn.ConvTranspose2d(256, 128,
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(128),
                      nn.ReLU(True)]
        
        model += [nn.ConvTranspose2d(128, 128,
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(128),
                      nn.ReLU(True)]
        
        model += [nn.ConvTranspose2d(128, 64,
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(64),
                      nn.ReLU(True)]
        model += [nn.Conv2d(64, 3, kernel_size=7, padding=3)]
        model += [nn.Tanh()]
        self.img_decoder = nn.Sequential(*model)
        
        
    
    def forward(self,lip_base, lmark ):
        lip_feature = self.lip_encoder(lip_base)
      
        lmark_feature = self.landmark_encoder(lmark)

        lmark_feature = lmark_feature.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4,4)
        merge_feature = torch.cat([lip_feature, lmark_feature], dim = 1)
        merge_feature = self.bottle(merge_feature)
        img = self.img_decoder(merge_feature)
                
        return img
    
    
class Lmark2rgb_single(nn.Module):
    def __init__(self ):
        super(Lmark2rgb_single,self).__init__()
        dtype            = torch.FloatTensor
        self.facer_encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            conv2d(3, 64, 7,1, 0),
            conv2d(64,128,3,2,1),
            conv2d(128,128,3,2,1),
            conv2d(128,256, 3,2,1),) #32
        self.facer_encoder2 = nn.Sequential(
            conv2d(256,256, 3,2,1), #16
            conv2d(256,512,3,2,1), #8
            conv2d(512,512,3,2,1)  #4
            )
#         self.fc = nn.Linear(512 , 256)
            
        use_bias = 'False'
        norm_layer = nn.BatchNorm2d
        self.landmark_encoder = nn.Sequential(
            nn.Linear(204, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            )
        self.landmark_encoder2 = nn.Sequential(
            conv2d(64,256, 3,1,1), #4
            conv2d(256,512,3,1,1), #8
            )
        self.bottle =  nn.Sequential(
            conv2d(1024,512,3,1,1), #4
            )
        model = []
        for i in range(9):
            model += [ResnetBlock(512, padding_type='zero', norm_layer=norm_layer, use_dropout=True, use_bias=False)]

        model += [nn.ConvTranspose2d(512, int(512),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(512),
                      nn.ReLU(True)]
        
        model += [nn.ConvTranspose2d(512, int(256),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(256),
                      nn.ReLU(True)]
        
        model += [nn.ConvTranspose2d(256, int(256),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(256),
                      nn.ReLU(True)]
        self.img_decoder1= nn.Sequential(*model)
        model = []
        for i in range(9):
            model += [ResnetBlock(512, padding_type='zero', norm_layer=norm_layer, use_dropout=True, use_bias=False)]
        model += [nn.ConvTranspose2d(512, 256,
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(256),
                      nn.ReLU(True)]
        
        model += [nn.ConvTranspose2d(256, 128,
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(128),
                      nn.ReLU(True)]
        
        model += [nn.ConvTranspose2d(128, 64,
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(64),
                      nn.ReLU(True)]
        model += [nn.Conv2d(64, 3, kernel_size=7, padding=3)]
        model += [nn.Tanh()]
        self.img_decoder2 = nn.Sequential(*model)
        
        
        self.sigmoid = nn.Sigmoid()
        self.maxpool2d = nn.MaxPool2d(kernel_size = 64, stride = (1,1))
        
        self.lmark_fc  = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(True))
            
     
    def forward(self, reference_img, target_lmark ):
        
        ff1 = self.facer_encoder1(reference_img)
        ff2 = self.facer_encoder2(ff1)
        target_lmark_feature = self.landmark_encoder(target_lmark)  

        target_lmark_feature = target_lmark_feature.unsqueeze(2).unsqueeze(3).repeat(1,1,4,4)

        target_lmark_feature  = self.landmark_encoder2(target_lmark_feature)
        img_feat = torch.cat([ff2, target_lmark_feature], 1)
        img_feat  = self.bottle(img_feat)

        img_feature2 = self.img_decoder1(img_feat)

        img_feature2 = torch.cat([ff1,img_feature2],1)

        img = self.img_decoder2(img_feature2)
  
        return img    

class Lmark2rgb_single_Discriminator(nn.Module):
    def __init__(self):
        super(Lmark2rgb_single_Discriminator, self).__init__()

        self.face_encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            conv2d(3 , 64, 7,1, 0),
            conv2d(64,64,3,2,1),  #128
            conv2d(64,128,3,2,1),  #64
            conv2d(128,128, 3,2,1),  #32
            conv2d(128,256, 3,2,1),  #16
            conv2d(256,256, 3,2,1),  #4
            conv2d(256,512,3,2,1),  #2
            # conv2d(512,512,3,2,1)  #1
            )
        self.landmark_encoder = nn.Sequential(
            nn.Linear(204, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            )
        self.landmark_encoder2 = nn.Sequential(
            conv2d(64,256, 3,1,1), #4
            conv2d(256,512,3,1,1), #8
            )
        
        
        
        self.decision = nn.Sequential(
            conv2d(1024,128, 3,1,1),  #4
            conv2d(128, 1, 4, 1, 0, activation=nn.Sigmoid, normalizer=None)
            )
        
    def forward(self, img  , lmark):
        
        lmark_feature= self.landmark_encoder(lmark)

        lmark_feature = lmark_feature.unsqueeze(2).unsqueeze(3).repeat(1,1,4,4)

        lmark_feature  = self.landmark_encoder2(lmark_feature)
        img_feature = self.face_encoder(img)
        merge_feature = torch.cat([img_feature, lmark_feature], 1)
        decision = self.decision(merge_feature)
        return decision.view(decision.size(0))
# from torch.autograd import Variable

# import numpy as np

# model = Lmark2rgb_single_Discriminator().cuda()
# torch.set_printoptions(precision=6)

# a = torch.zeros(2, 136).cuda()

# a = a + 0.0001
# a = Variable(a)

# # print (a.data)
# a.shape
# b = torch.FloatTensor(2, 3, 256, 256).cuda()
# b = Variable(b)

# g= model(b, a)
# # print ('================')

    
class MFCC_Face_single(nn.Module):
    def __init__(self ):
        super(MFCC_Face_single,self).__init__()
        dtype            = torch.FloatTensor
        self.facer_encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            conv2d(3, 64, 7,1, 0),
            conv2d(64,128,3,2,1), #128
            conv2d(128,128,3,2,1), # 64
            conv2d(128,256, 3,2,1)) #32
        self.facer_encoder2 = nn.Sequential(
            conv2d(256,256, 3,2,1), #16
            conv2d(256,512,3,2,1), #8
            conv2d(512,512,3,2,1)  #4
            )
#         self.fc = nn.Linear(512 , 256)
        self.mfcc_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(2,1)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            nn.MaxPool2d(3, stride=(2,2)),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            ) 
        self.audio_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True),
        )
        use_bias = 'False'
        norm_layer = nn.BatchNorm2d 
        # self.3d_encoder = nn.Sequential(
            # nn.Linear(136, 256),
            # nn.ReLU(True),
            # nn.Linear(256, 512),
            # nn.ReLU(inplace=True),
            # )
        self.bottle = nn.Sequential(
            conv2d(1024, 512,3,1,1)
        )
        model = []
        for i in range(9):
            model += [ResnetBlock(512, padding_type='zero', norm_layer=norm_layer, use_dropout=True, use_bias=False)]

        model += [nn.ConvTranspose2d(512, int(512),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(512),
                      nn.ReLU(True)]
        
        model += [nn.ConvTranspose2d(512, int(256),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(256),
                      nn.ReLU(True)]
        
        model += [nn.ConvTranspose2d(256, int(256),
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(256),
                      nn.ReLU(True)]
        self.img_decoder1= nn.Sequential(*model)
        model = []
        for i in range(9):
            model += [ResnetBlock(512, padding_type='zero', norm_layer=norm_layer, use_dropout=True, use_bias=False)]
        model += [nn.ConvTranspose2d(512, 256,
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(256),
                      nn.ReLU(True)]
        
        model += [nn.ConvTranspose2d(256, 128,
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(128),
                      nn.ReLU(True)]
        
        model += [nn.ConvTranspose2d(128, 64,
                                         kernel_size=3, stride=(2),
                                         padding=(1), output_padding=1,
                                         bias=use_bias),
                      norm_layer(64),
                      nn.ReLU(True)]
        model += [nn.Conv2d(64, 3, kernel_size=7, padding=3)]
        model += [nn.Tanh()]
        self.img_decoder2 = nn.Sequential(*model)
        
      
            
     
    def forward(self, warped_img, mfcc  ):
        mfcc_feature = self.mfcc_eocder(mfcc)
        ff1 = self.facer_encoder1(warped_img)
        ff2 = self.facer_encoder2(ff1)
        mfcc_feature = mfcc_feature.view(mfcc_feature.size(0), -1)
        mfcc_feature = self.audio_fc(mfcc_feature)
        mfcc_feature = mfcc_feature.unsqueeze(2).unsqueeze(3).repeat(1,1,4,4)
        new_img_f = torch.cat([ff2, mfcc_feature], 1)
        new_img_f  = self.bottle(new_img_f)
        new_img_ff = self.img_decoder1(new_img_f)
        new_img = self.img_decoder2( torch.cat([new_img_ff, ff1 ], 1) )
        
        
        return new_img
        
class VG_base_Discriminator(nn.Module):
    def __init__(self):
        super(VG_base_Discriminator, self).__init__()

        self.face_encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            conv2d(3 , 64, 7,1, 0),
            conv2d(64,64,3,2,1),  #128
            conv2d(64,128,3,2,1),  #64
            conv2d(128,128, 3,2,1),  #32
            conv2d(128,256, 3,2,1),  #16
            conv2d(256,256, 3,2,1),  #4
            conv2d(256,512,3,2,1),  #2
            # conv2d(512,512,3,2,1)  #1
            )
        self.mfcc_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(2,1)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            nn.MaxPool2d(3, stride=(2,2)),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            ) 
        self.mfcc_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True),
        )
        
        
        self.decision = nn.Sequential(
        conv2d(1024,128, 3,1,1),  #4
        conv2d(128, 1, 4, 1, 0, activation=nn.Sigmoid, normalizer=None)
        )
        
    def forward(self, img , mfcc):
        
        mfcc_feature= self.mfcc_eocder(mfcc)
        mfcc_feature = mfcc_feature.view(mfcc_feature.shape[0], -1)
        mfcc_feature = self.mfcc_fc(mfcc_feature)
        mfcc_feature = mfcc_feature.unsqueeze(2).unsqueeze(3).repeat(1,1,4,4)



        img_feature = self.face_encoder(img)
        merge_feature = torch.cat([img_feature, mfcc_feature], dim = 1)
        decision = self.decision(merge_feature)
        return decision.view(decision.size(0))


      
# from torch.autograd import Variable

# import numpy as np

# model = FaceNet_Discriminator().cuda()
# torch.set_printoptions(precision=6)

# a = torch.zeros(2, 1, 28,12).cuda()

# a = a + 0.0001
# a = Variable(a)

# # print (a.data)
# a.shape
# b = torch.FloatTensor(2, 3, 256, 256).cuda()
# b = Variable(b)

# g= model(b, a)
# # print ('================')

class AT_single(nn.Module):
    def __init__(self):
        super(AT_single, self).__init__()
        self.lmark_encoder = nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),

            )
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
       
            )
        
        self.fuse_fc = nn.Sequential(
            nn.Linear(512 + 256,6),
            )

    def forward(self, audio, example_landmark):
       
        example_landmark_f = self.lmark_encoder(example_landmark)
        current_audio = audio.unsqueeze(1)
        current_feature = self.audio_eocder(current_audio)
        current_feature = current_feature.view(current_feature.size(0), -1)
        current_feature = self.audio_eocder_fc(current_feature)
        features = torch.cat([example_landmark_f,  current_feature], 1)
        
        output = self.fuse_fc(features)
        return output
    

# from torch.autograd import Variable

# import numpy as np
    
# model = AT_single().cuda()
# torch.set_printoptions(precision=6)

# a = torch.zeros(2,28,12).cuda()
# a = a + 0.0001
# a = Variable(a)

# # print (a.data)
# a.shape
# b = torch.FloatTensor(2, 6).cuda()
# b = Variable(b)

# g= model(a, b)
# # print ('================')

    
    

class AT_single2(nn.Module):   # 7 chunks audio -> 1 frame
    def __init__(self):
        super(AT_single2, self).__init__()
       
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1,normalizer = None),
            conv2d(64,128,3,1,1,normalizer = None),
            nn.MaxPool2d(3, stride=(2,2)),
            conv2d(128,256,3,1,1,normalizer = None),
            conv2d(256,256,3,1,1,normalizer = None),
            conv2d(256,512,3,1,1,normalizer = None),
            nn.MaxPool2d(3, stride=(2,2))
            )
        
        self.face_encoder = nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(True))
        
        self.lip_encoder = nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(True))
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(512 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
        )
        self.lip_decoder = nn.Sequential(
            nn.Linear(512, 6),
#             nn.Tanh()
            )
        self.other_decoder = nn.Sequential(
            nn.Linear(512, 6)
            )


    def forward(self, audio, face, lip):
        current_audio = audio.unsqueeze(1)
        current_feature = self.audio_eocder(current_audio)
        current_feature = current_feature.view(current_feature.size(0), -1)
        output = self.audio_eocder_fc(current_feature)
        
        face_feature = self.face_encoder(face)
        
        lip_feature = self.lip_encoder(lip)
        
        merged = torch.cat([output, face_feature], 1)
        lip_merged = torch.cat([output, lip_feature], 1)
        
        lip_t = self.lip_decoder(lip_merged)
        
        other = self.other_decoder(merged)
        
        
        
        return lip_t, other
    

    
class AT_single2_no_pca(nn.Module):    # 7 chunks audio -> 1 frame
    def __init__(self):
        super(AT_single2_no_pca, self).__init__()
       
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1,normalizer = None),
            conv2d(64,128,3,1,1,normalizer = None),
            nn.MaxPool2d(3, stride=(2,2)),
            conv2d(128,256,3,1,1,normalizer = None),
            conv2d(256,256,3,1,1,normalizer = None),
            conv2d(256,512,3,1,1,normalizer = None),
            nn.MaxPool2d(3, stride=(2,2))
            )
        
        self.face_encoder = nn.Sequential(
            nn.Linear(144,256),
            nn.ReLU(True))
        
        self.lip_encoder = nn.Sequential(
            nn.Linear(60,256),
            nn.ReLU(True))
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(512 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
        )
        self.lip_decoder = nn.Sequential(
            nn.Linear(512, 60),
#             nn.Tanh()
            )
        self.other_decoder = nn.Sequential(
            nn.Linear(512, 144)
            )


    def forward(self, audio, face, lip):
        current_audio = audio.unsqueeze(1)
        current_feature = self.audio_eocder(current_audio)
        current_feature = current_feature.view(current_feature.size(0), -1)
        output = self.audio_eocder_fc(current_feature)
        
        face_feature = self.face_encoder(face)
        
        lip_feature = self.lip_encoder(lip)
        
        merged = torch.cat([output, face_feature], 1)
        lip_merged = torch.cat([output, lip_feature], 1)
        
        lip_t = self.lip_decoder(lip_merged)
        
        other = self.other_decoder(merged)
        
        
        
        return lip_t, other
        
    
class AT_single3(nn.Module):    # 3 chunks audio -> 1 frame
    def __init__(self):
        super(AT_single3, self).__init__()
       
        self.audio_encoder = nn.Sequential(
            conv2d(1,64,3,1,1,normalizer = None),
            conv2d(64,128,3,1,1,normalizer = None),
            nn.MaxPool2d(3, stride=(2,2)),
            conv2d(128,256,3,1,1,normalizer = None),
            conv2d(256,256,3,1,1,normalizer = None),
            conv2d(256,512,3,1,1,normalizer = None),
            nn.MaxPool2d(3, stride=(2,2))
            )
        
        self.face_encoder = nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(True))
        
        self.lip_encoder = nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(True))
        self.audio_encoder_fc = nn.Sequential(
            nn.Linear(512 *4,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
        )
        self.lip_decoder = nn.Sequential(
            nn.Linear(512, 6),
#             nn.Tanh()
            )
        self.other_decoder = nn.Sequential(
            nn.Linear(512, 6)
            )


    def forward(self, audio, face, lip):
        current_audio = audio.unsqueeze(1)
        current_feature = self.audio_encoder(current_audio)
        current_feature = current_feature.view(current_feature.size(0), -1)
        output = self.audio_encoder_fc(current_feature)
        
        face_feature = self.face_encoder(face)
        
        lip_feature = self.lip_encoder(lip)
        
        merged = torch.cat([output, face_feature], 1)
        lip_merged = torch.cat([output, lip_feature], 1)
        
        lip_t = self.lip_decoder(lip_merged)
        
        other = self.other_decoder(merged)
        return lip_t, other

class AT_net(nn.Module):
    def __init__(self):
        super(AT_net, self).__init__()
        self.lmark_encoder = nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),

            )
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
       
            )
        self.lstm = nn.LSTM(256*3,256,3,batch_first = True)
        self.lstm_fc = nn.Sequential(
            nn.Linear(256,6),
            )

    def forward(self, example_landmark, audio):
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()))
        example_landmark_f = self.lmark_encoder(example_landmark)
        lstm_input = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio)
            current_feature = current_feature.view(current_feature.size(0), -1)
            current_feature = self.audio_eocder_fc(current_feature)
            features = torch.cat([example_landmark_f,  current_feature], 1)
            lstm_input.append(features)
        lstm_input = torch.stack(lstm_input, dim = 1)
        lstm_out, _ = self.lstm(lstm_input, hidden)
        fc_out   = []
        for step_t in range(audio.size(1)):
            fc_in = lstm_out[:,step_t,:]
            fc_out.append(self.lstm_fc(fc_in))
        return torch.stack(fc_out, dim = 1)        
    
    
class AT_lstm(nn.Module):
    def __init__(self):
        super(AT_lstm, self).__init__()
       
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1,normalizer = None),
            conv2d(64,128,3,1,1,normalizer = None),
            nn.MaxPool2d(3, stride=(2,2)),
            conv2d(128,256,3,1,1,normalizer = None),
            conv2d(256,256,3,1,1,normalizer = None),
            conv2d(256,512,3,1,1,normalizer = None),
            nn.MaxPool2d(3, stride=(2,2))
            )
        
        self.face_encoder = nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(True))
        
        self.lip_encoder = nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(True))
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(512 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),
        )
        self.lstm = nn.LSTM(256 * 2,128,3,batch_first = True)
        
        

        self.lip_decoder = nn.Sequential(
            nn.Linear(128, 6),
#             nn.Tanh()
            )
        self.other_decoder = nn.Sequential(
            nn.Linear(128, 6)
            )


    def forward(self, audio, face, lip):
        hidden1 = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 128).cuda()),
                  torch.autograd.Variable(torch.zeros(3, audio.size(0), 128).cuda()))
        
        hidden2 = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 128).cuda()),
                  torch.autograd.Variable(torch.zeros(3, audio.size(0), 128).cuda()))
        
        face_feature = self.face_encoder(face)
        lip_feature = self.lip_encoder(lip)
        lip_input = []
        face_input =[]
        for step_t in range(audio.shape[1]):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)
            current_feature = self.audio_eocder(current_audio)
            current_feature = current_feature.view(current_feature.size(0), -1)
            output = self.audio_eocder_fc(current_feature)
            face_merged = torch.cat([output, face_feature], 1)
            lip_merged = torch.cat([output, lip_feature], 1)
            lip_input.append(lip_merged)
            face_input.append(face_merged)
            
        lip_input = torch.stack(lip_input, 1)
        face_input = torch.stack(face_input, 1)
        
        lip_outputs, _ = self.lstm(lip_input, hidden1)
        
        face_outputs, _ = self.lstm(face_input, hidden2)
        
        lip_output_fc = []
        face_output_fc = []
        for step_t in range(audio.size(1)):            
            lip_output_fc.append(self.lip_decoder(lip_outputs[:,step_t,:]))
            face_output_fc.append(self.other_decoder(face_outputs[:,step_t,:]))
        return torch.stack(lip_output_fc, dim = 1), torch.stack(face_output_fc, dim = 1)    
    
 
