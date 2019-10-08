from network.vgg import Cropped_VGG19
from torchvision.models import vgg19

import torch
from torch import nn
from torch.nn import functional as F

import config

import imp

root = '/data2/lchen63/voxceleb'


class LossCnt(nn.Module):
    def __init__(self):
        super(LossCnt, self).__init__()
        
        self.VGG19 = vgg19(pretrained=True)
        self.VGG19.eval()
        
        VGGFace_body_path = os.path.join(root, 'vggface' , 'Pytorch_VGGFACE_IR.py')
        VGGFace_weight_path = os.path.join(root, 'vggface' , 'Pytorch_VGGFACE.pth')
        MainModel = imp.load_source('MainModel', VGGFace_body_path)
        full_VGGFace = torch.load(VGGFace_weight_path)
        cropped_VGGFace = Cropped_VGG19()
        cropped_VGGFace.load_state_dict(full_VGGFace.state_dict(), strict = False)
        self.VGGFace = cropped_VGGFace
        self.VGGFace.eval()

    def forward(self, x, x_hat, vgg19_weight=1e-2, vggface_weight=2e-3):
        l1_loss = nn.L1Loss()

        """Retrieve vggface feature maps"""
        with torch.no_grad(): #no need for gradient compute
            vgg_x_features = self.VGGFace(x) #returns a list of feature maps at desired layers

        vgg_xhat_features = self.VGGFace(x_hat)

        lossface = 0
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            lossface += l1_loss(x_feat, xhat_feat)


        """Retrieve vggface feature maps"""
        #define hook
        def vgg_x_hook(module, input, output):
            output.detach_() #no gradient compute
            vgg_x_features.append(output)
        def vgg_xhat_hook(module, input, output):
            vgg_xhat_features.append(output)
            
        vgg_x_features = []
        vgg_xhat_features = []

        vgg_x_handles = []
        
        conv_idx_list = [2,7,12,21,30] #idxes of conv layers in VGG19 cf.paper
        conv_idx_iter = 0
        
        
        #place hooks
        for i,m in enumerate(self.VGG19.features.modules()):
            if i == conv_idx_list[conv_idx_iter]:
                if conv_idx_iter < len(conv_idx_list)-1:
                    conv_idx_iter += 1
                vgg_x_handles.append(m.register_forward_hook(vgg_x_hook))

        #run model for x
        self.VGG19(x)

        #retrieve features for x
        for h in vgg_x_handles:
            h.remove()

        #retrieve features for x_hat
        conv_idx_iter = 0
        for i,m in enumerate(self.VGG19.modules()):
            if i <= 30: #30 is last conv layer
                if type(m) is not torch.nn.Sequential and type(m) is not torchvision.models.vgg.VGG:
                #only pass through nn.module layers
                    if i == conv_idx_list[conv_idx_iter]:
                        if conv_idx_iter < len(conv_idx_list)-1:
                            conv_idx_iter += 1
                        x_hat = m(x_hat)
                        vgg_xhat_features.append(x_hat)
                        x_hat.detach_() #reset gradient from output of conv layer
                    else:
                        x_hat = m(x_hat)
        
        loss19 = 0
        for x_feat, xhat_feat in zip(vgg_x_features, vgg_xhat_features):
            loss19 += l1_loss(x_feat, xhat_feat)

        loss = vgg19_weight * loss19 + vggface_weight * lossface

        return loss