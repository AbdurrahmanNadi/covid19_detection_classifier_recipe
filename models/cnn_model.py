import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):

    def __init__(self, layers_conf):
        super().__init__()
        
        conv_layers_conf = layers_conf['conv']
        conv_layers = []
        for idx in range(len(conv_layers_conf)):
            outchn, k1, k2, s1, s2, p1, p2, norm, act, pooling, kp1, kp2, sp1, sp2, pp1, pp2 = conv_layers_conf[idx]
            kernel_size = (k1,k2)
            s = (s1, s2)
            p = (p1, p2)
            pooling_kernel = (kp1, kp2)
            pooling_stride = (sp1, sp2)
            pooling_padding = (pp1, pp2)

            if idx > 0:
                prev_chn = conv_layers_conf[idx-1][0]
                conv_layers.append(nn.Conv2d(prev_chn, outchn, kernel_size, s, p))
            else:
                conv_layers.append(nn.Conv2d(layers_conf['inchn'],outchn, kernel_size, s, p))
            
            if norm == 'bn':
                conv_layers.append(nn.BatchNorm2d(outchn))
            elif norm == 'inst':
                conv_layers.append(nn.InstanceNorm2d(outchn))
            
            if act == 'relu':
                conv_layers.append(nn.ReLU())
            elif act == 'prelu':
                conv_layers.append(nn.PReLU(outchn))
            elif act == 'leakyrelu':
                conv_layers.append(nn.LeakyReLU())
            
            if pooling == 'avg':
                conv_layers.append(nn.AvgPool2d(pooling_kernel, pooling_stride, pooling_padding))
            elif pooling == 'max':
                conv_layers.append(nn.MaxPool2d(pooling_kernel, pooling_stride, pooling_padding))
        
        self.conv_enc = nn.Sequential(*conv_layers)
        conv_outchn = conv_layers_conf[-1][0]
        
        dense_layers_conf = layers_conf['dense']
        dense_layers = []
        for idx in range(len(dense_layers_conf)):
            outdim, act = dense_layers_conf[idx]

            if idx > 0:
                prev_dim = dense_layers_conf[idx-1][0]
                dense_layers.append(nn.Linear(prev_dim, outdim))
            else:
                dense_layers.append(nn.Linear(conv_outchn, outdim))
            
            if act == 'relu':
                dense_layers.append(nn.ReLU())
            elif act == 'prelu':
                dense_layers.append(nn.PReLU(outdim))
            elif act == 'leakyrelu':
                dense_layers.append(nn.LeakyReLU())
        
        self.dense_enc = nn.Sequential(*dense_layers)


    def print(self):
        print(self.conv_enc)
        idx = len(list(self.conv_enc.modules())) - 1
        print(f'({idx}): Global Average Pooling')
        print(self.dense_enc)

    def forward(self, x):
        x = self.conv_enc(x)
        x = torch.mean(x, dim=(2,3))
        x = self.dense_enc(x)
        return x
    

if __name__ == '__main__':
    conf_file = 'conf/cnn_model.yaml'

    with open(conf_file) as f:
        conf = yaml.load(f, Loader=yaml.Loader)
    mdl_conf = conf['model']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model = CNNClassifier(mdl_conf['conf']).to(device)
    cnn_model.print()

    with torch.no_grad():
        x = torch.randn(8,3,224,224, device=device)
        y = cnn_model.conv_enc(x)
        print(y.shape)
        y = torch.mean(y, dim=(2,3))
        print(y.shape)
        y = cnn_model.dense_enc(y)
        y = y.to(torch.device('cpu'))
        print(y.shape)
    print(y)
