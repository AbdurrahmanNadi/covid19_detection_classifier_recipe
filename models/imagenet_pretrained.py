import yaml

import torch
import torch.nn as nn
from torchvision import models


class ImageNetPretrainedModel(nn.Module):
    def __init__(self, model_type='resnet50', n_cls=2):
        super(ImageNetPretrainedModel, self).__init__()
        
        if model_type == 'resnet18':
            self.cnn = models.resnet18(pretrained=True)
            outlayer_indim = self.cnn.fc.in_features
            self.cnn.fc = nn.Linear(outlayer_indim, n_cls)
        elif model_type == 'resnet50':
            self.cnn = models.resnet50(pretrained=True)
            outlayer_indim = self.cnn.fc.in_features
            self.cnn.fc = nn.Linear(outlayer_indim, n_cls)
        elif model_type == 'resnext50':
            self.cnn = models.resnext50_32x4d(pretrained=True)
            outlayer_indim = self.cnn.fc.in_features
            self.cnn.fc = nn.Linear(outlayer_indim, n_cls)
        elif model_type == 'efficientnet_b0':
            self.cnn = models.efficientnet_b0(pretrained=True)
            outlayer_indim = self.cnn.classifier[1].in_features
            self.cnn.classifier[1] = nn.Linear(outlayer_indim, n_cls)
        elif model_type == 'efficientnet_b3':
            self.cnn = models.efficientnet_b3(pretrained=True)
            outlayer_indim = self.cnn.classifier[1].in_features
            self.cnn.classifier[1] = nn.Linear(outlayer_indim, n_cls)

    def print(self):
        print(self.cnn)
    
    def forward(self, x):
        return self.cnn(x)

if __name__ == '__main__':
    conf_file = 'conf/imagenet_pretrained.yaml'
    with open(conf_file) as f:
        conf = yaml.load(f, yaml.Loader)
    print(conf)
    mdl_conf = conf['model']['conf']
    model = ImageNetPretrainedModel(mdl_conf['type'])

    model.to('cuda')
    x = torch.randn(8,3,224,224, device='cuda')

    y = model(x)
    print(y)
