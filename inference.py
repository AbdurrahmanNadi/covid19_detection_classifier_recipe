import yaml
import argparse

import albumentations as A
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.cnn_model import CNNClassifier
from models.covid_net import CovidNet
from models.imagenet_pretrained import ImageNetPretrainedModel


class InferenceModel():
    def __init__(self, conf_file, device, cls_map_file):
        with open(conf_file) as f:
            conf = yaml.load(f, yaml.Loader)
        with open(cls_map_file) as f:
            self.cls2idx = yaml.load(f, yaml.Loader)
        self.idx2cls = {v:k for k,v in self.cls2idx.items()}
        self.labels = [self.idx2cls[i] for i in range(len(self.idx2cls))]
        
        model_conf = conf['model']
        data_conf = conf['dataset']

        mdl_type = model_conf['type']
        if mdl_type == 'cnn':
            model = CNNClassifier(model_conf['conf'])
        elif mdl_type == 'pretrained':
            model = ImageNetPretrainedModel(**model_conf['conf'])
        elif mdl_type == 'covid_net':
            model = CovidNet(**model_conf['conf'])
        else:
            raise NotImplementedError(f'Model type {mdl_type} not implemented')

        self.model = model.to(device)
        self.model.eval()
        self.device = device

        self.use_CLAHE = data_conf['use_CLAHE']
        self.clahe = A.Compose([A.CLAHE((1,10), p=1)])
        self.transform = transforms.Compose(
            [
                transforms.Resize(data_conf['img_size']),
                transforms.CenterCrop(data_conf['crop_size'])
            ]
        )
    
    def load_model(self, ckpt):
        state_dict = torch.load(ckpt)
        self.model.load_state_dict(state_dict)
    
    def read_image(self, img_path):
        img_pil = Image.open(img_path).convert('RGB')
        return np.array(img_pil)

    @torch.no_grad()
    def preprocess(self, img):
        if len(img.shape) < 3 or img.shape[2] == 1:
            img = np.tile(img,(1,1,3))
        
        if self.use_CLAHE:
            img = self.clahe(image=img)['image']
        img_tensor = torch.from_numpy(np.transpose(img,(2,0,1)))
        img_tensor = self.transform(img_tensor) / 255
        return img_tensor.unsqueeze(0)

    @torch.no_grad()
    def infer(self, x):
        y_hat = self.model(x.to(self.device))
        y_hat = F.softmax(y_hat, dim=1)
        return y_hat.cpu().numpy().squeeze()
    
    def infer_label(self,x):
        y_pred = self.infer(x)
        cls_idx = np.argmax(y_pred)
        cls_lbl = self.idx2cls[cls_idx]
        return cls_lbl, y_pred, self.labels


def main(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Cuda specified as device however torch.cuda.isavailable returns False')
    device = torch.device(args.device)

    infernece_model = InferenceModel(args.conf, device, args.cls_map)
    infernece_model.load_model(args.ckpt)
   
    if args.mode == 'single':
        inputs = [args.input]
    else:
        with open(args.input) as f:
            inputs = [line.strip().split() for line in f.readlines()]

    for img_path in inputs:
        img = infernece_model.read_image(img_path)
        img_t = infernece_model.preprocess(img)
        label, _, _ = infernece_model.infer_label(img_t)
    
        print(f'{img_path}, {label}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'list'], type=str, default='single')
    parser.add_argument('--device', choices=['cuda', 'cpu'], type=str, default='cpu')
    parser.add_argument('--conf', type=str, default='conf/CNN5x.yaml')
    parser.add_argument('--ckpt', type=str, default='exp/cnn_model/CNN5x/model1/CNN5x_60.pt')
    parser.add_argument('--cls-map', type=str, default='conf/class_map.yaml')
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    main(args)