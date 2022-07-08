import os
import argparse
import json
import yaml

import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn.functional as F

from data.dataset import DevDataset, TestDataset
from models.cnn_model import CNNClassifier
from models.covid_net import CovidNet
from models.imagenet_pretrained import ImageNetPretrainedModel


class EvalModel():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def load_model(self, checkpoint):
        model_state_dict = torch.load(checkpoint)
        self.model.load_state_dict(model_state_dict)
    
    def count_model_param(self):
        return sum([p.numel() for p in self.model.parameters()])

    @torch.no_grad()
    def infer(self, dataset, cls_map=None):
        pred = {}
        
        for data in tqdm.tqdm(dataset):
            if type(dataset) is DevDataset:
                x, _, file_name = data
            elif type(dataset) is TestDataset:
                x, file_name = data
            else:
                raise ValueError('Inference in EvalModel supports only DevDataset and TestDataset only')
            
            x_tensor = x.unsqueeze(0).to(self.device)
            y_out = self.model(x_tensor)

            y_pred = F.softmax(y_out, dim=1).cpu().numpy().squeeze()
            y_cls = np.argmax(y_pred)

            if cls_map is not None:
                pred[file_name] = cls_map[y_cls]
            else:
                pred[file_name] = y_cls
        
        return pred

    @torch.no_grad()
    def eval(self, dataset, cls_map=None, print_report=False):
        if type(dataset) is not DevDataset:
            raise ValueError('Evaluation in EvalModel supports only DevDataset')

        pred = {}
        y_pred = []
        y_true = []

        for x, label, file_name in tqdm.tqdm(dataset):
            x_tensor = x.unsqueeze(0).to(self.device)
            y_true.append(np.argmax(label.cpu().numpy().squeeze()))

            y_out = self.model(x_tensor)
            y_softmax = F.softmax(y_out, dim=1).cpu().squeeze().numpy()
            y_cls = np.argmax(y_softmax)
            y_pred.append(y_cls)

            if cls_map is not None:
                pred[file_name] = cls_map[y_cls]
            else:
                pred[file_name] = y_cls
        
        if cls_map is not None:
            labels = []
            for i in range(len(cls_map)):
                labels.append(cls_map[i])
            y_pred = [cls_map[p] for p in y_pred]
            y_true = [cls_map[p] for p in y_true]
        else:
            labels = None

        conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
        cls_report = classification_report(y_true, y_pred, labels=labels, output_dict=(not print_report))

        if print_report:
            print(cls_report)
            return pred, conf_mat
        else:
            return pred, conf_mat, cls_report


def main(args):
    with open(args.conf) as f:
        conf = yaml.load(f, yaml.Loader)
    
    with open(args.cls_map) as f:
        cls2idx = yaml.load(f, yaml.Loader)
        idx2cls = {v:k for k,v in cls2idx.items()}

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Device cuda specified but torch.cuda.is_available indicates no cuda')
    device = torch.device(args.device)
    
    model_conf = conf['model']
    mdl_type = model_conf['type']
    if mdl_type == 'cnn':
        model = CNNClassifier(model_conf['conf'])
    elif mdl_type == 'pretrained':
        model = ImageNetPretrainedModel(**model_conf['conf'])
    elif mdl_type == 'covid_net':
        model = CovidNet(**model_conf['conf'])
    else:
        raise NotImplementedError(f'Model type {mdl_type} not implemented')
    
    inference_model = EvalModel(model, device)
    inference_model.load_model(args.ckpt)

    os.makedirs(args.result_dir, exist_ok=True)
    print(f'[Number of Parameters] {inference_model.count_model_param()}')

    dataset_conf = conf['dataset']
    if args.dataset_type == 'dev':
        dataset = DevDataset(args.dataset, cls2idx, **dataset_conf)
        output = inference_model.eval(dataset, idx2cls, args.print_report)
        
        if not args.print_report:
            preds, conf_mat, cls_report = output
            cls_report_file = os.path.join(args.result_dir, 'cls_report.json')
            with open(cls_report_file, 'w') as f:
                json.dump(cls_report, f)
        else:
            preds, conf_mat = output

        conf_mat_file = os.path.join(args.result_dir, 'conf_mat.npy')
        np.save(conf_mat_file, conf_mat)
        print(conf_mat)

        pred_file = os.path.join(args.result_dir, 'preds.csv')
        with open(pred_file, 'w') as f:
            for file_name, pred in preds.items():
                print(f'{file_name},{pred}', file=f)
    else:
        dataset = TestDataset(args.dataset, **dataset_conf)
        preds = inference_model.infer(dataset, idx2cls)
        pred_file = os.path.join(args.result_dir, 'preds.csv')
        with open(pred_file, 'w') as f:
            for file_name, pred in preds.items():
                print(f'{file_name},{pred}', file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--dataset-type', type=str, choices=['dev', 'test'], default='dev')
    parser.add_argument('--cls-map', type=str, default='conf/class_map.yaml')
    parser.add_argument('--conf', type=str, default='conf/CNN5x.yaml')
    parser.add_argument('--ckpt', type=str, default='exp/cnn_model/CNN5x/model1/CNN5x_60.pt')
    parser.add_argument('--print-report', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='all_data/Chext-X-ray-Images-Data-Set/DataSet/Data/test')
    parser.add_argument('--result-dir', type=str, default='exp/cnn_model/CNN5x/model1/results')
    args = parser.parse_args()
    main(args)
