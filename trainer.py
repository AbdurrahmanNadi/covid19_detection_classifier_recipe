import os
import sys
import yaml
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import tqdm
import numpy as np

from models.cnn_model import CNNClassifier
from models.covid_net import CovidNet
from models.imagenet_pretrained import ImageNetPretrainedModel
from data.dataset import TrainDataset, DevDataset


class ClassifierTrainer():
    def __init__(self, model, train_dataset, loss, device, save_dir, optim='Adam',
                scheduler=None, schd_params=None, model_affix='CNNCls', 
                optim_affix='Adam', dev_dataset=None, lr=0.001, batch_size=16,
                n_epochs=120, start_epoch=0, n_workers=8, summary_writer=None,
                log_interval=100, save_interval=10):

        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                        num_workers=n_workers, shuffle=True)
        self.criterion = loss.to(device)
        self.device = device
        self.save_dir = save_dir
        self.model_affix = model_affix
        self.optim_affix = optim_affix

        if optim == 'Adam':
            self.optim = Adam(self.model.parameters(), lr=lr)
        elif optim =='SGD':
            self.optim = SGD(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError(f'Specified optimizer {optim} not used. Only Adam and SGD supported')

        if scheduler == 'plateau':
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, **schd_params)
            self.use_scheduler = True
        elif scheduler == 'linear':
            self.scheduler = lr_scheduler.LinearLR(self.optim, **schd_params)
            self.use_scheduler = True
        elif scheduler == 'step':
            self.scheduler = lr_scheduler.StepLR(self.optim, **schd_params)
            self.use_scheduler = True
        else:
            self.use_scheduler = False

        if not dev_dataset:
            self.use_valid = False
        else:
            self.valid_dataset = dev_dataset
            self.use_valid = True
        
        if not summary_writer:
            self.use_logger = False
        else:
            self.use_logger = True
            self.logger = summary_writer
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.n_epochs = n_epochs
        self.n_iter = 0
        self.iters_per_epoch = len(list(iter(self.train_loader)))
        self.start_epoch = start_epoch
        self.log_interval = log_interval
        self.save_interval = save_interval
    
    def load_epoch(self, epoch):
        model_state_dict_path = os.path.join(self.save_dir, self.model_affix+f'_{epoch}.pt')
        optim_state_dict_path = os.path.join(self.save_dir, self.optim_affix+f'_{epoch}.pt')
        self.model.load_state_dict(torch.load(model_state_dict_path))
        self.optim.load_state_dict(torch.load(optim_state_dict_path))
    
    def save_epoch(self, epoch):
        mdl_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()

        model_state_dict_path = os.path.join(self.save_dir, self.model_affix+f'_{epoch}.pt')
        optim_state_dict_path = os.path.join(self.save_dir, self.optim_affix+f'_{epoch}.pt')

        torch.save(mdl_state_dict, model_state_dict_path)
        torch.save(optim_state_dict, optim_state_dict_path)
    
    def count_model_param(self):
        return sum([p.numel() for p in self.model.parameters()])
    
    def run_one_iter(self, x, target):
        self.optim.zero_grad()
        x = self.model(x)
        loss = self.criterion(x, target)
        loss.backward()
        self.optim.step()
        return loss
    
    @torch.no_grad()
    def eval(self, dataset):
        correct_preds = 0
        mean_loss = 0
        for x, label, _ in dataset:
            true_pred = np.argmax(label)

            x = x.to(self.device).unsqueeze(0)
            label = label.to(self.device).unsqueeze(0)
            
            y = self.model(x)
            loss = self.criterion(y, label).cpu().squeeze().numpy()
            
            y_norm = F.softmax(y, dim=1).cpu().squeeze().numpy()
            pred = np.argmax(y_norm)
            correct_preds += 1 if pred == true_pred else 0
            mean_loss += loss
        
        mean_loss = mean_loss / len(dataset)
        acc = correct_preds / len(dataset)

        return acc, mean_loss

    def train(self):

        if self.start_epoch > 0:
            load_epoch = (self.start_epoch - 1) / self.save_interval
            print(f'[INFO]: Loading model from epoch {load_epoch}', file=sys.stderr)
            self.load_epoch(load_epoch)
            self.n_iter = self.iters_per_epoch * load_epoch
        else:
            load_epoch = 0

        print('[Starting Training]', file=sys.stderr)
        print('[Model Architecture]', file=sys.stderr)
        self.model.print()
        print(f'[Number of Parameters:]{self.count_model_param()}')

        for epoch in range(load_epoch, self.n_epochs):
            
            self.model.train()
            train_loss = 0
            print(f'[Epoch({epoch+1}/{self.n_epochs})]', file=sys.stderr)
            for data in tqdm.tqdm(iter(self.train_loader)):
                x , targets, _ = data
                x = x.to(self.device)
                targets = targets.to(self.device)

                loss = self.run_one_iter(x, targets)
                self.n_iter += 1
                train_loss += loss.detach().cpu().numpy()

                if self.use_logger and (self.n_iter % self.log_interval == 0):
                    self.logger.add_scalar('Loss/Train', train_loss / self.log_interval, self.n_iter - 1)
                    train_loss = 0
                
            if epoch % self.save_interval == 0:
                print(f'[Saving epoch {epoch}]', file=sys.stderr)
                self.save_epoch(epoch)
            
            if self.use_logger:
                
                self.model.eval()

                train_acc, train_loss = self.eval(self.train_dataset)
                self.logger.add_scalar('Accuracy/Train', train_acc, epoch)
                
                if self.use_valid:
                    valid_acc, valid_loss = self.eval(self.valid_dataset)
                    self.logger.add_scalar('Loss/Valid', valid_loss, self.n_iter - 1)
                    self.logger.add_scalar('Accuracy/Valid', valid_acc, epoch)
            
            if type(self.scheduler) is lr_scheduler.ReduceLROnPlateau:
                if self.use_valid:
                    self.scheduler.step(valid_loss)
                else:
                    self.scheduler.step(train_loss)
            else:
                self.scheduler.step()

        print('[Training Done]', file=sys.stderr)
        print('[Saving Last Epoch]', file=sys.stderr)
        self.save_epoch(epoch)


def main(args):
    with open(args.conf) as f:
        conf = yaml.load(f, yaml.Loader)
    with open(args.class_map) as f:
        cls_map = yaml.load(f, yaml.Loader)
    
    dataset_conf = conf['dataset']
    model_conf = conf['model']
    trainer_conf = conf['trainer']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = TrainDataset(args.train_dataset, class_map=cls_map, **dataset_conf)
    dev_set = DevDataset(args.dev_dataset, class_map=cls_map, **dataset_conf)

    mdl_type = model_conf['type']
    if mdl_type == 'cnn':
        model = CNNClassifier(model_conf['conf'])
    elif mdl_type == 'pretrained':
        model = ImageNetPretrainedModel(**model_conf['conf'])
    elif mdl_type == 'covid_net':
        model = CovidNet(**model_conf['conf'])
    else:
        raise NotImplementedError(f'Model type {mdl_type} not implemented')
    
    criterion = nn.CrossEntropyLoss()
    logger = SummaryWriter(args.log_dir)

    trainer = ClassifierTrainer(model, train_set, criterion, device, args.save_dir,
                                dev_dataset=dev_set, summary_writer=logger, **trainer_conf)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dataset', default='all_data/Chext-X-ray-Images-Data-Set/DataSet/Data/train')
    parser.add_argument('--dev-dataset', default='all_data/Chext-X-ray-Images-Data-Set/DataSet/Data/test')
    parser.add_argument('--save-dir', default='exp/cnn_model/CNN5x/model1')
    parser.add_argument('--log-dir', default='logs/cnn_model/CNN5x/model1')
    parser.add_argument('--conf', default='conf/CNN5x.yaml')
    parser.add_argument('--class-map', default='conf/class_map.yaml')
    args = parser.parse_args()
    main(args)