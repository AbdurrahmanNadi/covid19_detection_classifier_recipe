import os
import glob

import cv2
import albumentations as A
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_dataloader(dataset, batch_size, shuffle=True, n_workers=8):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)


def get_one_hot(label, num_classes):
    one_hot = np.zeros((num_classes))
    one_hot[label] = 1
    return one_hot


class BaseImgDataset(Dataset):
    def __init__(self, dataset_dir, class_map, img_size=256, crop_size=224, img_affixs=['png'], use_augmentation=True, use_CLAHE=True):
        super().__init__()
        self.n_cls = len(class_map)
        
        self.labels = []
        self.imgs = []

        dirs = [os.path.join(dataset_dir, cls_dir) for cls_dir in class_map]

        for cls_dir in dirs:
            label = os.path.basename(cls_dir)
            for img_affix in img_affixs:
                imgs = glob.glob(os.path.join(cls_dir, '**.'+img_affix))
                self.imgs.extend(imgs)
                self.labels.extend([class_map[label] for _ in imgs])
     
        self.use_CLAHE = use_CLAHE
        self.clahe = A.Compose([A.CLAHE(clip_limit=(1,10), p=1)])

        if use_augmentation:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.RandomCrop(crop_size),
                    transforms.RandomHorizontalFlip(p=0.25)
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.CenterCrop(crop_size)
                ]
            )

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_file = self.imgs[index]
        
        label = torch.from_numpy(get_one_hot(self.labels[index], self.n_cls)).to(torch.float32)
        
        img_np = cv2.imread(img_file)
        if len(img_np.shape) < 3 or img_np.shape[2] == 1:
            img_np = img_np.squeeze()
            img_np = np.tile(img_np, (1,1,3))
        
        if self.use_CLAHE:
            img_np = self.clahe(image=img_np)['image']
        
        img_tensor = torch.from_numpy(img_np.transpose((2,0,1)))
        img_aug = self.transform(img_tensor).to(torch.float32) / 255
        
        return img_aug, label, img_file


class TrainDataset(BaseImgDataset):
    def __init__(self, dataset_dir, class_map, img_size=256, crop_size=224, img_affixs=['png'], use_augmentation=True, use_CLAHE=True):
        super().__init__(dataset_dir, class_map, img_size, crop_size, img_affixs, use_augmentation, use_CLAHE)

class DevDataset(BaseImgDataset):
    def __init__(self, dataset_dir, class_map, img_size=256, crop_size=224, img_affixs=['png'], use_CLAHE=True):
        super().__init__(dataset_dir, class_map, img_size, crop_size, img_affixs, False, use_CLAHE)

class TestDataset(Dataset):
    def __init__(self, dataset_dir, img_size=256, crop_size=224, img_affixs=['png'], use_CLAHE=True):
        super().__init__()

        self.imgs = []
        for img_affix in img_affixs:
            self.imgs.extend(glob.glob(os.path.join(dataset_dir, '**.'+img_affix)))

        self.use_CLAHE = use_CLAHE
        self.clahe = A.Compose([A.CLAHE(clip_limit=(1,10), p=1)])

        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(crop_size)
            ]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_file = self.imgs[index]
        img_np = cv2.imread(img_file)
        if len(img_np.shape) < 3 or img_np.shape[2] == 1:
            img_np = img_np.squeeze()
            img_np = np.tile(img_np, (1,1,3))
        
        if self.use_CLAHE:
            img_np = self.clahe(image=img_np)['image']
        
        img_tensor = torch.from_numpy(img_np.transpose((2,0,1)))
        img_aug = self.transform(img_tensor).to(torch.float32) / 255
        
        return img_aug, img_file


if __name__ == '__main__':
    class_map = {
        'NORMAL': 0,
        'COVID19': 1
    }
    train_dataset = TrainDataset('all_data/Chext-X-ray-Images-Data-Set/DataSet/Data/train', class_map, img_affixs=['png', 'jpeg', 'jpg'])
    valid_dataset = DevDataset('all_data/Chext-X-ray-Images-Data-Set/DataSet/Data/test', class_map, img_affixs=['png', 'jpeg', 'jpg'])
    test_dataset = TestDataset('all_data/Chext-X-ray-Images-Data-Set/DataSet/Data/train/NORMAL', img_affixs=['png', 'jpeg', 'jpg'])

    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))

    sample_dir = 'samples'

    train_samples = os.path.join(sample_dir, 'train')
    os.makedirs(train_samples, exist_ok=True)
    for i in np.random.randint(0, len(train_dataset), size=5):
        y, label, utt_id = train_dataset[i]
        y_np = y.cpu().numpy().transpose((1,2,0)) * 255
        y_np = y_np.astype(np.uint8)
        cv2.imwrite(os.path.join(train_samples, os.path.basename(utt_id)), y_np)
        print(label)
        print(y.max())
        print(y.min())
        print(utt_id)
        print(y.shape)
        print('--'*20)

    valid_samples = os.path.join(sample_dir, 'valid')
    os.makedirs(valid_samples, exist_ok=True)
    for i in np.random.randint(0, len(valid_dataset), size=5):
        y, label, utt_id = valid_dataset[i]
        y_np = y.cpu().numpy().transpose((1,2,0)) * 255
        y_np = y_np.astype(np.uint8)
        cv2.imwrite(os.path.join(valid_samples, os.path.basename(utt_id)), y_np)
        print(label)
        print(y.max())
        print(y.min())
        print(utt_id)
        print(y.shape)
        print('--'*20)

    test_samples = os.path.join(sample_dir, 'test')
    os.makedirs(test_samples, exist_ok=True)
    for i in np.random.randint(0, len(test_dataset), size=5):
        y, label, utt_id = test_dataset[i]
        y_np = y.cpu().numpy().transpose((1,2,0)) * 255
        y_np = y_np.astype(np.uint8)
        cv2.imwrite(os.path.join(test_samples, os.path.basename(utt_id)), y_np)
        print(label)
        print(y.max())
        print(y.min())
        print(utt_id)
        print(y.shape)
        print('--'*20)

    train_dataloader = get_dataloader(train_dataset, 8)

    y_batch, label_batch, _ = next(iter(train_dataloader))
    print(y_batch.shape)
    print(label_batch.shape)