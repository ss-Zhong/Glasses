from torchvision import datasets, transforms
import torch
from .Focura import Focura
from .contrastDataset import *
import os
from torch.utils.data import Subset
from lib.utils import logmsg
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

def build_dataset(dataset_type, data_set, data_path, transform = None):
    """
    n_shot is not useful for IMNET
    """

    if data_set == 'IMNET':
        root = os.path.join(data_path, dataset_type)
        dataset = datasets.ImageFolder(root, transform = transform)
        nb_classes = 1000

    return dataset, nb_classes

class Get_Dataloader():
    def __init__(self, data_set, data_path, args):
        self.set_name = data_set
        self.data_path = data_path
        self.args = args
        self.nb_classes = -1

    # 加载数据集
    def load_data(self, is_train: bool, sub_indices = None, dataset_type = None, n_shot = 10, contrast = False):
        '''
            is_train: dataset is RandomSampler or SequentialSampler
            dataset_type: which dataset is Loader (train set or test set)
        '''
        if self.set_name == 'Focura':
            transform = build_transform(is_train = is_train, args = self.args)
            return Focura(self.data_path, transform = transform, n_shot = n_shot, n_way = self.args.n_way, n_episodes = 100)

        if dataset_type is None:
            dataset_type = 'train' if is_train else 'val'

        if contrast:
            transform_clr = build_transform(is_train = is_train, args = self.args, contrast = True)
            root = os.path.join(self.data_path, dataset_type)
            if self.args.loss_type == 'iBot':
                dataset = maskDataset(root=root, transform = transform_clr)
            else:
                dataset = contrastDataset(root=root, transform = transform_clr)
        
        else:
            transform = build_transform(is_train = is_train, args = self.args)
            dataset,  self.nb_classes = build_dataset(dataset_type = dataset_type,
                                                  data_set = self.set_name, data_path=self.data_path,
                                                  transform = transform)
        
        if sub_indices is not None:
            dataset = Subset(dataset, sub_indices)

        logmsg(f"len of {dataset_type} dataset: {len(dataset)}")
        if is_train:
            logmsg("RandomSampler")
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            logmsg("SequentialSampler")
            sampler = torch.utils.data.SequentialSampler(dataset)

        data_loader = torch.utils.data.DataLoader(
            dataset, sampler = sampler,
            batch_size = self.args.batch_size,
            num_workers = self.args.num_workers,
            pin_memory = True,
            drop_last = False
        )

        return data_loader

def build_transform(is_train, args, contrast = False, patchmix = False, cropscale = (1, 1)):
    if contrast:
        transform = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=3),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomRotation((90, 90)),          # 90 度旋转
                    transforms.RandomRotation((180, 180)),        # 180 度旋转
                    transforms.RandomRotation((270, 270))         # 270 度旋转
                ])
            ], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=7),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        return transform
    
    elif patchmix:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=args.input_size, scale=cropscale, ratio=(0.9, 1.4)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomRotation((90, 90)),          # 90 度旋转
                    transforms.RandomRotation((180, 180)),        # 180 度旋转
                    transforms.RandomRotation((270, 270))         # 270 度旋转
                ])
            ], p=0.5),
            transforms.GaussianBlur(kernel_size=7),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        return transform

    else:
        resize_im = args.input_size > 32
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                # color_jitter=args.color_jitter,
                # auto_augment=args.aa,
                # interpolation=args.train_interpolation,
                # re_prob=args.reprob,
                # re_mode=args.remode,
                # re_count=args.recount,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)
            return transform

        else:
            t = []
            if resize_im:
                size = int((256 / 224) * args.input_size)
                t.append(
                    transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(args.input_size))

            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
            return transforms.Compose(t)
