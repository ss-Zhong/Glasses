import models
from models.iBotHead import *
import dataset
import torch
from pathlib import Path
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.data import Mixup
from lib.utils import logmsg, view_model_param
from lib import ibotutils
from lib.engine import *
from lib.all_loss import NTXentLoss, MixAwareLoss, iBOTLoss, DINOLoss
import json
import re
import random

# draw lib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid

class EdgeWorker():
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.output_dir = Path(args.output_dir)
        self.args = args
        
        # init edge model
        self.loss_type = args.loss_type

        self.classifier = models.create_peft_model(model = args.model, lora_rank=args.rank_edge, student=True)
        self.classifier_t = None
        
        if self.loss_type in ['iBot', 'Dino']:
            
            # teacher model
            self.classifier_t = models.create_peft_model(model = args.model, lora_rank=args.rank_edge, student=False)

            # Syn LoRA
            with torch.no_grad():
                for name, param_q in self.classifier.named_parameters():
                    if "lora" in name:
                        param_k = dict(self.classifier_t.named_parameters()).get(name)
                        if param_k is not None:
                            param_k.data = param_q.detach().data

        # view_model_param(self.classifier, log_file_path="output/console.txt")
        
        # get D_sub dataloader
        self.get_data = dataset.Get_Dataloader(data_set = args.edge_dataset, data_path = args.edge_data_path, args = args)
        self.edge_data_type = 'val'
        self.data_loader_base = self.get_data.load_data(is_train = False, dataset_type = self.edge_data_type)

        total_samples = len(self.data_loader_base.dataset)
        img_indices = random.sample(range(total_samples), self.args.subset_size) # sample D_sub
        self.data_loader_sub = self.get_data.load_data(sub_indices = img_indices, dataset_type = self.edge_data_type, is_train = True, contrast = True)


    def get_env(self, sample_img_path):
        if isinstance(sample_img_path, list) and len(sample_img_path)>1:  # Check if it's a list (multiple image paths)
            images = self.classifier.processor(images=image, return_tensors="pt")
            images = np.array([np.array(transforms.Resize((self.args.input_size, self.args.input_size))(Image.open(img_path).convert('RGB'))) for img_path in sample_img_path])
            mean_image = np.mean(images, axis=0).astype(np.uint8)
            image = Image.fromarray(mean_image)
        elif isinstance(sample_img_path, list):
            image = Image.open(sample_img_path[0]).convert('RGB')
        else:
            image = Image.open(sample_img_path).convert('RGB')
        
        image.save("output/env_image.png")

        return image
        

    def train_on_edge(self, scene_path = None):
        # get scene img
        if scene_path is not None:
            env_image =self.get_env(sample_img_path = scene_path)
        else:
            env_image = self.get_env(sample_img_path = self.args.sample_img_path)
        
        # prepare for train
        mixup_fn = None
        transform_mix = dataset.build_transform(is_train = True, args = self.args, patchmix=True, cropscale=(0.2, 0.7))
        optimizer = create_optimizer(self.args, self.classifier)
        lr_scheduler, _ = create_scheduler(self.args, optimizer)

        if self.loss_type == 'Dino':
            criterion = DINOLoss(
                65536, # args.out_dim,
                2,  # total number of crops = 2 global crops + local_crops_number
                0.04, # args.warmup_teacher_temp,
                0.04, # args.teacher_temp,
                0, # args.warmup_teacher_temp_epochs,
                1, # args.epochs,
            )
        elif self.loss_type == 'iBot':
            criterion = iBOTLoss(
                8192, # args.out_dim,
                8192, # args.out_dim if same_dim else args.patch_out_dim,
                2, # args.global_crops_number,
                0, # args.local_crops_number,
                0.04, #args.warmup_teacher_temp,
                0.07, # args.teacher_temp,
                0.04, # args.warmup_teacher_patch_temp,
                0.07, # args.teacher_patch_temp,
                0, # args.warmup_teacher_temp_epochs,
                1,
                lambda1=1.0,
                lambda2=1.0,
                mim_start_epoch=0,
            )
        elif self.loss_type == 'CrossEntropy':
            criterion = torch.nn.CrossEntropyLoss()

        # nt_criterion = NTXentLoss(batch_size=self.args.batch_size, temperature=0.5)
        nt_criterion = MixAwareLoss(batch_size=self.args.batch_size, temperature_pos=0.25, temperature_neg=0.5)

        self.classifier.to(self.device)
        if self.classifier_t:
            self.classifier_t.to(self.device)

        nt_criterion.to(self.device)
        criterion.to(self.device)

        # train begin
        for epoch in tqdm(range(1), desc='Training layer on Edge'): # one epoch for all
            train_one_epoch_AllLoss(
                self.classifier, self.data_loader_sub, self.device, epoch,
                criterion, nt_criterion, optimizer, mixup_fn = mixup_fn, freq = 1000, logger=False,
                mix_img=env_image, transform_mix = transform_mix, model_t = self.classifier_t, loss_type = self.loss_type
            )
            lr_scheduler.step(epoch)

        self.classifier.to("cpu")
        if self.classifier_t:
            self.classifier_t.to("cpu")
        nt_criterion.to("cpu")
        criterion.to("cpu")

        glasses = {k: v for k, v in self.classifier.named_parameters() if 'lora' in k}
        
        torch.save({'glasses': glasses,}, self.args.glasses_resume)