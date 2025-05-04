import wandb
import torch
import dataset
import models
from lib.utils import logmsg, view_model_param
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.data import Mixup
from lib.engine import *
from tqdm import tqdm

import cv2
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np

class Device():
    def __init__(self, args, n_shot = 5):
        self.device = torch.device(args.device)
        self.args = args

        self.get_data = dataset.Get_Dataloader(data_set = args.dataset, data_path=args.data_path, args = args)
        self.dataloader = self.get_data.load_data(is_train = False, n_shot = n_shot)

        # init_model
        self.classifier = models.create_peft_model(model = self.args.model, lora_rank = self.args.rank_edge, student=False)
    
    def load_Glasses(self):
        # load Glasses
        if self.args.glasses_resume != '':
            logmsg("Loading Glasses...")
            glasses_checkpoint = torch.load(self.args.glasses_resume, map_location='cpu', weights_only=False)
            self.classifier.load_state_dict(glasses_checkpoint['glasses'], strict=False)

    # few_shot on device
    def fewshot_on_device(self):

        self.classifier.to(self.device)
        test_stats = fs_evaluate(self.classifier, self.dataloader, self.device, test_both = True, n_way=self.args.n_way)
        acc, cof, loss = test_stats['acc'], test_stats['cof'], test_stats['loss']
        # 0: baseline++ 1: protonet
        logmsg(f'Acc@1 Cos: {acc[0]} | Euclidean: {acc[1]} | cof: {cof} | loss: {loss}')
        self.classifier.to("cpu")

        return test_stats