import torch
from timm.utils import accuracy
from typing import Iterable
import torch.nn.functional as F

from .utils import MetricLogger, logmsg, SmoothedValue
import numpy as np
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import math

from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, device: torch.device, epoch: int, 
          criterion: torch.nn.Module, nt_criterion, optimizer: torch.optim.Optimizer, mixup_fn = None,
          freq: int = 10, logger = False, key_layer = -1):
    
    # init logger
    if logger:
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        if mixup_fn is None:
            metric_logger.add_meter('acc1', SmoothedValue(window_size=len(data_loader)))
            metric_logger.add_meter('acc5', SmoothedValue(window_size=len(data_loader)))

        header = 'Epoch: [{}]'.format(epoch)
        loader_ = metric_logger.log_every(data_loader, freq, header)
    else:
        loader_ = data_loader

    model.train()
    criterion.train()

    for data in loader_:
        
        samples = data[0].to(device, non_blocking=True)
        targets = data[-1].to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        logits = model(samples)
        loss = criterion(logits, targets)
        # logmsg(f"loss: {loss.item()}")
        loss_value = loss.item()
        
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if logger:
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if mixup_fn is None:
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
                batch_size = samples.shape[0]
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    ret_dict = None
    if logger:
        ret_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return ret_dict

def train_one_epoch_AllLoss(model: torch.nn.Module, data_loader: Iterable, device: torch.device, epoch: int, 
          criterion: torch.nn.Module, nt_criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, mixup_fn = None,
          freq: int = 10, logger = False, mix_img = None, transform_mix = None,
          model_t = None, loss_type = 'CrossEntropy'):
    
    # init logger
    if logger and loss_type == 'CrossEntropy':
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        if mixup_fn is None:
            metric_logger.add_meter('acc1', SmoothedValue(window_size=len(data_loader)))
            metric_logger.add_meter('acc5', SmoothedValue(window_size=len(data_loader)))

        header = 'Epoch: [{}]'.format(epoch)
        loader_ = metric_logger.log_every(data_loader, freq, header)
    else:
        loader_ = tqdm(data_loader) if not logger else data_loader

    model.train()
    criterion.train()
    nt_criterion.train()

    scaler = GradScaler()
    for load_idx, data in enumerate(loader_):

        samples0 = data[0].to(device, non_blocking=True)
        samples1 = data[1].to(device, non_blocking=True)
        targets = data[2].to(device, non_blocking=True)

        # mix_mask = None
        batch_size = samples1.size(0)
        if mix_img is not None:
            # new mix
            reso = 7
            p = 0.3 if loss_type != 'iBot' else 0.3 / (1-0.3)# mix ratio
            num_patches = reso * reso
            num_ones = int(num_patches * p)

            mask = torch.zeros(batch_size, num_patches)
            for i in range(batch_size):
                idx = torch.randperm(num_patches)[:num_ones]
                mask[i, idx] = 1

            mask = mask.bool()
            mask = mask.view(batch_size, reso, reso)

            if loss_type == 'iBot':
                mask_mim = data[3][0]
                mask_to_zero = mask & mask_mim
                mask[mask_to_zero] = False

            mask = mask.repeat_interleave(224//reso, dim=1).repeat_interleave(224//reso, dim=2)
            mask = mask.unsqueeze(1).expand(-1, 3, -1, -1).to(device, non_blocking=True)
            
            mix_tensor = transform_mix(mix_img).unsqueeze(0).to(device)
            samples0 = samples0 * (~mask) + mix_tensor * mask

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_type == 'CrossEntropy':  
            samples = torch.cat([samples0, samples1], dim=0)
            targets = torch.cat([targets, targets], dim=0)

            with autocast():
                logits, proj_x = model(samples, proj=True)
                loss = criterion(logits, targets)

        elif loss_type == 'iBot':
            samples = [samples0, samples1]
            masks = [msk.to(device, non_blocking=True) for msk in data[3]]

            with autocast():
                teacher_output = model_t(samples)
                teacher_output = tuple(tensor.detach() for tensor in teacher_output)

                student_output, proj_x = model(samples, proj=True, mask=masks)
                proj_x = torch.cat([proj_x], dim=0)

                loss = criterion(student_output, teacher_output, None, masks, epoch).pop('loss')
                
        elif loss_type == 'Dino':
            samples = [samples0, samples1]

            with autocast():
                teacher_output = model_t(samples)
                teacher_output = teacher_output.detach()

                student_output, proj_x = model(samples, proj=True)
                proj_x = torch.cat([proj_x], dim=0)

                loss = criterion(student_output, teacher_output, epoch)
        
        nt_loss = nt_criterion(proj_x)

        if logger:
            logmsg(f"loss: {loss.item()} | nt_loss: {nt_loss.item()}")

        loss = loss * 0.5 + nt_loss * 0.5
        
        loss_value = loss.item()
        
        optimizer.zero_grad()

        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # There is no need for EMA due to the limited amount of data
        if loss_type in ['Dino', 'iBot']:
            with torch.no_grad():
                m = 0.3
                for name, param_q in model.named_parameters():
                    if "lora" in name:  # 只对 LoRA 相关的参数进行更新
                        param_k = dict(model_t.named_parameters()).get(name)
                        if param_k is not None:
                            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


        if logger and loss_type == 'CrossEntropy':
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if mixup_fn is None and loss_type == 'CrossEntropy':
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
                batch_size = samples.shape[0]
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    ret_dict = None
    if logger and loss_type == 'CrossEntropy':
        ret_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    return ret_dict

@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device, criterion: torch.nn.Module, freq: int = 10,  logger = False):

    metric_logger = MetricLogger(logger = logger, delimiter="  ")
    metric_logger.add_meter('acc1', SmoothedValue(window_size=len(data_loader)))
    metric_logger.add_meter('acc2', SmoothedValue(window_size=len(data_loader)))

    model.eval()
    
    correct_targets = []
    incorrect_targets = []

    for (samples, targets) in metric_logger.log_every(data_loader, freq, 'evaluate:'):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(samples)

        loss = criterion(logits, targets)
        loss_value = loss.item()
        
        # 计算预测正确的数量
        acc1, acc2 = accuracy(logits, targets, topk=(1, 2))
        batch_size = samples.shape[0]

        metric_logger.update(loss=loss_value)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)

    #     _, predicted = torch.max(logits, 1)
    #     correct_mask = (predicted == targets)
    #     correct_targets.append(targets[correct_mask])
    #     incorrect_mask = (predicted != targets)
    #     incorrect_targets.append([targets[incorrect_mask], predicted[incorrect_mask]])

    # logmsg("Yes: ", correct_targets)
    # logmsg("No: ", incorrect_targets)
    # logmsg('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc2))

    ret_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    ret_dict['acc_std'] = metric_logger.meters['acc1'].std

    return ret_dict

@torch.no_grad()
def fs_evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device, cosine_sim=False, test_both=False, n_way=5):

    model.eval()
    total_acc, total_loss = [[], []], np.array([0.0, 0.0])

    for i, data in enumerate(tqdm(data_loader)):
        if i == len(data_loader):
            break

        support_data = data['support_data'].to(device)      # [N*K, C, H, W]
        support_labels = data['support_labels'].to(device)  # [N*K]
        query_data = data['query_data'].to(device)          # [N*Q, C, H, W]
        query_labels = data['query_labels'].to(device)      # [N*Q]

        support_emb = model.forward_feature(support_data)   # [N*K, D]
        query_emb = model.forward_feature(query_data)       # [N*Q, D]

        prototypes = []
        for i in range(n_way):
            cls_emb = support_emb[support_labels == i]
            prototypes.append(cls_emb.mean(0))
        prototypes = torch.stack(prototypes)  # [N, D]

        if cosine_sim or test_both: # 0: baseline++ 1: protonet
            query_emb_ = F.normalize(query_emb, p=2, dim=1)
            prototypes_ = F.normalize(prototypes, p=2, dim=1)
            logits = torch.matmul(query_emb_, prototypes_.T)  # [N*Q, N]

            log_p_y = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(log_p_y, query_labels)
            acc = (log_p_y.argmax(1) == query_labels).float().mean()

            total_loss[0] += loss.item()
            total_acc[0].append(acc.item())

        if not cosine_sim or test_both:
            dists = torch.cdist(query_emb, prototypes)      # [N*Q, N]
            logits = -dists  # 越近越好

            log_p_y = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(log_p_y, query_labels)
            acc = (log_p_y.argmax(1) == query_labels).float().mean()

            total_loss[1] += loss.item()
            total_acc[1].append(acc.item())

    mean_acc, confidence_acc = np.array([0.0, 0.0]), np.array([0.0, 0.0])
    # print("total_acc", total_acc)
    for i, acc in enumerate(total_acc):
        acc_array = np.array(acc)
        mean_acc[i] = np.mean(acc_array)
        confidence_acc[i] = (np.std(acc_array) * 1.96) / np.sqrt(len(data_loader))
    
    ret_dict = {
        'loss': total_loss / len(data_loader),
        'acc': 100 * mean_acc,
        'cof': 100 * confidence_acc,
    }
    return ret_dict