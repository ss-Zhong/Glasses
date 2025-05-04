import os
import GPUtil
import inspect
import random
import numpy as np
import torch
import time
from collections import defaultdict, deque
import datetime

def compare_models(model1, model2):
    # 获取模型1和模型2的参数字典
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    
    # 遍历模型1的参数字典，比较它与模型2中相同命名的参数
    mismatched_params = []
    
    for name, param1 in params1.items():
        if name in params2:
            param2 = params2[name]
            # 判断参数值是否相同
            if not torch.equal(param1, param2):
                mismatched_params.append(name)
    
    logmsg(mismatched_params, blue=True)

def logmsg(*args, show_where = True, blue = False, log_file_path = None):
    """
    Log message with the place of log
    show_where: show the place of log
    blue: print in blue color
    log_file_path: log into a file
    """
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    filename = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno

    if blue:
        blue_bold = "\033[1m\033[34m"
        reset = "\033[0m"
    else:
        blue_bold = ""
        reset = ""

    message = ''.join(map(str, args))

    if log_file_path is not None:
        try:
            with open(log_file_path, 'a') as log_file:  # Open in append mode
                log_file.write(message + "\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")
    elif show_where:
        print(f"{blue_bold}[{filename}:{line_number}] {message}{reset}")
    else:
        print(f"{blue_bold}{message}{reset}")

def view_model_param(model, log_file_path = None):
    for k, v in model.named_parameters():
        logmsg(f"{k}: {v.requires_grad}", show_where = False, log_file_path = log_file_path)

def select_gpu_with_most_free_memory():
    """
    select gpu with most free memory
    """
    gpus = GPUtil.getGPUs()
    if not gpus:
        raise RuntimeError("No GPU found!")
    gpu_id = max(gpus, key=lambda gpu: gpu.memoryFree).id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logmsg(f"Selected GPU: {gpu_id}, Free Memory: {gpus[gpu_id].memoryFree} MB", blue=True)

def set_seed(seed):
    """
    set seed for reproducibility
    """
    if seed != -1:
        random.seed(seed)  # 固定 Python 随机种子
        np.random.seed(seed)  # 固定 NumPy 随机种子
        torch.manual_seed(seed)  # 固定 PyTorch CPU 随机种子
        torch.cuda.manual_seed(seed)  # 固定 PyTorch GPU 随机种子
        torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU，固定所有 GPU 的种子
        torch.backends.cudnn.deterministic = True  # 保证每次卷积运算结果相同
        torch.backends.cudnn.benchmark = False  # 禁止动态选择最佳卷积算法

class MetricLogger(object):
    def __init__(self,  logger = True, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if (i % print_freq == 0 or i == len(iterable) - 1) and self.logger:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        if self.logger:
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)))
    
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def std(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.std().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)