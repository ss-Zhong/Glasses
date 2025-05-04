import sys
from lib.utils import logmsg

args = {
    "model": "vit_tiny", 
    # ["vit_base", "deit_base", "vit_tiny", 'deit_small', 'swin_tiny', 'swin_small', | 'iBotvit_small', 'Dinovit_small']
    "loss_type": "CrossEntropy", 
    # ["CrossEntropy", "Dino", "iBot"]
    
    "config-module": f"hyper_parameters",
    
    "edge_dataset": "IMNET",
    "edge_data_path": "/share/imagenet",
    
    "dataset": "Focura",
    "data_path": "/share/Focura",

    "seed": 0,
    "n_way": 5,
    # "upload": True,

    "subset_size": 5120,
    "batch_size": 128,
    "epochs": 0,
    "input_size": 224,
    "num_workers": 64,

    "rank_edge": 10,
    "glasses_resume": "./output/glasses.pth",
    
    "sched": "step",
    "decay-epochs": 2,
    "warmup-epochs": 0,
    "lr": 8e-4,
    "decay-rate": 0.8,
}

def set_args():
    cmd_args = []
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                cmd_args.append(f'--{key}')
        else:
            cmd_args.append(f'--{key}')
            cmd_args.append(str(value))
    
    # 使用模拟的命令行参数
    sys.argv = ['main.py'] + cmd_args

    command_str = ' '.join(cmd_args)
    logmsg("[Command] ", command_str, blue=True, show_where=False)