from lib import _args_, utils

from pathlib import Path
import wandb
from glasses.edgeWorker import EdgeWorker
from glasses.device import Device
from dataset.Focura import Focura

if __name__ == '__main__':
    """
    initialize
    """
    utils.select_gpu_with_most_free_memory()

    parser = _args_.get_args_parser()
    if True:# if you want to run the code without args
        from lib.set_args import set_args
        set_args()
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    wandb.init(
        # set the wandb project where this run will be logged
        project = "Glasses",
        name = f"[{args.dataset}]-[{args.model}]-[{args.loss_type}]-[seed]{args.seed}",
        mode=('online' if args.upload else 'offline'),
        group='main'
    )

    utils.set_seed(seed = args.seed)
    scene_list = Focura.__list_scene__('/share/Focura')
    
    for scene in scene_list:
        Focura._scene_ = scene
        scene_path = args.data_path + '/scene/' + scene + '.jpg'
        
        utils.logmsg(f"################# {scene} #################", blue=True)
        

        if args.glasses_resume != '':
            utils.set_seed(seed = args.seed)
            edge_worker = EdgeWorker(args)
            edge_worker.train_on_edge(scene_path=scene_path)
        
        test_stats = {}
        for n_shot in [1, 5]:
            utils.set_seed(seed = args.seed) # if seed == -1, Random seed
            device = Device(args, n_shot = n_shot)
            
            utils.logmsg(f"################# unuse Glasses #################", blue=True)
            utils.set_seed(seed = args.seed)
            test_stats[f"{n_shot}_base"] = device.fewshot_on_device()
  
            utils.logmsg(f"################# use Glasses #################", blue=True)
            utils.set_seed(seed = args.seed)
            device.load_Glasses()
            test_stats[f"{n_shot}_ours"] = device.fewshot_on_device()

        wandb.log({
            # with Glasses
            f"Acc1_ours_1_ProtoNet": test_stats["1_ours"]['acc'][1],
            f"Acc1_ours_5_ProtoNet": test_stats["5_ours"]['acc'][1],
            
            f"Acc1_ours_1_Baseline++": test_stats["1_ours"]['acc'][0],
            f"Acc1_ours_5_Baseline++": test_stats["5_ours"]['acc'][0],
            
            # base
            f"Acc1_base_1_ProtoNet": test_stats["1_base"]['acc'][1],
            f"Acc1_base_5_ProtoNet": test_stats["5_base"]['acc'][1],

            f"Acc1_base_1_Baseline++": test_stats["1_base"]['acc'][0],
            f"Acc1_base_5_Baseline++": test_stats["5_base"]['acc'][0],

            # confidence
            f"cof_ours_1_ProtoNet": test_stats["1_ours"]['cof'][1],
            f"cof_ours_5_ProtoNet": test_stats["5_ours"]['cof'][1],
            
            f"cof_ours_1_Baseline++": test_stats["1_ours"]['cof'][0],
            f"cof_ours_5_Baseline++": test_stats["5_ours"]['cof'][0],
            
            f"cof_base_1_ProtoNet": test_stats["1_base"]['cof'][1],
            f"cof_base_5_ProtoNet": test_stats["5_base"]['cof'][1],

            f"cof_base_1_Baseline++": test_stats["1_base"]['cof'][0],
            f"cof_base_5_Baseline++": test_stats["5_base"]['cof'][0],
        })