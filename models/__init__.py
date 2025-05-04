from peft import get_peft_model, LoraConfig
import torch
import warnings

url = {
    'vit_base': 'google/vit-base-patch16-224',
    'vit_tiny': 'facebook/deit-tiny-patch16-224',
    'deit_base': 'facebook/deit-base-distilled-patch16-224',
    'deit_small': 'facebook/deit-small-distilled-patch16-224',
    'swin_tiny': 'microsoft/swin-tiny-patch4-window7-224',
    'swin_small': 'microsoft/swin-small-patch4-window7-224',
    'iBotvit_small': '/share/models/vit-small/ibot_checkpoint.pth', # down from https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint.pth 
    'Dinovit_small': '/share/models/vit-small/dino_checkpoint.pth', # down from https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth
}

def create_model(model, student = True):
    model_family = model.split('_')[0]

    if model_family == 'deit':
        from .DeiT import ImageClassifier
    elif model_family == 'vit':
        from .ViT import ImageClassifier
    elif model_family == 'swin':
        from .SwinT import ImageClassifier
    elif model_family == 'iBotvit':
        from .iBotViT import MultiCropWrapper, vit_small
        from .iBotHead import iBOTHead

        backbone = vit_small(return_all_tokens=True, patch_size=16, masked_im_modeling = student, drop_path_rate=0.1 if student else 0)
        embed_dim = backbone.num_features
        classifier = MultiCropWrapper(backbone, iBOTHead(
            embed_dim,
            8192, # 8192 is args.out_dim in iBot
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer= ~student,
            shared_head=True,
        ))

        chkpt = torch.load(url[model], weights_only=False)
        state_dict = chkpt['student' if student else 'student']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        msg = classifier.load_state_dict(state_dict, strict=False)
        
        return classifier

    elif model_family == 'Dinovit':
        from .iBotViT import MultiCropWrapper, vit_small
        from .iBotHead import DINOHead

        backbone = vit_small(patch_size=16, drop_path_rate=0.1 if student else 0)
        embed_dim = backbone.num_features
        classifier = MultiCropWrapper(backbone, DINOHead(
            embed_dim,
            65536, # args.out_dim,
            use_bn= False, # args.use_bn_in_head,
            norm_last_layer= True,
        ))

        chkpt = torch.load(url[model], weights_only=False)
        
        state_dict = chkpt['student' if student else 'student'] # both student

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        msg = classifier.load_state_dict(state_dict, strict=False)

        return classifier
        
    return ImageClassifier.from_pretrained(url[model])

def create_peft_model(model, classifier = None, lora_rank = 0, student = True):
    model_family = model.split('_')[0]

    if model_family == 'iBotvit':
        from .iBotViT import MultiCropWrapper, vit_small
        from .iBotHead import iBOTHead

        backbone = vit_small(return_all_tokens=True, patch_size=16, masked_im_modeling = student, drop_path_rate=0.1 if student else 0)
        embed_dim = backbone.num_features
        classifier = MultiCropWrapper(backbone, iBOTHead(
            embed_dim,
            8192, # 8192 is args.out_dim in iBot
            patch_out_dim=8192,
            norm=None,
            act='gelu',
            norm_last_layer= ~student,
            shared_head=True,
        ))

        chkpt = torch.load(url[model], weights_only=False)
        # print(list(chkpt.keys()))
        state_dict = chkpt['student' if student else 'student'] # both student
        # for key in state_dict.keys():
        #     print(key)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        msg = classifier.load_state_dict(state_dict, strict=False)
        # print(f"Model parameters successfully loaded. Details: {msg}")

    elif model_family == 'Dinovit':
        from .iBotViT import MultiCropWrapper, vit_small
        from .iBotHead import DINOHead

        backbone = vit_small(patch_size=16, drop_path_rate=0.1 if student else 0)
        embed_dim = backbone.num_features
        classifier = MultiCropWrapper(backbone, DINOHead(
            embed_dim,
            65536, # args.out_dim,
            use_bn= False, # args.use_bn_in_head,
            norm_last_layer= True,
        ))

        chkpt = torch.load(url[model], weights_only=False)
        # print(list(chkpt.keys()))
        state_dict = chkpt['student' if student else 'student'] # both student
        # for key in state_dict.keys():
        #     print(key)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        msg = classifier.load_state_dict(state_dict, strict=False)
        # print(f"Model parameters successfully loaded. Details: {msg}")
        
    else:
        if model_family == 'deit':
            from .DeiT import ImageClassifier
        elif model_family == 'vit':
            from .ViT import ImageClassifier
        elif model_family == 'swin':
            from .SwinT import ImageClassifier
        
        if classifier is None:
            classifier = ImageClassifier.from_pretrained(url[model])
    
    if model_family in ['deit', 'vit', 'swin', 'iBotvit', 'Dinovit']:
        if lora_rank == 0:
            return classifier
        
        target_modules = classifier.get_lora_target_modules()
        
        lora_config = LoraConfig(
            r = lora_rank,
            lora_alpha = 16,
            lora_dropout = 0.1,  # dropout
            target_modules = target_modules,
            init_lora_weights = True
        )

        classifier = get_peft_model(classifier, lora_config)

        for k, v in classifier.named_parameters():
            if 'projection_head' in k or 'classifier' in k or 'head' in k:
            # if 'projection_head' in k:
                v.requires_grad = True
        
    return classifier
