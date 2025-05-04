# ✨Glasses✨

![GitHub stars](https://img.shields.io/github/stars/ss-Zhong/Glasses?style=flat&color=5caaf3)
![Visits](https://badges.pufler.dev/visits/ss-Zhong/Glasses?color=47bdae)
![License](https://img.shields.io/github/license/ss-Zhong/Glasses)
![Last commit](https://img.shields.io/github/last-commit/ss-Zhong/Glasses)

<!-- ![Citation Count](https://img.shields.io/semantic-release/citation?url=https://your-papers-link) -->

Official PyTorch implementation of the paper **Glasses: Environment-Aware Deployment of Few-Shot Classification via Device-Cloud Collaboration**.

## Datasets

The paper introduces **Focura**, which can be accessed in the appendix of the paper. Additionally, you need to download the **ImageNet** dataset. Once downloaded, place it in the appropriate directory. You can specify the paths for both datasets by setting the `edge_data_path` and `data_path` in `lib.set_args` to your respective paths.

## Backbone

For the model backbone, we provide support for various pre-trained models, including those from Hugging Face. If the model is hosted on Hugging Face, the code will automatically download it for you. For other models such as **DINO** or **iBOT**, please refer to the original research papers to download the model.

The following models are supported:

```yaml
vit_base: "google/vit-base-patch16-224"
vit_tiny: "facebook/deit-tiny-patch16-224"
deit_base: "facebook/deit-base-distilled-patch16-224"
deit_small: "facebook/deit-small-distilled-patch16-224"
swin_tiny: "microsoft/swin-tiny-patch4-window7-224"
swin_small: "microsoft/swin-small-patch4-window7-224"
iBotvit_small: "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint.pth"
Dinovit_small: "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth"
```

## Quick Start 🔥

### Step 1: Prepare Datasets

Download and prepare **Focura** and **ImageNet** datasets. Ensure they are placed in the correct locations on your system.

### Step 2: Run the Main Script

Once the datasets are ready, simply run the main script to train or deploy the model:

```bash
python main.py
```

This command will execute the model training or inference process based on the configuration settings.