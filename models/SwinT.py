from transformers import SwinConfig, SwinForImageClassification
import torch.nn as nn
import torch

class ImageClassifier(SwinForImageClassification):
    def __init__(self, config, projection_dim=128):
        super().__init__(config)
        
        self.projection_head = nn.Linear(config.hidden_size, projection_dim)

    def forward(self, pixel_values, proj = False):
        
        x = self.swin(pixel_values)[1]
        
        logits = self.classifier(x)

        if proj:
            proj_x = self.projection_head(x)
            return logits, proj_x
        else:
            return logits

    def forward_feature(self, pixel_values):
        
        prototypes = self.swin(pixel_values)[1]

        return prototypes

    def get_depth(self):
        return len(self.swin.encoder.layers)
    
    @staticmethod
    def get_lora_target_modules():
        # Define target modules in the Swin model (similar to ViT)
        target_component = ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense", "output.dense"]
        return target_component
    
if __name__ == '__main__':
    
    model = ImageClassifier.from_pretrained("microsoft/swin-small-patch4-window7-224")
    # print(f"Number of attention heads: {model.config.num_attention_heads}")
    for k, v in model.named_parameters():
        print(k)