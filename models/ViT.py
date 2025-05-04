from transformers import ViTForImageClassification
import torch.nn as nn

class ImageClassifier(ViTForImageClassification):
    def __init__(self, config, projection_dim=128):
        super().__init__(config)
        
        # 定义投影头
        self.projection_head = nn.Linear(config.hidden_size, projection_dim)

    def forward(self, pixel_values, proj = False):
        x = self.vit.embeddings(pixel_values)
        x = self.vit.encoder(x)[0]
        x = self.vit.layernorm(x)
        logits = self.classifier(x[:, 0])

        if proj:
            proj_x = self.projection_head(x[:, 0])
            return logits, proj_x
        else:
            return logits
    
    def forward_attention(self, pixel_values):
        x = self.vit.embeddings(pixel_values)
        for i, layer in enumerate(self.vit.encoder.layer):
            if i < self.get_depth()-1:
                x = layer(x)[0]
            else:
                x, attention = layer(x, output_attentions=True)

        return attention[:, :, 0, 1:]

    def forward_feature(self, pixel_values):
        x = self.vit.embeddings(pixel_values)
        x = self.vit.encoder(x)[0]
        x = self.vit.layernorm(x)
        prototypes = x[:, 0]
        return prototypes

    def get_depth(self):
        return len(self.vit.encoder.layer)
    
    @staticmethod
    def get_lora_target_modules():
        target_component = ["attention.attention.query", "attention.attention.key", "attention.attention.value", "attention.output.dense", "intermediate.dense", "output.dense"]
        return target_component
    
if __name__ == '__main__':
    
    model = ImageClassifier.from_pretrained("google/vit-base-patch16-224")
    print(f"Number of attention heads: {model.config.num_attention_heads}")
    # for k, v in model.named_parameters():
    #     print(k)