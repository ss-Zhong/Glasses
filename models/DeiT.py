from transformers import DeiTForImageClassificationWithTeacher
import torch
import torch.nn as nn
from collections import Counter

class ImageClassifier(DeiTForImageClassificationWithTeacher):
    def __init__(self, config, projection_dim=128):
        super().__init__(config)
        
        # 定义投影头
        self.projection_head = nn.Linear(config.hidden_size, projection_dim)
        self.vit = self.deit
        self.classifier = self.cls_classifier
        # self.projection_head = nn.Linear(config.hidden_size, projection_dim)

    def forward(self, pixel_values, proj = False):
        x = self.deit.embeddings(pixel_values)
        x = self.deit.encoder(x)[0]
        x = self.deit.layernorm(x)
        logits = self.cls_classifier(x[:, 0])
        
        if proj:
            proj_x = self.projection_head(x[:, 0])
            return logits, proj_x
        else:
            return logits
    
    def forward_attention(self, pixel_values):
        x = self.deit.embeddings(pixel_values)
        for i, layer in enumerate(self.deit.encoder.layer):
            if i < self.get_depth()-1:
                x = layer(x)[0]
            else:
                x, attention = layer(x, output_attentions=True)

        return attention[:, :, 0, 2:]

    def forward_feature(self, pixel_values):
        x = self.deit.embeddings(pixel_values)
        x = self.deit.encoder(x)[0]
        x = self.deit.layernorm(x)
        prototypes = x[:, 0]
        return prototypes

    def get_depth(self):
        return len(self.deit.encoder.layer)

    @staticmethod
    def get_lora_target_modules():
        target_component = ["attention.attention.query", "attention.attention.key", "attention.attention.value", "attention.output.dense", "intermediate.dense", "output.dense"]
        return target_component