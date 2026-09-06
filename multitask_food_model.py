import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MultiTaskFoodCNN(nn.Module):
    def __init__(self, num_cuisines=7, num_freshness=2, dropout_p=0.2):
        super(MultiTaskFoodCNN, self).__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        in_features = backbone.classifier[0].in_features
        
        # Task 1: Cuisine (Indian, American, British, French, Italian, Chinese, Thai)
        self.cuisine_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_cuisines)
        )
        
        # Task 2: Spoilage / Freshness (Fresh vs Spoiled)
        self.freshness_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_freshness)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.cuisine_head(x), self.freshness_head(x)

class MultiTaskFoodDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # List of tuples: (image_path, cuisine_idx, freshness_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cuisine_idx, freshness_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            'image': image,
            'cuisine': torch.tensor(cuisine_idx, dtype=torch.long),
            'freshness': torch.tensor(freshness_idx, dtype=torch.long)
        }

if __name__ == "__main__":
    model = MultiTaskFoodCNN()
    dummy_input = torch.randn(1, 3, 224, 224)
    out_c, out_f = model(dummy_input)
    print(f"Model functional. Output shapes: Cuisine={out_c.shape}, Freshness={out_f.shape}")
