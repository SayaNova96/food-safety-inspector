import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from multitask_food_model import MultiTaskFoodCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training using compute device: {device}")

# 1. Initialize MultiTask Model
model = MultiTaskFoodCNN(num_cuisines=7, num_freshness=2)

weights_file = "multitask_food_weights.pth"
if os.path.exists(weights_file):
    try:
        model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
        print("Loaded base model weights.")
    except Exception:
        pass

# 2. Freeze backbone features to train fast on CPU
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze the freshness head
for param in model.freshness_head.parameters():
    param.requires_grad = True

model.to(device)

# 3. Augmentations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_path = "dataset/train"
if not os.path.exists(dataset_path):
    print(f"Error: Could not find directory '{dataset_path}'. Please create and populate it first.")
    exit(1)

train_data = datasets.ImageFolder(dataset_path, transform=transform)
print(f"Classes indexed: {train_data.class_to_idx}")  # Ensure {'fresh': 0, 'rotten': 1}

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.freshness_head.parameters(), lr=1e-3)

# 4. Fine-Tuning Loop
print("\nStarting fine-tuning...")
model.train()
for epoch in range(5):
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        _, logits_f = model(images)
        loss = criterion(logits_f, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits_f, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    print(f"Epoch [{epoch+1}/5] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

# 5. Overwrite the weights with real fine-tuned weights
torch.save(model.state_dict(), weights_file)
print(f"\nSuccess: Updated '{weights_file}' with fine-tuned freshness parameters!")
