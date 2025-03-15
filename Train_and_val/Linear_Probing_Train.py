import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import json
import numpy as np
import cv2

# ✅ Load Pretrained ResNet Model (Remove CLIP)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50 as feature extractor
backbone = models.resnet50(pretrained=True)
feature_dim = backbone.fc.in_features
backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove final classification layer
backbone.to(device).eval()  # Move model to GPU/CPU and set to evaluation mode

# ✅ Define Linear Probe (Maps ResNet Features to Saliency Maps)
class LinearProbe(nn.Module):
    def __init__(self, feature_dim, output_size=(256, 256)):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(feature_dim, output_size[0] * output_size[1])

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten feature map
        x = self.fc(x)  # Linear mapping
        return x.view(x.size(0), 1, 256, 256)  # Reshape to saliency map

# ✅ Combine ResNet and Linear Probe
class SaliencyPredictor(nn.Module):
    def __init__(self, backbone, feature_dim):
        super(SaliencyPredictor, self).__init__()
        self.backbone = backbone
        self.probe = LinearProbe(feature_dim)

    def forward(self, x):
        features = self.backbone(x)
        return torch.sigmoid(self.probe(features))

# ✅ Define Dataset Class
class SaliencyDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.fixations = {}
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in self.fixations:
                self.fixations[image_id] = []
            self.fixations[image_id].extend(ann['fixations'])
        self.image_id_to_file = {img['id']: (img['file_name'], img['height'], img['width']) for img in self.annotations['images']}
        self.image_ids = list(self.fixations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_filename, original_height, original_width = self.image_id_to_file[image_id]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")
        points = self.fixations.get(image_id, [])
        saliency_map = points_to_saliency(points, image_size=(256, 256), original_image_height=original_height, original_image_width=original_width)
        saliency_map = Image.fromarray((saliency_map * 255).astype(np.uint8))
        if self.transform:
            image = self.transform(image)
            saliency_map = self.transform(saliency_map)
        return image, saliency_map

# ✅ Convert Fixation Points to Saliency Map
def points_to_saliency(points, image_size=(256, 256), original_image_height=None, original_image_width=None, sigma=10):
    saliency_map = np.zeros(image_size, dtype=np.float32)
    for (row, col) in points:
        row = int((row - 1) * (image_size[0] / original_image_height))
        col = int((col - 1) * (image_size[1] / original_image_width))
        row = min(max(row, 0), image_size[0] - 1)
        col = min(max(col, 0), image_size[1] - 1)
        saliency_map[row, col] += 1.0
    saliency_map = cv2.GaussianBlur(saliency_map, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if saliency_map.max() > 0:
        saliency_map /= saliency_map.max()
    return saliency_map

# ✅ Training Function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, saliency_maps in dataloader:
        images, saliency_maps = images.to(device), saliency_maps.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, saliency_maps)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# ✅ Main Script
if __name__ == "__main__":
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 10
    image_dir = "/home/selma_akarsu/salicon_data/train"
    annotation_file = "/home/selma_akarsu/fixations_train2014.json"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = SaliencyDataset(image_dir, annotation_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SaliencyPredictor(backbone, feature_dim).to(device)
    criterion = nn.MSELoss()
 
    for epoch in range(num_epochs):
        epoch_loss = train(model, dataloader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # ✅ Save Trained Model
    torch.save(model.state_dict(), "/home/selma_akarsu/resnet_saliency.pth")
    print("✅ Model saved!")
