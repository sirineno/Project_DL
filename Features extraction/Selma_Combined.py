import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from tqdm import tqdm

# Detect available device (MPS for Mac, CUDA for GPU, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths (Change these paths to your actual dataset locations)
train_images_dir = "/Users/selmaakarsu/PycharmProjects/SaliencyProject/salicon_data/train"
train_annotations_file = "/Users/selmaakarsu/PycharmProjects/SaliencyProject/salicon_data/fixations_train2014.json"

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ResNet normalization
])


# Define Salicon Dataset
class SaliconDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_ids = [img['id'] for img in self.annotations['images']]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_data = next(img for img in self.annotations['images'] if img['id'] == img_id)
        img_path = os.path.join(self.image_dir, img_data['file_name'])
        image = Image.open(img_path).convert('RGB')

        ann = next(ann for ann in self.annotations['annotations'] if ann['image_id'] == img_id)
        fixations = ann['fixations']

        # Create a saliency heatmap
        heatmap = torch.zeros((img_data['height'], img_data['width']))
        for (row, col) in fixations:
            heatmap[row - 1, col - 1] += 1

        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / (heatmap.max() + 1e-8)  # Prevent division by zero

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, heatmap


# Load full dataset (can subset later for testing)
train_dataset = SaliconDataset(image_dir=train_images_dir, annotation_file=train_annotations_file, transform=transform)

# Subset for quick testing (first 10 images)
small_dataset = Subset(train_dataset, range(50))
train_loader = DataLoader(small_dataset, batch_size=2, shuffle=True, num_workers=0)

# Load pretrained ResNet-50 model (feature extractor)
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*list(resnet.children())[:-2])  # Remove final classification layers
resnet.to(device).eval()


class LinearProbeModel(nn.Module):
    def __init__(self, feature_dim):
        super(LinearProbeModel, self).__init__()
        self.fc1 = nn.Linear(feature_dim * 7 * 7, 128)  # First hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(128, 1)  # Output layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))  # Apply activation
        return self.fc2(x)



# Instantiate Linear Probing Model
model = LinearProbeModel(2048).to(device)

# Define loss function and optimizer
#criterion = nn.MSELoss()
criterion = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=0.00005)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for images, heatmaps in progress_bar:
        images, heatmaps = images.to(device), heatmaps.to(device)

        # Extract features using ResNet-50
        with torch.no_grad():
            features = resnet(images)

        features = features.view(features.size(0), -1)  # Flatten before passing to Linear layer
        outputs = model(features)

        # Normalize heatmap targets correctly
        target = heatmaps.view(heatmaps.size(0), -1).mean(dim=1)  # Compute mean saliency
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)  # Normalize between 0-1
        target = target * 10  # Scale up values to avoid tiny numbers
        # Compute loss (compare predicted saliency with ground-truth heatmaps)
        loss = criterion(outputs.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {epoch_loss / len(train_loader):.4f}")

# Evaluate Model with Pearson Correlation
with torch.no_grad():
    sample_images, sample_heatmaps = next(iter(train_loader))
    sample_images = sample_images.to(device)
    features = resnet(sample_images)  # Extract features
    features = features.view(features.size(0), -1)  # Flatten
    predicted_heatmaps = model(features).cpu().detach().numpy().flatten()

true_heatmaps = sample_heatmaps.view(sample_heatmaps.size(0), -1).mean(dim=1).numpy().flatten()
corr, _ = pearsonr(predicted_heatmaps, true_heatmaps)
print(f"Pearson Correlation: {corr:.4f}")
print("Predicted Saliency Values:", predicted_heatmaps)
print("True Saliency Values:", true_heatmaps)


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)  # Convert back to valid range

# Visualization
def visualize(image, heatmap):
    plt.subplot(1, 2, 1)
    plt.imshow(denormalize(image).permute(1, 2, 0).cpu().numpy())
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    smooth_heatmap = gaussian_filter(heatmap.numpy(), sigma=5)
    plt.imshow(smooth_heatmap, cmap="hot")
    plt.title("Smoothed Saliency Map")
    plt.axis("off")
    plt.show()



# Show one sample
image, heatmap = train_dataset[0]
visualize(image, heatmap)

#%%
with torch.no_grad():
    features = resnet(sample_images)  # Extract features
    print("Feature Map Shape:", features.shape)  # Should be (batch_size, 2048, 7, 7)
    print("Feature Map Example:", features[0, :5, :, :])  # Print part of the feature map
