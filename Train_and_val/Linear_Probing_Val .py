
import torch
import numpy as np
import os
import json
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

# ✅ Load Pretrained ResNet Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define Linear Probe (Mapping Features → Saliency Maps)
class LinearProbe(torch.nn.Module):
    def __init__(self, feature_dim, output_size=(256, 256)):
        super(LinearProbe, self).__init__()
        self.fc = torch.nn.Linear(feature_dim, output_size[0] * output_size[1])

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten feature map
        x = self.fc(x)  # Linear mapping
        return x.view(x.size(0), 1, 256, 256)  # Reshape to saliency map

# ✅ Define Saliency Predictor Model (ResNet + Linear Probe)
class SaliencyPredictor(torch.nn.Module):
    def __init__(self, backbone, feature_dim):
        super(SaliencyPredictor, self).__init__()
        self.backbone = backbone
        self.probe = LinearProbe(feature_dim)

    def forward(self, x):
        features = self.backbone(x)
        return torch.sigmoid(self.probe(features))  # Sigmoid for saliency map prediction

# ✅ Load Pretrained ResNet Model
backbone = models.resnet50(pretrained=True)
feature_dim = backbone.fc.in_features
backbone = torch.nn.Sequential(*list(backbone.children())[:-1])  # Remove classification layer
backbone.to(device).eval()

# ✅ Load the full model instead of just the probe
model = SaliencyPredictor(backbone, feature_dim).to(device)
model.load_state_dict(torch.load("/Users/nouira/Desktop/resnet_saliency.pth", map_location=device))
model.eval()

# ✅ Validation Data Paths
val_image_dir = "/Users/nouira/Desktop/deeplearning/project/val"
val_annotation_file = "/Users/nouira/Desktop/deeplearning/project/fixations_val2014.json"

# ✅ Define Dataset Class
class SaliencyDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        # Create fixation mapping
        self.fixations = {}
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in self.fixations:
                self.fixations[image_id] = []
            self.fixations[image_id].extend(ann['fixations'])

        # Map image_id to file_name and original dimensions
        self.image_id_to_file = {img['id']: (img['file_name'], img['height'], img['width']) for img in self.annotations['images']}
        self.image_ids = list(self.fixations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Load image
        image_id = self.image_ids[idx]
        image_filename, original_height, original_width = self.image_id_to_file[image_id]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        # Convert fixations to saliency map
        fixations = self.fixations.get(image_id, [])
        saliency_map = points_to_saliency(fixations, image_size=(256, 256), original_image_height=original_height, original_image_width=original_width)

        if self.transform:
            image = self.transform(image)
            saliency_map = torch.tensor(saliency_map).unsqueeze(0).float()

        return image, saliency_map, image_id

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

# ✅ Load Validation Dataset
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

val_dataset = SaliencyDataset(val_image_dir, val_annotation_file, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

# ✅ Evaluation Metrics
mse_loss = torch.nn.MSELoss()
mse_total = 0
pearson_corrs = []
auc_scores = []

# ✅ Directory to Save Visualizations
output_dir = "/Users/nouira/Desktop/Predicted_saliency_maps"
os.makedirs(output_dir, exist_ok=True)

# ✅ Dictionary to store metrics for each image
image_metrics = {}

# ✅ Run Model on Validation Set
print("🔹 Running validation...")
with torch.no_grad():
    for images, targets, image_ids in val_loader:
        images, targets = images.to(device), targets.to(device)

        # ✅ Use full model instead of just the probe
        predictions = model(images)

        # Compute MSE Loss
        mse = mse_loss(predictions, targets)
        mse_total += mse.item()

        # ✅ Save Visualizations and Compute Metrics for Each Image
        for i in range(len(image_ids)):
            # Get the image ID
            image_id = image_ids[i].item()

            # Extract the ground truth and predicted saliency maps
            gt_map = targets[i].cpu().squeeze().numpy()
            pred_map = predictions[i].cpu().squeeze().numpy()

            # Compute Pearson Correlation
            pred_flat = pred_map.flatten()
            target_flat = gt_map.flatten()
            if np.any(target_flat > 0):  # Avoid division by zero
                pearson_corr, _ = pearsonr(pred_flat, target_flat)
            else:
                pearson_corr = np.nan  # Avoid division by zero

            # Compute AUC Score
            auc = roc_auc_score((target_flat > 0.5).astype(int), pred_flat)

            # Store metrics in the dictionary
            image_metrics[image_id] = {
                "mse": float(np.mean((pred_flat - target_flat) ** 2)),  # MSE for this image
                "pearson_corr": float(pearson_corr),  # Pearson Correlation for this image
                "auc": float(auc),  # AUC for this image
            }

            # Save Visualization
            img = images[i].cpu().permute(1, 2, 0).numpy()
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(gt_map, cmap="jet")
            axes[1].set_title("Ground Truth Saliency")
            axes[1].axis("off")

            axes[2].imshow(pred_map, cmap="jet")
            axes[2].set_title("Predicted Saliency")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"saliency_{image_id}.png"))
            plt.close()

# ✅ Compute Final Metrics (averages)
avg_mse = mse_total / len(val_loader)
avg_pearson = np.mean([metrics["pearson_corr"] for metrics in image_metrics.values()])
avg_auc = np.mean([metrics["auc"] for metrics in image_metrics.values()])

# ✅ Save Results to File
results_path = "/Users/nouira/Desktop/validation_results_resnet.json"
with open(results_path, "w") as f:
    json.dump({
        "average_metrics": {
            "mse": avg_mse,
            "pearson_corr": avg_pearson,
            "auc": avg_auc,
        },
        "image_metrics": image_metrics,  # Metrics for each image
    }, f, indent=4)

print(f"✅ Validation results saved to {results_path}")
