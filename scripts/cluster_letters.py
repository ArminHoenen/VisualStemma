import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import shutil
from sklearn.cluster import KMeans
import numpy as np
import sys

# Feature Extractor Setup
# Using ResNet18 as a feature extractor (stripping the final layer)
model = models.resnet18(weights='DEFAULT')
model.fc = nn.Identity()
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_vector(img_path):
    img = Image.open(img_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        vector = model(tensor)
    return vector.squeeze().numpy()

def run_clustering(input_dir, k_value, output_base):
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    img_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(img_paths) == 0:
        print(f"No images found in {input_dir}")
        return

    # K cannot be larger than the number of samples
    k_actual = min(int(k_value), len(img_paths))
    
    vectors = []
    valid_paths = []

    print(f"Vectorizing {len(img_paths)} images...")
    for path in img_paths:
        try:
            vectors.append(get_vector(path))
            valid_paths.append(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    print(f"Clustering into k={k_actual} groups...")
    kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(np.array(vectors))

    # Clean and create output directory
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    os.makedirs(output_base)

    for i in range(k_actual):
        os.makedirs(os.path.join(output_base, f"cluster_{i}"), exist_ok=True)

    for path, cluster_id in zip(valid_paths, clusters):
        shutil.copy(path, os.path.join(output_base, f"cluster_{cluster_id}"))
    
    print(f"Success: {len(valid_paths)} letters grouped into {output_base}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 cluster_letters.py <input_dir> <k> <output_dir>")
    else:
        run_clustering(sys.argv[1], sys.argv[2], sys.argv[3])
