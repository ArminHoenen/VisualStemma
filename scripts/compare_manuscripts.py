import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import sys

# Setup Feature Extractor
model = models.resnet18(weights='DEFAULT')
model.fc = nn.Identity()
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_cluster_centroid(cluster_path):
    vectors = []
    files = [os.path.join(cluster_path, f) for f in os.listdir(cluster_path) if f.lower().endswith('.png')]
    if not files: return None, 0
    
    for f in files:
        img = Image.open(f).convert('RGB')
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            vectors.append(model(tensor).squeeze().numpy())
    
    centroid = np.mean(vectors, axis=0)
    return centroid, len(files)

def compare_pair(ms1_path, ms2_path, log_file):
    clusters_ms1 = sorted([d for d in os.listdir(ms1_path) if os.path.isdir(os.path.join(ms1_path, d))])
    clusters_ms2 = sorted([d for d in os.listdir(ms2_path) if os.path.isdir(os.path.join(ms2_path, d))])
    
    data1 = {}
    data2 = {}
    total_letters_ms1 = 0
    total_letters_ms2 = 0

    for c in clusters_ms1:
        cent, count = get_cluster_centroid(os.path.join(ms1_path, c))
        if cent is not None:
            data1[c] = (cent, count)
            total_letters_ms1 += count

    for c in clusters_ms2:
        cent, count = get_cluster_centroid(os.path.join(ms2_path, c))
        if cent is not None:
            data2[c] = (cent, count)
            total_letters_ms2 += count

    # Build Similarity Matrix
    keys1 = list(data1.keys())
    keys2 = list(data2.keys())
    sim_matrix = np.zeros((len(keys1), len(keys2)))

    for i, k1 in enumerate(keys1):
        for j, k2 in enumerate(keys2):
            sim_matrix[i, j] = cosine_similarity(data1[k1][0].reshape(1, -1), 
                                                 data2[k2][0].reshape(1, -1))[0][0]

    # Solve Assignment Problem (Max Similarity = Min Cost)
    # Since linear_sum_assignment finds the minimum, we use 1 - similarity
    row_ind, col_ind = linear_sum_assignment(1 - sim_matrix)

    all_mappings = []
    for r, c in zip(row_ind, col_ind):
        k1, k2 = keys1[r], keys2[c]
        freq1 = data1[k1][1] / total_letters_ms1
        freq2 = data2[k2][1] / total_letters_ms2
        all_mappings.append({
            'ms1_c': k1, 'ms2_c': k2,
            'similarity': sim_matrix[r, c],
            'diff': abs(freq1 - freq2)
        })

    # Sort and discard the bottom 5% most dissimilar mappings
    all_mappings.sort(key=lambda x: x['similarity'], reverse=True)
    num_to_keep = int(len(all_mappings) * 0.9)
    kept_mappings = all_mappings[:num_to_keep]
    discarded = all_mappings[num_to_keep:]

    # Logging
    log_file.write(f"\nComparing {os.path.basename(ms1_path)} vs {os.path.basename(ms2_path)}\n")
    log_file.write(f"Strict One-to-One Mapping Results:\n")
    for m in kept_mappings:
        log_file.write(f"  MATCH: {m['ms1_c']} <-> {m['ms2_c']} (Sim: {m['similarity']:.4f}, FreqDiff: {m['diff']:.4f})\n")
    for d in discarded:
        log_file.write(f"  DISCARDED: {d['ms1_c']} <-> {d['ms2_c']} (Sim: {d['similarity']:.4f})\n")

    avg_dist = sum(m['diff'] for m in kept_mappings) / len(kept_mappings) if kept_mappings else 0
    return avg_dist

if __name__ == "__main__":
    cluster_base = "./clusters"
    if not os.path.exists(cluster_base):
        print("Error: ./clusters folder not found.")
        sys.exit(1)
        
    manuscripts = sorted([d for d in os.listdir(cluster_base) if os.path.isdir(os.path.join(cluster_base, d))])
    
    with open("comparison_log.txt", "w") as log_f:
        print("\n--- Final Manuscript Distances (Strict 1-to-1 Mapping) ---")
        for i in range(len(manuscripts)):
            for j in range(i + 1, len(manuscripts)):
                ms1, ms2 = manuscripts[i], manuscripts[j]
                distance = compare_pair(os.path.join(cluster_base, ms1), os.path.join(cluster_base, ms2), log_f)
                print(f"{ms1} <-> {ms2}: {distance:.6f}")
