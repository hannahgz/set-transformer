from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import umap
from model import GPT
from data_utils import initialize_loaders
import os
import numpy as np
from matplotlib.colors import ListedColormap 

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'


def run_pca_analysis(embeddings, labels, layer, n_components=2):
    """
    Run PCA on input embeddings and create visualization
    Args:
        embeddings: torch.Tensor of shape (n_samples, n_features)
        labels: torch.Tensor of shape (n_samples,) containing class labels
        n_components: Number of PCA components to keep
    """
    print(f"Running PCA with {n_components} components for layer {layer}")
    # Convert embeddings to numpy if they're torch tensors
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Run PCA
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings)

    # Create scatter plot
    plt.figure(figsize=(10, 8))

    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Default matplotlib colors
    
    # Ensure labels are 0-4
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in labels])
    
    scatter = plt.scatter(embeddings_pca[:, 0], 
                          embeddings_pca[:, 1],
                          c=mapped_labels, 
                          cmap=ListedColormap(distinct_colors[:len(unique_labels)]),
                          alpha=0.3)
    
    # Create colorbar with integer ticks
    colorbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
    colorbar.set_ticklabels([f'Class {i}' for i in range(len(unique_labels))])

    plt.title('PCA visualization of embeddings')
    plt.xlabel(
        f'PC1 (variance explained: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(
        f'PC2 (variance explained: {pca.explained_variance_ratio_[1]:.3f})')

    # Save plot

    base_dir = f"figs/classify/pca/components_{n_components}"
    os.makedirs(base_dir, exist_ok=True)

    plt.savefig(f'figs/classify/pca/components_{n_components}/layer_{layer}.png')
    plt.close()

    return embeddings_pca, pca.explained_variance_ratio_


def run_umap_analysis(embeddings, labels, layer, n_components=2):
    """
    Run UMAP on input embeddings and create visualization
    Args:
        embeddings: torch.Tensor of shape (n_samples, n_features)
        labels: torch.Tensor of shape (n_samples,) containing class labels
        n_components: Number of UMAP components
    """
    print(f"Running UMAP with {n_components} components for layer {layer}")
    # Convert embeddings to numpy if they're torch tensors
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    print("Applying PCA first...")
    n_pca = min(50, embeddings.shape[1])  # Aggressive dimension reduction
    pca = PCA(n_components=n_pca)
    embeddings_pca = pca.fit_transform(embeddings)

    # Run UMAP
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embeddings_umap = reducer.fit_transform(embeddings_pca)

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
    #                       c=labels, cmap='tab10', alpha=0.6)
    
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Default matplotlib colors
    
    # Ensure labels are 0-4
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in labels])
    
    scatter = plt.scatter(embeddings_umap[:, 0], 
                          embeddings_umap[:, 1],
                          c=mapped_labels, 
                          cmap=ListedColormap(distinct_colors[:len(unique_labels)]),
                          alpha=0.3)
    
    # Create colorbar with integer ticks
    colorbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
    colorbar.set_ticklabels([f'Class {i}' for i in range(len(unique_labels))])

    plt.title('UMAP visualization of embeddings')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')

    # Save plot
    base_dir = f"figs/classify/umap/components_{n_components}"
    os.makedirs(base_dir, exist_ok=True)

    plt.savefig(f'figs/classify/umap/components_{n_components}/layer_{layer}.png')
    plt.close()

    return embeddings_umap
