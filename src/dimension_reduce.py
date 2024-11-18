from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import umap
from model import GPT
from data_utils import initialize_loaders

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

# def get_raw_input_embeddings(config, dataset_name, capture_layer):
#     dataset_path = f"{PATH_PREFIX}/{dataset_name}.pth"
#     dataset = torch.load(dataset_path)
#     train_loader, val_loader = initialize_loaders(config, dataset)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = GPT(config).to(device)

#     all_flattened_input_embeddings = []

#     for index, batch in enumerate(val_loader):
#         batch = batch.to(device)
#         _, _, _, captured_embedding = model(batch, True, capture_layer)

#         input_embeddings = captured_embedding[:, 1:(config.input_size-1):2, :]
#         flattened_input_embeddings = input_embeddings.reshape(-1, 64)
#         breakpoint()

#         all_flattened_input_embeddings.append(flattened_input_embeddings.detach().cpu())

#     return torch.cat(all_flattened_input_embeddings)

def run_pca_analysis(embeddings, labels, n_components=2):
    """
    Run PCA on input embeddings and create visualization
    Args:
        embeddings: torch.Tensor of shape (n_samples, n_features)
        labels: torch.Tensor of shape (n_samples,) containing class labels
        n_components: Number of PCA components to keep
    """
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
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('PCA visualization of embeddings')
    plt.xlabel(f'PC1 (variance explained: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 (variance explained: {pca.explained_variance_ratio_[1]:.3f})')
    
    # Save plot
    plt.savefig(f'{PATH_PREFIX}/classify/pca_visualization.png')
    plt.close()

    return embeddings_pca, pca.explained_variance_ratio_

def run_umap_analysis(embeddings, labels, n_components=2):
    """
    Run UMAP on input embeddings and create visualization
    Args:
        embeddings: torch.Tensor of shape (n_samples, n_features)
        labels: torch.Tensor of shape (n_samples,) containing class labels
        n_components: Number of UMAP components
    """

    # Convert embeddings to numpy if they're torch tensors
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Run UMAP
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embeddings_umap = reducer.fit_transform(embeddings)

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('UMAP visualization of embeddings')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    
    # Save plot
    plt.savefig(f'{PATH_PREFIX}/classify/umap_visualization.png')
    plt.close()

    return embeddings_umap
