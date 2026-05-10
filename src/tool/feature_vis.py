import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']


def pil_to_tensor(img_pil, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    tensor = (tensor - mean_t) / std_t
    return tensor


def load_model(model_type, weight_path):
    if model_type == 'base':
        from model.base import CNNModel
        model = CNNModel()
    elif model_type == 'deeper':
        from model.deeper import CNNModel
        model = CNNModel()
    elif model_type == 'res':
        from model.res import CNNResModel
        model = CNNResModel()
    elif model_type == 'se':
        from model.SEBlock import CNN_SE
        model = CNN_SE()
    elif model_type == 'cbam':
        from model.SEBlock import CNN_CBAM
        model = CNN_CBAM()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=True)
    model.eval()
    return model


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = None
        self.model.gap.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        feat = input[0].detach()
        self.features = feat.view(feat.size(0), -1)

    def forward(self, x):
        self.model(x)
        return self.features


def extract_features(model, image_dir, max_per_class=100, batch_size=32):
    extractor = FeatureExtractor(model)

    all_features = []
    all_labels = []

    class_dirs = sorted([d for d in os.listdir(image_dir)
                         if os.path.isdir(os.path.join(image_dir, d))])

    for cls_idx, cls_name in enumerate(class_dirs):
        cls_dir = os.path.join(image_dir, cls_name)
        img_files = sorted([f for f in os.listdir(cls_dir) if f.endswith('.png')])
        img_files = img_files[:max_per_class]

        print(f'Extracting: {cls_name} ({len(img_files)} images)', flush=True)

        batch_tensors = []
        for i, img_file in enumerate(img_files):
            img_path = os.path.join(cls_dir, img_file)
            img_pil = Image.open(img_path).convert('RGB')
            input_tensor = pil_to_tensor(img_pil)
            batch_tensors.append(input_tensor)

            if len(batch_tensors) == batch_size or i == len(img_files) - 1:
                batch = torch.stack(batch_tensors)
                with torch.no_grad():
                    feat = extractor(batch)
                feat = feat.cpu().numpy()
                all_features.append(feat)
                all_labels.extend([cls_idx] * feat.shape[0])
                batch_tensors = []

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.array(all_labels)

    print(f'Feature shape: {all_features.shape}', flush=True)
    return all_features, all_labels


def plot_scatter(features_2d, labels, title, save_path, classes_list=None):
    num_classes = len(np.unique(labels))
    cmap = plt.colormaps.get_cmap('tab10').resampled(num_classes)

    fig, ax = plt.subplots(figsize=(10, 8))

    for cls_idx in range(num_classes):
        mask = labels == cls_idx
        color = cmap(cls_idx)
        label = classes_list[cls_idx] if classes_list else str(cls_idx)
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[color],
            label=label,
            alpha=0.6,
            s=20,
            edgecolors='none'
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.legend(loc='best', fontsize=10, markerscale=2)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}', flush=True)


def visualize_pca_tsne(model_type, weight_path, image_dir, output_dir,
                       max_per_class=100, tsne_perplexity=30):
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_type, weight_path)
    features, labels = extract_features(model, image_dir, max_per_class)

    print('Running PCA...', flush=True)
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    print(f'PCA variance ratio: {pca.explained_variance_ratio_}', flush=True)

    pca_path = os.path.join(output_dir, f'pca_{model_type}.png')
    plot_scatter(features_pca, labels, f'PCA - {model_type}', pca_path, classes)

    print('Running t-SNE...', flush=True)
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42,
                n_iter=1000, learning_rate='auto', init='pca')
    features_tsne = tsne.fit_transform(features)

    tsne_path = os.path.join(output_dir, f'tsne_{model_type}.png')
    plot_scatter(features_tsne, labels, f't-SNE - {model_type}', tsne_path, classes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    num_classes = len(np.unique(labels))
    cmap = plt.colormaps.get_cmap('tab10').resampled(num_classes)

    for cls_idx in range(num_classes):
        mask = labels == cls_idx
        color = cmap(cls_idx)
        label = classes[cls_idx]

        ax1.scatter(features_pca[mask, 0], features_pca[mask, 1],
                    c=[color], label=label, alpha=0.6, s=20, edgecolors='none')
        ax2.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                    c=[color], label=label, alpha=0.6, s=20, edgecolors='none')

    ax1.set_title('PCA', fontsize=14)
    ax1.set_xlabel('PC1', fontsize=11)
    ax1.set_ylabel('PC2', fontsize=11)
    ax1.legend(loc='best', fontsize=9, markerscale=2)
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2.set_title('t-SNE', fontsize=14)
    ax2.set_xlabel('Dim 1', fontsize=11)
    ax2.set_ylabel('Dim 2', fontsize=11)
    ax2.legend(loc='best', fontsize=9, markerscale=2)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle(f'Feature Space Visualization - {model_type}', fontsize=16, y=1.02)
    plt.tight_layout()
    combined_path = os.path.join(output_dir, f'pca_tsne_{model_type}.png')
    plt.savefig(combined_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {combined_path}', flush=True)

    np.save(os.path.join(output_dir, f'features_{model_type}.npy'), features)
    np.save(os.path.join(output_dir, f'labels_{model_type}.npy'), labels)

    print(f'\nAll visualizations saved to: {output_dir}', flush=True)


if __name__ == '__main__':
    MODEL_TYPE = 'res'
    WEIGHT_PATH = '../../out/train/transform/best_model.pth'
    IMAGE_DIR = '../../STL10/test'
    OUTPUT_DIR = '../../out/feature_vis'

    visualize_pca_tsne(
        model_type=MODEL_TYPE,
        weight_path=WEIGHT_PATH,
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        max_per_class=100,
        tsne_perplexity=30
    )
