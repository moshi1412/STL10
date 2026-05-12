import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']


def pil_to_tensor(img_pil, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    tensor = (tensor - mean_t) / std_t
    return tensor


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = np.uint8(cam * 255)

        return cam, target_class


def get_target_layer(model, model_type):
    if model_type == 'base':
        return model.conv4
    elif model_type == 'deeper':
        return model.conv6
    elif model_type == 'res':
        return model.res3.conv2
    elif model_type == 'se':
        return model.conv4
    elif model_type == 'cbam':
        return model.conv4
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


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

    model.load_state_dict(torch.load(weight_path, map_location='cpu')['model'], strict=True)
    model.eval()
    return model


def overlay_cam_on_image(img_pil, cam, img_size=(96, 96)):
    img = img_pil.resize(img_size)
    img_np = np.array(img)

    cam_resized = np.array(Image.fromarray(cam).resize(img_size, Image.BILINEAR))

    heatmap = plt.cm.jet(cam_resized / 255.0)[:, :, :3]
    heatmap = np.uint8(heatmap * 255)

    overlay = np.uint8(img_np * 0.5 + heatmap * 0.5)

    return img_np, heatmap, overlay


def visualize_gradcam(model_type, weight_path, image_dir, output_dir, num_samples=3):
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_type, weight_path)
    target_layer = get_target_layer(model, model_type)
    grad_cam = GradCAM(model, target_layer)

    class_dirs = sorted([d for d in os.listdir(image_dir)
                         if os.path.isdir(os.path.join(image_dir, d))])

    for cls_name in class_dirs:
        cls_dir = os.path.join(image_dir, cls_name)
        img_files = sorted([f for f in os.listdir(cls_dir) if f.endswith('.png')])

        if len(img_files) == 0:
            continue

        selected = img_files[:num_samples]

        fig, axes = plt.subplots(len(selected), 4, figsize=(16, 4 * len(selected)))
        if len(selected) == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(f'Grad-CAM Visualization - Class: {cls_name}', fontsize=16, y=1.02)

        for i, img_file in enumerate(selected):
            img_path = os.path.join(cls_dir, img_file)
            img_pil = Image.open(img_path).convert('RGB')

            input_tensor = pil_to_tensor(img_pil).unsqueeze(0)
            input_tensor.requires_grad_(True)

            cam, pred_class = grad_cam.generate(input_tensor)
            img_np, heatmap, overlay = overlay_cam_on_image(img_pil, cam)

            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title('Original', fontsize=11)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(cam, cmap='jet')
            axes[i, 1].set_title('CAM', fontsize=11)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(heatmap)
            axes[i, 2].set_title('Heatmap', fontsize=11)
            axes[i, 2].axis('off')

            axes[i, 3].imshow(overlay)
            pred_name = classes[pred_class]
            color = 'green' if pred_name == cls_name else 'red'
            axes[i, 3].set_title(f'Overlay (Pred: {pred_name})', fontsize=11, color=color)
            axes[i, 3].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'gradcam_{cls_name}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'Saved: {save_path}')

    print(f'\nAll Grad-CAM visualizations saved to: {output_dir}')


def visualize_single(model_type, weight_path, image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_type, weight_path)
    target_layer = get_target_layer(model, model_type)
    grad_cam = GradCAM(model, target_layer)

    img_pil = Image.open(image_path).convert('RGB')
    input_tensor = pil_to_tensor(img_pil).unsqueeze(0)
    input_tensor.requires_grad_(True)

    cam, pred_class = grad_cam.generate(input_tensor)
    img_np, heatmap, overlay = overlay_cam_on_image(img_pil, cam)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Grad-CAM - Pred: {classes[pred_class]}', fontsize=14)

    axes[0].imshow(img_np)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('CAM')
    axes[1].axis('off')

    axes[2].imshow(heatmap)
    axes[2].set_title('Heatmap')
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title(f'Overlay (Pred: {classes[pred_class]})')
    axes[3].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'gradcam_single.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


if __name__ == '__main__':
    MODEL_TYPE = 'base'
    WEIGHT_PATH = '../../out/train/avg/checkpoint.pth'
    IMAGE_DIR = '../../STL10/test'
    OUTPUT_DIR = '../../out/gradcam/avg'

    visualize_gradcam(
        model_type=MODEL_TYPE,
        weight_path=WEIGHT_PATH,
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        num_samples=3
    )
