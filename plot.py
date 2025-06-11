import matplotlib.pyplot as plt
import numpy as np
import torch

def show_selected_images_labels(images, labels, descriptions, title_prefix="Label"):
    """
    Display selected images and their corresponding labels.

    Args:
        images (Tensor): shape (B, C, H, W)
        labels (Tensor or np.ndarray): shape (B, ...)
        indices (list or array): indices to show
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    n = images.shape[0]
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.flatten() if n > 1 else [axs]

    for i, (img, label, desc) in enumerate(zip(images, labels, descriptions)):
        img_np = img.permute(1, 2, 0).numpy()
        label_str = np.array2string(np.asarray(label), precision=3, separator=', ')
        axs[i].imshow(img_np)
        axs[i].set_title(f"{title_prefix}: {desc}, {label_str}", fontsize=9)
        axs[i].axis("off")

    for j in range(n, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()