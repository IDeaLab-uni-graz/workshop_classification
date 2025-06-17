import matplotlib.pyplot as plt
import numpy as np
import torch

def show_selected_images_labels(images, labels, descriptions, title_prefix="Label", vmin=-1, vmax=1, rows=3, cols=3):
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

    fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axs = axs.flatten() if n > 1 else [axs]

    for i, (img, label, desc) in enumerate(zip(images, labels, descriptions)):
        img_np = img.permute(1, 2, 0).numpy()
        label_str = np.array2string(np.asarray(label), precision=3, separator=', ')
        axs[i].imshow(img_np, vmin=vmin, vmax=vmax, cmap='gray' if img_np.shape[-1] == 1 else None)
        axs[i].set_title(f"{title_prefix}: {desc}, {label_str}", fontsize=9)
        axs[i].axis("off")

    for j in range(n, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_predictions(x_test, y_test, y_pred, vmin=-1, vmax=1, rows=3, cols=3):
    true_labels = y_test
    images = x_test
    pred_labels = y_pred

    num_images = images.shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(rows * cols):
        if i < num_images:
            ax = axes[i]
            img = images[i, :].permute(1, 2, 0).numpy()
            true_label = true_labels[i].item()
            pred_label = pred_labels[i].item()

            # Show the image
            ax.imshow(img, vmin=vmin, vmax=vmax, cmap='gray' if img.shape[-1] == 1 else None)
            ax.axis("off")

            # Set title with correct or incorrect prediction
            title_color = "green" if true_labels[i] == pred_labels[i] else "red"
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10, color=title_color)

    plt.tight_layout()
    plt.show()