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

def plot_loss(ob_val, acc):
    def moving_average(x, window=50):
        return np.convolve(x, np.ones(window)/window, mode='valid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    # --- Plot Loss ---
    ax1.plot(ob_val, linewidth=1, alpha=0.5, color='blue')
    smoothed_loss = moving_average(ob_val)
    ax1.plot(range(len(smoothed_loss)), smoothed_loss, linewidth=2, color='darkblue')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Training Steps')
    ax1.set_ylim([0, 1.0])
    ax1.set_xlim([0, len(ob_val)]) 

    # --- Plot Accuracy ---
    ax2.plot(acc, linewidth=1, alpha=0.5, color='red', label='Raw Accuracy')
    smoothed_acc = moving_average(acc)
    ax2.plot(range(len(smoothed_acc)), smoothed_acc, linewidth=2, color='darkred')
    ax2.set_title('Accuracy over Training Steps')
    
    # Add horizontal milestone lines
    milestones=[0.5, 0.75]
    for milestone in milestones:
        ax2.axhline(y=milestone, color='gray', linestyle='--', linewidth=1)
        ax2.text(len(acc)*0.01, milestone + 0.01, f"{milestone:.1%}", color='gray', fontsize=8)

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0.25, 1.0])
    ax2.set_xlim([0, len(acc)]) 

    plt.tight_layout()
    #plt.pause(0.001)

def compare_to_target(function_to_learn, type):
    xmin, xmax = -3, 3
    ymin, ymax = -8, 8

    x = np.linspace(xmin, xmax, 100)

    if type == 'function':
        target_function = lambda x: 2*x + 1
        target_x = x
        target_y = target_function(target_x)
    elif type == 'data':
        target_function = lambda x: -2*x - 1
        target_x = np.array([-2, 0, 2])
        noise = np.array([1.3, -0.6, 1.0])
        target_y = target_function(target_x) + noise

    # Plot code
    fig, ax = plt.subplots(figsize=(8, 6))

    def add_arrow(ax, x1, y1, x2, y2, i):
        # Draw arrow from (x1, y1) to (x2, y2)
        ax.annotate(
            '',  # No text
            xy=(x2, y2),  # End of arrow
            xytext=(x1, y1),  # Start of arrow
            arrowprops=dict(
                arrowstyle='<->',
                color='black',
                linewidth=2,
            )
        )

        mid_y = (y1 + y2) / 2
        err = np.abs(y2 - y1)
        ax.text(x1 + 0.1, mid_y, f'$Δ_{i} = {err:.2f}$', va='center', ha='left', fontsize=12)

    ax.plot(x, function_to_learn(x), label='Our function to learn', color='red', zorder=11, linewidth=3)

    if type == 'function':
        ax.plot(x, target_function(x), label='Unknown target function $f$', color='blue', zorder=10, linewidth=2)

    elif type == 'data':
        my_y = function_to_learn(target_x)
        ax.scatter(target_x, target_y, label='Data from our unknown target function $f$', color='blue', zorder=10, linewidth=2)
        ax.scatter(target_x, my_y, color='red', zorder=11, linewidth=3)

        sum = np.sum((my_y - target_y) ** 2)
        ax.text(xmin + 0.2, ymin + 1.0, f'$\\Delta_1^2 + \\Delta_2^2 + \\Delta_3^2 = {sum:.2f}$', va='center', ha='left',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.5), fontsize=18)

        add_arrow(ax, target_x[0], target_y[0], target_x[0], my_y[0], 1)
        add_arrow(ax, target_x[1], target_y[1], target_x[1], my_y[1], 2)
        add_arrow(ax, target_x[2], target_y[2], target_x[2], my_y[2], 3)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.arange(xmin, xmax + 1, 1))
    ax.set_yticks(np.arange(ymin, ymax + 1, 1))
    ax.grid()
    ax.legend()
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_title(f'Comparison of our function to learn with the target {type}')
    plt.show()


def plot_loss_landscape(func, sequence, x_range, y_range):
    xmin, xmax = x_range
    ymin, ymax = y_range

    x = np.linspace(xmin, xmax, 400)
    y = np.linspace(ymin, ymax, 400)

    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    opt_a, opt_b = -2.075, -0.43333333
    f_opt = func(opt_a, opt_b)

    # Plot code

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection='3d')

    def plot_line_on_surface(ax, p0, p1, func, n=20):
        x0, y0 = p0
        x1, y1 = p1

        if not isinstance(p0, np.ndarray):
            p0 = np.array([p0])
        if not isinstance(p1, np.ndarray):
            p1 = np.array([p1])

        x = p0 + (p1 - p0) * np.linspace(0, 1, n).reshape(-1, 1)
        z = func(x[:, 0], x[:, 1])

        ax.plot(x[:, 0], x[:, 1], z, color='red', alpha=1.0, zorder=5)
        ax.plot([x0], [y0], [func(x0, y0)], color='red', alpha=1.0, zorder=5, marker='o', markersize=6)
        ax.plot([x1], [y1], [func(x1, y1)], color='red', alpha=1.0, zorder=5, marker='o', markersize=6)

    surface = ax.plot_surface(X, Y, Z, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                              alpha=1.0)  # Use default matplotlib blue with transparency

    for p0, p1 in zip(sequence[:-1], sequence[1:]):
        plot_line_on_surface(ax, p0, p1, func)

    ax.plot([opt_a], [opt_b], [f_opt], color='orange', alpha=1.0, marker='*', markersize=10, zorder=5)
    ax.plot([opt_a, opt_a], [opt_b, opt_b], [0.0, f_opt], color='orange', alpha=1.0)
    ax.plot([opt_a, x_range[1] + 0.01], [opt_b, opt_b], [0.0, 0.0], color='orange', alpha=1.0)
    ax.plot([opt_a, opt_a], [opt_b, y_range[1] + 0.03], [0.0, 0.0], color='orange', alpha=1.0)

    Z = func(opt_a, opt_b) * np.ones_like(Z)
    surface = ax.plot_surface(X, Y, Z, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1], alpha=0.2)

    ax.set_xlabel("a", fontsize=14)
    ax.set_ylabel("b", fontsize=14)
    ax.view_init(elev=20, azim=75, roll=0)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(0.0, 4.0)

    ax.text(opt_a + 0.1, opt_b, f_opt + 0.3, "global minimum", fontsize=12, color='orange', zorder=100)

    ax.set_zlabel("$\\text{loss}(a,b)$", fontsize=14)
    plt.title("3D plot of our loss in dependence of the two parameters $a,b$", fontsize=14)
    plt.show()

def plot_2d_function(func, x_range, y_range, filename=""):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                              alpha=1)  # Use default matplotlib blue with transparency

    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    # ax.set_zlabel("z", fontsize=14)
    # ax.set_title("Surface Plot of the Function", fontsize=16)
    ax.invert_xaxis()  # Flip the x-axis

    if filename:
        plt.savefig(filename)
    plt.show()