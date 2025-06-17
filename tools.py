import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_2d_function(func, x_range, y_range, filename=""):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], alpha=1)  # Use default matplotlib blue with transparency
    
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    #ax.set_zlabel("z", fontsize=14)
    #ax.set_title("Surface Plot of the Function", fontsize=16)
    ax.invert_xaxis()  # Flip the x-axis
    
    if filename:
        plt.savefig(filename)
    plt.show()
    
    
    
    
    

def visualize_predictions(x_test, y_test,y_pred, rows=3, cols=3):

    true_labels = y_test
    pred_lables = y_pred
    images = x_test
    pred_labels = np.argmax(y_pred, axis=1)  # Get class index with highest probability
    softmax_outputs = np.max(y_pred, axis=1) 
    
    num_images = images.shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(rows * cols):
        if i < num_images:
            ax = axes[i]
            img = images[i,:].reshape(16,16)
            true_label = true_labels[i]
            pred_label = pred_labels[i]
            softmax_score = softmax_outputs[i]

            # Show the image
            ax.imshow(img, cmap='gray_r' if img.ndim == 2 else None)
            ax.axis("off")
            
            # Set title with correct or incorrect prediction
            title_color = "green" if true_labels[i] == pred_labels[i] else "red"
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10, color=title_color)
            
            # Add softmax confidence score
            ax.text(0.5, -0.15, f"Conf: {softmax_score:.2f}", 
                    fontsize=9, ha="center", va="top", transform=ax.transAxes, color='blue')

    plt.tight_layout()
    plt.show()
    
