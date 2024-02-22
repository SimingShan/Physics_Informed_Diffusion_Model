import matplotlib.pyplot as plt
import torch

def normalize(sample):
    sample = torch.from_numpy(sample).float().unsqueeze(0)  # Add channel dimension (N, C, H, W)
    sample = (sample - sample.min()) / (sample.max() - sample.min())  # Scale to [0, 1]
    sample = (sample * 2) - 1  # Scale to [-1, 1]
    return sample

def show_tensor_image(image_tensor):
    """
    Display a tensor as an image, inversely normalizing it.
    """
    image_tensor = (image_tensor + 1) / 2  # Scale from [-1, 1] to [0, 1]
    image_np = image_tensor.squeeze().cpu().numpy()  # Remove channel dimension and convert to numpy
    plt.imshow(image_np, cmap='twilight')  # 'viridis' works well for single-channel data
    plt.axis('off')
    plt.show()