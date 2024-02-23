import matplotlib.pyplot as plt
import torch
from utils.data_utils import prepare_device
from Forward_Process.noising import get_index_from_list
device = prepare_device()
def normalize(sample):
    sample = torch.from_numpy(sample).float().unsqueeze(0)  # Add channel dimension (N, C, H, W)
    sample = (sample - sample.min()) / (sample.max() - sample.min())  # Scale to [0, 1]
    sample = (sample * 2) - 1  # Scale to [-1, 1]
    return sample

def show_tensor_image(image_tensor, ax=None):
    """
    Display a tensor as an image, inversely normalizing it, on a given axis.
    """
    if ax is None:
        ax = plt.gca()
    image_tensor = (image_tensor + 1) / 2  # Scale from [-1, 1] to [0, 1]
    image_np = image_tensor.squeeze().cpu().numpy()  # Remove channel dimension and convert to numpy
    ax.imshow(image_np, cmap='twilight')
    ax.axis('off')


