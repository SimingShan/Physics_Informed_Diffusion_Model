import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Functions Definitions
def load_data(file_path):
    hi_res = np.load(file_path)
    train_data = hi_res[:32].reshape(-1, hi_res.shape[2], hi_res.shape[3])
    dev_data = hi_res[32:36]
    test_data = hi_res[36:]
    return train_data, dev_data, test_data

class VorticityDataset(Dataset):
    def __init__(self, numpy_data, transform=None):
        self.data = numpy_data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

# Normalize function as transform
def normalize(sample):
    sample = torch.from_numpy(sample).float().unsqueeze(0)  # Add channel dimension (N, C, H, W)
    sample = (sample - sample.min()) / (sample.max() - sample.min())  # Scale to [0, 1]
    sample = (sample * 2) - 1  # Scale to [-1, 1]
    return sample

def normalize_data(data):
    data = torch.from_numpy(data).float()
    data = data.unsqueeze(1)  # Add channel dimension if needed (N, C, H, W)
    data = (data - data.min()) / (data.max() - data.min())  # Scale to [0, 1]
    data = (data * 2) - 1  # Scale to [-1, 1]
    return data


def prepare_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def show_tensor_image(image_tensor):
    """
    Display a tensor as an image, inversely normalizing it.
    """
    image_tensor = (image_tensor + 1) / 2  # Scale from [-1, 1] to [0, 1]
    image_np = image_tensor.squeeze().cpu().numpy()  # Remove channel dimension and convert to numpy
    plt.imshow(image_np, cmap='twilight')  # 'viridis' works well for single-channel data
    plt.axis('off')
    plt.show()

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def setup_diffusion(timesteps, start, end):
    betas = linear_beta_schedule(timesteps=timesteps, start=start, end=end)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod
    }

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    # Ensure t is on the same device as vals before gathering
    t = t.to(vals.device)  # Move t to the device of vals
    batch_size = t.shape[0]
    out = vals.gather(-1, t)  # Now both tensors are on the same device
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def forward_diffusion_sample(x_0, t, diffusion_params, device="cuda"):
    """
    Takes an image and a timestep as input and returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod = diffusion_params['sqrt_alphas_cumprod']
    sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod']

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


def apply_diffusion_to_batch(batch, t, diffusion_params, device):
    """
    Applies the forward diffusion process to a batch of data at a single timestep.
    """
    # Move the batch to the device
    batch = batch.to(device)

    # Apply the diffusion process to the batch
    noisy_batch, _ = forward_diffusion_sample(batch, t, diffusion_params, device=device)

    return noisy_batch


# Load the data
train_data, dev_data, test_data = load_data(file_path="../data/kf_2d_re1000_256_40seed.npy")

# Create VorticityDataset
train_dataset = VorticityDataset(train_data, transform=normalize)

# Create DataLoader for the training data
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, drop_last=True)

# Set up diffusion parameters
device = prepare_device()
diffusion_params = setup_diffusion(timesteps=200, start=0.0001, end=0.02)
diffusion_params = {k: v.to(device) for k, v in diffusion_params.items()}

# Specify the timestep for diffusion
timestep_tensor = torch.tensor([10], dtype=torch.long, device=device)  # Single timestep tensor

# Iterate over batches and apply diffusion
batch = next(iter(train_loader))
noisy_batch = apply_diffusion_to_batch(batch, timestep_tensor, diffusion_params, device)
show_tensor_image(noisy_batch[0])

