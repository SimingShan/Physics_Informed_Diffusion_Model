import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn.functional as F

hi_res = np.load("../data/kf_2d_re1000_256_40seed.npy")

# Assuming hi_res is already loaded and reshaped
train_data = hi_res[:32].reshape(-1, hi_res.shape[2], hi_res.shape[3])
print(train_data.shape)

# Normalize the training data to the range [-1, 1]
# Assuming the vorticity data is already in a range such as [0, 1] or similar
train_data = torch.from_numpy(train_data).float()
train_data = train_data.unsqueeze(1)  # Add channel dimension if needed (N, C, H, W)
train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())  # Scale to [0, 1]
train_data = (train_data * 2) - 1  # Scale to [-1, 1]
# Assuming you're running this on a system with CUDA available. If not, set device = "cpu".
device = "cuda" if torch.cuda.is_available() else "cpu"
# Convert train_data to the selected device
x_0 = train_data.to(device)
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

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

def forward_diffusion_sample(x_0, t, device="cuda"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 200
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# Make sure the betas and pre-calculated terms are on the same device
betas = betas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)


def show_tensor_image(image_tensor):
    """
    Display a tensor as an image, inversely normalizing it.
    """
    image_tensor = (image_tensor + 1) / 2  # Scale from [-1, 1] to [0, 1]
    image_np = image_tensor.squeeze().cpu().numpy()  # Remove channel dimension and convert to numpy
    plt.imshow(image_np, cmap='twilight')  # 'viridis' works well for single-channel data
    plt.axis('off')
    plt.show()

# After generating the noisy image
t = torch.tensor([199], dtype=torch.long, device=device)  # A single timestep for demonstration
noisy_image, noise = forward_diffusion_sample(x_0[:1], t, device=device)

# Convert to CPU and inverse normalize for plotting
show_tensor_image(noisy_image)

# Plot the noisy image - convert it back to CPU and NumPy for plotting
noisy_image_np = noisy_image.squeeze().cpu().numpy()
plt.imshow(noisy_image_np, cmap='twilight')
plt.title("Noisy Image at Timestep 100")
plt.axis('off')
plt.show()

