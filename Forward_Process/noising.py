import torch
import torch.nn.functional as F

def linear_beta_schedule(timesteps, start, end):
    return torch.linspace(start, end, timesteps)

def setup_diffusion(timesteps, start, end):
    betas = linear_beta_schedule(timesteps=timesteps, start=start, end=end)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'posterior_variance': posterior_variance
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
    noisy_batch, noise = forward_diffusion_sample(batch, t, diffusion_params, device=device)

    return noisy_batch, noise


