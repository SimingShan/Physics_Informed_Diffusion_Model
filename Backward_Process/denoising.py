import torch
import matplotlib.pyplot as plt
from Forward_Process.noising import setup_diffusion, forward_diffusion_sample, get_index_from_list  # Adjust path as necessary
from utils.data_transform_utils import show_tensor_image
import os
@torch.no_grad()
def sample_timestep(x, t, model, diffusion_params, device):
    # Ensure tensors are on the correct device
    betas = diffusion_params['betas'].to(device)
    sqrt_one_minus_alphas_cumprod = diffusion_params['sqrt_one_minus_alphas_cumprod'].to(device)
    sqrt_recip_alphas = (1 / torch.sqrt(diffusion_params['alphas_cumprod'])).to(device)
    posterior_variance = diffusion_params['posterior_variance'].to(device)

    betas_t = get_index_from_list(betas, t, x.shape).to(device)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape).to(device)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape).to(device)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape).to(device)

    # Assuming model(x, t) returns noise prediction
    noise_pred = model(x.to(device), t).to(device)  # Ensure model's input is on the correct device
    model_mean = sqrt_recip_alphas_t * (x.to(device) - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

    if t.item() == 0:
        return model_mean
    else:
        noise = torch.randn_like(x).to(device)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def sample_plot_image(model, device, diffusion_params, output_num, IMG_SIZE=256, T=200):
    model.eval()
    img = torch.randn((1, 1, IMG_SIZE, IMG_SIZE), device=device)
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))  # Adjust figsize as needed

    num_images = 10
    stepsize = max(int(T / num_images), 1)

    for idx, i in enumerate(range(T, 0, -stepsize)[:num_images]):
        t = torch.full((1,), i - 1, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model, diffusion_params, device)
        img = torch.clamp(img, -1.0, 1.0)
        ax = axes[idx]
        show_tensor_image(img, ax)
        ax.axis('off')  # Ensure axis is turned off for each subplot

    # Ensure the output directory exists
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the figure
    save_path = f'{output_dir}/output_{output_num}.png'
    plt.savefig(save_path)
    print(f"Image saved to {save_path}")
    plt.close(fig)  # Close the figure to free memory

