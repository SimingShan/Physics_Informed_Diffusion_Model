from Forward_Process.noising import forward_diffusion_sample
from utils.data_utils import prepare_device
import torch.nn.functional as F

device = prepare_device()

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)
