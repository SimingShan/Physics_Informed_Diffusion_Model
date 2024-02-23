from tqdm import tqdm
import torch.nn.functional as F
from Forward_Process.noising import forward_diffusion_sample
import torch
from Backward_Process.denoising import sample_plot_image, sample_timestep
import os
def train(model, dataloader, optimizer, device, epochs, T, diffusion_params):
    model.train()
    output_num = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # Wrap dataloader with tqdm for a progress bar
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}')
        for batch_idx, batch in progress_bar:
            # Assuming your dataset returns a batch in the format (data, timestep)
            x_0 = batch
            x_0 = x_0.to(device)

            # Generate random timesteps for each image in the batch
            t = torch.randint(0, T, (dataloader.batch_size,), device=device).long()

            optimizer.zero_grad()

            # Perform the forward pass and calculate the loss
            x_noisy, noise = forward_diffusion_sample(x_0, t, diffusion_params, device)
            noise_pred = model(x_noisy, t)
            loss = F.l1_loss(noise, noise_pred)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update the progress bar with the current average loss
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}')
        sample_plot_image(model, device, diffusion_params, output_num)
        output_num = + 1


