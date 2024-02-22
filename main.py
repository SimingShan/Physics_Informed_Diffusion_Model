from model.u_net import SimpleUnet
from utils.data_utils import load_data, prepare_device, VorticityDataset
from torch.utils.data import DataLoader
from utils.data_transform_utils import normalize, show_tensor_image
from training.training import train
from torch.optim import Adam
from Forward_Process.noising import setup_diffusion,  forward_diffusion_sample, get_index_from_list
def main():
    # Configuration
    file_path = "data\kf_2d_re1000_256_40seed.npy"
    device = prepare_device()

    # Load and prepare data
    train_data, dev_data, test_data = load_data(file_path)
    train_dataset = VorticityDataset(train_data, transform=normalize)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

    # Model
    model = SimpleUnet().to(device)

    # set up optimizer
    optimizer = Adam(model.parameters(), lr=0.001)

    # set up epoch
    epoch = 10

    # set up diffusion
    diffusion_params = setup_diffusion(timesteps=200, start=0.0001, end=0.02)

    # Training
    train(model, train_loader, optimizer, device, epoch, 200, diffusion_params)


if __name__ == "__main__":
    main()
