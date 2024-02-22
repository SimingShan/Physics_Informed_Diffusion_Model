import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import imageio
from PIL import Image

# Load high-resolution data
high_res_data = np.load("..\data\kf_2d_re1000_256_40seed.npy")

def downsample_data(data, target_shape):
    """
    Downsamples a given 4D dataset to a specified spatial resolution.
    """
    zoom_factors = (1, 1, target_shape[0] / data.shape[2], target_shape[1] / data.shape[3])
    downsampled_data = zoom(data, zoom_factors, order=1)
    return downsampled_data


def create_gif(data, target_dir, filename, fps=5):
    """
    Creates a GIF from a set of 2D slices (frames) in data.
    """
    images = []
    for i in range(data.shape[0]):
        fig, ax = plt.subplots()
        ax.imshow(data[i], cmap='twilight', interpolation='none')
        plt.axis('off')
        fname = f'{target_dir}/tmp_{i}.png'
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        images.append(imageio.imread(fname))
    imageio.mimsave(f'{target_dir}/{filename}', images, fps=fps)


def randomly_draw_points(data, percentage):
    """
    Randomly draws a percentage of spatial points from the 4D dataset.

    Parameters:
    - data: 4D numpy array, shape (num_simulations, timesteps, height, width)
    - percentage: float, the percentage of points to draw

    Returns:
    - drawn_points: 4D numpy array, same shape as data, with randomly drawn points
                     and the rest set to zero (or NaN to signify absence).
    """
    # Calculate the number of points to draw
    total_points = data.shape[2] * data.shape[3]
    points_to_draw = int(total_points * percentage / 100)

    # Initialize an array to store the drawn points
    drawn_points = np.zeros_like(data)

    # Iterate through each simulation and timestep
    for sim in range(data.shape[0]):
        for time in range(data.shape[1]):
            # Flatten the spatial dimensions
            flat_data = data[sim, time].flatten()

            # Randomly select indices
            selected_indices = np.random.choice(data.shape[2] * data.shape[3], points_to_draw, replace=False)

            # Set the selected points in the drawn_points array
            np.put(drawn_points[sim, time], selected_indices, np.take(flat_data, selected_indices))

    return drawn_points


# Directory for saving temporary images and final GIFs
target_dir = "..\\data_preprocessing\\output_image"

# Downsample to 64x64 and create GIF
low_res_data_64 = downsample_data(high_res_data, (64, 64))
create_gif(low_res_data_64[0], target_dir, 'visualization_low_64.gif')
print(low_res_data_64.shape)
# Downsample to 32x32 and create GIF
low_res_data_32 = downsample_data(low_res_data_64, (32, 32))
create_gif(low_res_data_32[0], target_dir, 'visualization_low_32.gif')
print(low_res_data_32.shape)
# Draw 5% of the points
drawn_points_5_percent = randomly_draw_points(high_res_data, 5)
create_gif(drawn_points_5_percent[0], target_dir, 'visualization_low_5p.gif')
print(drawn_points_5_percent.shape)
# Draw 1.5625% of the points
drawn_points_1_5625_percent = randomly_draw_points(high_res_data, 1.5625)
create_gif(drawn_points_1_5625_percent[0], target_dir, 'visualization_low_1_5625p.gif')
print(drawn_points_1_5625_percent.shape)
# Save the 32*32 down-sampled dataset
np.save("../data/low_res_data_32.npy", low_res_data_32)

# Save the 64*64 down-sampled dataset
np.save("../data/low_res_data_64.npy", low_res_data_64)

# Save the 5% drawn points dataset
np.save("../data/drawn_points_5.npy", drawn_points_5_percent)

# Save the 1.5625% drawn points dataset
np.save("../data/drawn_points_1_5625.npy", drawn_points_1_5625_percent)
