import numpy as np
import matplotlib.pyplot as plt
import imageio

data = np.load("../../data/kf_2d_re1000_256_40seed.npy")
data_0 = np.load("../../data/kmflow_sampled_data_irregnew.npz")

# Select the first sample
sample_data = data[0]
print(sample_data.shape)
# Initialize a list to store images
images = []

for i in range(sample_data.shape[0]):  # Loop through the 2nd dimension
    fig, ax = plt.subplots()
    im = ax.imshow(sample_data[i], cmap='viridis', interpolation='none')
    plt.axis('off')  # Optional: Remove axes for a cleaner look

    # Save the plot to a PNG and store the filename
    fname = f'tmp_{i}.png'
    plt.savefig(fname)
    plt.close(fig)  # Close the figure to free memory
    images.append(imageio.imread(fname))

# Create GIF
imageio.mimsave('visualization_high_0.gif', images, fps=5)

