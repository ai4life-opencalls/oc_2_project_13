import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import imageio

def create_gif_from_numpy(array, output_file, scale_factor=1.0, frame_skip=10, fps=10):
    """
    Create a GIF from a numpy array using matplotlib.
    
    Parameters:
    - array: numpy array of shape [1000, 1024, 1024]
    - output_file: str, the filename of the output GIF
    - scale_factor: float, the factor by which to scale the frames
    - frame_skip: int, the number of frames to skip
    - fps: int, frames per second for the gif
    
    Returns:
    - None (Saves a GIF file)
    """
    frames = []

    # Loop through the array with frame skipping
    for i in range(0, array.shape[0], frame_skip):
        frame = array[i]

        # Rescale the frame using nearest-neighbor interpolation
        if scale_factor != 1.0:
            frame = zoom(frame, zoom=(scale_factor, scale_factor), order=0)  # nearest-neighbor interpolation

        # Plot the frame using matplotlib
        fig, ax = plt.subplots(figsize=(frame.shape[1] / 100, frame.shape[0] / 100), dpi=100)
        ax.imshow(frame)
        ax.axis('off')  # Turn off the axis

        # Adjust the figure layout to remove margins
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save the plot to a buffer instead of showing it
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        plt.close(fig)
    
    # Create GIF using imageio
    imageio.mimsave(output_file, frames, fps=fps)
    
    # Create GIF using imageio
    imageio.mimsave(output_file, frames, fps=fps)