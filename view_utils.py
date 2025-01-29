import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import imageio
from typing import Literal
import ipywidgets as widgets
from IPython.display import display

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
        ax.imshow(frame, vmin=array.min(), vmax=array.max(), cmap='gray')
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



def display_image_with_slider(images, vlim: Literal["stack", "frame", "all"]="stack", titles: np.ndarray = None, **fig_kw):
    """
        Shows a set of images side-by-side with a common slider for the time dimension.
        Args:
            - images: A np.ndarray of images of shape [N, T, H, W] or [T, H, W]

    """
    if images.ndim == 3:
        images = images[None, ...]

    N, T, Y, X = images.shape
    all_vmin = [i.min() for i in images] if vlim == "stack" else [images.min() for i in range(N)] 
    all_vmax = [i.max() for i in images] if vlim == "stack" else [images.max() for i in range(N)]

    def update_image(t):
        fig, axs = plt.subplots(1, N, squeeze=False, **fig_kw)
        for n, ax in enumerate(axs[0]):
            img_to_show = images[n, t]
            if vlim == "frame":
                vmin, vmax = img_to_show.min(), img_to_show.max()
            else:
                vmin, vmax = all_vmin[n], all_vmax[n]
            ax.imshow(img_to_show, cmap='gray', vmin=vmin, vmax=vmax)

            if titles is not None:
                ax.set_title(titles[n])
                
        fig.tight_layout()
        plt.show()

    slider = widgets.IntSlider(min=0, max=T-1, step=1, description='T')
    widgets.interact(update_image, t=slider)

