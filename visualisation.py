import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Subplot:
    title: str
    img: np.ndarray
    cmap: str | None = None


def plot(subplots_data, plot_sync_zoom: bool = True):

    num_rows = int(np.ceil(len(subplots_data) / 3))
    _, axes = plt.subplots(num_rows, 3, figsize=(16, 8), sharex=plot_sync_zoom, sharey=plot_sync_zoom)

    # plt.get_current_fig_manager().full_screen_toggle() # Fullscreen mode
    # plt.get_current_fig_manager().window.state('zoomed') # WARNING: This may not work in all environments

    axes = axes.flatten()

    for i, subplot in enumerate(subplots_data):
        ax = axes[i]
        ax.imshow(subplot.img, cmap=subplot.cmap)
        ax.set_title(subplot.title)
        ax.axis('off')

    # Hide any remaining empty subplots
    for i in range(len(subplots_data), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
