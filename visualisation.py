from typing import Callable, Optional, Tuple
from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Subplot:
    title: str
    img: np.ndarray
    cmap: Optional[str] = None
    slider_label: Optional[str] = None
    slider_range: Tuple[float, float] = (0, 1)
    slider_initial_value: float = 1
    slider_update_function: Optional[Callable[[float], np.ndarray]] = None

    def __post_init__(self):
        self.img = self.slider_update_function(self.slider_initial_value) if self.slider_update_function else self.img


def plot(subplots_data: list[Subplot], plot_sync_zoom: bool = True):

    num_rows = int(np.ceil(len(subplots_data) / 3))
    fig, axes = plt.subplots(num_rows, 3, figsize=(16, 8), sharex=plot_sync_zoom, sharey=plot_sync_zoom, constrained_layout=True)

    axes = axes.flatten()
    sliders = []
    images = []

    for i, subplot_data in enumerate(subplots_data):
        ax = axes[i]
        im = ax.imshow(subplot_data.img, cmap=subplot_data.cmap)
        ax.set_title(subplot_data.title)
        ax.axis('off')
        images.append(im)

        if (subplot_data.slider_update_function):
            bbox = ax.get_position()
            slider_ax = fig.add_axes([bbox.x0, bbox.y0 - 0.05, bbox.width, 0.03]) # Adjust y0 and height as needed

            slider = Slider(
                ax=slider_ax,
                label=subplot_data.slider_label,
                valmin=subplot_data.slider_range[0],
                valmax=subplot_data.slider_range[1],
                valinit=subplot_data.slider_initial_value
            )
            def update(val, current_im=im, current_update_func=subplot_data.slider_update_function):
                new_img = current_update_func(val)
                current_im.set_data(new_img)
                fig.canvas.draw_idle()

            slider.on_changed(update)
            sliders.append(slider)

    # Hide any remaining empty subplots
    for i in range(len(subplots_data), len(axes)):
        axes[i].axis('off')

    plt.show()
