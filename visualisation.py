from typing import Callable, Dict, Optional, Tuple, List # Added List
from matplotlib.widgets import Slider
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class SliderConfig:
    label: str = ""
    range: Tuple[float, float] = (0, 1)
    initial_value: float = 1

@dataclass
class Subplot:
    title: str
    img: np.ndarray
    cmap: Optional[str] = None
    slider_configs: Optional[List[SliderConfig]] = None
    slider_update_function: Optional[Callable[[Tuple[float, ...]], np.ndarray]] = None

    def __post_init__(self):
        # Only initialize img if slider_configs and slider_update_function are provided
        if self.slider_configs and self.slider_update_function:
            initial_values = [c.initial_value for c in self.slider_configs]
            self.img = self.slider_update_function(tuple(initial_values))


def plot(subplots_data: List[Subplot], plot_sync_zoom: bool = True):

    num_rows = int(np.ceil(len(subplots_data) / 3))
    max_sliders_per_subplot = max([len(s.slider_configs) if s.slider_configs else 0 for s in subplots_data], default=0)
    extra_height_per_row = max_sliders_per_subplot * 0.05 * 3
    fig, axes = plt.subplots(num_rows, 3, figsize=(16, 8 + extra_height_per_row), sharex=plot_sync_zoom, sharey=plot_sync_zoom, constrained_layout=True)

    axes = axes.flatten()

    # Dictionaries to store references keyed by subplot index
    image_artists: Dict[int, plt.AxesImage] = {}
    subplot_slider_value_lists: Dict[int, List[float]] = {}
    subplot_update_functions: Dict[int, Callable[[Tuple[float, ...]], np.ndarray]] = {}
    all_sliders = [] 

    def _global_update_callback(new_val: float, slider_idx: int, subplot_idx: int):
        # Retrieve the specific list of slider values for this subplot
        current_values_list = subplot_slider_value_lists[subplot_idx]
        current_values_list[slider_idx] = new_val # Update the specific slider's value

        im_ref = image_artists[subplot_idx]
        update_func = subplot_update_functions[subplot_idx]

        new_img = update_func(tuple(current_values_list))
        im_ref.set_data(new_img)
        fig.canvas.draw_idle()


    for subplot_index, subplot_data in enumerate(subplots_data):
        ax = axes[subplot_index]
        im = ax.imshow(subplot_data.img, cmap=subplot_data.cmap)
        ax.set_title(subplot_data.title)
        ax.axis('off')
        image_artists[subplot_index] = im

        if subplot_data.slider_configs and subplot_data.slider_update_function:
            bbox_img = ax.get_position()
            current_slider_y = bbox_img.y0 - 0.04
            slider_height = 0.03

            current_subplot_slider_values = [c.initial_value for c in subplot_data.slider_configs]
            subplot_slider_value_lists[subplot_index] = current_subplot_slider_values
            subplot_update_functions[subplot_index] = subplot_data.slider_update_function # Store the update function


            for slider_idx, slider_config in enumerate(subplot_data.slider_configs):
                slider_ax = fig.add_axes([bbox_img.x0, current_slider_y, bbox_img.width, slider_height])
                slider = Slider(
                    ax=slider_ax,
                    label=slider_config.label,
                    valmin=slider_config.range[0],
                    valmax=slider_config.range[1],
                    valinit=slider_config.initial_value
                )

                slider.on_changed(lambda val, s_idx=slider_idx, p_idx=subplot_index: _global_update_callback(val, s_idx, p_idx))
                all_sliders.append(slider)

                current_slider_y -= (slider_height + 0.01)

    # Hide any remaining empty subplots
    for i in range(len(subplots_data), len(axes)):
        axes[i].axis('off')

    plt.show()