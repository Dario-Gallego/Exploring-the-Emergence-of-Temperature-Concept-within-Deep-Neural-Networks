import matplotlib.pyplot as plt
import numpy as np


def get_grayscale_ball_plot(
    ax: plt.Axes, x: float, y: float, ax_min: float, ax_max: float, pixels_per_axis: int
) -> np.ndarray:
    """
    Plot a ball at specified coordinates and return the grayscale image.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib Axes object to plot on.
    x : float
        X-coordinate of the ball.
    y : float
        Y-coordinate of the ball.
    ax_min : float
        Minimum value of the axes.
    ax_max : float
        Maximum value of the axes.
    pixels_per_axis : int
        Number of pixels per axis in the image.

    Returns
    -------
    plot_gray : np.ndarray
        Grayscale image array.
    """
    ax.scatter(x, y, c="k", s=2)
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.axis("off")
    ax.figure.tight_layout(pad=0.0)
    ax.figure.set_size_inches(0.01 * pixels_per_axis, 0.01 * pixels_per_axis)
    ax.figure.canvas.draw()
    plot_array = np.array(ax.figure.canvas.renderer.buffer_rgba())
    plot_gray = np.dot(plot_array[..., :3], [0.2989, 0.5870, 0.1140])
    plot_gray /= np.max(plot_gray)
    ax.figure.canvas.renderer.clear()
    ax.clear()
    return plot_gray
