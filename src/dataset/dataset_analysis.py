import matplotlib.pyplot as plt
import numpy as np


def plot_ball_positions(ax: plt.Axes, dataset_row: np.ndarray, num_balls: int) -> None:
    """
    Plot the positions of the balls in the three frames on the given axes.

    Parameters
    ----------
    ax : plt.Axes
        The axes on which to plot the positions.
    dataset_row : np.ndarray
        A row from the dataset containing ball positions and velocities.
    num_balls : int
        The number of balls in each system.
    """
    colors = ["r", "y", "b"]
    for frame in range(3):
        for j in range(num_balls):
            ax.scatter(
                dataset_row[2 * (j + frame * num_balls)],
                dataset_row[2 * (j + frame * num_balls) + 1],
                c=colors[frame],
                s=10,
            )


def calculate_length(dataset_row: np.ndarray, num_balls: int) -> float:
    """
    Calculate the length between initial and second frame positions of the first ball.
    All balls share the same velocity magnitude, so length is equal for all balls.

    Parameters
    ----------
    dataset_row : np.ndarray
        A row from the dataset containing ball positions and velocities.
    num_balls : int
        The number of balls in each system.

    Returns
    -------
    float
        The calculated length.
    """
    return np.sqrt(
        (dataset_row[0] - dataset_row[2 * num_balls]) ** 2
        + (dataset_row[1] - dataset_row[2 * num_balls + 1]) ** 2
    )


def calculate_speed_length_ratio(dataset_row: np.ndarray, num_balls: int) -> float:
    """
    Calculate the ratio of speed to length for the first ball.
    All balls share the same speed, so it can be computed from any ball.

    Parameters
    ----------
    dataset_row : np.ndarray
        A row from the dataset containing ball positions and velocities.
    num_balls : int
        The number of balls in each system.

    Returns
    -------
    float
        The ratio of speed to length.
    """
    length = np.sqrt(
        (dataset_row[0] - dataset_row[4 * num_balls]) ** 2
        + (dataset_row[1] - dataset_row[4 * num_balls + 1]) ** 2
    )
    speed = dataset_row[-1]
    return speed / length


def analyze_and_plot_dataset(dataset: np.ndarray, num_samples: int = 5) -> None:
    """
    Analyze and plot the positions of balls from the dataset.

    Parameters
    ----------
    dataset : np.ndarray
        The dataset containing ball positions and velocities.
    num_samples : int
        The number of samples to analyze and plot, by default 5.
    """
    num_balls = (dataset.shape[1] - 2) // 6

    for i in range(num_samples):
        fig, ax = plt.subplots()

        plot_ball_positions(ax, dataset[i], num_balls)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")
        plt.show()

        length = calculate_length(dataset[i], num_balls)
        speed_length_ratio = calculate_speed_length_ratio(dataset[i], num_balls)

        print(f"Length: {length}")
        print(f"Speed: {dataset[i][-1]}")
        print(f"Speed * Length: {speed_length_ratio}")
