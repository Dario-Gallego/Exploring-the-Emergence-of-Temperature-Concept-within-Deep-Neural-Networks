import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from tqdm import tqdm

from src.dataset.utils import get_grayscale_ball_plot


def create_ball(density: float, radius: float) -> int:
    """
    Create and initialize a ball in the simulation.

    Parameters
    ----------
    density : float
        The density of the ball.
    radius : float
        The radius of the ball.

    Returns
    -------
    ball_id : int
        The unique ID of the created ball.
    """
    # Create a sphere collision shape
    sphere_shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    # Create a multi-body object with the sphere visual shape
    visual_shape_id = -1
    ball_id = p.createMultiBody(density, sphere_shape_id, visual_shape_id)

    return ball_id


def initialize_ball(ball_id: int, velocity_magnitude: float) -> None:
    """
    Set random initial position and velocity for a ball.

    Parameters
    ----------
    ball_id : int
        The unique ID of the ball.
    velocity_magnitude : float
        The magnitude of the velocity vector for the ball.
    """
    # Set random position of ball on the plane
    p.resetBasePositionAndOrientation(
        ball_id,
        (np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0),
        p.getQuaternionFromEuler([0, 0, 0]),
    )
    # Set velocity direction
    velocity_x = np.random.uniform(-1, 1)
    velocity_y = np.sqrt(1 - velocity_x**2) * (-1) ** np.random.randint(2)
    base_velocity = velocity_magnitude * np.array([velocity_x, velocity_y, 0])
    p.resetBaseVelocity(ball_id, base_velocity, (0, 0, 0))


def remove_external_factors(ball_id: int) -> None:
    """
    Remove external factors affecting the ball's velocity.

    Parameters
    ----------
    ball_id : int
        The unique ID of the ball.
    """
    p.changeDynamics(ball_id, -1, linearDamping=0, angularDamping=0)
    p.applyExternalForce(ball_id, -1, [0, 0, 0], [0, 0, 0], p.WORLD_FRAME)
    p.applyExternalTorque(ball_id, -1, [0, 0, 0], p.WORLD_FRAME)
    p.changeDynamics(ball_id, -1, lateralFriction=0, restitution=0)
    p.setCollisionFilterGroupMask(ball_id, -1, 0, 0)


def get_ball_positions_and_velocities(
    balls: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the positions and velocities of the balls.

    Parameters
    ----------
    balls : list[int]
        A list of balls identifiers.

    Returns
    -------
    positions : np.ndarray
        Positions of balls.
    velocities : np.ndarray
        Velocities of balls.
    """
    num_balls = len(balls)
    positions = np.zeros((num_balls, 2))
    velocities = np.zeros((num_balls, 2))
    for ball_id in balls:
        idx = ball_id % num_balls
        positions[idx] = p.getBasePositionAndOrientation(ball_id)[0][:2]
        velocities[idx] = p.getBaseVelocity(ball_id)[0][:2]
    return positions, velocities


def record_positions_and_velocities(
    balls: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Record the positions and velocities of the balls at different frames.

    Parameters
    ----------
    balls : list of int
        A list of unique IDs for the balls.

    Returns
    -------
    image_t0 : np.ndarray
        Positions of balls at initial frame.
    image_t1 : np.ndarray
        Positions of balls at second frame.
    image_t2 : np.ndarray
        Positions of balls at third frame.
    velocities_vector : np.ndarray
        Velocities of balls.
    """
    # Initial frame
    image_t0, velocities_vector = get_ball_positions_and_velocities(balls)
    p.stepSimulation()
    # Second frame
    image_t1, _ = get_ball_positions_and_velocities(balls)
    p.stepSimulation()
    # Third frame
    image_t2, _ = get_ball_positions_and_velocities(balls)

    return image_t0, image_t1, image_t2, velocities_vector


def generate_dataset(
    num_systems: int, num_balls: int, density: float, radius: float
) -> np.ndarray:
    """
    Generate a dataset by simulating multiple systems of balls.

    Parameters
    ----------
    num_systems : int
        The number of systems to simulate.
    num_balls : int
        The number of balls in each system.
    density : float
        The density of the balls.
    radius : float
        The radius of the balls.

    Returns
    -------
    dataset : np.ndarray
        The generated dataset.
    """
    # 8 goes to (2 coordinates x (3 positions + 1 velocity vector))
    # and the +1 is the velocity magnitude
    dataset = np.zeros((num_systems, 8 * num_balls + 1))

    for i in tqdm(range(num_systems)):
        # Initialize the simulation
        p.connect(p.DIRECT)
        p.setGravity(0, 0, 0)
        p.setPhysicsEngineParameter(enableConeFriction=0, enableSAT=0)

        # Balls creation
        balls = []
        velocity_magnitude = np.random.uniform(20, 80)

        for _ in range(num_balls):
            ball_id = create_ball(density, radius)
            initialize_ball(ball_id, velocity_magnitude)
            remove_external_factors(ball_id)
            balls.append(ball_id)

        image_t0, image_t1, image_t2, velocities_vector = (
            record_positions_and_velocities(balls)
        )

        # Save image and velocity data on dataset
        dataset[i, : 2 * num_balls] = image_t0.flatten()
        dataset[i, 2 * num_balls : 4 * num_balls] = image_t1.flatten()
        dataset[i, 4 * num_balls : 6 * num_balls] = image_t2.flatten()
        dataset[i, 6 * num_balls : 8 * num_balls] = velocities_vector.flatten()
        dataset[i, -1] = velocity_magnitude

        # Clean up
        p.disconnect()

    return dataset


def calculate_ball_coordinates(
    dataset: np.ndarray, frames: int = 3, pixels_per_axis: int = 32
) -> tuple[np.ndarray, float, float]:
    """
    Calculate the coordinates and velocity vectors of balls in a system.

    Parameters
    ----------
    dataset : np.ndarray
        Input dataset containing ball coordinates.
    frames : int
        Number of frames in the dataset (default is 3).
    pixels_per_axis : int
        Number of pixels per axis in the image (default is 32).

    Returns
    -------
    coordinates : np.ndarray
        Array containing the calculated coordinates and velocity vectors.
    ax_min : float
        Minimum value of the axes to have a square grid that contains all balls
        for any system.
    ax_max : float
        Maximum value of the axes to have a square grid that contains all balls
        for any system.
    """
    num_systems = dataset.shape[0]
    num_balls = (dataset.shape[1] - 2) // 6
    coordinates = np.zeros((num_systems, 8 * num_balls))
    # Limits of axes to have a square grid that contains all balls for any system
    ax_max = np.round(np.max(dataset[:, : 6 * num_balls]) + 0.1, 1)
    ax_min = np.round(np.min(dataset[:, : 6 * num_balls]) - 0.1, 1)

    for system_index in tqdm(range(num_systems)):
        for ball_index in range(num_balls):
            non_white_pixels = {frame: ([], []) for frame in range(frames)}

            for frame in range(frames):
                x = dataset[system_index, 2 * ball_index + 2 * num_balls * frame]
                y = dataset[system_index, 2 * ball_index + 2 * num_balls * frame + 1]
                plot_gray = get_grayscale_ball_plot(
                    plt.gca(), x, y, ax_min, ax_max, pixels_per_axis
                )
                plt.close()

                # Store all pixels in ball region in the three frames
                for m in range(pixels_per_axis):
                    for n in range(pixels_per_axis):
                        if plot_gray[m, n] != 1:
                            non_white_pixels[frame][0].append(m)
                            non_white_pixels[frame][1].append(n)

                # Calculate the center pixel of each ball
                coordinates[
                    system_index,
                    2 * frame * num_balls
                    + 2 * ball_index : 2 * frame * num_balls
                    + 2 * (ball_index + 1),
                ] = np.mean(non_white_pixels[frame][1]), np.mean(
                    non_white_pixels[frame][0]
                )

            # Calculate the velocity vector
            coordinates[
                system_index,
                6 * num_balls + 2 * ball_index : 6 * num_balls + 2 * (ball_index + 1),
            ] = np.mean(non_white_pixels[1][1]) - np.mean(
                non_white_pixels[0][1]
            ), np.mean(
                non_white_pixels[1][0]
            ) - np.mean(
                non_white_pixels[0][0]
            )

    return coordinates, ax_min, ax_max


def generate_images(dataset, ax_min, ax_max, folder_path, use_color):
    """
    Generate input and output images from the dataset.

    Parameters
    ----------
    dataset : np.ndarray
        Input dataset containing ball coordinates.
    ax_min : float
        Minimum value of the axes.
    ax_max : float
        Maximum value of the axes.
    folder_path : str
        Path to the directory to save images.
    use_color : bool
        Flag indicating whether to use color for balls.
    """
    num_systems = dataset.shape[0]
    num_balls = (dataset.shape[1] - 2) // 6

    # Set directories name
    folder_path = os.path.join(folder_path, f"{num_balls}ball{'_color' * use_color}")
    input_path = os.path.join(folder_path, "Input")
    output_path = os.path.join(folder_path, "Output")

    # Create directories if needed
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Only thought for 3 balls case
    colors = ["r", "b", "g"] if use_color else ["k"]

    for i in tqdm(range(num_systems)):
        # Generate input and output images
        for j, frame_index in enumerate([0, 2 * num_balls, 4 * num_balls]):
            balls_coordinates = (
                dataset[i, frame_index : frame_index + 2 * num_balls : 2],
                dataset[i, frame_index + 1 : frame_index + 2 * num_balls : 2],
            )
            plt.figure(figsize=(0.42, 0.42))
            plt.scatter(balls_coordinates[0], balls_coordinates[1], color=colors, s=2)
            plt.xlim(ax_min, ax_max)
            plt.ylim(ax_min, ax_max)
            plt.axis("off")

            image_path = (
                os.path.join(input_path, f"image{j+1}_{i:04d}.png")
                if j < 2
                else os.path.join(output_path, f"image{i:04d}.png")
            )
            plt.savefig(image_path, bbox_inches="tight", pad_inches=0.0)
            plt.close()


# We can also generate the dataset by calling this script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a dataset of ball systems and images."
    )
    parser.add_argument(
        "--num_systems",
        type=int,
        default=10000,
        help="Number of systems to generate (default: 10000)",
    )
    parser.add_argument(
        "--num_balls",
        type=int,
        default=3,
        help="Number of balls per system (default: 3). It also accepts 1.",
    )
    parser.add_argument(
        "--density", type=float, default=1.0, help="Density of the balls (default: 1.0)"
    )
    parser.add_argument(
        "--radius", type=float, default=5.0, help="Radius of each ball (default: 5.0)"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        default="data",
        help="Path to the folder where images and data will be stored (default: data)",
    )
    parser.add_argument(
        "--use_color",
        type=bool,
        default=False,
        help="Whether to use color for the balls (default: False)",
    )

    args = parser.parse_args()

    # Generate dataset
    print("Generating dataset...")
    dataset = generate_dataset(
        num_systems=args.num_systems,
        num_balls=args.num_balls,
        density=args.density,
        radius=args.radius,
    )
    # Calculate coordinates
    print("Calculating coordinates...")
    coordinates, ax_min, ax_max = calculate_ball_coordinates(dataset=dataset)
    # Generate and store images
    print("Generating images...")
    generate_images(
        dataset=dataset,
        ax_min=ax_min,
        ax_max=ax_max,
        folder_path=args.folder_path,
        use_color=args.use_color,
    )
    # Save coordinates data
    folder_name = f"{args.num_balls}ball{'_color' * args.use_color}"
    np.savetxt(
        os.path.join(args.folder_path, folder_name, "coordinates.csv"),
        coordinates,
        delimiter=",",
    )
    print("Images dataset has been generated!")
