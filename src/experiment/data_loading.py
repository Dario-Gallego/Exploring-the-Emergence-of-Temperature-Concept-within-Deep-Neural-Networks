import os
from glob import glob

import cv2
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_images(pattern: str, colored_balls: bool = False) -> np.ndarray:
    """
    Load images from the specified pattern.

    Parameters
    ----------
    pattern : str
        Pattern to match the image filenames.
    colored_balls : bool
        If True, load images in color. Default is False (grayscale).

    Returns
    -------
    images : np.ndarray
        Array containing the loaded images.
    """
    if colored_balls:
        imread_flag = cv2.IMREAD_COLOR
    else:
        imread_flag = cv2.IMREAD_GRAYSCALE

    images = [cv2.imread(file, imread_flag) for file in tqdm(glob(pattern))]
    images = (
        np.array([keras.utils.img_to_array(img) for img in images], dtype="float32")
        / 255
    )

    return images


def load_image_set(
    data_folder: str, colored_balls: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load sets of input and output images.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the image sets.
    colored_balls : bool
        If True, load images in color. Default is False (grayscale).

    Returns
    -------
    x1 : np.ndarray
        Array containing input images from the first frame.
    x2 : np.ndarray
        Array containing input images from the second frame.
    y : np.ndarray
        Array containing output images from the third and final frame.
    """
    x1 = load_images(os.path.join(data_folder, "Input/image1_*.png"), colored_balls)
    x2 = load_images(os.path.join(data_folder, "Input/image2_*.png"), colored_balls)
    y = load_images(os.path.join(data_folder, "Output/*.png"), colored_balls)

    return x1, x2, y


def compute_temperature(data_folder: str, num_balls: int) -> np.ndarray:
    """
    Compute the temperature for each system based on the dataset coordinates.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the coordinates.
    num_balls : int
        Number of balls in each system.

    Returns
    -------
    temperature : np.ndarray
        Temperature of each system.
    """
    # Load the coordinates from the CSV file
    coordinates = np.loadtxt(
        os.path.join(data_folder, "coordinates.csv"), delimiter=","
    )
    # Set velocity vector coordinates between -1 and 1 (they are between -4 and 4)
    coordinates = coordinates[:, -2 * num_balls :] / 4
    num_systems = coordinates.shape[0]
    temperature = np.zeros(shape=(num_systems))

    # Compute temperature for each system
    for i in range(num_systems):
        for j in range(num_balls):
            temperature[i] += np.sqrt(
                np.sum(np.square(coordinates[i, 2 * j : 2 * (j + 1)]))
            )
        temperature[i] /= num_balls

    return temperature


def load_and_process_coordinates(
    data_folder: str, num_balls: int, pixels_per_axis: int
) -> np.ndarray:
    """
    Load and process the coordinates from a CSV file.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the coordinates.
    num_balls : int
        Number of balls in each system.
    pixels_per_axis : int
        The number of pixels per axis on images.

    Returns
    -------
    processed_coordinates : np.ndarray
        Processed coordinates with balls position in second frame and velocity vectors.
    """
    coordinates = np.loadtxt(
        os.path.join(data_folder, "coordinates.csv"), delimiter=","
    )
    processed_coordinates = np.concatenate(
        (
            # Normalized balls position in second frame
            coordinates[:, 2 * num_balls : 4 * num_balls] / (pixels_per_axis - 1),
            # Set velocity vector coordinates between -1 and 1
            # (they are between -4 and 4)
            coordinates[:, -2 * num_balls :] / 4,
        ),
        axis=1,
    )
    
    return processed_coordinates


def compute_optical_flow_and_temperature_from_images(
    x1: np.ndarray, x2: np.ndarray, central_pixel_only: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute optical flow and temperature for a given dataset by using the images.

    Parameters
    ----------
    x1 : np.ndarray
        Array with first frames of the dataset with shape
        (num_systems, pixels_per_axis, pixels_per_axis, 1).
    x2 : np.ndarray
        Array with second frames of the dataset with shape
        (num_systems, pixels_per_axis, pixels_per_axis, 1).
    central_pixel_only : bool
        If True, assign the optical flow to the central pixel of the ball region only,
        otherwise to all ball pixels. Default is False.

    Returns
    -------
    optical_flow : np.ndarray
        Computed optical flow with shape
        (num_systems, pixels_per_axis, pixels_per_axis, 2).
    temperature : np.ndarray
        Computed temperature with shape (num_systems,).
    """
    num_systems, pixels_per_axis, _, _ = x1.shape

    # Initialization of the needed data
    optical_flow = np.zeros((num_systems, pixels_per_axis, pixels_per_axis, 2))
    temperature = np.zeros(num_systems)

    # For each system we compute the optical flow and temperature
    for i in tqdm(range(num_systems)):
        # Flatten the last dimension since it's always 1
        frame_t0 = x1[i, :, :, 0]
        frame_t1 = x2[i, :, :, 0]

        # Check if there's ball in that pixel (first frame)
        mask_t0 = frame_t0 != 1
        # Check if there's ball in that pixel (second frame)
        mask_t1 = frame_t1 != 1

        # Get the coordinates of the ball region pixels
        non_white_y_t0, non_white_x_t0 = np.where(mask_t0)
        non_white_y_t1, non_white_x_t1 = np.where(mask_t1)

        # Compute the vectors and divide by 4 to have (or close to have)
        # them between -1 and 1
        velocity_vector = (
            (np.mean(non_white_x_t1) - np.mean(non_white_x_t0)) / 4,
            (np.mean(non_white_y_t1) - np.mean(non_white_y_t0)) / 4,
        )

        # Compute temperature value
        temperature[i] = np.sqrt(velocity_vector[0] ** 2 + velocity_vector[1] ** 2)

        if central_pixel_only:
            # Assign optical flow to balls center pixels
            central_x = int(np.mean(non_white_x_t0))
            central_y = int(np.mean(non_white_y_t0))
            optical_flow[i, central_y, central_x] = velocity_vector
        else:
            # Assign optical flow to ball pixels
            optical_flow[i][mask_t0] = velocity_vector

    return optical_flow, temperature


def compute_optical_flow_and_temperature_from_coordinates(
    data_folder: str, num_balls: int, pixels_per_axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute optical flow and temperature for a given dataset by using the coordinates.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the coordinates.
    num_balls : int
        Number of balls in each system.
    pixels_per_axis : int
        The number of pixels per axis on images.

    Returns
    -------
    optical_flow : np.ndarray
        Computed optical flow with shape
        (num_systems, pixels_per_axis, pixels_per_axis, 2).
    temperature : np.ndarray
        Computed temperature with shape (num_systems,).
    """
    # Load coordinates from the CSV file
    coordinates = np.loadtxt(
        os.path.join(data_folder, "coordinates.csv"), delimiter=","
    )

    # Determine the number of systems from the coordinates array
    num_systems = coordinates.shape[0]

    # Adjust velocity vectors to be between -1 and 1
    coordinates = np.concatenate(
        (coordinates[:, : 2 * num_balls], coordinates[:, -2 * num_balls :] / 4), axis=1
    )

    # Initialization of the needed data
    optical_flow = np.zeros((num_systems, pixels_per_axis, pixels_per_axis, 2))
    temperature = np.zeros(num_systems)

    # For each system we compute the optical flow and temperature
    for i in range(num_systems):
        for j in range(num_balls):
            y_coord = int(coordinates[i, 2 * j])
            x_coord = int(coordinates[i, 2 * j + 1])
            vel_x = coordinates[i, 2 * num_balls + 2 * j]
            vel_y = coordinates[i, 2 * num_balls + 2 * j + 1]

            # Store optical flow information
            optical_flow[i, y_coord, x_coord] = (vel_x, vel_y)

            # Compute temperature value
            temperature[i] += np.sqrt(vel_x**2 + vel_y**2)

    return optical_flow, temperature


def prepare_train_val_test_splits(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray, temperature: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Prepare train, validation, and test splits for input images, output images, and
    temperature data.

    Parameters
    ----------
    x1 : np.ndarray
        Array containing input images from the first frame.
    x2 : np.ndarray
        Array containing input images from the second frame.
    y : np.ndarray
        Array containing output images from the third and final frame.
    temperature : np.ndarray
        Temperature value for each system.

    Returns
    -------
    x_train : np.ndarray
        Train split for input images.
    x_val : np.ndarray
        Validation split for input images.
    x_test : np.ndarray
        Test split for input images.
    y_train : np.ndarray
        Train split for output images.
    y_val : np.ndarray
        Validation split for output images.
    y_test : np.ndarray
        Test split for output images.
    temperature_train : np.ndarray
        Train split for temperature value.
    temperature_val : np.ndarray
        Validation split for temperature value.
    temperature_test : np.ndarray
        Test split for temperature value.
    """
    # Concatenate both frames along the channel axis
    x = np.concatenate((x1, x2), axis=3)

    # Train-val-test split (80-10-10)
    x_train, x_test, y_train, y_test, temperature_train, temperature_test = (
        train_test_split(x, y, temperature, test_size=0.2)
    )
    x_val, x_test, y_val, y_test, temperature_val, temperature_test = train_test_split(
        x_test, y_test, temperature_test, test_size=0.5
    )

    return (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        temperature_train,
        temperature_val,
        temperature_test,
    )
