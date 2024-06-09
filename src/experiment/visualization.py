import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model


def plot_frames(
    model: Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pixels_per_axis: int,
    colored_balls: bool = False,
    num_images: int = 10,
) -> None:
    """
    Plot the three frames, the model predictions and an overlay of the initial and final
    frame, and another with the true and the predicted image.

    Parameters
    ----------
    model : Model
        The trained Keras model used for making predictions.
    x_test : np.ndarray
        The test dataset containing input images.
    y_test : np.ndarray
        The test dataset containing true output images.
    pixels_per_axis : int
        The number of pixels per axis for reshaping images.
    colored_balls : bool
        Indicates whether the balls are colored. Default is False.
    num_images : int
        The number of images to plot. Default is 10.
    """
    last_layer_shape = 3 if colored_balls else 1

    # Generate predictions using the model
    predictions = model.predict(x_test)

    # Loop through the specified number of images
    for i in range(num_images):
        # Create a new figure with specified size
        plt.figure(figsize=(pixels_per_axis / 2, pixels_per_axis * 3))

        # List of images to be plotted
        images = [
            # First frame
            x_test[i, :, :, :last_layer_shape].reshape(
                pixels_per_axis, pixels_per_axis, last_layer_shape
            ),
            # Second frame
            x_test[i, :, :, last_layer_shape:].reshape(
                pixels_per_axis, pixels_per_axis, last_layer_shape
            ),
            # Third frame
            y_test[i].reshape(pixels_per_axis, pixels_per_axis, last_layer_shape),
            # Overlay of final and first frames
            y_test[i].reshape(pixels_per_axis, pixels_per_axis, last_layer_shape)
            + x_test[i, :, :, :last_layer_shape].reshape(
                pixels_per_axis, pixels_per_axis, last_layer_shape
            ),
            # Predicted image
            predictions[i].reshape(pixels_per_axis, pixels_per_axis, last_layer_shape),
            # Overlay of true and predicted image
            y_test[i].reshape(pixels_per_axis, pixels_per_axis, last_layer_shape)
            + predictions[i].reshape(pixels_per_axis, pixels_per_axis, last_layer_shape)
            / 2,
        ]

        # Plot each image in a subplot
        for j, img in enumerate(images):
            plt.subplot(1, 6, j + 1)
            if colored_balls:
                plt.imshow(img)
            else:
                plt.imshow(img, cmap="gray")

        # Show the figure
        plt.show()


def visualize_layer_outputs(
    model: Model, data: np.ndarray, pixels_per_axis: int
) -> None:
    """
    Visualize the output of each layer of the model.

    Parameters
    ----------
    model : Model
        Trained Keras model.
    data : np.ndarray
        Input data for model.
    pixels_per_axis : int
        Number of pixels per axis.
    """
    layers = model.layers
    outputs = [layer.output for layer in layers]
    layer_outputs_model = Model(inputs=model.input, outputs=outputs)
    layer_outputs = layer_outputs_model.predict(
        data[0].reshape(1, pixels_per_axis, pixels_per_axis, 2)
    )

    for i, output in enumerate(layer_outputs):
        if layers[i].name not in ["dense", "reshape", "flatten", "dense_1"]:
            print(layers[i].name)
            for j in range(output.shape[3]):
                plt.imshow(
                    output[:, :, :, j].reshape(output.shape[1], output.shape[2]),
                    cmap="gray",
                )
                plt.show()
                plt.close()


def plot_frames_and_optical_flow(
    x1: np.ndarray, x2: np.ndarray, optical_flow: np.ndarray, num_samples: int = 5
) -> None:
    """
    Plot frames and optical flow for the given dataset.

    Parameters
    ----------
    x1 : np.ndarray
        Array with first frames of the dataset with shape (num_systems, pix, pix, 1).
    x2 : np.ndarray
        Array with second frames of the dataset with shape (num_systems, pix, pix, 1).
    optical_flow : np.ndarray
        Computed optical flow with shape (num_systems, pix, pix, 2).
    num_samples : int
        Number of samples to plot, by default 5.
    """
    for i in range(num_samples):
        plt.figure(figsize=(20, 5))

        plt.subplot(141)
        plt.title("Frame 1")
        plt.imshow(x1[i, :, :, 0], cmap="gray")

        plt.subplot(142)
        plt.title("Frame 2")
        plt.imshow(x2[i, :, :, 0], cmap="gray")

        plt.subplot(143)
        plt.title("Both frames")
        plt.imshow(x1[i, :, :, 0], cmap="gray")
        plt.imshow(x2[i, :, :, 0], alpha=0.5)

        plt.subplot(144)
        plt.ylim(32, 0)
        plt.title("Optical Flow")
        plt.quiver(
            optical_flow[i, ::1, ::1, 0],
            optical_flow[i, ::1, ::1, 1],
            angles="xy",  # use 'xy' angles for consistent arrow orientation
            scale_units="xy",  # use 'xy' scale units for consistent arrow length
            scale=0.2,
        )

        plt.show()

    plt.close()
