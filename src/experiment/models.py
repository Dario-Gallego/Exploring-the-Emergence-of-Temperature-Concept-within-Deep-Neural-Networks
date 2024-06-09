import keras
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input
from keras.models import Model


def build_conv_and_fc_regressor(input_layer: keras.Layer) -> Model:
    """
    Build a Keras model for temperature regression using convolutional and fully
    connected layers.

    Parameters
    ----------
    input_layer : keras.Layer
        Input layer.

    Returns
    -------
    model : Model
        The constructed Keras model for temperature regression.
    """
    # Hidden layers
    x = Conv2D(1, (3, 3), activation="relu")(input_layer)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(16, activation="relu")(x)

    # Output layer
    outputs = Dense(1, activation="sigmoid")(x)

    # Construct and return model
    model = Model(input_layer, outputs, name="ConvAndFC")

    return model


def build_fc_regressor(input_layer: keras.Layer) -> Model:
    """
    Build a Keras model for temperature regression using dense layers.

    Parameters
    ----------
    input_layer : keras.Layer
        Input layer.

    Returns
    -------
    model : Model
        The constructed Keras model for temperature regression.
    """
    # Hidden layers
    x = Dense(128, activation="relu")(input_layer)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.3)(x)

    # Output layer
    outputs = Dense(1, activation="sigmoid")(x)

    # Construct and return model
    model = Model(input_layer, outputs, name="FC")

    return model


def build_img_to_coordinates_model(
    num_balls: int, pixels_per_axis: int, colored_balls: bool
) -> Model:
    """
    Build a Keras model for mapping images to coordinates.

    Parameters
    ----------
    num_balls : int
        The number of balls in each image.
    pixels_per_axis : int
        The number of pixels per axis in the input images.
    colored_balls : bool
        Indicates whether the input images have colored balls.

    Returns
    -------
    model : Model
        The constructed Keras model.
    """
    # Calculate input shape based on whether balls are colored
    input_shape = (pixels_per_axis, pixels_per_axis, 2 + 4 * int(colored_balls))

    # Input layer
    inputs = Input(shape=input_shape)

    # Hidden layers
    x = Conv2D(1, (3, 3), activation="relu")(inputs)
    x = Conv2D(1, (3, 3), activation="relu")(x)
    x = Conv2D(6, (28, 28), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(12 * num_balls)(x)

    # Output layer
    outputs = Dense(4 * num_balls)(x)

    # Construct and return model
    model = Model(inputs, outputs, name="ImgToCoordinates")
    return model
