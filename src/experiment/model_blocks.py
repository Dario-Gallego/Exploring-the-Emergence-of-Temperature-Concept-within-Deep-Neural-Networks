import keras
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    MaxPooling2D,
    UpSampling2D,
    ZeroPadding2D,
)


def double_conv_block(
    input_layer: keras.Layer, n_filters: int, drop_pct: float = 0.0
) -> keras.Layer:
    """
    Double convolutional block consisting of two Conv2D layers with ReLU activation.

    Parameters
    ----------
    input_layer : np.ndarray
        Input layer.
    n_filters : int
        Number of filters for the Conv2D layers.
    drop_pct : float
        Dropout rate. No dropout is applied by default.

    Returns
    -------
    x : keras.Layer
        Output layer after passing through the double convolutional block.
    """
    x = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(input_layer)
    if drop_pct > 0:
        x = Dropout(drop_pct)(x)
    x = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(x)

    return x


def downsample_block(
    input_layer: keras.Layer,
    n_filters: int,
    drop_pct: float = 0.0,
    max_pooling: bool = False,
) -> tuple[keras.Layer, keras.Layer]:
    """
    Downsample block for reducing spatial dimensions.

    Parameters
    ----------
    input_layer : keras.Layer
        Input layer.
    n_filters : int
        Number of filters for the Conv2D layers.
    drop_pct : float
        Dropout rate. No dropout is applied by default.
    max_pooling : bool
        Indicates whether a max pooling layer is used. If not, it will add a (1,1)
        padding and then a Conv2D layer. Default is False.

    Returns
    -------
    f : keras.Layer
        Output layer before downsampling.
    p : keras.Layer
        Output layer after downsampling.
    """
    f = double_conv_block(input_layer, n_filters, drop_pct)
    if max_pooling:
        p = MaxPooling2D((2, 2))(f)
    else:
        p = ZeroPadding2D(padding=(1, 1))(f)
        p = Conv2D(n_filters, (3, 3), activation="relu", strides=(2, 2))(p)

    return f, p


def upsample_block_without_skip_connection(
    input_layer: keras.Layer, n_filters: int, drop_pct: float = 0.0
) -> keras.Layer:
    """
    Upsample block (without skip connection) for increasing spatial dimensions.

    Parameters
    ----------
    input_layer : keras.Layer
        Input layer.
    n_filters : int
        Number of filters for the Conv2D layers.
    drop_pct : float
        Dropout rate. No dropout is applied by default.

    Returns
    -------
    x : keras.Layer
        Output layer after passing through the upsample block.
    """
    x = UpSampling2D(size=(2, 2), interpolation="nearest")(input_layer)
    x = double_conv_block(x, n_filters, drop_pct)

    return x


def upsample_block_with_skip_connection(
    input_layer: keras.Layer,
    conv_features: keras.Layer,
    n_filters: int,
    drop_pct: float = 0.0,
    batch_norm: bool = False,
) -> keras.Layer:
    """
    Upsample block (with skip connection) for increasing spatial dimensions.

    Parameters
    ----------
    input_layer : keras.Layer
        Input layer.
    conv_features : keras.Layer
        Features layer to be concatenated with the upsampled layer.
    n_filters : int
        Number of filters for the Conv2D layers.
    drop_pct : float
        Dropout rate. No dropout is applied by default.
    batch_norm : bool
        Indicates if batch normalization is applied at the end of the block.
        Default is False.

    Returns
    -------
    x : keras.Layer
        Output layer after passing through the upsample block.
    """
    x = Conv2DTranspose(n_filters, (2, 2), (2, 2), padding="same")(input_layer)
    x = keras.layers.concatenate([x, conv_features])
    x = double_conv_block(x, n_filters, drop_pct)
    if batch_norm:
        x = BatchNormalization()(x)

    return x


def encoder_block(input_layer: keras.Layer, n_filters: int) -> keras.Layer:
    """
    Create an encoder block consisting of a Conv2D layer, MaxPooling2D layer, and
    BatchNormalization layer.

    Parameters
    ----------
    input_layer : keras.Layer
        Input layer for the encoder block.
    n_filters : int
        Number of filters for the Conv2D layer.

    Returns
    -------
    x : keras.Layer
        Output layer after applying Conv2D, MaxPooling2D, and BatchNormalization.
    """
    x = Conv2D(n_filters, (3, 3), activation="relu", padding="same")(input_layer)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = BatchNormalization()(x)

    return x


def decoder_block(input_layer: keras.Layer, n_filters: int) -> keras.Layer:
    """
    Create a decoder block consisting of a Conv2DTranspose layer, an UpSampling2D layer
    and a BatchNormalization layer.

    Parameters
    ----------
    input_layer : keras.Layer
        Input layer for the decoder block.
    n_filters : int
        Number of filters for the Conv2DTranspose layer.

    Returns
    -------
    x : keras.Layer
        Output layer after applying Conv2DTranspose, UpSampling2D, and
        BatchNormalization.
    """
    x = Conv2DTranspose(n_filters, (3, 3), activation="relu", padding="same")(
        input_layer
    )
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)

    return x
