import numpy as np
from keras.models import Model
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def get_encoded_images(model: Model, data: np.ndarray, layer_name: str) -> np.ndarray:
    """
    Get the encoded representation of images from a specific layer of the model.

    Parameters
    ----------
    model : Model
        The trained Keras model.
    data : np.ndarray
        The data to encode.
    layer_name : str
        The name of the layer from which to extract the encoded representation.

    Returns
    -------
    encoded_data : np.ndarray
        Encoded representations of input data.
    """
    encoder_model = Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output
    )
    encoded_data = encoder_model.predict(data)

    return encoded_data


def flatten_encoded_images(encoded_data: np.ndarray) -> np.ndarray:
    """
    Flatten the encoded representation of images.

    Parameters
    ----------
    encoded_data : np.ndarray
        Encoded representations.

    Returns
    -------
    encoded_data_flat : np.ndarray
        Flattened encoded representations.
    """
    encoded_data_flat = encoded_data.reshape((encoded_data.shape[0], -1))

    return encoded_data_flat


def train_and_evaluate_linear_temperature_regressor(
    encoded_train: np.ndarray,
    encoded_val: np.ndarray,
    temperature_train: np.ndarray,
    temperature_val: np.ndarray,
) -> RegressorMixin:
    """
    Train the temperature linear regressor model and evaluate it on the validation set.

    Parameters
    ----------
    encoded_train : np.ndarray
        Encoded representations of training samples.
    encoded_val : np.ndarray
        Encoded representations of validation samples.
    temperature_train : np.ndarray
        Training temperature data.
    temperature_val : np.ndarray
        Validation temperature data.

    Returns
    -------
    temperature_regressor : RegressorMixin
        The trained temperature regression model.
    """
    temperature_regressor = LinearRegression()
    temperature_regressor.fit(encoded_train, temperature_train)
    val_predictions = temperature_regressor.predict(encoded_val)
    print(f"MSE: {mean_squared_error(temperature_val, val_predictions)}")

    return temperature_regressor


def regression_evaluation(
    regressor_model: RegressorMixin | Model, x_test: np.ndarray, y_test: np.ndarray
) -> None:
    """
    Evaluate the regressor model and print evaluation results.

    Parameters
    ----------
    regressor_model : RegressorMixin | Model
        The regressor model, either sklearn or Keras.
    x_test : np.ndarray
        Input data from test samples.
    y_test : np.ndarray
        Test target data.
    """
    test_predictions = regressor_model.predict(x_test)

    for i in range(5):
        print(f"Sample {i}")
        print(f"Prediction: {test_predictions[i]}")
        print(f"True values: {y_test[i]}")
