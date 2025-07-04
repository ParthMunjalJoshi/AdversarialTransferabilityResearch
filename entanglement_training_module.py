import tensorflow as tf
from tensorflow import keras 
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import json

with open('lib/expt_config.json', 'r') as f:
    data = json.load(f)
val_frac = data["training_parameters"]["val_set_size"]
erly_stp_patience = data["training_parameters"]["early_stop_patience"]
lr_factor = data["training_parameters"]["reduce_lr_factor"]
lr_patience = data["training_parameters"]["reduce_lr_patience"]
batch_sz = data["training_parameters"]["batch_size"]
epchs = data["training_parameters"]["epochs"]

def preprocess(x_train, y_train, details):
    """Normalizes and one-hot encodes the training data, then splits it into training and validation sets.

    Args:
        x_train (np.ndarray): Raw input images.
        y_train (np.ndarray): Corresponding labels.
        details (dict): Dictionary containing dataset metadata like number of classes.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
            A tuple of (x_train, y_train) and (x_val, y_val), both properly processed.
    """
    x_train = x_train.astype('float32') / 255.0
    y_train = to_categorical(y_train, details["n_classes"])
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, random_state=127, test_size=val_frac, shuffle=True
    )
    return (x_train, y_train), (x_val, y_val)


def load_mnist_dataset():
    """Loads and preprocesses the MNIST dataset.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
            Processed (x_train, y_train) and (x_val, y_val) sets.
    """
    dataset_details = {"n_classes": 10, "shape": (28, 28, 1), "vrange": (0.0, 1.0)}
    (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    return preprocess(x_train, y_train, dataset_details)


def load_fmnist_dataset():
    """Loads and preprocesses the Fashion MNIST dataset.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
            Processed (x_train, y_train) and (x_val, y_val) sets.
    """
    dataset_details = {"n_classes": 10, "shape": (28, 28, 1), "vrange": (0.0, 1.0)}
    (x_train, y_train), (_, _) = keras.datasets.fashion_mnist.load_data()
    return preprocess(x_train, y_train, dataset_details)


def load_cifar10_dataset():
    """Loads and preprocesses the CIFAR-10 dataset.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
            Processed (x_train, y_train) and (x_val, y_val) sets.
    """
    dataset_details = {"n_classes": 10, "shape": (32, 32, 3), "vrange": (0.0, 1.0)}
    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
    return preprocess(x_train, y_train, dataset_details)


def load_dataset(name):
    """Loads the specified dataset by name.

    Args:
        name (str): Name of the dataset. One of ['mnist', 'fmnist', 'cifar10'].

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
            Processed training and validation data.

    Raises:
        ValueError: If the specified dataset name is not supported.
    """
    if name == 'mnist':
        return load_mnist_dataset()
    elif name == 'fmnist':
        return load_fmnist_dataset()
    elif name == 'cifar10':
        return load_cifar10_dataset()
    else:
        raise ValueError("Dataset Not Supported")


def train_model(model, dataset):
    """Compiles and trains a Keras model on the specified dataset with early stopping and learning rate reduction.

    Args:
        model (tf.keras.Model): The Keras model to train.
        dataset (str): The name of the dataset to use (e.g., 'mnist', 'fmnist', or 'cifar10').

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: The trained model and the training history object.
    """
    (x_train, y_train), (x_val, y_val) = load_dataset(dataset)
    
    # Convert to tensors
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    import logging
    tf.get_logger().setLevel(logging.ERROR)

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=erly_stp_patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('tmp/temp.keras', save_best_only=True, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, min_lr=1e-5)
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_sz,
        epochs=epchs,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )

    return model, history