import tensorflow as tf
from tensorflow import keras 
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

def preprocess(x_train,y_train,details):
    x_train = x_train.astype('float32') / 255.0
    y_train = to_categorical(y_train, details["n_classes"])
    # using the train test split function
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=127, test_size=0.1, shuffle=True)
    return (x_train,y_train),(x_val,y_val)

def load_mnist_dataset():
    dataset_details = {"n_classes":10,"shape":(28, 28, 1),"vrange":(0.0, 1.0)}
    (x_train,y_train),(_,_)= keras.datasets.mnist.load_data()
    return preprocess(x_train,y_train,dataset_details)

def load_fmnist_dataset():
    dataset_details = {"n_classes":10,"shape":(28, 28, 1),"vrange":(0.0, 1.0)}
    (x_train,y_train),(_,_) = keras.datasets.fashion_mnist.load_data()
    return preprocess(x_train,y_train,dataset_details)

def load_cifar10_dataset():
    dataset_details = {"n_classes":10,"shape":(32, 32, 3),"vrange":(0.0, 1.0)}
    (x_train,y_train),(_,_) = keras.datasets.cifar10.load_data()
    return preprocess(x_train,y_train,dataset_details)


#Load Test set from dataset according to name of apt size
def load_dataset(name):
    if name == 'mnist':
        return load_mnist_dataset()
    elif name == 'fmnist':
        return load_fmnist_dataset()
    elif name =='cifar10':
        return load_cifar10_dataset()
    else:
        raise ValueError("Dataset Not Supported")
    

def train_model(model,dataset):
    (x_train,y_train),(x_val,y_val) = load_dataset(dataset)
    #Convert all to tensors
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    import logging
    # Suppress all informational and warning messages
    tf.get_logger().setLevel(logging.ERROR) 
    #callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('tmp/temp.keras', save_best_only=True, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=32,
        epochs=25,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )
    return model,history
