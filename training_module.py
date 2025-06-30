from tensorflow import keras 
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def load_mnist_dataset():
    ...

def load_fmnist_dataset():
    ...

def load_cifar10_dataset():
    ...

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
    (x_train,y_train),(x_val,y_val),(x_test,y_test) = load_dataset(dataset)