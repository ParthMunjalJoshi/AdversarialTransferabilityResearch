import entanglement_model_factory as em
import evaluation_pipeline as ep
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

classical_model = em.entanglement_model_factory(5,0,1,["none"])
print(classical_model.summary())
hqcnn = em.entanglement_model_factory(5,4,3,["circularx"]*3)
print(hqcnn.summary())

# Load MNIST
(x_train, y_train_pure), (x_test, y_test_pure) = mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape to 28x28x1
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# One-hot encode labels
y_train = to_categorical(y_train_pure, 10)
y_test = to_categorical(y_test_pure, 10)

# Convert to TensorFlow Tensors
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

classical_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
hqcnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

import tensorflow as tf
import logging
# Suppress all informational and warning messages
tf.get_logger().setLevel(logging.ERROR) 

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('emnist_hyperparam.keras', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
print("Starting model training...")
history1 = classical_model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=100,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)
history2 = hqcnn.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=100,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)
print("\nModels trained successfully!")
print(ep.eval_pipeline("mnist",classical_model,hqcnn,0.1,1,1))

