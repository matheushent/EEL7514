import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = Sequential(
    [
        Flatten(input_shape=(28, 28)),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.005), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test accuracy in the Fashion MNIST dataset:", test_acc)

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test accuracy in the MNIST dataset:", test_acc)