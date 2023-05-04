import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
import argparse

"""
Write a ML classification model using keras with the following steps. 
"""

"""
Data loading.
Load the fashion mnist dataset from tensorflow.keras.datasets.fashion_mnist. Study the dataset size (pixel shape) and 
plot some sample images. This dataset contains the following classes 
['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'].
"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize training and test images
x_train = x_train[:]/255.
x_test = x_test[:]/255.

"""
NN MODEL 
"""

def NNmodelClassif():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    return model

"""
MAIN
"""

def main(epochs):
    model = NNmodelClassif()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    # model.summary()

    # Model fit 
    model.fit(x_train, y_train, epochs=epochs)

    print(model.get_metrics_result())
    print(model.evaluate(x_test, y_test))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line options.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    args = parser.parse_args()
    main(args.epochs)



