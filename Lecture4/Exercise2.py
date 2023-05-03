import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt

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
dim = x_train[0,0].shape[0]
X_train = []

for i in range(x_train.shape[0]):
    X_train.append(x_train[i].flatten())


"""
NN MODEL 
"""

def NNmodelClassif():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=dim*dim))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    return model

"""
MAIN
"""

def main():
    model = NNmodelClassif()
    # model.summary()

    # Model fit 
    model.fit(x=X_train, y=y_train, batch_size=len(y_train), epochs=5)





if __name__ == "__main__":
    main()



