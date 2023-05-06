import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
import argparse


"""
Write a ML classification model using keras with the following steps. 
"""

"""
DATA LOADING.
Load the fashion mnist dataset from tensorflow.keras.datasets.fashion_mnist. Study the dataset size (pixel shape) and 
plot some sample images. This dataset contains the following classes 
['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'].
"""

class_names = ['T-shirt/top', 'Trouser', 'Pullover',
               'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
               'Bag', 'Ankle boot'] # categories 10 objects

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize training and test images
x_train = x_train[:]/255.
x_test = x_test[:]/255.



"""
FUNCTIONS
"""

def plot_sample(images, labels):
    """Plot utils."""
    plt.figure(figsize=(10,10))
    for i in range(36):
        plt.subplot(6, 6, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()


def plot_predictions(predictions, images, labels):
    rows = 5
    cols = 3
    plt.figure(figsize=(4*cols, 2*rows))
    for i in range(rows * cols):
        plt.subplot(rows, 2*cols, 2 * i + 1)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.yticks([])
        plt.xticks([])
        predicted_label = np.argmax(predictions[i])
        if predicted_label == labels[i]:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel(f"{class_names[predicted_label]} {100*np.max(predictions[i]):2.0f}% ({class_names[labels[i]]})", color=color)

        plt.subplot(rows, 2*cols, 2 * i + 2)
        tp = plt.bar(range(10), predictions[i], color='grey')
        tp[predicted_label].set_color('red')
        tp[labels[i]].set_color('blue')
        plt.yticks([])
        plt.xticks([])
        plt.ylim([0,1])
    plt.show()



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

    plot_sample(x_train, y_train)
    model = NNmodelClassif()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'],)
    
    # model.summary()
    # Model fit 
    model.fit(x_train, y_train, epochs=epochs)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test loss:' , test_loss)
    print('Test accuracy:' , test_acc)

    predictions = model.predict(x_test)
    plot_predictions(predictions, x_test, y_test)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command line options.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    args = parser.parse_args()
    main(args.epochs)



