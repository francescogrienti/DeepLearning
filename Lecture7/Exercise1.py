# A simple CNN classifier
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# TODO review parameters nodes of the NN 

"""
DATA LOADING
"""

(train_set, train_label),(test_set, test_label) = tf.keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

img_height = len(train_set[0][0])
img_width = len(train_set[0][0])


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


def plot_metrics(history, metric:str, title:str):
    x = range(1, len(history.history[metric])+1)
    yt = history.history[metric]
    yv = history.history['val_'+ metric]
    plt.plot(x, yt, label='Training ' + metric)
    plt.plot(x, yv, label='Validation ' + metric)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.ylim([0,1])
    plt.title(title)
    plt.legend()
    plt.show()


def image_classifier():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Rescaling(1/255., input_shape=(img_width, img_height, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(len(class_names), activation='softmax'))
    
    return model 


def image_cnn_classifier():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Rescaling(1/255., input_shape=(img_width, img_height, 3)))
    model.add(tf.keras.layers.Conv2D(16, 3, padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(32, 3, padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Conv2D(64, 3, padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(len(class_names), activation='softmax'))

    return model

"""
MAIN 
"""

def main():

    # Plot a sample of images
    plot_sample(train_set, train_label.flatten())
    # Simple model 
    model1 = image_classifier()
    model1.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history1 = model1.fit(train_set, train_label, validation_data=[test_set, test_label], epochs=10)
    test_loss, test_accuracy = model1.evaluate(test_set, test_label)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    # Loss and accuracy plots
    # plot_metrics(history1, 'loss', 'Simple classifier')
    plot_metrics(history1, 'accuracy', 'Simple classifier')

    # Convolutional model 
    model = image_cnn_classifier()
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_set, train_label, validation_data=[test_set, test_label], epochs=10)
    test_loss, test_accuracy = model.evaluate(test_set, test_label)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    # Loss and accuracy plots
    # plot_metrics(history, 'loss', 'CNN classifier')
    plot_metrics(history, 'accuracy', 'CNN classifier')


if __name__ == "__main__":
    main()
