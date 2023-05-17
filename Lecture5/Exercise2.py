# Hyperparameter scan for classifier 
import tensorflow as tf
import numpy as np
from hyperopt import hp, tpe, Trials, fmin, space_eval, STATUS_OK
import matplotlib.pyplot as plt
import datetime

"""
DATA LOADING 
"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/ 255
x_test = x_test / 255

print("Training dataset size:", x_train.shape[0], "images with dimensions", x_train.shape[1], "x", x_train.shape[2])
print("Test dataset size:", x_test.shape[0], "images with dimensions", x_test.shape[1], "x", x_test.shape[2])
print("Maximum label value:", np.max(y_train))
print("Minimum label value:", np.min(y_train))
print("The images are classified in", np.max(y_train)-np.min(y_train)+1, "categories")


"""
NN Design Model 
"""

def image_classifier():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    
    return model


def train_hyper_param_model(train_images, train_labels, params, epochs):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(params['layer_size'], activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    
    log_dir = "home/francescogrienti/logs/trainHyperParamModelFit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_images, train_labels, epochs=epochs, callbacks=[tensorboard_callback])

    return model

# Hyperfunction
def hyperfunc(params):
    model = train_hyper_param_model(x_train, y_train, params, epochs=5)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    return {'loss': test_loss, 'accuracy': test_acc, 'status': STATUS_OK}

"""
MAIN 
"""

def main():
    # Train model with a starting layer_size and default learning_rate
    model = image_classifier()
    model.fit(x_train, y_train, epochs=10)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Starting model with fixed layer_size and learning rate of the optimizer ----> ")
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    # Hyperparameter optimization using HyperOpt
    search_space = {
        'layer_size': hp.choice('layer_size', np.arange(10, 100, 20)),
        'learning_rate': hp.loguniform('learning_rate', -10, 0)
    }

    trials = Trials()
    best = fmin(hyperfunc, search_space, algo=tpe.suggest, max_evals=5, trials=trials)
    print(space_eval(search_space, best))

    # Plots 

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    xs = [t['tid'] for t in trials.trials]
    ys = [t['result']['accuracy'] for t in trials.trials]
    ax1.set_xlim(xs[0]-1, xs[-1]+1)
    ax1.scatter(xs, ys, s=20)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')

    xs = [t['misc']['vals']['layer_size'] for t in trials.trials]
    ys = [t['result']['accuracy'] for t in trials.trials]

    ax2.scatter(xs, ys, s=20)
    ax2.set_xlabel('Layers')
    ax2.set_ylabel('Accuracy')

    xs = [t['misc']['vals']['learning_rate'] for t in trials.trials]
    ys = [t['result']['accuracy'] for t in trials.trials]

    ax3.scatter(xs, ys, s=20)
    ax3.set_xlabel('learning_rate')
    ax3.set_ylabel('Accuracy')
    plt.show()

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    xs = [t['tid'] for t in trials.trials]
    ys = [t['result']['loss'] for t in trials.trials]
    ax1.set_xlim(xs[0]-1, xs[-1]+1)
    ax1.scatter(xs, ys, s=20)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')

    xs = [t['misc']['vals']['layer_size'] for t in trials.trials]
    ys = [t['result']['loss'] for t in trials.trials]

    ax2.scatter(xs, ys, s=20)
    ax2.set_xlabel('Layers')
    ax2.set_ylabel('Loss')

    xs = [t['misc']['vals']['learning_rate'] for t in trials.trials]
    ys = [t['result']['loss'] for t in trials.trials]

    ax3.scatter(xs, ys, s=20)
    ax3.set_xlabel('learning_rate')
    ax3.set_ylabel('Loss')
    plt.show()

    
if __name__ == "__main__":
    main()


