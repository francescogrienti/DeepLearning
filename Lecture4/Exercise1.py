import tensorflow as tf, numpy as np 
import matplotlib.pyplot as plt

"""

DATA LOADING 

"""

# Write a ML regression model using TensorFlow/Keras's sequential model with the following steps. 
# Data loading 
dataset = np.loadtxt("data.dat")
x_training = dataset[:, 0]
y_training = dataset[:, 1]
x_validation = dataset[:, 2]
y_validation = dataset[:, 3]

"""

FUNCTIONS

"""

def plot_data(x, y, label):

    plt.figure()
    plt.scatter(x, y, label=label)
    plt.xlabel("X-set")
    plt.ylabel("Y-set")
    plt.legend()
    plt.show()


def plot_history(epochs, t_loss, v_loss):
    plt.figure()
    plt.scatter(epochs, t_loss, label="Training loss")
    plt.scatter(epochs, v_loss, label="Validation loss")
    plt.title("Training and Validation losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss(MSE)")
    plt.legend()
    plt.show()


# Baseline linear fit
# Create a baseline linear model (dense layer with 1 unit node) and store 
# the instance of this class in a variable called model.
def BaselineLinearModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1, activation='linear', input_dim=(1)))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss=tf.keras.losses.mean_squared_error)
    return model


def NN_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10, activation='relu', input_dim=(1)))
    model.add(tf.keras.layers.Dense(10, activation='relu', input_dim=(1)))
    model.add(tf.keras.layers.Dense(10, activation='relu', input_dim=(1)))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss=tf.keras.losses.mean_squared_error)
    return model


"""

MAIN

"""

def main():

    # Plot training and validation data
    plot_data(x_training, y_training, "Training set")
    plot_data(x_validation, y_validation, "Validation set")

    """
    BASELINE LINEAR MODEL
    """

    with tf.device('CPU:0'):
        # Instantiate model
        model = BaselineLinearModel()
        
        # Perform a fit with model.fit with full batch size and 500 epochs. Monitor the validation data during epochs.
        history = model.fit(x=tf.convert_to_tensor(x_training), y=tf.convert_to_tensor(y_training), batch_size=len(dataset), epochs=500, 
                validation_data=[tf.convert_to_tensor(x_validation), tf.convert_to_tensor(y_validation)])

        # Plot the loss function for training and validation using the history object returned by model.fit
        plot_history(history.epoch, np.array(history.history["loss"]), np.array(history.history["val_loss"]))

        # Plot the model prediction on top of data.
        plt.figure()
        plt.scatter(x_validation, y_validation, label="Validation data")
        plt.scatter(x_validation, model.predict(x=x_validation, batch_size=len(x_validation)), label="Model prediction")
        plt.xlabel("X-set")
        plt.ylabel("Y-set")
        plt.title("Model prediction")
        plt.legend()
        plt.show()

    """
    NEURAL NETWORK MODEL
    """

    with tf.device('CPU:1'):
        # Instantiate NNmodel
        model = NN_model()

        # Perform a fit with model.fit with full batch size and 500 epochs. Monitor the validation data during epochs.
        history = model.fit(x=tf.convert_to_tensor(x_training), y=tf.convert_to_tensor(y_training), batch_size=len(dataset), epochs=500, 
                validation_data=[tf.convert_to_tensor(x_validation), tf.convert_to_tensor(y_validation)])

        # Plot the loss function for training and validation using the history object returned by model.fit
        plot_history(history.epoch, np.array(history.history["loss"]), np.array(history.history["val_loss"]))

        # Plot the model prediction on top of data.
        plt.figure()
        plt.scatter(x_validation, y_validation, label="Validation data")
        plt.scatter(x_validation, model.predict(x=x_validation, batch_size=len(x_validation)), label="Model prediction")
        plt.xlabel("X-set")
        plt.ylabel("Y-set")
        plt.title("Model prediction")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()


