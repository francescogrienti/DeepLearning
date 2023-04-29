import tensorflow as tf, numpy as np 
import matplotlib.pyplot as plt

# Global values, dimension of the dataset taken into account
# and seed for the data generation
N = 200
#tf.random.set_seed(0)
np.random.seed(0)
w = tf.Variable(5.0)
b = tf.Variable(1.0)
epochs = range(10)
    

def truth_model(x):
    return 3 * x + 2

# Define a loss function matching the mean squared error.
def loss_func(y_1, y_2):
    l = tf.reduce_mean(tf.square(y_1-y_2))
    return l 

#Define a train function which computes the loss function gradient 
# and performs a full batch SGD (manually).
def train_func(model, x, y, learning_rate):
    with tf.GradientTape() as tape: 
        current_loss = loss_func(y, model(x))
    
    grad = tape.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate*grad[0])
    model.b.assign_sub(learning_rate*grad[1])


def report(model, loss):
    return f"W = {model.w.numpy()}, b = {model.b.numpy()}, loss={loss.numpy()}"

#Def training loop 
def train_loop(model, x, y, epochs):
    
    weights = []
    bias = []

    for epoch in epochs:
        train_func(model, x, y, 0.1)
        weights.append(model.w)
        bias.append(model.b)
        loss_value = loss_func(y, model(x))

        print(report(model, loss_value))
    
    return weights, bias
    
    
# Define a custom model using tf.Module inheritance which returns 
# the functional form w * x + b where w and b are tensor variables initialized with random values.
class LinearModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.w = w
        self.b = b
    
    def __call__(self, x): 
        return self.w * x + self.b
    

# My custom model
class MyKerasModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = w
        self.b = b

    def __call__(self, x, training=False):
        return self.w * x + self.b


def main(): 

    # Data generation 
    # Generate predictions of f(x) = 3 * x + 2 for 200 linearly spaced x points between [-2, 2] in single precision.
    x_test = np.linspace(-2., 2., N, dtype=np.float32)
    
    # Include random normal noise (mu=0, sigma=1) to all predictions.
    y_true = truth_model(x_test) + np.random.normal(loc=0.0, scale=1.0, size=N)

    # Prediction with the untrained model 
    linear = LinearModule()
    y_pred = linear(x_test.reshape(N,1))
    
    # Plot data, ground truth model, predictions and loss function for the untrained model.
    plt.figure()
    
    plt.title("Truth model vs Untrained model prediction")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x_test, y_true, 'o', label="$3*x + 2 + \epsilon $")
    plt.plot(x_test, y_pred, 'ro', label="untrained_model prediction")
    plt.legend()

    #Show canvas
    plt.show()

    #Print loss function 
    print("Untrained model loss:", loss_func(y_true, y_pred).numpy())

    weights, bias = train_loop(linear, x_test, y_true, epochs)

    plt.figure()
    plt.plot(epochs, weights, 'o', label="Trained weights")
    plt.plot(epochs, [3.0]*len(epochs), label="True weight")
    plt.plot(epochs, bias, 'o', label="Trained bias")
    plt.plot(epochs, [2.0]*len(epochs), label="True bias")
    plt.legend()
    plt.show()

    plt.figure()
    plt.scatter(x_test, y_true, label="Data")
    plt.plot(x_test, truth_model(x_test), "orange", label="Ground truth")
    plt.plot(x_test, y_pred, "green", label="Untrained predictions")
    plt.plot(x_test, linear(x_test), "red", label="Trained predictions")
    plt.title("Functional API")
    plt.legend()
    plt.show()

    # keras model
    keras_model = MyKerasModel()
    #weights, biases = training_loop(keras_model, x, y, epochs)
    keras_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                        loss=tf.keras.losses.mean_squared_error)
    keras_model.fit(x_test, y_true, epochs=10, batch_size=len(x_test))

    plt.figure()
    plt.scatter(x_test, y_true, label="Data")
    plt.plot(x_test, truth_model(x_test), "orange", label="Ground truth")
    plt.plot(x_test, y_pred, "green", label="Untrained predictions")
    plt.plot(x_test, keras_model(x_test), "red", label="Trained predictions")
    plt.title("Keras Model")
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    main()
