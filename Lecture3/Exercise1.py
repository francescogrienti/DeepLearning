import tensorflow as tf 
import numpy as np


def MLP_pred_function(input_tensor, weights, bias):
    return tf.nn.sigmoid(input_tensor @ weights + bias)


def MultiLayerPerceptronFirstModelRandom(n_input, n_hidden_1, n_hidden_2, n_output, starting_input):
    
    # Allocate random normal variables for weight and bias representation of a multi-layer perceptron (MLP) with n_input size, 
    # two hidden layers with n_hidden_1 and n_hidden_2 neurons respectively and n_output size.

    tf.random.set_seed(0)
    # Bias
    bias01 = tf.Variable(tf.random.normal([n_hidden_1]))
    bias12 = tf.Variable(tf.random.normal([n_hidden_2]))
    bias23 = tf.Variable(tf.random.normal([n_output]))

    # Weights
    weight01 = tf.Variable(tf.random.normal([n_input, n_hidden_1]))
    weight12 = tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2]))
    weight23 = tf.Variable(tf.random.normal([n_hidden_2, n_output]))

    # Define a function which takes a tensor as input and returns the MLP prediction. 
    # Use the sigmoid function as activation function for all nodes in the network except for the output layer, which should be linear.

    first_layer = MLP_pred_function(starting_input, weight01, bias01)
    second_layer = MLP_pred_function(first_layer, weight12, bias12)
    output_layer = bias23 + tf.matmul(second_layer, weight23)

    return output_layer

def main():

    x_test = np.linspace(-1, 1, 10, dtype=np.float32).reshape(10, 1)
    print(MultiLayerPerceptronFirstModelRandom(1, 5, 2, 1, x_test))

if __name__ == "__main__":
    main()