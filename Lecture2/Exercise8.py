#Exercise 7 task using TensorFlow primitives

import tensorflow as tf, numpy as np 
import math
import matplotlib.pyplot as plt
import scipy.optimize
import time 

def true_fun(t):
    return tf.cos(1.5*tf.constant(math.pi*t))


def main():
    
    n_samples = 30
    np.random.seed(0)
    x = np.sort(np.random.rand(n_samples))
    y = true_fun(x) + np.random.randn(n_samples) * 0.1
    x_test = np.linspace(0, 1, 100)

    plt.figure()
    degrees = [1, 4, 15]

    def loss(p, func):
        ypred = func(list(p), x)
        return tf.reduce_mean(tf.square(ypred - y)).numpy()

    for degree in degrees:
        s_t = time.time()
        res = scipy.optimize.minimize(loss, np.zeros(degree+1), args=(tf.math.polyval), method='BFGS')
        e_t = time.time()
        print("Execution TensorFlow computation:", s_t-e_t, "seconds")
        
        plt.plot(x_test, np.poly1d(res.x)(x_test), label=f"Poly degree={degree}")
        plt.plot(x_test, true_fun(x_test), label="True function")
        plt.scatter(x, y, color='b', label="Samples")
        plt.title("TensorFlow")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim([0,1])
        plt.ylim([-2,2])
        plt.legend()

        plt.show()


if __name__ == "__main__":
    main()