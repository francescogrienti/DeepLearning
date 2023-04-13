import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

def true_fun(x):
    return np.cos(1.5 * np.pi * x)


def main():

    n_samples = 30
    np.random.seed(0)
    x = np.sort(np.random.rand(n_samples))
    y = true_fun(x) + np.random.randn(n_samples) * 0.1
    x_test = np.linspace(0, 1, 100)
    degrees = [1, 4, 15]

    # Mode 3 - scipy minimize
    ax = plt.subplot(1, len(degrees), 3)

    def loss(p, func):
        ypred = func(p)
        return np.mean(np.square(ypred(x) - y))

    for degree in degrees:
        res = minimize(loss, np.zeros(degree+1), args=(np.poly1d), method='BFGS')
        plt.plot(x_test, np.poly1d(res.x)(x_test), label=f"Poly degree={degree}")

    plt.plot(x_test, true_fun(x_test), label="True function")
    plt.scatter(x, y, color='b', label="Samples")
    plt.title("Scipy.minimize")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([0,1])
    plt.ylim([-2,2])
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()