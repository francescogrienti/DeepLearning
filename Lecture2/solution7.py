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

  plt.figure(figsize=(14, 5))
  degrees = [1, 4, 15]
  ax = plt.subplot(1, len(degrees), 1)

  # Mode 1 - using least squares
  for degree in degrees:
    p = np.polyfit(x, y, degree)
    z = np.poly1d(p)
    plt.plot(x_test, z(x_test), label=f"Poly degree={degree}")
    plt.plot(x_test, true_fun(x_test), label="True function")
    plt.scatter(x, y, color='b', label="Samples")
    plt.title("Polyfit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([0,1])
    plt.ylim([-2,2])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()