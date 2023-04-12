import numpy as np 
from numpy import polynomial
import matplotlib.pyplot as plt


def true_function(t):
    return np.cos(1.5*np.pi*t)


def main():
    x = np.random.rand(100)
    y = true_function(x)+np.random.rand() * 0.1

    #Polynomial fit with degree = 1
    param_1 = np.polyfit(x, y, 1)
    #Construct the polynomial function
    fit_1 = np.poly1d(param_1)

    #Polynomial fit with degree = 4
    param_4 = np.polyfit(x, y, 4)
    #Construct the polynomial function
    fit_4 = np.poly1d(param_4)

    #Polynomial fit with degree = 15
    param_15 = np.polyfit(x, y, 15)
    #Construct the polynomial function
    fit_15 = np.poly1d(param_15)

    #Plots
    plt.plot(x, y, 'bo')
    plt.plot(x, fit_1(x), 'ro')
    plt.plot(x, fit_4(x), 'go')
    plt.plot(x, fit_15(x), 'yo')
    plt.show()



if __name__ == "__main__":
    main()


#Comments
#Use a x-test set to test the polynomial fits
#Use a for-loop to write cleaner code