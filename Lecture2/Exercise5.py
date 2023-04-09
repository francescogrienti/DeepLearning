import matplotlib.pyplot as plt, numpy as np


def func(t):
    return -np.sin(t*t)/t + 0.01 * t*t


def main():
    x = np.linspace(-3.0, 3.0, num=100)
    y = np.array([func(i) for i in x])
    np.savetxt("/home/francescogrienti/DL/DeepLearning/Lecture2/output.dat", np.transpose([x,y]))

    #Plot points
    plt.title("Damping oscillating function")
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.plot(x,y, 'go', scalex=True, linestyle='dashed', label = '$-\\frac{sin(x^2)}{x} + 0.01 * x^2$')  #you can write equations using latex syntax
    plt.xlim([-3,3])
    plt.legend()
    
    
    #Save plot to disk
    plt.savefig("/home/francescogrienti/DL/DeepLearning/Lecture2/output5.png")

    #Show canvas
    plt.show()



if __name__ == "__main__":
    main()
