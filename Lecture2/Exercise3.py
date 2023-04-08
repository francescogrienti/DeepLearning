import matplotlib.pyplot as plt, math, numpy as np


def func(x):
    return math.exp(-x) * math.cos(2*math.pi*x)

def main():
    x = np.linspace(0, 2, num=1000)
    y = np.array([func(i) for i in x]) 

    plt.title("Damped function")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    main()