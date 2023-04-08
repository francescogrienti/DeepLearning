import matplotlib.pyplot as plt, numpy as np 


def main():
    imported_data = np.loadtxt("data4.dat")
    x = imported_data[:, 0]
    y = imported_data[:, 1]

    #Scatter plot
    plt.title("Charged particles")
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.scatter(x, y, c="green")
    
    #Save plot to disk
    plt.savefig("/home/francescogrienti/DL/DeepLearning/Lecture2/output5.png")

    #Show canvas
    plt.show()




if __name__ == "__main__":
    main()