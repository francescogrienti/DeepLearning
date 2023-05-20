# Bayesian Optimization
from hyperopt import hp, fmin, tpe, Trials, rand
import hyperopt.pyll.stochastic
import matplotlib.pyplot as plt
import numpy as np

# Fix evals number
max_evals = 2000

"""
FUNCTIONS
"""

def objective_function(x):
    return 0.05*(x**6 - 2*x**5 - 28*x**4 + 28*x**3 + 12*x**2 - 26*x + 100)


def fun_plot(grid_points, fun, label):
    plt.figure()
    plt.plot(grid_points, fun(grid_points), 'go', label=label)
    plt.xlabel('X')
    plt.ylabel('f(X)')
    plt.legend()
    plt.show()    


def hist_plot(sample, title, bins, range): 
    plt.hist(sample, bins=bins, range=range)
    plt.xlabel("Sampled points")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


def minimization(fun, space, algo, title):
    
    # Perform minimization 
    trials = Trials()
    best = fmin(fun, space, algo=algo, max_evals=max_evals, trials=trials)
    print('Best x-value:', best)

    # Scatterplot iteration vs x-value
    plt.figure()
    plt.scatter(trials.idxs_vals[0]['x'], trials.idxs_vals[1]['x'])
    plt.axhline(best['x'], color='red')
    plt.xlabel('Iteration')
    plt.ylabel('X-value')
    plt.title(title)
    plt.show()

    # Histogram of x-values sampled
    hist_plot(trials.idxs_vals[1]['x'], 'X-values sampled', 100, (-5., 6.))

"""
MAIN 
"""

def main():

    x = np.linspace(-5., 6., 100)
    fun_plot(x, objective_function, '$ f(x) = 0.05(x^6 - 2x^5 - 28x^4 + 28x^3 + 12x^2 - 26x + 100)$')
    space = hp.uniform('x', -5., 6.)
    
    # Histogram plot of sampled points from search space
    hist_plot([hyperopt.pyll.stochastic.sample(space) for _ in range(1000)], 'sampled points', 100, (-5., 6))
   
    # TPE minimization
    minimization(objective_function, space, tpe.suggest, 'TPE')

    # Random minimization 
    minimization(objective_function, space, rand.suggest, 'Random')


if __name__ == "__main__":
    main()