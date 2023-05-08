# Bayesian Optimization

from hyperopt import hp, fmin, tpe, Trials, rand
import hyperopt.pyll.stochastic
import matplotlib.pyplot as plt
import numpy as np

# TODO refactoring code!!!

max_evals = 2000

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
    plt.title(title)
    plt.show()


def main():

    x = np.linspace(-5., 6., 100)
    sample = []
    fun_plot(x, objective_function, '$ f(x) = 0.05(x^6 - 2x^5 - 28x^4 + 28x^3 + 12x^2 - 26x + 100)$')
    space = hp.uniform('x', -5., -2.)
    
    for i in range(1000):
        sample.append(hyperopt.pyll.stochastic.sample(space))
    
    sample = np.array(sample)

    # Histogram plot of sampled points
    hist_plot(sample, 'sampled points', 50, (-5., -2))
   
    # Perform minimization 
    trials = Trials()
    best = fmin(objective_function, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print('Best x-value:', best)

    # Scatterplot iteration vs x-value
    x_values = []
    for i in range(max_evals):
        x_values.append(trials.trials[i]['misc']['vals']['x'])
    x_values = np.array(x_values)
    iteration = np.array([i for i in range(max_evals)])
    plt.figure()
    plt.scatter(iteration, x_values)
    plt.xlabel('Iteration')
    plt.ylabel('X-value')
    plt.title('iteration vs x-value')
    plt.show()

    # Histogram of x-values extracted 
    hist_plot(x_values, 'X-values sampled', 50, (-5., -2.))

    # Perform minimization 
    trials = Trials()
    best = fmin(objective_function, space, algo=rand.suggest, max_evals=max_evals, trials=trials)
    print('Best x-value:', best)

    # Scatterplot iteration vs x-value
    x_values = []
    for i in range(max_evals):
        x_values.append(trials.trials[i]['misc']['vals']['x'])
    x_values = np.array(x_values)
    iteration = np.array([i for i in range(max_evals)])
    plt.figure()
    plt.scatter(iteration, x_values)
    plt.xlabel('Iteration')
    plt.ylabel('X-value')
    plt.title('iteration vs x-value')
    plt.show()

    # Histogram of x-values extracted 
    hist_plot(x_values, 'X-values sampled', 50, (-5., -2.))



if __name__ == "__main__":
    main()