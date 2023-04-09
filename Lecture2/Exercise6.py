import pandas as pd 
import numpy as np

def main():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names,
                            na_values='?', comment='\t',
                            sep=' ', skipinitialspace=True)


    #Mean values of each column
    print("Mean values:") 
    print("\n")
    print(raw_dataset.mean(axis=0))


    #Filter results by selecting only entries where the number of cylinders is equal to 3.
    print("\nFilter by cylinders == 3")
    print("\n")
    print(raw_dataset.loc[raw_dataset['Cylinders'] == 3])


if __name__ == "__main__":
    main()
