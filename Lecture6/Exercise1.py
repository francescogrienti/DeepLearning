from sklearn.datasets import load_iris
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorboard
"""
DATA LOADING AND MANIPULATION
"""

iris = load_iris()
# Pandas DataFrame
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
iris_df['label'] = iris['target']

# One-hot encoding 
last_indeces = []
for i in range(len(iris['target'])-1):
    if iris['target'][i] != iris['target'][i+1]:
        last_indeces.append(i)

iris_df.drop(['label'], axis=1, inplace=True)
iris_df['label_setosa'] = [0 for i in range(len(iris['target']))]
iris_df['label_versicolor'] = [0 for i in range(len(iris['target']))]
iris_df['label_virginica'] = [0 for i in range(len(iris['target']))]

iris_df.loc[[i for i in range(0,last_indeces[0]+1)], 'label_setosa'] = 1
iris_df.loc[[i for i in range(last_indeces[0]+1,last_indeces[1]+1)], 'label_versicolor'] = 1
iris_df.loc[[i for i in range(last_indeces[1]+1,len(iris['target']))], 'label_virginica'] = 1

# Extract 80% of the data for training and keep 20% for test, using the DataFrame.sample
training_df = iris_df.sample(frac=0.8, random_state=1)
test_df = iris_df.drop(training_df.index)

# Creating X and y
X_train = training_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

# Convert DataFrame into np array
y_train = training_df[['label_setosa', 'label_versicolor', 'label_virginica']]

"""
NN_MODEL
"""

def image_classifier():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_dim=(4)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model 


"""
MAIN
"""

def main():
    flower_model = image_classifier()
    history = flower_model.fit(X_train, 
                     y_train,  
                     batch_size=32,
                     epochs=200, 
                     validation_split=0.4,
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                     tf.keras.callbacks.TensorBoard(log_dir='./log')]
                     )
 
    # Plot the learning curves (loss vs epochs) for the training and validation datasets.
    x = range(1, len(history.history['loss'])+1)
    yt = history.history['loss']
    yv = history.history['val_loss']
    plt.plot(x, yt, label="Training loss")
    plt.plot(x, yv, label="Validation loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

