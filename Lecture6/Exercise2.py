"""
RNN
"""

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 


"""
DATA LOADING 
"""

train_set = np.load('./training_data/training_data.npy')
train_label = np.load('./training_data/training_label.npy')

test_set = np.load('./test_data/test_data.npy')
test_label = np.load('./test_data/test_label.npy')


"""
LSTM MODEL 
"""

def lstm_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(30, activation='relu', input_shape=(train_set.shape[1],1)))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model 

"""
MAIN 
"""

def main():
    model = lstm_model()
    model.fit(train_set, train_label, batch_size=32, epochs=25)
    test_loss, test_acc = model.evaluate(test_set, test_label)
    print("MSE on test mode:", test_loss)
    print("Accuracy on test mode:", test_acc)

    y_pred = np.array(model.predict(test_set))

    # Plots
    plt.figure()
    plt.plot(train_label, color='purple')
    plt.xlabel("Day")
    plt.ylabel("Temperature")
    plt.title("Training data")
    plt.show()

    plt.figure()
    plt.plot(test_label, color='red', label="True value")
    plt.plot(y_pred, color='green', label="Prediction")
    plt.xlabel("Day")
    plt.ylabel("Temperature")
    plt.legend()
    plt.title("Full test set comparison true value-prediction")
    plt.show()


    plt.figure()
    plt.plot(test_label[0:100], color='red', label="True value")
    plt.plot(y_pred[0:100], color='green', label="Prediction")
    plt.xlabel("Day")
    plt.ylabel("Temperature")
    plt.legend()
    plt.title("First 100 days true value-prediction comparison")
    plt.show()

    plt.figure()
    plt.plot(test_label-y_pred, color='k')
    plt.ylabel("Residual")
    plt.xlabel("Day")
    plt.title("Residual plot")
    plt.show()

    plt.figure()
    plt.scatter(y_pred, test_label, s=2, color='black')
    plt.ylabel("Y true")
    plt.xlabel("Y predicted")
    plt.title("Scatter plot")
    plt.show()



if __name__ == "__main__":
    main()





