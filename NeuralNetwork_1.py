import tensorflow as tf
import pandas as pd
import numpy as np
import FeatureScaling
import matplotlib.pyplot as plt

path = r'C:\Users\h8611\OneDrive\Desktop\MLS Slides\HousePrice\house.csv'  # Location of the dataset and read it
dataset = pd.read_csv(path)


# As usual, use
x_train = np.array(dataset.iloc[0:500, 0:6])   # consider the first 6 columns as features of a single training example
x_train = FeatureScaling.meanNormal(x_train)
y_train = np.array(dataset.iloc[0:500, -1])  # The last column is the target

x_test = np.array(dataset.iloc[501:-1, 0:6])   # consider the first 6 columns as features of a single test example
x_test = FeatureScaling.meanNormal(x_test)
y_test = np.array(dataset.iloc[501:-1, -1])  # The last column is the target


#  Define the model.
#  Units means number of neuros
#  activation means a "sub model" that evaluate a training examples
#  "Sub models" frequently used are relu, sigmoid.
#  Where relu has a fast processing speed， sigmoid used to evaluate probability
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units = 10, activation='relu'),
    tf.keras.layers.Dense(units = 1, activation='relu')
])


#  Compile trains the model
#  optimizer = adam means utilize adam as the algorithmn to optimize the model's performance
#  According to lecture, adam has dynamic learning rate. Start at a large learning rate and decrease gradually. That helps saving time
#  loss = MeanSquaredError refer to the classic loss function which is MSE = sum(i=1, m)[(f_wb(x_i) - y_i)]^2, m is total training examples
model.compile(optimizer = 'adam',
              loss = 'MeanSquaredError')

# model.fit start to iterate and optimize the model, epochs = total iterations
fiter = model.fit(x_train, y_train, epochs=100)


# Use the test set to evalute the performance
# Notice, my training and test set is not randomly choose from dataset
# This is for the purpose of utilize model.evaluate function.
evale = model.evaluate(x_test, y_test)
print(fiter, "fiter")
print(evale,"eval")
