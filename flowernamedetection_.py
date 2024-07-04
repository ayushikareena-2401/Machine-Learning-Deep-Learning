# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

((x_Train,y_Train) , (x_Test,y_Test)) = keras.datasets.mnist.load_data()

len(x_Train)

len(x_Test)

len(y_Train)

len(y_Test)

x_Test[1]

plt.matshow(x_Train[9])
plt.show()#TO REMOVE THE EXTRA COMMENT OVER THE IMAGE

x_Train[9]

#We Want to flatten the array that is we need to covert the 2d array to 1d array, we will use the reshape method
x_Train.shape

x_Train=x_Train/255
x_Test=x_Test/255
#Now the values will be in between the range 0 to 1

x_Train_flattened=x_Train.reshape(60000,28*28)
x_Test_flattened=x_Test.reshape(10000,28*28)

#Now both the required array are flattened to fed as an input to the model
x_Train_flattened.shape
x_Test_flattened.shape

#now we are making the neural network and storing it in the variable by name model
model=keras.Sequential([keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')])

model.compile(optimizer='adam',
              loss ='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_Train_flattened,y_Train,epochs=5)

model.evaluate(x_Train_flattened,y_Train)
