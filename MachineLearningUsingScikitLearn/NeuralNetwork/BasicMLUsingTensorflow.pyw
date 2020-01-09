#Import Relevant Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Create a datasets
x_axis=np.random.uniform(-10,10,(1000,1))#this will create a matrix of 1000x1 with random number ranging from -10 to 10
z_axis=np.random.uniform(-10,10,(1000,1))#this will create a matrix of 1000x1 with random number ranging from -10 to 10

#As each data contains noise so 
noise =x_axis=np.random.uniform(-1,1,(1000,1))#this will create a matrix of 1000x1 with random number ranging from -1 to 1
#Create a matrix for inputs
generated_inputs=np.column_stack((x_axis,z_axis))
#Our Targetted Linear Equation
targets= 2*x_axis - 3*z_axis + 5 + noise

#Save data into file with .pyz format
np.savez('Tf_Intro.pyz',inputs =generated_inputs, outputs=targets)

#Load the training data
training_data=np.load('Tf_Intro.pyz')
# Now create a model
model=tf.keras.Sequential(tf.keras.layers.Dense(1))# will create a targetted equation y=mx+c

#compile the model using the requred optimizer and loss function
model.compile(optimizer='sgd',loss='mean_squared_error')

#model  fits with data
#epochs means iteration of full datasets
#verbose 0: hide the processing
######### 1: show the processing
model.fit(training_data['inputs'],training_data['outputs'],epochs=100,verbose=0)

#check the weights nw
weights=model.layers[0].get_weights()[0]
weights
#check bias now
bias=model.layers[0].get_weights()[1]
bias
#Make the prediction now
model.predict_on_batch(training_data['inputs'])

#Plot the line .to see whether it make a 45% line
plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])),np.squeeze(training_data['outputs']))