#import relevant libraries

import numpy as np
import tensorflow as tp
import tensorflow_datasets as tfds

#import data 
#Data will be installed in default path C:/User/username/tensorflow_datasets
#as_supervised=True will load data into 2-tuples structure[input,target]
#With_info =True will provides info containing about version features etc
mnist_dataset=tfds.load(name='mnist',with_info=True,as_supervised=True)

#Loading data into training and test data
#By default tensorflow has only test and training data, we can practice splitting of the training data to validation and training data

mnist_train,mnist_test=mnist_dataset['train'],mnist_dataset['test']

#Load 10% of training data as validation data
number_validation_Sample=0.1*mnist_info.splits['train'].num_examples
#in order to get integer value as a output for number_validation_Sample we will cast it witj int64
number_validation_Sample=tf.cast(number_validation_Sample,tf.int64)

#Split test samples
number_test_samples=mnist_info.splits['test'].num_examples
number_test_samples=tf.cast(number_test_samples.tf.int64)

#Write a function to scale data
def scale(image,label):
    image=tf.cast(image,tf.float32)
	image/=255. # This means we need a output in float and output to be 0 and 1
	return image.label
scaled_train_and_validation_data=mnist_train.map(scale)
test_data=mnist_test.map(scale)

#Applying shuffling to avoid patterns in batching
#if data is big then we add buffer size of data to shuffle
#if BUFFER_SIZE=1 then no shuffling will happen
#if BUFFER_SIZE>num_of_samples then shuffling will happen only once
BUFFER_SIZE=10000
#This will shuffle data based on the buffer sizes
shuffled_train_and_validation_data=scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

#Now after shuffling then split validation and train data 
#take(no_of_sample) : will take that chunk of data and put into a variable

validation_data=shuffled_train_and_validation_data.take(number_validation_Sample) 

#This will skip first chunk of specified number of data as it is part of validation data and it will keep the rest of the data
train_data=shuffled_train_and_validation_data.skip(number_validation_Sample)

#Mention the BATCH Size
BATCH_SIZE=100

#Now format train data into different batches
#batch() : Will create batches of data , with input size each
#We dont need batching for validation data
train_data=train_data.batch(BATCH_SIZE)