import tensorflow
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	print('yo')
	return trainX, trainY, testX, testY

# summarize loaded dataset
def summarize_data():
  trainX, trainY, testX, testY = load_dataset()	
  print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
  print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
  # plot first few images
  for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(trainX[i])
  # show the figure
  plt.show()
  plt.close()

  # normalize data
def normalize(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm - train_norm.min() / train_norm.max() - train_norm.min() 
	test_norm = test_norm - test_norm.min() / test_norm.max() - test_norm.min()
	# return normalized images
	return train_norm, test_norm

load_dataset()