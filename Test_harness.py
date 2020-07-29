from Data_helper import load_dataset, normalize
from def_model import define_model
from Summary_plots import summary_plots
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
# run the test harness for evaluating a model
def run_model():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# normalize data
	trainX, testX = normalize(trainX, testX)
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	# prepare iterator
	it_train = datagen.flow(trainX, trainY, batch_size=64)
	# fit model
	# steps = int(trainX.shape[0] / 128)
	history = model.fit(it_train, steps_per_epoch=300, epochs=100, validation_data=(testX, testY), verbose=1)
	# evaluate model
	_, acc = model.evaluate(testX, testY, verbose=1)
	print('\n \n \n')
	print('> %.3f' % (acc * 100.0), "\n \n \n")
	# learning curves
	summary_plots(history)

  # save model
	# uncomment to save the model
	# model.save('path to folder')

# entry point, run the test harness
run_model()