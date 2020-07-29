import matplotlib.pyplot as plt
import seaborn as sns
import sys

# plot the learning curves for loss and accuracy metrics
def summary_plots(history):
	sns.set(context='notebook', palette='bright', style='dark')
	# plot loss
	plt.subplot(211)
	plt.title('Cross Entropy Loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.plot(history.history['loss'], color='green', label='train')
	plt.plot(history.history['val_loss'], color='red', label='test')
	# plot accuracy
	plt.subplot(212)
	plt.title('Classification Accuracy')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.plot(history.history['accuracy'], color='green', label='train')
	plt.plot(history.history['val_accuracy'], color='red', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	plt.savefig(filename + '_plot.png')
	plt.show()
	plt.close()