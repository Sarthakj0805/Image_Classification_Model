import numpy as np
import matplotlib.pyplot as plt
from Data_helper import load_dataset, normalize
from tensorflow.keras.models import load_model
''' to check the model using predictions on the test dataset using the saved model.pb file'''

# load dataset
trainX, trainY, testX, testY = load_dataset()

# normalize data
trainX, testX = normalize(trainX, testX)

#enter any index between 0-10000 for the test image
n = int(input("enter n: "))
img = testX[n]

# reshape into a single sample with 3 channels
img = img.reshape(1, 32, 32, 3)

#enter the path to the saved model folder to load the model
model = load_model('path to saved model.pb folder')

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]
result = model.predict(img)

print('the selected image is: \n')
plt.imshow(testX[n].astype('uint8'))
plt.show()
plt.close()

print("the predicted class is : ", classes[np.argmax(result)], "\n")
print("the correct class is : " , classes[np.argmax(testY[n])], "\n")